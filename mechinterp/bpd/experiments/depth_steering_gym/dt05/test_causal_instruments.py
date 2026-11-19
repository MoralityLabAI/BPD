"""Deterministic regressions for DT05 causal instruments on tiny engineering fixtures."""
from __future__ import annotations

import unittest

import torch

from mechinterp.bpd.adapters.trm import build_mini_trm
from ..dt03_causal import DualState, flat_observer_jacobian, native_resume_recipient, select_full_coordinate_block
from ..dt04.causal import build_dual_protected_basis, extract_dual_state_at_depth, sudoku4_task_probs
from ..dt04.tasks import encode_sudoku_cases
from ..worlds import sudoku4
from .causal_instruments import (
    TransportInterventionProtocol,
    batched_observer_jacobian_vjp,
    benchmark_autonomous_routing,
    evaluate_transport_protocol_with_evalmath,
    multistep_transport_audit,
    probe_depth_causal_rank,
    run_transport_vs_static_unit,
    select_protected_task_gramian_vjp,
    topk_set_overlap_fraction,
)
from .resources import BoundedTensorCache, MemoryBudget
from .techniques import principal_angles_deg
from .training import fit_online_factors


class CausalInstrumentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.model = build_mini_trm(
            hidden_size=16, num_heads=2, l_layers=1,
            h_cycles=1, l_cycles=1, puzzle_emb_len=1,
            seq_len=16, seed=2605950,
        ).eval()
        cases = [sudoku4(860_600 + i, "engineering", blanks=4 + (i % 3)) for i in range(10)]
        cls.tokens, cls.gold = encode_sudoku_cases(cases, "cpu")
        cls.budget = MemoryBudget(max_host_rss_gib=12.0, min_host_available_gib=0.1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_chunked_vjp_jacobian_and_gramian_match_dt03_reference(self):
        cal_tok = self.tokens[:3]
        dual = extract_dual_state_at_depth(self.model, cal_tok, 1)
        state_shape = tuple(dual.z_h.shape[1:])
        packed = dual.flatten()
        emb = self.model.encode_inputs(cal_tok).detach()

        def ref_obs(s_batch: torch.Tensor) -> torch.Tensor:
            d_obj = DualState.unflatten(s_batch, state_shape)
            return sudoku4_task_probs(native_resume_recipient(self.model, cal_tok, d_obj, 2))

        def single_obs(s_1d: torch.Tensor, e_1d: torch.Tensor) -> torch.Tensor:
            t_len, d_hid = state_shape
            pair = s_1d.reshape(2, t_len, d_hid)
            zh, zl = pair[0].unsqueeze(0), pair[1].unsqueeze(0)
            for _ in range(2):
                zh, zl = self.model.act_step(zh, zl, e_1d.unsqueeze(0))
            return torch.softmax(self.model.digit_logits(zh)[0, :, :4].float(), dim=-1)

        j_ref = flat_observer_jacobian(ref_obs, packed)
        j_vjp = batched_observer_jacobian_vjp(single_obs, packed, emb, chunk_size=11)
        torch.testing.assert_close(j_vjp, j_ref, rtol=1e-5, atol=1e-6)

        # Verify matrix-free protected task Gramian matches dense build_dual_protected_basis + select_full_coordinate_block
        prot_dense, _ = build_dual_protected_basis(self.model)
        ref_cont = lambda p_s, off: native_resume_recipient(self.model, cal_tok, DualState.unflatten(p_s, state_shape), off)
        b_ref, m_ref = select_full_coordinate_block(
            ref_cont, sudoku4_task_probs, packed, prot_dense, (1, 3), rank=4,
        )
        b_vjp, m_vjp = select_protected_task_gramian_vjp(
            self.model, cal_tok, packed, fit_offsets=(1, 3), rank=4, chunk_size=16,
        )
        self.assertFalse(m_vjp["dense_state_gramian_allocated"])
        torch.testing.assert_close(
            torch.tensor(m_vjp["top_eigenvalues"][:4], dtype=torch.float64),
            torch.tensor(m_ref["eigenvalues"][:4], dtype=torch.float64),
            rtol=1e-4, atol=1e-7,
        )
        overlap = principal_angles_deg(b_ref.double(), b_vjp.double())["subspace_overlap"]
        self.assertGreater(overlap, 0.9999)

    def test_multistep_transport_audit_and_linearization_convergence(self):
        audit = multistep_transport_audit(
            self.model, self.tokens[:1], calib_tokens=self.tokens[:2],
            basis_mode="task_gramian", rank=3, start_depth=1, num_steps=3,
            perturbation_doses=(1e-4, 1e-1),
        )
        self.assertFalse(audit["dense_state_jacobian_allocated"])
        self.assertEqual(len(audit["steps"]), 3)
        step1 = audit["steps"][0]["finite_perturbation_vs_jacobian"]
        self.assertLess(step1["0.0001"]["relative_linearization_error"], step1["0.1"]["relative_linearization_error"])
        self.assertGreater(step1["0.0001"]["cosine_alignment"], 0.999)

    def test_transport_vs_static_unit_and_evalmath_preflight(self):
        proto = TransportInterventionProtocol(
            selection_depth=1, patch_depth=2, fit_offsets_from_selection=(1, 3),
            eval_total_depths=(4, 8), subspace_rank=3, n_shams=2,
        )
        unit1 = run_transport_vs_static_unit(
            self.model, self.tokens[:2], self.tokens[2:5], self.gold[2:5],
            unit_id="eng_unit_1", seed=9101, protocol=proto,
        )
        unit2 = run_transport_vs_static_unit(
            self.model, self.tokens[1:3], self.tokens[5:8], self.gold[5:8],
            unit_id="eng_unit_2", seed=9102, protocol=proto,
        )
        self.assertLess(unit1["observer_max_abs_error"], proto.observer_tolerance)
        d8 = unit1["by_depth"]["8"]
        self.assertAlmostEqual(
            d8["transported_task_basis"]["delivered_l2_dose_mean"],
            d8["static_task_basis_dose_matched"]["delivered_l2_dose_mean"],
            places=5,
        )
        for cert in unit1["signed_sham_certificates"]:
            self.assertLess(cert["signed_full_task_jacobian_max_error"], 1e-4)
            self.assertLess(cert["delivered_l2_error"], 1e-4)

        pf = evaluate_transport_protocol_with_evalmath([unit1, unit2], protocol=proto, n_eval_boards_per_unit=3)
        self.assertEqual(pf["evalmath_result"]["schema"], "EvalMath/mechinterp-power-preflight/v1")
        self.assertEqual(pf["evalmath_result"]["decision"], "REDESIGN_OR_PILOT_ONLY")

    def test_topk_overlap_uses_unordered_set_membership(self):
        # Previously a positionwise same-slot check produced 0 for permutations.
        online = torch.tensor([[[0, 1, 2, 3], [3, 2, 1, 0]]])
        same_set = torch.tensor([[[3, 2, 1, 0], [0, 1, 2, 3]]])
        self.assertEqual(topk_set_overlap_fraction(online, same_set), 1.0)

        half_shared = torch.tensor([[[0, 1, 4, 5], [4, 5, 2, 3]]])
        self.assertEqual(topk_set_overlap_fraction(online, half_shared), 0.5)

        disjoint = torch.tensor([[[4, 5, 6, 7], [4, 5, 6, 7]]])
        self.assertEqual(topk_set_overlap_fraction(online, disjoint), 0.0)

        with self.assertRaises(ValueError):
            topk_set_overlap_fraction(online, online[..., :3])
        with self.assertRaises(ValueError):
            topk_set_overlap_fraction(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))

    def test_autonomous_routing_benchmark_shares_vu_and_tracks_teacher_fidelity(self):
        cache = BoundedTensorCache(4 * 1024**2)
        student, _ = fit_online_factors(
            self.model, self.tokens[:4], steps=1, components=6, top_k=3,
            logical_batch=4, microbatch=2, seed=77, budget=self.budget, cache=cache,
        )
        bench = benchmark_autonomous_routing(
            self.model, student, self.tokens[4:8], self.gold[4:8],
            depths=(2, 4), microbatch=2, seed=77, budget=self.budget,
        )
        self.assertTrue(bench["shared_vu_verified"])
        self.assertEqual(set(bench["arms"]), {"online", "teacher_scheduled", "fixed", "random"})
        for arm in bench["arms"].values():
            self.assertAlmostEqual(arm["active_component_telemetry"]["mean_active_per_token"], 3.0)

    def test_depth_causal_rank_separates_three_dimensions(self):
        report = probe_depth_causal_rank(
            self.model, self.tokens[:6], self.gold[:6], self.tokens[6:10], self.gold[6:10],
            probe_depths=(1, 2), horizon_depth=4, intervention_ranks=(1, 2, 4), n_gramian_cases=3,
        )
        self.assertEqual(set(report["by_depth"]), {"1", "2"})
        d1 = report["by_depth"]["1"]
        self.assertGreater(d1["statistical_information_dimension"]["full_dual_state"]["participation_ratio"], 0.0)
        self.assertIn("2", d1["linearly_decodable_dimension"]["full_dual_state"]["heldout_by_rank"])
        self.assertIn("2", d1["native_future_causal_dimension"]["interventions_by_rank"])


if __name__ == "__main__":
    unittest.main()
