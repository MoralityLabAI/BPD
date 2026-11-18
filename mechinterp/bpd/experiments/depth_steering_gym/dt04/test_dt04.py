"""Deterministic source and contract tests for BPD-DT04.

Uses synthetic or toy fixtures only; never touches `DEV_ROOTS` (`2604000..2604058`),
never opens `ARG-03-N`, and verifies all historical DT01/DT02/DT03 locks remain intact.
"""
from __future__ import annotations

import unittest
import torch

from mechinterp.bpd.adapters.trm import build_mini_trm

from . import DEV_ROOTS, SEALED_CONFIRM_ROOTS, SMOKE_ROOT
from .causal import (
    build_dual_protected_basis,
    corrupt_in_protected_kernel,
    extract_dual_state_at_depth,
    full_vocab_observer,
)
from .memory_guard import host_memory_bytes, memory_snapshot
from .runner import guard_root
from .tasks import (
    _encode_story_public,
    audit_arc_color_support,
    build_root_sudoku_splits,
    encode_sudoku_cases,
)
from .verify_source import verify_historical_locks
from ..worlds import HiddenEndingWorld


class DT04ContractTests(unittest.TestCase):
    def test_historical_locks_untouched(self):
        res = verify_historical_locks()
        self.assertTrue(res["dt02_archive_verified"])
        self.assertGreaterEqual(res["historical_files_checked"], 12)

    def test_root_guard_blocks_sealed_and_foreign_roots(self):
        guard_root(SMOKE_ROOT)
        guard_root(DEV_ROOTS[0])
        guard_root(DEV_ROOTS[-1])
        self.assertEqual(len(DEV_ROOTS), 59)
        self.assertTrue(set(DEV_ROOTS).isdisjoint(SEALED_CONFIRM_ROOTS))
        for bad in (901, 902, SEALED_CONFIRM_ROOTS[0], -1, 2604059):
            with self.assertRaises(ValueError):
                guard_root(bad)

    def test_canonical_sudoku_split_has_zero_isomorphic_leakage(self):
        sp = build_root_sudoku_splits(SMOKE_ROOT)
        self.assertTrue(sp["audit_train_vs_eval"]["admissible"])
        self.assertEqual(sp["audit_train_vs_eval"]["cross_split_isomorphic_items"], 0)
        self.assertTrue(sp["audit_train_vs_calib"]["admissible"])
        self.assertTrue(sp["audit_calib_vs_eval"]["admissible"])
        self.assertEqual(len(sp["eval_cases"]), 64)
        self.assertEqual(len(sp["calib_cases"]), 16)

    def test_arc_color_support_audit_flags_missing_color3(self):
        tr = torch.tensor([[0, 1, 2, 1]])
        ho = torch.tensor([[0, 3, 2, 3]])
        rep = audit_arc_color_support(tr, ho)
        self.assertEqual(rep["unsupported_heldout_colors"], [3])
        self.assertFalse(rep["full_4color_support_in_train"])
        rep_ok = audit_arc_color_support(torch.tensor([[0, 1, 2, 3]]), ho)
        self.assertTrue(rep_ok["full_4color_support_in_train"])

    def test_story_public_encoder_rejects_secret_leak(self):
        w = HiddenEndingWorld(12345, "ui", horizon=12)
        obs = w.observe()
        toks = _encode_story_public(obs, 12)
        self.assertEqual(len(toks), 16)
        self.assertTrue(all(1 <= t <= 10 for t in toks))
        with self.assertRaises(AssertionError):
            _encode_story_public({**obs, "secret_cause": 1}, 12)

    def test_dual_protected_basis_conserves_all_11_vocab_logits_and_prefix(self):
        model = build_mini_trm(
            hidden_size=32, num_heads=4, l_layers=1, h_cycles=1, l_cycles=1, puzzle_emb_len=2, seq_len=16, seed=7
        )
        q, meta = build_dual_protected_basis(model)
        self.assertLess(meta["orthogonality_error"], 1e-6)
        sp = build_root_sudoku_splits(SMOKE_ROOT)
        x, _ = encode_sudoku_cases(sp["eval_cases"][:4], "cpu")
        dual = extract_dual_state_at_depth(model, x, 1)
        shape = tuple(dual.z_h.shape[1:])
        packed = dual.flatten()
        corrupted = corrupt_in_protected_kernel(packed, q, rel_dose=0.9, seed=19)
        p_clean = full_vocab_observer(model, packed, shape)
        p_corr = full_vocab_observer(model, corrupted, shape)
        self.assertLess(float((p_clean - p_corr).abs().max().item()), 1e-6)
        self.assertGreater(float((corrupted - packed).norm().item()), 0.1)

    def test_memory_guard_telemetry(self):
        mem = host_memory_bytes()
        self.assertGreaterEqual(mem["rss_bytes"], 0)
        snap = memory_snapshot("cpu")
        self.assertIn("host_rss_mib", snap)


if __name__ == "__main__":
    unittest.main()
