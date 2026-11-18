"""BPD-DT04 complete-state causal steering, BPD (V, U) refits, gauge checks, and shams.

Implements the full DT01 + DT03 causal contract on competent native PureSudokuTRM roots:
1. Complete `DualState(z_h, z_l)` packing with puzzle-prefix lock and exact 11-vocab
   normalized softmax readout protection across all 16 board cells (`ker(lm_head)` on z_h,
   full unheaded board fiber on z_l).
2. Train-depth-only (`depths 2, 4, 6`, patched at `t0 = 1` with `delta_d = (1, 3, 5)`)
   full-coordinate physical task Gramian via `dt03_causal.select_full_coordinate_block`.
3. Actual `BPDModel / TiedComponentLinear` physical `(V, U)` factorization trained strictly
   on supervised depths `<= 6` (`delta_mask = 0`, sparse top-k routing), invertible GL(C)
   factor-gauge invariance verification, RMS-matched Haar read-rotation null, and physical
   output-write subspace extraction.
4. Privileged signed full-task-Jacobian and L2-dose matched shams (`8` per root) verified
   casewise by `dt03_causal.validate_delivered_sham`, plus left output-weight SVD and
   random protected-subspace controls.
5. Same-case evaluation at supervised depth `6` and unseen depths `8, 12` (`delta_d = 5, 7, 11`),
   recording whole-case donor recovery, independent gold board exactness, blank accuracy,
   wrong-to-correct cell gains, correct-to-wrong corruptions, principal-angle subspace drift,
   and one-step recurrent commutator / closure defect.
"""
from __future__ import annotations

import copy
import math
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.nulls import sample_per_pass_rms_matched_rotations
from mechinterp.loopscope02.scripts.trm_sudoku import PureSudokuTRM

from ..dt03_causal import (
    DualState,
    flat_observer_jacobian,
    native_resume_recipient,
    output_weight_basis,
    select_full_coordinate_block,
    validate_delivered_sham,
    whole_case_recovery,
)
from ..dt03_tasks import blank_accuracy
from ..steering import signed_task_matched_sham

PATCH_DEPTH = 1
FIT_OFFSETS = (1, 3, 5)       # Total ACT depths: (2, 4, 6) -- strictly supervised
EVAL_OFFSETS = (5, 7, 11)     # Total ACT depths: (6, 8, 12) -- same-case fit-boundary + unseen
SUBSPACE_RANK = 8
N_SHAMS = 8
DONOR_PROB_TOL = 0.10
OBSERVER_TOL = 1e-6


def build_dual_protected_basis(model: PureSudokuTRM) -> tuple[Tensor, dict]:
    """Construct exact orthonormal basis Q [2*T*D, K] for ker(DP) with locked prefix."""
    dev = model.embed_tokens.weight.device
    t_len = int(model.total_len)
    puz = int(model.puzzle_emb_len)
    seq = int(model.seq_len)
    d = int(model.hidden_size)
    half = t_len * d
    full_dim = 2 * half

    w_head = model.lm_head.weight.detach().double().cpu()  # [11, D]
    _u, s, vh = torch.linalg.svd(w_head, full_matrices=True)
    rank_head = int((s > 1e-8 * max(1.0, float(s.max().item()))).sum().item())
    ker_cell = vh[rank_head:].T.contiguous()  # [D, D - rank_head]
    k_h_cell = int(ker_cell.shape[1])
    if k_h_cell < 1:
        raise RuntimeError("lm_head has trivial kernel; cannot form dark z_H subspace")

    k_h_total = seq * k_h_cell
    k_l_total = seq * d
    k_total = k_h_total + k_l_total

    q = torch.zeros((full_dim, k_total), dtype=torch.float64)
    col = 0
    # Board z_H dark cells (prefix 0..puz-1 stays identically zero)
    for cell in range(seq):
        row0 = (puz + cell) * d
        q[row0 : row0 + d, col : col + k_h_cell] = ker_cell
        col += k_h_cell
    # Board z_L unheaded cells (prefix 0..puz-1 stays identically zero)
    eye_d = torch.eye(d, dtype=torch.float64)
    for cell in range(seq):
        row0 = half + (puz + cell) * d
        q[row0 : row0 + d, col : col + d] = eye_d
        col += d

    q32 = q.float().to(dev)
    orth_err = float((q.T @ q - torch.eye(k_total, dtype=torch.float64)).abs().max().item())
    if orth_err > 1e-6:
        raise AssertionError(f"Protected dual basis lost orthogonality: {orth_err}")
    return q32, {
        "full_dual_state_dim": full_dim,
        "protected_dim": k_total,
        "z_h_dark_dim": k_h_total,
        "z_l_board_dim": k_l_total,
        "lm_head_rank": rank_head,
        "orthogonality_error": orth_err,
    }


def extract_dual_state_at_depth(
    model: PureSudokuTRM, tokens: Tensor, depth: int
) -> DualState:
    """Roll out clean model to `depth` and return detached `DualState(z_h, z_l)`."""
    with torch.no_grad():
        emb = model.encode_inputs(tokens)
        zh, zl = model.init_states(tokens.shape[0])
        for _ in range(depth):
            zh, zl = model.act_step(zh, zl, emb)
        return DualState(zh.detach().clone(), zl.detach().clone())


def full_vocab_observer(
    model: PureSudokuTRM, packed_state: Tensor, state_shape: tuple[int, ...]
) -> Tensor:
    """Complete normalized 11-vocabulary softmax over all board cells [B, seq_len, 11]."""
    dual = DualState.unflatten(packed_state, state_shape)
    logits = model.lm_head(dual.z_h)[:, model.puzzle_emb_len :, :].float()
    return torch.softmax(logits, dim=-1)


def sudoku4_task_probs(logits_9: Tensor) -> Tensor:
    """Complete normalized 4-digit Sudoku cell probabilities [B, 16, 4]."""
    return torch.softmax(logits_9[:, :, :4].float(), dim=-1)


def corrupt_in_protected_kernel(
    clean_packed: Tensor,
    protected_basis: Tensor,
    *,
    rel_dose: float = 0.90,
    seed: int,
) -> Tensor:
    """Perturb state strictly within span(Q) at fixed relative norm `rel_dose`."""
    dev = clean_packed.device
    bsz = clean_packed.shape[0]
    k_dim = protected_basis.shape[1]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    coords = torch.randn((bsz, k_dim), generator=gen, dtype=torch.float32).to(dev)
    coords = coords / coords.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # Scale to clean protected norm so corruption meaningfully challenges multi-step rollout
    clean_proj_norm = (clean_packed @ protected_basis).norm(dim=-1, keepdim=True).clamp_min(1.0)
    delta = (coords * (rel_dose * clean_proj_norm)) @ protected_basis.T
    return clean_packed + delta


def _cell_broadcast_subspace(
    cell_dirs_h: Tensor,
    cell_dirs_l: Tensor,
    model: PureSudokuTRM,
    protected_basis: Tensor,
    rank: int,
) -> Tensor:
    """Lift cell-level output write directions into full DualState, project to Q, orthonormalize."""
    dev = protected_basis.device
    t_len = int(model.total_len)
    puz = int(model.puzzle_emb_len)
    seq = int(model.seq_len)
    d = int(model.hidden_size)
    half = t_len * d
    full_dim = 2 * half

    # Pool across board cells (uniform read/write mode + first spatial Fourier modes) so rank-r
    # spans coherent multi-cell physical writes inside Q.
    cols = []
    pos = torch.arange(seq, dtype=torch.float64)
    spatial_modes = [
        torch.ones(seq, dtype=torch.float64),
        torch.cos(math.pi * (pos + 0.5) / seq),
        torch.sin(math.pi * (pos + 0.5) / seq),
        torch.cos(2.0 * math.pi * (pos + 0.5) / seq),
    ]
    dh64 = cell_dirs_h.detach().double().cpu()  # [D, R0]
    dl64 = cell_dirs_l.detach().double().cpu()  # [D, R0]
    for mode_vec in spatial_modes:
        for r_idx in range(dh64.shape[1]):
            v_full = torch.zeros(full_dim, dtype=torch.float64)
            for c_idx in range(seq):
                w = float(mode_vec[c_idx].item())
                h0 = (puz + c_idx) * d
                l0 = half + (puz + c_idx) * d
                v_full[h0 : h0 + d] = w * dh64[:, r_idx]
                v_full[l0 : l0 + d] = w * dl64[:, r_idx]
            cols.append(v_full)
            if len(cols) >= max(rank * 3, 24):
                break
        if len(cols) >= max(rank * 3, 24):
            break

    raw_mat = torch.stack(cols, dim=1).to(device=dev, dtype=torch.float64)  # [full_dim, M]
    q64 = protected_basis.double()
    in_q = q64.T @ raw_mat  # [K, M]
    u_q, _s_q, _vh_q = torch.linalg.svd(in_q, full_matrices=False)
    basis = (q64 @ u_q[:, :rank]).float()
    orth_err = float((basis.double().T @ basis.double() - torch.eye(rank, dtype=torch.float64, device=dev)).abs().max().item())
    if orth_err > 1e-4:
        raise AssertionError(f"Constructed physical subspace lost orthogonality: {orth_err}")
    return basis


def _project_and_match_dose(
    diff: Tensor, basis: Tensor, target_norms: Tensor | None = None
) -> Tensor:
    """Project `diff` [B, D] onto orthonormal `basis` [D, r]; optionally match per-case L2 norm."""
    proj = (diff @ basis) @ basis.T
    if target_norms is None:
        return proj
    cur = proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return proj * (target_norms.unsqueeze(-1) / cur)


def fit_bpd_physical_factors_and_gauge(
    model: PureSudokuTRM,
    train_inputs: Tensor,
    eval_inputs: Tensor,
    eval_gold: Tensor,
    protected_basis: Tensor,
    *,
    seed: int,
    c_components: int = 24,
    sparse_top_k: int = 12,
    steps: int = 80,
) -> tuple[Tensor, dict]:
    """Fit physical BPD (V, U) on shared TRM matrices at depths <=6, check GL(C) gauge & OOD depth."""
    dev = model.embed_tokens.weight.device
    target_copy = copy.deepcopy(model).eval()
    c_map = {}
    for idx in range(len(target_copy.layers)):
        c_map[f"layers.{idx}.o_proj"] = c_components
        c_map[f"layers.{idx}.down_proj"] = c_components

    calls_per_step = int(target_copy.h_cycles * (target_copy.l_cycles + 1))
    max_eval_passes = 12 * calls_per_step
    fit_passes = 6 * calls_per_step

    bpd = BPDModel(
        target_copy,
        c_map,
        max_passes=max_eval_passes + 4,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=True,
    ).to(dev)

    def _run_clean_logits_and_states(m_obj: PureSudokuTRM, inp: Tensor, max_d: int):
        emb = m_obj.encode_inputs(inp)
        zh, zl = m_obj.init_states(inp.shape[0])
        logits_by_d = {}
        states_by_d = {}
        for d_step in range(1, max_d + 1):
            zh, zl = m_obj.act_step(zh, zl, emb)
            logits_by_d[d_step] = m_obj.digit_logits(zh)[:, :, :4]
            states_by_d[d_step] = (zh, zl)
        return logits_by_d, states_by_d

    def _build_pass_schedules(inp: Tensor, max_d: int, top_k: int | None, delta_val: float | None):
        bsz = inp.shape[0]
        n_p = max_d * calls_per_step
        seq_t = int(target_copy.total_len)
        if top_k is None or top_k >= c_components:
            scheds = {}
            for name in bpd.module_order:
                ones = torch.ones((bsz, n_p, seq_t, c_components), device=dev, dtype=torch.float32)
                d_mask = None if delta_val is None else torch.full((bsz, n_p, seq_t), float(delta_val), device=dev)
                scheds[name] = PassMaskSchedule(comp_mask=ones, delta_mask=d_mask)
            return scheds

        with torch.no_grad(), bpd.target_capture(pass_offset=0, detach_inputs=True):
            _ = _run_clean_logits_and_states(target_copy, inp, max_d)
            rec_map = bpd.collect_recorded_inputs(n_p)
        scheds = {}
        for name in bpd.module_order:
            w = bpd.wrappers[name]
            x_rec = rec_map[name]  # [B, T, P, d_in]
            u_norm = w.U.detach().norm(dim=-1).view(1, 1, 1, c_components)
            scores = (x_rec.float() @ w.V.detach().float()).abs() * u_norm
            topk_idx = scores.topk(top_k, dim=-1).indices
            mask = torch.zeros_like(scores)
            mask.scatter_(-1, topk_idx, 1.0)
            d_mask = torch.zeros((bsz, n_p, seq_t), device=dev, dtype=torch.float32)
            scheds[name] = PassMaskSchedule(comp_mask=mask, delta_mask=d_mask)
        return scheds

    # Train (V, U) strictly on supervised depths (2, 4, 6) under sparse top-k (delta=0)
    opt = torch.optim.AdamW(bpd.component_parameters(), lr=3e-3)
    gen = torch.Generator(device=dev).manual_seed(int(seed) * 13 + 7)
    n_train = train_inputs.shape[0]
    batch_sz = min(64, n_train)

    for _step in range(steps):
        idx = torch.randint(0, n_train, (batch_sz,), generator=gen, device=dev)
        bx = train_inputs[idx]
        with torch.no_grad():
            bpd.clear_schedules()
            clean_logits, clean_states = _run_clean_logits_and_states(model, bx, 6)

        scheds = _build_pass_schedules(bx, 6, sparse_top_k, 0.0)
        with bpd.masked_components(scheds, pass_offset=0):
            comp_logits, comp_states = _run_clean_logits_and_states(target_copy, bx, 6)

        loss = torch.tensor(0.0, device=dev)
        for d_fit in (2, 4, 6):
            p_clean = torch.softmax(clean_logits[d_fit], dim=-1)
            log_p_comp = torch.log_softmax(comp_logits[d_fit], dim=-1)
            kl = (p_clean * (torch.log(p_clean.clamp_min(1e-12)) - log_p_comp)).sum(dim=-1).mean()
            zl_mse = (comp_states[d_fit][1] - clean_states[d_fit][1]).pow(2).mean()
            loss = loss + kl + 0.05 * zl_mse
        loss = loss / 3.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bpd.component_parameters(), 1.0)
        opt.step()

    # 1. Invertible GL(C) factor-gauge invariance audit across depths (6, 8, 12)
    with torch.no_grad():
        pre_scheds = _build_pass_schedules(eval_inputs[:16], 12, c_components, 0.0)
        with bpd.masked_components(pre_scheds, pass_offset=0):
            pre_logits, _ = _run_clean_logits_and_states(target_copy, eval_inputs[:16], 12)

        saved_factors = {}
        max_op_rel_err = 0.0
        g_cpu = torch.Generator(device="cpu").manual_seed(int(seed) * 29 + 101)
        for name in bpd.module_order:
            w = bpd.wrappers[name]
            v_orig = w.V.detach().clone()
            u_orig = w.U.detach().clone()
            saved_factors[name] = (v_orig, u_orig)
            q1, _ = torch.linalg.qr(torch.randn((c_components, c_components), generator=g_cpu, dtype=torch.float64))
            q2, _ = torch.linalg.qr(torch.randn((c_components, c_components), generator=g_cpu, dtype=torch.float64))
            scales = torch.linspace(0.75, 1.35, c_components, dtype=torch.float64)
            g_mat = (q1 * scales.unsqueeze(0)) @ q2
            g_inv = torch.linalg.inv(g_mat)
            v_new = (v_orig.double().cpu() @ g_mat).float().to(dev)
            u_new = (g_inv @ u_orig.double().cpu()).float().to(dev)
            op_before = v_orig.double().cpu() @ u_orig.double().cpu()
            op_after = v_new.double().cpu() @ u_new.double().cpu()
            rel_err = float((op_after - op_before).norm().item() / op_before.norm().clamp_min(1e-12).item())
            max_op_rel_err = max(max_op_rel_err, rel_err)
            w.V.copy_(v_new)
            w.U.copy_(u_new)

        with bpd.masked_components(pre_scheds, pass_offset=0):
            post_logits, _ = _run_clean_logits_and_states(target_copy, eval_inputs[:16], 12)
        max_rollout_logit_err = max(
            float((post_logits[d_ev] - pre_logits[d_ev]).abs().max().item())
            for d_ev in (6, 8, 12)
        )
        # Restore exact original learned factors
        for name, (v_orig, u_orig) in saved_factors.items():
            bpd.wrappers[name].V.copy_(v_orig)
            bpd.wrappers[name].U.copy_(u_orig)

    # 2. Evaluate sparse top-k BPD (delta=0) vs clean & vs RMS-matched Haar read-rotation null at (6, 8, 12)
    with torch.no_grad():
        clean_ev_logits, _ = _run_clean_logits_and_states(model, eval_inputs, 12)
        sparse_scheds = _build_pass_schedules(eval_inputs, 12, sparse_top_k, 0.0)
        with bpd.masked_components(sparse_scheds, pass_offset=0):
            sparse_ev_logits, _ = _run_clean_logits_and_states(target_copy, eval_inputs, 12)

        # RMS-matched Haar read-rotation control on top-1 component of final o_proj across all passes
        last_o = f"layers.{len(target_copy.layers) - 1}.o_proj"
        with bpd.target_capture(pass_offset=0, detach_inputs=True):
            _ = _run_clean_logits_and_states(target_copy, eval_inputs[:16], 12)
            rec_12 = bpd.collect_recorded_inputs(max_eval_passes)
        x_passes = rec_12[last_o]
        v0 = bpd.wrappers[last_o].V[:, 0].detach()
        rot_gen = torch.Generator(device=dev).manual_seed(int(seed) * 41 + 3)
        rot_reads = sample_per_pass_rms_matched_rotations(
            x_passes, v0, 4, generator=rot_gen
        )  # [T, d_in, 4]
        with bpd.intervene(last_o, (0,), mode="ablate"):
            ablate_c0_logits, _ = _run_clean_logits_and_states(target_copy, eval_inputs[:16], 12)
        rot_kls = {6: [], 8: [], 12: []}
        for r_idx in range(4):
            with bpd.intervene(last_o, (0,), mode="override_read", read_override=rot_reads[:, :, r_idx : r_idx + 1]):
                rot_logits, _ = _run_clean_logits_and_states(target_copy, eval_inputs[:16], 12)
            for d_ev in (6, 8, 12):
                p_c = torch.softmax(clean_ev_logits[d_ev][:16], dim=-1)
                kl_r = float((p_c * (torch.log(p_c.clamp_min(1e-12)) - torch.log_softmax(rot_logits[d_ev], dim=-1))).sum(-1).mean().item())
                rot_kls[d_ev].append(kl_r)

        bpd_depth_metrics = {}
        for d_ev in (6, 8, 12):
            p_c = torch.softmax(clean_ev_logits[d_ev], dim=-1)
            kl_sp = float((p_c * (torch.log(p_c.clamp_min(1e-12)) - torch.log_softmax(sparse_ev_logits[d_ev], dim=-1))).sum(-1).mean().item())
            pred_sp = sparse_ev_logits[d_ev].argmax(-1)
            exact_sp = float((pred_sp == eval_gold).all(-1).float().mean().item())
            p_c16 = p_c[:16]
            kl_ab = float((p_c16 * (torch.log(p_c16.clamp_min(1e-12)) - torch.log_softmax(ablate_c0_logits[d_ev], dim=-1))).sum(-1).mean().item())
            rot_mean = sum(rot_kls[d_ev]) / len(rot_kls[d_ev])
            bpd_depth_metrics[str(d_ev)] = {
                "sparse_topk_kl_nats": kl_sp,
                "sparse_topk_exact_board": exact_sp,
                "c0_raw_ablation_kl_nats": kl_ab,
                "c0_rms_rotation_null_kl_nats": rot_mean,
                "c0_net_importance_kl_nats": kl_ab - rot_mean,
            }

        # Extract top output-write physical directions U [C, D] from final block o_proj & down_proj
        last_d = f"layers.{len(target_copy.layers) - 1}.down_proj"
        u_o = bpd.wrappers[last_o].U.detach()  # [C, D]
        u_d = bpd.wrappers[last_d].U.detach()  # [C, D]
        order_o = u_o.norm(dim=-1).argsort(descending=True)
        order_d = u_d.norm(dim=-1).argsort(descending=True)
        cell_h = u_o[order_o[:SUBSPACE_RANK]].T  # [D, R]
        cell_l = u_d[order_d[:SUBSPACE_RANK]].T  # [D, R]
        bpd_basis = _cell_broadcast_subspace(cell_h, cell_l, model, protected_basis, SUBSPACE_RANK)

    gauge_passed = bool(max_op_rel_err <= 1e-4 and max_rollout_logit_err <= 5e-5)
    return bpd_basis, {
        "c_per_module": c_components,
        "sparse_top_k": sparse_top_k,
        "fit_steps": steps,
        "fit_depths_only": [2, 4, 6],
        "gauge_invariance": {
            "passed": gauge_passed,
            "physical_operator_rel_error": max_op_rel_err,
            "max_rollout_logit_abs_error": max_rollout_logit_err,
        },
        "by_depth": bpd_depth_metrics,
    }


def principal_angle_and_commutator_audit(
    model: PureSudokuTRM,
    calib_tokens: Tensor,
    calib_states: Tensor,
    state_shape: tuple[int, ...],
    protected_basis: Tensor,
    selected_block: Tensor,
) -> dict:
    """Record post-freeze principal-angle drift to depths 8/12 and 1-step commutator defect."""
    def _cont(packed: Tensor, offset: int) -> Tensor:
        dual = DualState.unflatten(packed, state_shape)
        return native_resume_recipient(model, calib_tokens, dual, offset)

    # Post-freeze oracle blocks at individual continuation offsets (5->d6, 7->d8, 11->d12)
    angle_drift = {}
    for total_d, offset in ((6, 5), (8, 7), (12, 11)):
        oracle_b, _ = select_full_coordinate_block(
            _cont, sudoku4_task_probs, calib_states, protected_basis, (offset,), SUBSPACE_RANK
        )
        svals = torch.linalg.svdvals(selected_block.double().T @ oracle_b.double()).clamp(0.0, 1.0)
        angles_deg = [float(math.degrees(math.acos(float(sv.item())))) for sv in svals]
        overlap_frob = float(svals.pow(2).mean().item())
        angle_drift[str(total_d)] = {
            "subspace_overlap_fraction": overlap_frob,
            "min_principal_angle_deg": min(angles_deg),
            "mean_principal_angle_deg": sum(angles_deg) / len(angles_deg),
            "max_principal_angle_deg": max(angles_deg),
        }

    # One-step recurrent closure/commutator defect ||(I - Pi_E) DF(S) Pi_E||_F / ||DF(S) Pi_E||_F
    emb = model.encode_inputs(calib_tokens[:4]).detach()

    def _one_step_packed(packed: Tensor) -> Tensor:
        dual = DualState.unflatten(packed, state_shape)
        zh_next, zl_next = model.act_step(dual.z_h, dual.z_l, emb)
        return DualState(zh_next, zl_next).flatten()

    s0 = calib_states[:4].detach()
    e_cols = selected_block.detach()  # [D_full, r]
    jvp_cols = []
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for r_i in range(e_cols.shape[1]):
            v_dir = e_cols[:, r_i].unsqueeze(0).expand_as(s0)
            _, jvp_out = torch.func.jvp(_one_step_packed, (s0,), (v_dir,))
            jvp_cols.append(jvp_out.detach().double())
    df_e = torch.stack(jvp_cols, dim=-1)  # [B, D_full, r]
    e64 = e_cols.double()
    in_e = torch.einsum("dr,bdk->brk", e64, df_e)  # [B, r, r]
    proj_df_e = torch.einsum("dr,brk->bdk", e64, in_e)
    leak = (df_e - proj_df_e).norm() / df_e.norm().clamp_min(1e-12)
    # Also leak outside full protected kernel Q:
    q64 = protected_basis.double()
    in_q = torch.einsum("dk,bdr->bkr", q64, df_e)
    proj_q = torch.einsum("dk,bkr->bdr", q64, in_q)
    observer_escape = (df_e - proj_q).norm() / df_e.norm().clamp_min(1e-12)

    return {
        "principal_angle_drift_by_depth": angle_drift,
        "one_step_subspace_closure_defect": float(leak.item()),
        "one_step_protected_kernel_escape": float(observer_escape.item()),
    }


def evaluate_root_causal_steering(
    model: PureSudokuTRM,
    train_tokens: Tensor,
    calib_tokens: Tensor,
    eval_tokens: Tensor,
    eval_gold: Tensor,
    cross_donor_tokens: Tensor,
    *,
    seed: int,
) -> dict:
    """Run complete DT04 causal steering + sham + BPD (V,U) evaluation for one model root."""
    gpu_dev = model.embed_tokens.weight.device
    model.eval()
    protected_basis_gpu, _ = build_dual_protected_basis(model)

    # 1. Fit physical BPD (V, U) on shared weights strictly at depths <= 6 on GPU + gauge audit
    bpd_basis_gpu, bpd_report = fit_bpd_physical_factors_and_gauge(
        model, train_tokens, eval_tokens, eval_gold, protected_basis_gpu, seed=seed
    )
    bpd_basis = bpd_basis_gpu.detach().cpu()
    del protected_basis_gpu, bpd_basis_gpu

    # Run exact DT03/DT01 autograd Jacobian & sham routines on CPU (`dt03_causal` & `steering`
    # allocate CPU-default `torch.zeros`/`torch.eye`/`torch.randn` and loop elementwise).
    cpu_model = copy.deepcopy(model).cpu().eval()
    protected_basis, basis_meta = build_dual_protected_basis(cpu_model)
    cal_tok = calib_tokens[:4].detach().cpu()
    ev_tok = eval_tokens[:16].detach().cpu()
    ev_gold = eval_gold[:16].detach().cpu()
    cd_tok = cross_donor_tokens[:8].detach().cpu()

    # 2. Calibration states at patch depth t0 = 1 (B_cal = 4)
    cal_clean = extract_dual_state_at_depth(cpu_model, cal_tok, PATCH_DEPTH)
    state_shape = tuple(cal_clean.z_h.shape[1:])
    cal_clean_packed = cal_clean.flatten()
    cal_recip_packed = corrupt_in_protected_kernel(
        cal_clean_packed, protected_basis, rel_dose=0.90, seed=seed * 101 + 11
    )

    def _cal_continuation(packed: Tensor, offset: int) -> Tensor:
        dual = DualState.unflatten(packed, state_shape)
        return native_resume_recipient(cpu_model, cal_tok, dual, offset)

    # Select top-8 physical block strictly using supervised continuation offsets (1, 3, 5) -> depths (2, 4, 6)
    gram_basis, gram_meta = select_full_coordinate_block(
        _cal_continuation,
        sudoku4_task_probs,
        cal_recip_packed,
        protected_basis,
        FIT_OFFSETS,
        SUBSPACE_RANK,
    )

    # 3. Output-side left SVD physical subspace (repaired DT03 left singular vectors!)
    last_blk = cpu_model.layers[-1]
    left_o = output_weight_basis(last_blk.o_proj.weight.detach(), SUBSPACE_RANK)
    left_d = output_weight_basis(last_blk.down_proj.weight.detach(), SUBSPACE_RANK)
    svd_left_basis = _cell_broadcast_subspace(left_o, left_d, cpu_model, protected_basis, SUBSPACE_RANK)

    # 4. Random orthonormal rank-8 protected subspace
    g_rand = torch.Generator(device="cpu").manual_seed(seed * 131 + 17)
    raw_rand = torch.randn((protected_basis.shape[1], SUBSPACE_RANK), generator=g_rand, dtype=torch.float64)
    q_rand, _ = torch.linalg.qr(raw_rand)
    rand_basis = (protected_basis.double() @ q_rand).float()

    # 5. Principal-angle drift and 1-step commutator diagnostics
    transport_diag = principal_angle_and_commutator_audit(
        cpu_model, cal_tok, cal_recip_packed, state_shape, protected_basis, gram_basis
    )

    # 6. Disjoint evaluation cohort (B_eval = 16) at t0 = 1, evaluated at same-case depths (6, 8, 12)
    n_ev = int(ev_tok.shape[0])
    ev_donor_dual = extract_dual_state_at_depth(cpu_model, ev_tok, PATCH_DEPTH)
    ev_donor_packed = ev_donor_dual.flatten()
    ev_recip_packed = corrupt_in_protected_kernel(
        ev_donor_packed, protected_basis, rel_dose=0.90, seed=seed * 197 + 29
    )
    diff = ev_donor_packed - ev_recip_packed  # Lies strictly in span(Q)

    edit_zero = torch.zeros_like(diff)
    edit_full_dark = (diff @ protected_basis) @ protected_basis.T
    edit_gram = _project_and_match_dose(diff, gram_basis, None)
    gram_norms = edit_gram.norm(dim=-1)
    edit_bpd_dose = _project_and_match_dose(diff, bpd_basis, gram_norms)
    edit_svd_dose = _project_and_match_dose(diff, svd_left_basis, gram_norms)
    edit_rand_dose = _project_and_match_dose(diff, rand_basis, gram_norms)

    # 7. Privileged full signed-task-Jacobian + L2-dose matched shams (8 per root!)
    # Matched to `edit_gram` on the complete 64-coordinate task Jacobian at supervised boundary depth 6 (offset=5)
    def _ev_task_at_d6(packed: Tensor) -> Tensor:
        dual = DualState.unflatten(packed, state_shape)
        return sudoku4_task_probs(native_resume_recipient(cpu_model, ev_tok, dual, 5))

    jac_d6 = flat_observer_jacobian(_ev_task_at_d6, ev_recip_packed)  # [16, 64, D_full]
    sham_edits: list[Tensor] = []
    sham_ledger: list[dict] = []
    for s_idx in range(N_SHAMS):
        s_gen = torch.Generator(device="cpu").manual_seed(seed * 1009 + 503 + s_idx)
        per_case_ctrl = []
        for b_i in range(n_ev):
            ctrl_b, _ = signed_task_matched_sham(
                jac_d6[b_i], edit_gram[b_i : b_i + 1], protected_basis, s_gen
            )
            per_case_ctrl.append(ctrl_b[0])
        ctrl_full = torch.stack(per_case_ctrl, dim=0)
        cert = validate_delivered_sham(
            edit_gram, ctrl_full, protected_basis, jac_d6, tolerance=1e-4
        )
        sham_edits.append(ctrl_full)
        sham_ledger.append({"sham_index": s_idx, **cert})

    # Verify immediate full-11-vocab readout conservation across EVERY arm
    base_obs = full_vocab_observer(cpu_model, ev_recip_packed, state_shape)
    arms: dict[str, Tensor] = {
        "no_edit": edit_zero,
        "full_dark_donor": edit_full_dark,
        "task_gramian_r8": edit_gram,
        "bpd_vu_r8_dose": edit_bpd_dose,
        "weight_svd_left_r8_dose": edit_svd_dose,
        "random_protected_r8_dose": edit_rand_dose,
    }
    for s_idx, s_ed in enumerate(sham_edits):
        arms[f"signed_jac_sham_{s_idx}"] = s_ed

    max_obs_err = 0.0
    for arm_name, ed in arms.items():
        patched = ev_recip_packed + ed
        err = float((full_vocab_observer(cpu_model, patched, state_shape) - base_obs).abs().max().item())
        if err > OBSERVER_TOL:
            raise AssertionError(f"Protected 11-vocab readout violated on {arm_name}: {err}")
        d_rec = DualState.unflatten(ev_recip_packed, state_shape)
        d_pat = DualState.unflatten(patched, state_shape)
        puz = cpu_model.puzzle_emb_len
        pref_err = max(
            float((d_pat.z_h[:, :puz] - d_rec.z_h[:, :puz]).abs().max().item()),
            float((d_pat.z_l[:, :puz] - d_rec.z_l[:, :puz]).abs().max().item()),
        )
        if pref_err > 1e-7:
            raise AssertionError(f"Prefix modified on {arm_name}: {pref_err}")
        max_obs_err = max(max_obs_err, err)

    # 8. Same-case evaluation at total depths (6, 8, 12) -> offsets (5, 7, 11)
    by_depth = {}
    blanks_mask = (ev_tok == 1)  # [16, 16]
    with torch.no_grad():
        for offset in EVAL_OFFSETS:
            total_d = PATCH_DEPTH + offset
            donor_logits = native_resume_recipient(cpu_model, ev_tok, ev_donor_dual, offset)[:, :, :4]
            donor_probs = torch.softmax(donor_logits, dim=-1)
            no_edit_logits = native_resume_recipient(
                cpu_model, ev_tok, DualState.unflatten(ev_recip_packed, state_shape), offset
            )[:, :, :4]
            no_edit_pred = no_edit_logits.argmax(-1)
            no_edit_correct = (no_edit_pred == ev_gold)

            depth_arms = {}
            for arm_name, ed in arms.items():
                cand_dual = DualState.unflatten(ev_recip_packed + ed, state_shape)
                cand_logits = native_resume_recipient(cpu_model, ev_tok, cand_dual, offset)[:, :, :4]
                cand_probs = torch.softmax(cand_logits, dim=-1)
                cand_pred = cand_logits.argmax(-1)
                rec = whole_case_recovery(cand_probs, donor_probs, DONOR_PROB_TOL)
                b_acc = blank_accuracy(cand_pred, ev_gold, ev_tok, blank_token=1)
                cand_correct = (cand_pred == ev_gold)
                w2c_all = int((~no_edit_correct & cand_correct).sum().item())
                c2w_all = int((no_edit_correct & ~cand_correct).sum().item())
                w2c_blank = int((~no_edit_correct & cand_correct & blanks_mask).sum().item())
                c2w_blank = int((no_edit_correct & ~cand_correct & blanks_mask).sum().item())
                net_blank_gain = (w2c_blank - c2w_blank) / float(max(1, int(blanks_mask.sum().item())))

                depth_arms[arm_name] = {
                    "whole_case_recovery": rec["whole_case_recovery"],
                    "coordinate_recovery": rec["coordinate_recovery"],
                    "donor_prob_mse": rec["mean_squared_response_error"],
                    "gold_whole_board_exact": b_acc["whole_board_exact"],
                    "gold_blank_accuracy": b_acc["blank_accuracy"],
                    "gold_clue_preservation": b_acc["clue_preservation"],
                    "wrong_to_correct_cells": w2c_all,
                    "correct_to_wrong_cells": c2w_all,
                    "wrong_to_correct_blanks": w2c_blank,
                    "correct_to_wrong_blanks": c2w_blank,
                    "net_blank_cell_gain": float(net_blank_gain),
                    "delivered_l2_dose_mean": float(ed.norm(dim=-1).mean().item()),
                    "per_case_exact_gold": (cand_pred == ev_gold).all(dim=-1).tolist(),
                    "per_case_donor_recovered": rec["per_case_recovered"],
                    "predictions": cand_pred.cpu().tolist() if arm_name in (
                        "no_edit", "full_dark_donor", "task_gramian_r8", "bpd_vu_r8_dose", "signed_jac_sham_0"
                    ) else None,
                }
            by_depth[str(total_d)] = depth_arms

        # 9. Secondary diagnostic: cross-puzzle donor transplant bound to recipient tokens (DT03 §2.4)
        cr_tok = ev_tok[:8]
        cd_dual = extract_dual_state_at_depth(cpu_model, cd_tok, PATCH_DEPTH)
        cr_dual = extract_dual_state_at_depth(cpu_model, cr_tok, PATCH_DEPTH)
        cr_packed = cr_dual.flatten()
        cd_packed = cd_dual.flatten()
        cross_diff_q = ((cd_packed - cr_packed) @ protected_basis) @ protected_basis.T
        cross_donor_in_rec = DualState.unflatten(cr_packed + cross_diff_q, state_shape)
        cross_gram_ed = _project_and_match_dose(cross_diff_q, gram_basis, None)
        cross_summary = {}
        for offset in EVAL_OFFSETS:
            total_d = PATCH_DEPTH + offset
            ref_probs = sudoku4_task_probs(native_resume_recipient(cpu_model, cr_tok, cross_donor_in_rec, offset))
            zero_probs = sudoku4_task_probs(native_resume_recipient(cpu_model, cr_tok, cr_dual, offset))
            gram_probs = sudoku4_task_probs(
                native_resume_recipient(cpu_model, cr_tok, DualState.unflatten(cr_packed + cross_gram_ed, state_shape), offset)
            )
            cross_summary[str(total_d)] = {
                "no_edit_whole_case_recovery": whole_case_recovery(zero_probs, ref_probs, DONOR_PROB_TOL)["whole_case_recovery"],
                "task_gramian_r8_whole_case_recovery": whole_case_recovery(gram_probs, ref_probs, DONOR_PROB_TOL)["whole_case_recovery"],
                "no_edit_mse": whole_case_recovery(zero_probs, ref_probs, DONOR_PROB_TOL)["mean_squared_response_error"],
                "task_gramian_r8_mse": whole_case_recovery(gram_probs, ref_probs, DONOR_PROB_TOL)["mean_squared_response_error"],
            }
    del cpu_model
    _ = gpu_dev

    # Summarize root-level primary contrasts across depths (6, 8, 12) for EvalMath preflight
    root_contrasts = {}
    for d_key in ("6", "8", "12"):
        da = by_depth[d_key]
        gram_exact = da["task_gramian_r8"]["gold_whole_board_exact"]
        gram_rec = da["task_gramian_r8"]["whole_case_recovery"]
        gram_blank_gain = da["task_gramian_r8"]["net_blank_cell_gain"]
        base_exact = da["no_edit"]["gold_whole_board_exact"]
        bpd_exact = da["bpd_vu_r8_dose"]["gold_whole_board_exact"]
        sham_exacts = [da[f"signed_jac_sham_{i}"]["gold_whole_board_exact"] for i in range(N_SHAMS)]
        sham_recs = [da[f"signed_jac_sham_{i}"]["whole_case_recovery"] for i in range(N_SHAMS)]
        sham_blank_gains = [da[f"signed_jac_sham_{i}"]["net_blank_cell_gain"] for i in range(N_SHAMS)]
        root_contrasts[d_key] = {
            "gramian_minus_no_edit_exact": gram_exact - base_exact,
            "gramian_minus_mean_sham_exact": gram_exact - (sum(sham_exacts) / N_SHAMS),
            "gramian_minus_mean_sham_recovery": gram_rec - (sum(sham_recs) / N_SHAMS),
            "gramian_minus_mean_sham_blank_gain": gram_blank_gain - (sum(sham_blank_gains) / N_SHAMS),
            "bpd_minus_mean_sham_exact": bpd_exact - (sum(sham_exacts) / N_SHAMS),
            "sham_minus_no_edit_exact_list": [sx - base_exact for sx in sham_exacts],
            "sham_centered_exact_list": [sx - (sum(sham_exacts) / N_SHAMS) for sx in sham_exacts],
            "sham_centered_recovery_list": [sr - (sum(sham_recs) / N_SHAMS) for sr in sham_recs],
        }

    return {
        "patch_depth": PATCH_DEPTH,
        "fit_offsets": list(FIT_OFFSETS),
        "eval_offsets": list(EVAL_OFFSETS),
        "subspace_rank": SUBSPACE_RANK,
        "protected_basis_meta": basis_meta,
        "protected_all_vocab_max_error": max_obs_err,
        "gramian_selection_meta": {
            "fit_depths_total": [PATCH_DEPTH + o for o in FIT_OFFSETS],
            "rank": gram_meta["rank"],
            "top8_eigenvalues": gram_meta["eigenvalues"][:12],
            "full_output_coordinates": gram_meta["full_output_coordinates"],
            "full_state_coordinates": gram_meta["full_state_coordinates"],
        },
        "bpd_physical_factor_audit": bpd_report,
        "depth_transport_geometry": transport_diag,
        "sham_validation_ledger": sham_ledger,
        "same_board_corrupt_repair_by_depth": by_depth,
        "cross_puzzle_donor_by_depth": cross_summary,
        "root_contrasts_by_depth": root_contrasts,
    }
