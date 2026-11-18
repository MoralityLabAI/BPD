"""BPD-DT04 task admission, canonical split separation, and native TRM training.

Covers all three DT01/DT03 gym task families with zero cross-split leakage:
1. `sudoku4`: Canonical-equivalence-class partitioned 4x4 Sudoku (`blanks in {6, 8, 9}`),
   certified `cross_split_isomorphic_items == 0` via `dt03_tasks.audit_puzzle_split`,
   supervised only at ACT depths `(2, 4, 6)` with `blank_weighted_ce + 0.25 * clue_ce`,
   and evaluated on the same heldout boards at `(2, 4, 6, 8, 12)` plus period-2 tail
   oscillation and solver legality.
2. `arc_style`:
   - `arc_analogue_strict`: Exact `worlds.arc_analogue` (`flip_h, recolor_1_2` ->
     `rotate_180, fill_zero`) encoded via `dt03_tasks.arc_demo_tokens` (`t - 1` onto `0..10`),
     paired with an explicit DT03 color-support audit (`color 3` absent in train targets).
   - `arc_lut_induction`: 192 train vs 64 disjoint heldout 4-color lookup programs
     `f: {0..3} -> {0..3}` (holding out `fill_zero = (3, 1, 2, 3)` and 63 other unseen
     programs) serialized through the exact 127-token `arc_demo_tokens` format to measure
     genuine zero-shot demo-conditioned program induction at depths `(6, 8, 12)`.
3. `storyworld`: Disjoint `HiddenEndingWorld` root cohorts (`train`, `eval`), supervised at
   depths `(2, 4, 6)` on reachable public states labeled by `dt03_tasks.public_reachable_policy`,
   and evaluated via 12-step **closed-loop rollouts** at depths `(6, 8, 12)` against host-only
   terminal endings (`ending != "unfinished"`). Explicitly labeled as a public-policy
   closed-loop control-world test (`DT03_REVIEW.md`).
"""
from __future__ import annotations

import hashlib
from itertools import permutations, product
import random
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from mechinterp.bpd.adapters.trm import build_mini_trm
from mechinterp.loopscope02.scripts.trm_sudoku import PureSudokuTRM

from ..dt03_tasks import (
    arc_demo_tokens,
    audit_puzzle_split,
    blank_accuracy,
    blank_weighted_ce,
    native_arc_query_positions,
    public_reachable_policy,
    sudoku4_canonical,
)
from ..worlds import ArcCase, HiddenEndingWorld, SudokuCase, arc_analogue, sudoku4, sudoku_legal

FIT_DEPTHS = (2, 4, 6)
ALL_EVAL_DEPTHS = (2, 4, 6, 8, 12)
SAME_CASE_DEPTHS = (6, 8, 12)

# Precomputed legal 4x4 Sudoku symmetry group elements (within-canonical-class orbit)
_ROW_BAND_PERMS = tuple(
    tuple(b * 2 + j for b, inner in zip(bands, (i0, i1)) for j in inner)
    for bands in permutations((0, 1))
    for i0 in permutations((0, 1))
    for i1 in permutations((0, 1))
)  # 8 row/col band-stack permutations
_DIGIT_PERMS = tuple((0,) + tuple(p) for p in permutations((1, 2, 3, 4)))  # 24 relabelings

# Deterministic global canonical 4x4 Sudoku catalog cached in-process (~2 MB)
_CANONICAL_SUDOKU_CACHE: list[tuple[tuple[int, ...], list[SudokuCase]]] | None = None


def _get_canonical_sudoku_catalog() -> list[tuple[tuple[int, ...], list[SudokuCase]]]:
    """Build deterministic canonical-grouped 4x4 Sudoku catalog across blanks {6, 8, 9}."""
    global _CANONICAL_SUDOKU_CACHE
    if _CANONICAL_SUDOKU_CACHE is not None:
        return _CANONICAL_SUDOKU_CACHE
    by_canon: dict[tuple[int, ...], list[SudokuCase]] = {}
    for idx in range(1800):
        for blanks in (6, 8, 9):
            seed_val = 700_000 + idx * 13 + blanks
            try:
                case = sudoku4(seed_val, "catalog", blanks=blanks)
            except ValueError:
                continue
            canon = sudoku4_canonical(case.givens)
            bucket = by_canon.setdefault(canon, [])
            if len(bucket) < 3:
                bucket.append(case)
    _CANONICAL_SUDOKU_CACHE = sorted(by_canon.items(), key=lambda kv: kv[0])
    return _CANONICAL_SUDOKU_CACHE


def build_root_sudoku_splits(root: int) -> dict:
    """Partition canonical 4x4 Sudoku classes into strictly disjoint train/calib/eval/donor sets."""
    catalog = _get_canonical_sudoku_catalog()
    if len(catalog) < 400:
        raise RuntimeError(f"Insufficient canonical 4x4 Sudoku classes: {len(catalog)}")
    rng = random.Random(int(root) * 10007 + 31)
    indices = list(range(len(catalog)))
    rng.shuffle(indices)

    # Reserve strictly disjoint canonical classes: 64 eval, 16 calib, 16 cross-donor, rest train
    eval_idx = indices[:64]
    calib_idx = indices[64:80]
    donor_idx = indices[80:96]
    train_idx = indices[96:]

    eval_cases = [catalog[i][1][0] for i in eval_idx]
    calib_cases = [catalog[i][1][0] for i in calib_idx]
    donor_cases = [catalog[i][1][0] for i in donor_idx]
    train_cases = [c for i in train_idx for c in catalog[i][1]]

    audit_eval = audit_puzzle_split(train_cases, eval_cases)
    audit_calib = audit_puzzle_split(train_cases, calib_cases)
    audit_cal_ev = audit_puzzle_split(calib_cases, eval_cases)
    if not (audit_eval["admissible"] and audit_calib["admissible"] and audit_cal_ev["admissible"]):
        raise AssertionError("Canonical Sudoku isomorphism detected across splits")

    split_hash = hashlib.sha256(
        repr(([c.givens for c in eval_cases], [c.givens for c in calib_cases])).encode("utf-8")
    ).hexdigest()
    return {
        "train_cases": train_cases,
        "calib_cases": calib_cases,
        "eval_cases": eval_cases,
        "donor_cases": donor_cases,
        "canonical_class_counts": {
            "total": len(catalog),
            "train": len(train_idx),
            "calib": len(calib_idx),
            "eval": len(eval_idx),
            "cross_donor": len(donor_idx),
        },
        "audit_train_vs_eval": audit_eval,
        "audit_train_vs_calib": audit_calib,
        "audit_calib_vs_eval": audit_cal_ev,
        "split_sha256": split_hash,
    }


def encode_sudoku_cases(cases: Sequence[SudokuCase], device: torch.device | str) -> tuple[Tensor, Tensor]:
    """Encode 4x4 Sudoku givens (`0..4 -> 1..5`, blank=1) and solutions (`1..4 -> 0..3`)."""
    xs = [[v + 1 for row in c.givens for v in row] for c in cases]
    ys = [[v - 1 for row in c.solution for v in row] for c in cases]
    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
    )


def train_and_eval_sudoku4(
    root: int,
    splits: dict,
    device: torch.device | str,
    *,
    steps: int = 480,
    batch_size: int = 128,
) -> tuple[PureSudokuTRM, dict]:
    """Train native PureSudokuTRM at fit depths (2,4,6) and evaluate at (2,4,6,8,12)."""
    dev = torch.device(device)
    train_cases: list[SudokuCase] = splits["train_cases"]
    eval_cases: list[SudokuCase] = splits["eval_cases"]

    base_g = torch.tensor(
        [[[v for v in row] for row in c.givens] for c in train_cases], dtype=torch.long, device=dev
    )
    base_s = torch.tensor(
        [[[v for v in row] for row in c.solution] for c in train_cases], dtype=torch.long, device=dev
    )
    row_perms = torch.tensor(_ROW_BAND_PERMS, dtype=torch.long, device=dev)
    dig_perms = torch.tensor(_DIGIT_PERMS, dtype=torch.long, device=dev)

    def _sample_batch(gen: torch.Generator) -> tuple[Tensor, Tensor]:
        idx = torch.randint(0, len(base_g), (batch_size,), generator=gen, device=dev)
        g = base_g[idx]
        s = base_s[idx]
        tr = torch.randint(0, 2, (batch_size, 1, 1), generator=gen, device=dev).bool()
        g = torch.where(tr, g.transpose(1, 2), g)
        s = torch.where(tr, s.transpose(1, 2), s)
        rp = row_perms[torch.randint(0, 8, (batch_size,), generator=gen, device=dev)]
        cp = row_perms[torch.randint(0, 8, (batch_size,), generator=gen, device=dev)]
        b_idx = torch.arange(batch_size, device=dev)[:, None, None]
        g = g[b_idx, rp[:, :, None], cp[:, None, :]]
        s = s[b_idx, rp[:, :, None], cp[:, None, :]]
        dp = dig_perms[torch.randint(0, 24, (batch_size,), generator=gen, device=dev)]
        g_flat = dp.gather(1, g.reshape(batch_size, 16))
        s_flat = dp.gather(1, s.reshape(batch_size, 16))
        return g_flat + 1, s_flat - 1

    model = build_mini_trm(
        hidden_size=96,
        num_heads=4,
        l_layers=2,
        h_cycles=1,
        l_cycles=2,
        puzzle_emb_len=2,
        seq_len=16,
        seed=int(root) * 17 + 1,
    ).train().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator(device=dev).manual_seed(int(root) * 31 + 5)

    final_loss = 0.0
    for _step in range(steps):
        bx, by = _sample_batch(gen)
        emb = model.encode_inputs(bx)
        zh, zl = model.init_states(batch_size)
        clues = bx != 1
        loss = torch.tensor(0.0, device=dev)
        for d_step in range(1, max(FIT_DEPTHS) + 1):
            zh, zl = model.act_step(zh, zl, emb)
            if d_step in FIT_DEPTHS:
                logits = model.digit_logits(zh)[:, :, :4]
                b_loss = blank_weighted_ce(logits, by, bx, blank_token=1)
                per = F.cross_entropy(
                    logits.reshape(-1, 4), by.reshape(-1), reduction="none"
                ).reshape(by.shape)
                c_loss = (per * clues).sum() / clues.sum().clamp_min(1)
                loss = loss + b_loss + 0.25 * c_loss
        loss = loss / float(len(FIT_DEPTHS))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite Sudoku4 training loss on root {root}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        final_loss = float(loss.detach().item())

    model.eval()
    ev_x, ev_y = encode_sudoku_cases(eval_cases, dev)
    by_depth = {}
    tail_preds: list[Tensor] = []
    with torch.no_grad():
        emb = model.encode_inputs(ev_x)
        zh, zl = model.init_states(ev_x.shape[0])
        for d_step in range(1, 13):
            zh, zl = model.act_step(zh, zl, emb)
            pred = model.digit_logits(zh)[:, :, :4].argmax(-1)
            if d_step >= 10:
                tail_preds.append(pred)
            if d_step in ALL_EVAL_DEPTHS:
                acc = blank_accuracy(pred, ev_y, ev_x, blank_token=1)
                legal_cnt = sum(
                    sudoku_legal(
                        tuple(
                            tuple(int(v + 1) for v in row)
                            for row in pred[i].detach().cpu().reshape(4, 4)
                        ),
                        require_complete=True,
                    )
                    for i in range(pred.shape[0])
                )
                by_depth[str(d_step)] = {
                    **acc,
                    "board_legality": legal_cnt / float(pred.shape[0]),
                    "n": int(pred.shape[0]),
                    "per_case_exact": (pred == ev_y).all(dim=-1).cpu().tolist(),
                    "predictions": pred.cpu().tolist() if d_step in SAME_CASE_DEPTHS else None,
                }

    p10, p11, p12 = tail_preds
    flips_last = (p12 != p11).any(dim=-1)
    period2 = float((flips_last & (p12 == p10).all(dim=-1)).float().mean().item())
    tail_flip = float((p12 != p11).float().mean().item())

    return model, {
        "family": "sudoku4",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "updates": steps,
        "batch_size": batch_size,
        "final_fit_loss": final_loss,
        "supervised_depths": list(FIT_DEPTHS),
        "heldout_depths": [8, 12],
        "split_audit": splits["audit_train_vs_eval"],
        "canonical_class_counts": splits["canonical_class_counts"],
        "split_sha256": splits["split_sha256"],
        "period2_oscillation_d10_12": period2,
        "tail_cell_flip_d11_12": tail_flip,
        "eval_by_depth": by_depth,
    }


def _build_arc_lut_partition(root: int) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Partition all 256 4-color lookup functions {0..3}^4 into 192 train and 64 heldout."""
    all_luts = list(product((0, 1, 2, 3), repeat=4))
    # Force `fill_zero` = (3, 1, 2, 3) into heldout to mirror worlds.arc_analogue!
    fill_zero = (3, 1, 2, 3)
    remaining = [lut for lut in all_luts if lut != fill_zero]
    rng = random.Random(int(root) * 19 + 401)
    rng.shuffle(remaining)
    heldout = [fill_zero] + remaining[:63]
    train = remaining[63:]
    if set(train) & set(heldout):
        raise AssertionError("ARC LUT program overlap across splits")
    return train, heldout


def audit_arc_color_support(train_targets: Tensor, heldout_targets: Tensor) -> dict:
    """Explicit DT03 color-support audit comparing target color histograms."""
    tr_hist = {str(c): int((train_targets == c).sum().item()) for c in range(4)}
    ho_hist = {str(c): int((heldout_targets == c).sum().item()) for c in range(4)}
    unsupported = [c for c in range(4) if tr_hist[str(c)] == 0 and ho_hist[str(c)] > 0]
    return {
        "train_target_color_counts": tr_hist,
        "heldout_target_color_counts": ho_hist,
        "unsupported_heldout_colors": unsupported,
        "full_4color_support_in_train": len(unsupported) == 0,
    }


def train_and_eval_arc_family(
    root: int,
    device: torch.device | str,
    *,
    lut_steps: int = 520,
    legacy_steps: int = 320,
    batch_size: int = 128,
) -> dict:
    """Train and evaluate both (A) strict `worlds.arc_analogue` and (B) 192->64 LUT induction."""
    dev = torch.device(device)
    qpos = list(native_arc_query_positions(3))

    # --- Sub-study A: Strict `worlds.arc_analogue` (2 train rules -> 2 heldout rules) ---
    tr_cases_a = [arc_analogue(root * 100_000 + i, "train") for i in range(768)]
    id_cases_a = [arc_analogue(root * 100_000 + 50_000 + i, "train") for i in range(64)]
    ho_cases_a = [arc_analogue(root * 100_000 + 80_000 + i, "test") for i in range(64)]

    tx_a = torch.tensor([[t - 1 for t in arc_demo_tokens(c)] for c in tr_cases_a], dtype=torch.long, device=dev)
    ty_a = torch.tensor([[v for row in c.answer for v in row] for c in tr_cases_a], dtype=torch.long, device=dev)
    ix_a = torch.tensor([[t - 1 for t in arc_demo_tokens(c)] for c in id_cases_a], dtype=torch.long, device=dev)
    iy_a = torch.tensor([[v for row in c.answer for v in row] for c in id_cases_a], dtype=torch.long, device=dev)
    hx_a = torch.tensor([[t - 1 for t in arc_demo_tokens(c)] for c in ho_cases_a], dtype=torch.long, device=dev)
    hy_a = torch.tensor([[v for row in c.answer for v in row] for c in ho_cases_a], dtype=torch.long, device=dev)
    color_audit_a = audit_arc_color_support(ty_a, hy_a)

    torch.manual_seed(int(root) * 23 + 2)
    m_leg = PureSudokuTRM(
        hidden_size=96, num_heads=4, l_layers=2, h_cycles=1, l_cycles=1, puzzle_emb_len=1, seq_len=127
    ).to(dev)
    with torch.no_grad():
        m_leg.embed_tokens.weight.mul_(0.35)
    opt_a = torch.optim.AdamW(m_leg.parameters(), lr=2e-3, weight_decay=1e-4)
    gen_a = torch.Generator(device=dev).manual_seed(int(root) * 29 + 3)
    loss_a_val = 0.0
    for _s in range(legacy_steps):
        idx = torch.randint(0, len(tx_a), (batch_size,), generator=gen_a, device=dev)
        emb = m_leg.encode_inputs(tx_a[idx])
        zh, zl = m_leg.init_states(batch_size)
        loss = torch.tensor(0.0, device=dev)
        for d_step in range(1, 7):
            zh, zl = m_leg.act_step(zh, zl, emb)
            if d_step in FIT_DEPTHS:
                loss = loss + F.cross_entropy(
                    m_leg.digit_logits(zh)[:, qpos, :4].reshape(-1, 4), ty_a[idx].reshape(-1)
                )
        loss = loss / 3.0
        opt_a.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_leg.parameters(), 1.0)
        opt_a.step()
        loss_a_val = float(loss.detach().item())

    m_leg.eval()
    leg_eval = {"in_dist": {}, "heldout": {}}
    with torch.no_grad():
        for tag, ex, ey in (("in_dist", ix_a, iy_a), ("heldout", hx_a, hy_a)):
            emb = m_leg.encode_inputs(ex)
            zh, zl = m_leg.init_states(ex.shape[0])
            for d_step in range(1, 13):
                zh, zl = m_leg.act_step(zh, zl, emb)
                if d_step in SAME_CASE_DEPTHS:
                    pred = m_leg.digit_logits(zh)[:, qpos, :4].argmax(-1)
                    c3_mask = ey == 3
                    c3_rec = float((pred[c3_mask] == 3).float().mean().item()) if bool(c3_mask.any()) else None
                    leg_eval[tag][str(d_step)] = {
                        "cell_accuracy": float((pred == ey).float().mean().item()),
                        "exact_grid_accuracy": float((pred == ey).all(-1).float().mean().item()),
                        "color3_recall": c3_rec,
                        "n": int(ex.shape[0]),
                    }
    del m_leg

    # --- Sub-study B: Balanced 4-color 192->64 LUT induction via `arc_demo_tokens` ---
    train_luts, heldout_luts = _build_arc_lut_partition(root)
    tr_lut_t = torch.tensor(train_luts, dtype=torch.long, device=dev)   # [192, 4]
    ho_lut_t = torch.tensor(heldout_luts, dtype=torch.long, device=dev) # [64, 4]

    def _sample_lut_batch(g: torch.Generator, n_b: int, luts_t: Tensor) -> tuple[Tensor, Tensor]:
        grids = torch.randint(0, 4, (n_b, 4, 16), generator=g, device=dev)
        cm = luts_t[torch.randint(0, len(luts_t), (n_b,), generator=g, device=dev)]
        outs = cm[:, None, :].expand(n_b, 4, 4).gather(2, grids)
        parts = []
        for k in range(3):
            parts.extend([
                torch.full((n_b, 1), 6, dtype=torch.long, device=dev),
                grids[:, k],
                torch.full((n_b, 1), 9, dtype=torch.long, device=dev),
                torch.full((n_b, 1), 7, dtype=torch.long, device=dev),
                outs[:, k],
                torch.full((n_b, 1), 9, dtype=torch.long, device=dev),
            ])
        parts.extend([
            torch.full((n_b, 1), 8, dtype=torch.long, device=dev),
            grids[:, 3],
            torch.full((n_b, 1), 9, dtype=torch.long, device=dev),
            torch.full((n_b, 1), 10, dtype=torch.long, device=dev),
        ])
        return torch.cat(parts, dim=1), outs[:, 3]

    # Verify exact parity with `dt03_tasks.arc_demo_tokens` on deterministic test sample
    rng_chk = random.Random(int(root) + 77)
    lut0 = heldout_luts[0]
    demos0 = []
    for _ in range(4):
        g0 = tuple(tuple(rng_chk.randrange(4) for _ in range(4)) for _ in range(4))
        o0 = tuple(tuple(lut0[v] for v in r) for r in g0)
        demos0.append((g0, o0))
    case0 = ArcCase("lut-check", tuple(demos0[:3]), demos0[3][0], demos0[3][1], "heldout", "synthetic")
    tok0 = [t - 1 for t in arc_demo_tokens(case0)]
    if len(tok0) != 127 or max(tok0) > 10 or min(tok0) < 0:
        raise AssertionError("ARC LUT serializer violated 127-token 11-vocab contract")

    torch.manual_seed(int(root) * 37 + 4)
    m_lut = PureSudokuTRM(
        hidden_size=96, num_heads=4, l_layers=2, h_cycles=1, l_cycles=1, puzzle_emb_len=1, seq_len=127
    ).to(dev)
    with torch.no_grad():
        m_lut.embed_tokens.weight.mul_(0.35)
    opt_b = torch.optim.AdamW(m_lut.parameters(), lr=2.5e-3, weight_decay=1e-4)
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=lut_steps)
    gen_b = torch.Generator(device=dev).manual_seed(int(root) * 41 + 9)
    loss_b_val = 0.0
    sample_tr_y = None
    for _s in range(lut_steps):
        bx, by = _sample_lut_batch(gen_b, batch_size, tr_lut_t)
        if sample_tr_y is None:
            sample_tr_y = by.detach()
        emb = m_lut.encode_inputs(bx)
        zh, zl = m_lut.init_states(batch_size)
        loss = torch.tensor(0.0, device=dev)
        for d_step in range(1, 7):
            zh, zl = m_lut.act_step(zh, zl, emb)
            if d_step in FIT_DEPTHS:
                loss = loss + F.cross_entropy(
                    m_lut.digit_logits(zh)[:, qpos, :4].reshape(-1, 4), by.reshape(-1)
                )
        loss = loss / 3.0
        opt_b.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_lut.parameters(), 1.0)
        opt_b.step()
        sched_b.step()
        loss_b_val = float(loss.detach().item())

    m_lut.eval()
    gen_id = torch.Generator(device=dev).manual_seed(int(root) * 53 + 11)
    gen_ho = torch.Generator(device=dev).manual_seed(int(root) * 59 + 12)
    ix_b, iy_b = _sample_lut_batch(gen_id, 64, tr_lut_t)
    hx_b, hy_b = _sample_lut_batch(gen_ho, 64, ho_lut_t)
    color_audit_b = audit_arc_color_support(sample_tr_y, hy_b)
    lut_eval = {"in_dist": {}, "heldout": {}}
    with torch.no_grad():
        for tag, ex, ey in (("in_dist", ix_b, iy_b), ("heldout", hx_b, hy_b)):
            emb = m_lut.encode_inputs(ex)
            zh, zl = m_lut.init_states(ex.shape[0])
            for d_step in range(1, 13):
                zh, zl = m_lut.act_step(zh, zl, emb)
                if d_step in SAME_CASE_DEPTHS:
                    pred = m_lut.digit_logits(zh)[:, qpos, :4].argmax(-1)
                    lut_eval[tag][str(d_step)] = {
                        "cell_accuracy": float((pred == ey).float().mean().item()),
                        "exact_grid_accuracy": float((pred == ey).all(-1).float().mean().item()),
                        "per_case_exact": (pred == ey).all(-1).cpu().tolist(),
                        "n": int(ex.shape[0]),
                    }
    del m_lut

    return {
        "family": "arc_style_synthetic",
        "official_arc1_evaluated": False,
        "arc_analogue_strict_2rule": {
            "final_fit_loss": loss_a_val,
            "updates": legacy_steps,
            "color_support_audit": color_audit_a,
            "eval": leg_eval,
        },
        "arc_lut_192_to_64_induction": {
            "final_fit_loss": loss_b_val,
            "updates": lut_steps,
            "train_program_count": len(train_luts),
            "heldout_program_count": len(heldout_luts),
            "color_support_audit": color_audit_b,
            "eval": lut_eval,
        },
    }


def _encode_story_public(obs: dict, horizon: int = 12) -> list[int]:
    """Public-only story encoding (`1..10`); never accesses `secret_cause` or `ending`."""
    if "secret_cause" in obs or "ending" in obs:
        raise AssertionError("Host secret leaked into public storyworld observation")
    rem = max(0, min(9, horizon - int(obs["day"]) - 1))
    arr = [
        min(int(obs["day"]), 9),
        rem,
        max(0, min(9, int(obs["trust"]) + 2)),
        min(9, int(obs["evidence"])),
        min(9, int(obs["stability"])),
        int(bool(obs["token"])),
    ]
    return [x + 1 for x in (arr + [0] * 16)[:16]]


def train_and_eval_storyworld(
    root: int,
    device: torch.device | str,
    *,
    steps: int = 240,
    batch_size: int = 64,
    horizon: int = 12,
    n_eval_worlds: int = 64,
) -> dict:
    """Train on reachable public trajectories and evaluate closed-loop 12-step terminal endings."""
    dev = torch.device(device)
    actions = HiddenEndingWorld.ACTIONS

    # Build root-specific training pool from disjoint training world roots
    rng = random.Random(int(root) * 67 + 19)
    train_world_roots = [int(root) * 1_000_000 + 10_000 + i for i in range(512)]
    eval_world_roots = [int(root) * 1_000_000 + 900_000 + i for i in range(n_eval_worlds)]
    if set(train_world_roots) & set(eval_world_roots):
        raise AssertionError("Storyworld root overlap across splits")

    xs, ys = [], []
    for wr in train_world_roots:
        w = HiddenEndingWorld(wr, "train", horizon=horizon)
        for _ in range(horizon):
            obs = w.observe()
            if obs["terminal"]:
                break
            try:
                teacher_act = public_reachable_policy({**obs, "horizon": horizon})
            except ValueError:
                break
            xs.append(_encode_story_public(obs, horizon))
            ys.append(actions.index(teacher_act))
            chosen = teacher_act if rng.random() < 0.75 else rng.choice(actions)
            w.step(chosen)

    xp = torch.tensor(xs, dtype=torch.long, device=dev)
    yp = torch.tensor(ys, dtype=torch.long, device=dev)

    torch.manual_seed(int(root) * 71 + 6)
    model = PureSudokuTRM(
        hidden_size=64, num_heads=4, l_layers=1, h_cycles=1, l_cycles=1, puzzle_emb_len=1, seq_len=16
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator(device=dev).manual_seed(int(root) * 73 + 8)

    final_loss = 0.0
    for _s in range(steps):
        idx = torch.randint(0, len(xp), (batch_size,), generator=gen, device=dev)
        emb = model.encode_inputs(xp[idx])
        zh, zl = model.init_states(batch_size)
        loss = torch.tensor(0.0, device=dev)
        for d_step in range(1, 7):
            zh, zl = model.act_step(zh, zl, emb)
            if d_step in FIT_DEPTHS:
                loss = loss + F.cross_entropy(model.digit_logits(zh)[:, 0, :5], yp[idx])
        loss = loss / 3.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        final_loss = float(loss.detach().item())

    # Batched 12-step closed-loop evaluation across all 64 disjoint evaluation world roots!
    model.eval()
    by_depth = {}
    with torch.no_grad():
        for eval_d in SAME_CASE_DEPTHS:
            worlds = [HiddenEndingWorld(wr, "eval", horizon=horizon) for wr in eval_world_roots]
            teacher_matches = 0
            total_decisions = 0
            for _turn in range(horizon):
                obs_list = [w.observe() for w in worlds]
                enc_batch = torch.tensor(
                    [_encode_story_public(o, horizon) for o in obs_list], dtype=torch.long, device=dev
                )
                emb = model.encode_inputs(enc_batch)
                zh, zl = model.init_states(len(worlds))
                for _ in range(eval_d):
                    zh, zl = model.act_step(zh, zl, emb)
                pred_idx = model.digit_logits(zh)[:, 0, :5].argmax(-1).cpu().tolist()
                for w_obj, o_dict, p_i in zip(worlds, obs_list, pred_idx):
                    try:
                        t_act = public_reachable_policy({**o_dict, "horizon": horizon})
                        teacher_matches += int(actions[p_i] == t_act)
                    except ValueError:
                        pass
                    total_decisions += 1
                    w_obj.step(actions[p_i])

            endings = [w.ending() for w in worlds]
            succ_list = [e != "unfinished" for e in endings]
            hist = {k: endings.count(k) for k in sorted(set(endings))}
            by_depth[str(eval_d)] = {
                "terminal_episodes": len(worlds),
                "host_ending_success_rate": sum(succ_list) / float(len(worlds)),
                "closed_loop_teacher_agreement": teacher_matches / float(max(1, total_decisions)),
                "ending_histogram": hist,
                "per_episode_success": succ_list,
            }
    del model

    return {
        "family": "storyworld_closed_loop_control",
        "claim_boundary": "public-state multi-gate reachable control policy; not hidden-cause inference",
        "updates": steps,
        "final_fit_loss": final_loss,
        "horizon": horizon,
        "eval_worlds_count": n_eval_worlds,
        "eval_by_depth": by_depth,
    }
