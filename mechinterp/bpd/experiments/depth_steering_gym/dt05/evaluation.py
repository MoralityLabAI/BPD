"""Same-case depth trajectories and a physically compacted, public-validity policy.

Gold labels enter scoring only. The first-valid policy receives native input
tokens and public Sudoku constraints. Evaluation retains the full raw trajectory
as well as a separate policy replay so case-step savings can be checked directly.
"""
from __future__ import annotations

import math
import time

import torch
from torch import Tensor, nn

from .resources import MemoryBudget, assert_budget, memory_snapshot
from .techniques import FirstValidStopper, HeadNullProjector, sudoku_valid_mask, transport_basis_jvp


def _device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tokens_contract(model: nn.Module, tokens: Tensor) -> int:
    if tokens.ndim != 2 or len(tokens) == 0 or tokens.is_floating_point() or tokens.is_complex() or tokens.dtype == torch.bool:
        raise ValueError("Native input tokens must be a nonempty integer [batch,cells] tensor")
    size = math.isqrt(tokens.shape[1])
    if size not in (4, 9) or size * size != tokens.shape[1] or int(model.seq_len) != tokens.shape[1]:
        raise ValueError("Model and tokens must describe matching 4x4 or 9x9 boards")
    if bool(((tokens < 1) | (tokens > size + 1)).any()):
        raise ValueError("Native Sudoku tokens use blank1 and digit+1")
    # This checks public clue consistency before any model execution.
    FirstValidStopper(tokens - 1)
    return size


def _predict(model: nn.Module, high: Tensor, size: int, batch: int) -> Tensor:
    logits = model.digit_logits(high)
    if logits.ndim != 3 or logits.shape[:2] != (batch, size * size) or logits.shape[-1] < size:
        raise ValueError("Model digit logits have the wrong shape")
    if not bool(torch.isfinite(logits[..., :size]).all()):
        raise RuntimeError("Model produced nonfinite Sudoku logits")
    return logits[..., :size].argmax(dim=-1) + 1


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@torch.no_grad()
def rollout_first_valid(model: nn.Module, tokens: Tensor, *, max_depth: int) -> dict:
    """Run only unfinished boards; return real-digit predictions and work counts.

    The policy is deterministic given the model and inputs. All selection uses
    public constraints. ``executed_case_steps`` counts each active board at each
    native ACT step; it is not a CUDA FLOP or time estimate.
    """
    _positive_int(max_depth, "max_depth")
    size = _tokens_contract(model, tokens)
    device = _device(model)
    tokens = tokens.to(device=device)
    stopper = FirstValidStopper(tokens - 1)
    predictions = torch.zeros_like(tokens)
    active = torch.arange(len(tokens), device=device)
    case_steps = 0
    active_counts = []
    was_training = model.training
    model.eval()
    try:
        embed = model.encode_inputs(tokens)
        high, low = model.init_states(len(tokens))
        for depth in range(1, max_depth + 1):
            count = len(active)
            active_counts.append(count)
            case_steps += count
            high, low = model.act_step(high, low, embed)
            predictions[active] = _predict(model, high, size, count)
            stopper.update(predictions, depth)
            keep = ~stopper.valid_found[active]
            active = active[keep]
            if len(active) == 0:
                break
            high, low, embed = high[keep], low[keep], embed[keep]
        result = stopper.finish(predictions, depth)
        return {
            **result,
            "executed_case_steps": case_steps,
            "active_cases_per_step": active_counts,
            "executed_outer_steps": depth,
            "selection_uses_gold": False,
            "prediction_encoding": "digits1..N",
        }
    finally:
        model.train(was_training)


def _score(predictions: Tensor, givens: Tensor, gold_digits: Tensor) -> dict:
    correct = predictions.eq(gold_digits)
    blank = givens.eq(0)
    clues = ~blank
    valid = sudoku_valid_mask(predictions, givens)
    exact = correct.all(dim=-1)
    blank_correct = (correct & blank).sum(dim=-1)
    blank_total = blank.sum(dim=-1)
    clue_violations = ((predictions != givens) & clues).sum(dim=-1)
    clue_total = clues.sum(dim=-1)
    return {
        "n_cases": len(predictions),
        "exact_board_accuracy": float(exact.float().mean()),
        "blank_cell_accuracy": float(blank_correct.sum() / blank_total.sum()) if int(blank_total.sum()) else None,
        "clue_preservation": 1.0 - float(clue_violations.sum() / clue_total.sum()) if int(clue_total.sum()) else None,
        "valid_board_rate": float(valid.float().mean()),
        "per_case_exact": exact.tolist(),
        "per_case_valid": valid.tolist(),
        "per_case_blank_correct": blank_correct.tolist(),
        "per_case_blank_total": blank_total.tolist(),
        "per_case_clue_violations": clue_violations.tolist(),
        "per_case_clue_total": clue_total.tolist(),
        "predictions": predictions.tolist(),
        "prediction_encoding": "digits1..N",
    }


@torch.no_grad()
def evaluate_policy(model: nn.Module, tokens_cpu: Tensor, gold_cpu: Tensor, *,
                    depths: tuple[int, ...] = (2, 4, 6, 8, 12), microbatch: int = 16,
                    budget: MemoryBudget = MemoryBudget()) -> dict:
    """Score raw trajectories and compacted first-valid policy on every same case.

    Gold uses native zero-based digits0..N-1. No depth is selected using labels.
    Timings include Python validity checks and transfers, are synchronized on
    CUDA, and describe this evaluator call rather than a calibrated speedup.
    Memory ceilings are sampled between chunks and ACT steps.
    """
    _positive_int(microbatch, "microbatch")
    if tokens_cpu.device.type != "cpu" or gold_cpu.device.type != "cpu":
        raise ValueError("Evaluation inputs and labels must begin on CPU")
    size = _tokens_contract(model, tokens_cpu)
    if gold_cpu.shape != tokens_cpu.shape or gold_cpu.is_floating_point() or gold_cpu.is_complex() or gold_cpu.dtype == torch.bool:
        raise ValueError("Gold must be a matching zero-based integer tensor")
    if bool(((gold_cpu < 0) | (gold_cpu >= size)).any()):
        raise ValueError("Gold digits must lie in 0..N-1")
    if not bool(sudoku_valid_mask(gold_cpu + 1, tokens_cpu - 1).all()):
        raise ValueError("Gold must satisfy every public clue and Sudoku constraint")
    if not depths or any(type(d) is not int or d < 1 for d in depths) or tuple(sorted(set(depths))) != tuple(depths):
        raise ValueError("Depths must be distinct positive increasing integers")
    max_depth = max(depths)
    device = _device(model)
    initial_memory = memory_snapshot(str(device))
    assert_budget(initial_memory, budget)
    givens_cpu, gold_digits_cpu = tokens_cpu - 1, gold_cpu + 1
    raw_by_depth = {depth: [] for depth in range(1, max_depth + 1)}
    selected_predictions, selected_depths, found_valid = [], [], []
    final_period2, final_flips, ever_correct = [], [], []
    samples = [initial_memory]
    was_training = model.training
    model.eval()
    _sync(device)
    start = time.perf_counter()
    try:
        for offset in range(0, len(tokens_cpu), microbatch):
            inp = tokens_cpu[offset:offset + microbatch].to(device=device)
            gold_digits = gold_digits_cpu[offset:offset + microbatch].to(device=device)
            stopper = FirstValidStopper(inp - 1)
            embed = model.encode_inputs(inp)
            high, low = model.init_states(len(inp))
            history: list[Tensor] = []
            had_correct = torch.zeros(len(inp), device=device, dtype=torch.bool)
            for depth in range(1, max_depth + 1):
                high, low = model.act_step(high, low, embed)
                pred = _predict(model, high, size, len(inp))
                stopper.update(pred, depth)
                had_correct |= pred.eq(gold_digits).all(dim=-1)
                history.append(pred)
                history = history[-3:]
                if depth in raw_by_depth:
                    raw_by_depth[depth].append(pred.cpu())
                sample = memory_snapshot(str(device))
                assert_budget(sample, budget)
                samples.append(sample)
            stopped = stopper.finish(pred, max_depth)
            selected_predictions.append(stopped["predictions"].cpu())
            selected_depths.append(stopped["selected_depths"].cpu())
            found_valid.append(stopped["valid_found"].cpu())
            ever_correct.append(had_correct.cpu())
            flip = history[-1].ne(history[-2]).any(-1) if len(history) >= 2 else torch.zeros(len(inp), device=device, dtype=torch.bool)
            period = flip & history[-1].eq(history[-3]).all(-1) if len(history) >= 3 else torch.zeros_like(flip)
            final_flips.append(flip.cpu())
            final_period2.append(period.cpu())
        _sync(device)
        raw_seconds = time.perf_counter() - start
        stopped_predictions = torch.cat(selected_predictions)
        stopped_depths = torch.cat(selected_depths)
        stopped_valid = torch.cat(found_valid)
        raw_predictions = {d: torch.cat(values) for d, values in raw_by_depth.items()}

        compact_predictions, compact_depths, compact_valid = [], [], []
        compact_steps = 0
        compact_active_counts = []
        _sync(device)
        start = time.perf_counter()
        for offset in range(0, len(tokens_cpu), microbatch):
            sample = memory_snapshot(str(device))
            assert_budget(sample, budget)
            compact = rollout_first_valid(model, tokens_cpu[offset:offset + microbatch], max_depth=max_depth)
            compact_predictions.append(compact["predictions"].cpu())
            compact_depths.append(compact["selected_depths"].cpu())
            compact_valid.append(compact["valid_found"].cpu())
            compact_steps += compact["executed_case_steps"]
            compact_active_counts.append(compact["active_cases_per_step"])
            sample = memory_snapshot(str(device))
            assert_budget(sample, budget)
            samples.append(sample)
        _sync(device)
        compact_seconds = time.perf_counter() - start
        compact_predictions_t = torch.cat(compact_predictions)
        compact_depths_t = torch.cat(compact_depths)
        compact_valid_t = torch.cat(compact_valid)
        compact_agreement = bool(torch.equal(compact_predictions_t, stopped_predictions)
                                 and torch.equal(compact_depths_t, stopped_depths)
                                 and torch.equal(compact_valid_t, stopped_valid))
        compact_disagreements = (compact_predictions_t.ne(stopped_predictions).any(-1)
                                 | compact_depths_t.ne(stopped_depths) | compact_valid_t.ne(stopped_valid))
        final_valid = sudoku_valid_mask(raw_predictions[max_depth], givens_cpu)
        final_correct = raw_predictions[max_depth].eq(gold_digits_cpu).all(-1)
        valid_destroyed = stopped_valid & ~final_valid
        correct_destroyed = torch.cat(ever_correct) & ~final_correct
        fixed_steps = len(tokens_cpu) * max_depth
        return {
            "schema": "BPD-DT05/policy-evaluation/v1",
            "evidence_class": "development same-case policy comparison",
            "n_cases": len(tokens_cpu),
            "depths": list(depths),
            "observed_depths": list(range(1, max_depth + 1)),
            "microbatch": microbatch,
            "by_depth": {str(d): _score(raw_predictions[d], givens_cpu, gold_digits_cpu) for d in raw_predictions},
            "first_valid": {
                **_score(compact_predictions_t, givens_cpu, gold_digits_cpu),
                "selected_depths": compact_depths_t.tolist(),
                "valid_found": compact_valid_t.tolist(),
                "compact_case_steps": compact_steps,
                "fixed_case_steps": fixed_steps,
                "case_step_fraction": compact_steps / fixed_steps,
                "active_cases_per_step_by_chunk": compact_active_counts,
                "selection_uses_gold": False,
                "compact_matches_trajectory_policy": compact_agreement,
                "compact_disagreement_count": int(compact_disagreements.sum()),
                "per_case_compact_disagreement": compact_disagreements.tolist(),
                "raw_trajectory_selected_predictions": stopped_predictions.tolist(),
                "raw_trajectory_selected_depths": stopped_depths.tolist(),
                "raw_final_previously_valid_destruction_count": int(valid_destroyed.sum()),
                "raw_final_previously_correct_destruction_count": int(correct_destroyed.sum()),
                "per_case_raw_final_valid_destroyed": valid_destroyed.tolist(),
                "per_case_raw_final_correct_destroyed": correct_destroyed.tolist(),
            },
            "tail": {
                "period2_board_fraction": float(torch.cat(final_period2).float().mean()),
                "per_case_period2": torch.cat(final_period2).tolist(),
                "final_step_board_flip_fraction": float(torch.cat(final_flips).float().mean()),
                "per_case_final_step_flip": torch.cat(final_flips).tolist(),
            },
            "raw_fixed_seconds": raw_seconds,
            "compact_policy_seconds": compact_seconds,
            "timing_scope": "single synchronized end-to-end evaluation including validity checks and transfers; no speedup claim",
            "memory": {
                "before": initial_memory,
                "after": samples[-1],
                "sample_count": len(samples),
                "sampled_max_host_rss_bytes": max(s["host_rss_bytes"] for s in samples),
                "sampled_max_cuda_reserved_bytes": max((s["cuda_reserved_bytes"] for s in samples if s["cuda_reserved_bytes"] is not None), default=None),
            },
        }
    finally:
        model.train(was_training)


def transport_diagnostic(model: nn.Module, one_train_tokens: Tensor, rank: int = 8) -> dict:
    """Inspect one training case's local H/L transport with no future labels.

    A fixed seeded random protected basis supplies an engineering diagnostic.
    It is neither a fitted task Gramian nor a measured steering intervention.
    """
    _positive_int(rank, "rank")
    _tokens_contract(model, one_train_tokens)
    if len(one_train_tokens) != 1:
        raise ValueError("Transport diagnostic requires exactly one training case")
    device = _device(model)
    tokens = one_train_tokens.to(device)
    with torch.no_grad():
        embed = model.encode_inputs(tokens).detach()
        high, low = model.init_states(1)
        high, low = model.act_step(high, low, embed)
        state = torch.cat((high.flatten(1), low.flatten(1)), dim=-1).detach()
    if rank > state.numel():
        raise ValueError("Rank exceeds state dimension")
    projector = HeadNullProjector(model.lm_head.weight)
    project = lambda vectors: projector.project_dual(vectors, int(model.total_len), int(model.puzzle_emb_len))
    gen = torch.Generator(device="cpu").manual_seed(260500031)
    candidate = torch.randn((rank, state.numel()), generator=gen, dtype=torch.float32).to(device=device, dtype=state.dtype)
    projected = project(candidate).T
    work_dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
    left, singular, _ = torch.linalg.svd(projected.to(work_dtype), full_matrices=False)
    initial_rank = int((singular > (1e-9 + 1e-6 * float(singular.max()))).sum())
    if initial_rank == 0:
        raise RuntimeError("The requested protected transport basis has zero rank")
    basis = left[:, :initial_rank].to(dtype=state.dtype)

    def step(packed):
        dual = packed.reshape(1, 2, model.total_len, model.hidden_size)
        h, l = model.act_step(dual[:, 0], dual[:, 1], embed)
        return torch.cat((h.flatten(1), l.flatten(1)), dim=-1)

    _, metrics = transport_basis_jvp(step, state, basis, project=project)
    return {
        **metrics,
        "basis_source": "fixed-seed random protected training-state basis",
        "requested_rank": rank,
        "initial_projected_rank": initial_rank,
        "full_dual_state_dim": state.numel(),
        "head_row_basis_elements": projector.row_basis.numel(),
        "head_rank": projector.rank,
        "state_basis_elements": basis.numel(),
        "fitted_task_gramian": False,
        "future_labels_used": False,
        "measured_causal_benefit": False,
    }
