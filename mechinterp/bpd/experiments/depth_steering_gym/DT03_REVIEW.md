# BPD-DT03: frozen-gym critique and repaired source contracts

November 18, 2026. Review of [BPD-DT02-A PR #103](https://github.com/MoralityLabAI/BPD/pull/103), source-frozen GitHub Actions run 44277919962. The original first-run scientific evidence and prior source hash locks remain unchanged.

## Original result and scope

Six real 14,082-parameter mini Attention-TRMs trained on two excluded engineering seeds (901/902), fit depths 2/4/6, evaluated at 8/12. Sudoku4 and synthetic ARC-style grid-exact success: 0/64 per root/depth. Storyworld independent action accuracy 61/64 and 59/64; full terminal success 0/16 each. Full current Sudoku 11-class head was protected to ~2.4e-7, while full protected-dark and rank-eight physical SVD edits failed to improve donor agreements at depths 8/12. Neither BPD fitted components nor a full signed Jacobian matched control were evaluated. Official ARC-1 and real 7M checkpoint results were excluded.

## Source defects demonstrated

**Full-output causal Jacobian.** The original stage-0 Gramian selector loops over the final class dimension of [batch,cell,class] task output, indexes a cell and sums its classes. If classes are normalized probabilities, their sum is one, giving a zero gradient even when each output probability depends on the state. The repair differentiates every output coordinate, detects cross-example coupling and checks finite physical derivatives.

**Wrong recovery unit.** The old patch evaluator reduces only the last output axis. Two-cell cases with one failed cell can report 75% coordinate recovery when only 50% of cases recover. The repair scores complete case recovery, retaining coordinate accuracy as secondary.

**Physical null and output-weight SVD.** A sham must match delivered full-state L2 dose, protected span and signed whole-task response, with a recorded candidate/sham separation. Invalid state components must not disappear through projection before checking norm. For y=Wx, output-side physical singular vectors are the left vectors. The original pilot used right vectors of o_proj as an output-space edit. The repaired helper selects left singular vectors.

**Incomplete native causal comparison.** The pilot applied recipient H-state edits with original recipient L and input, while its donor reference used different donor H/L and input. The successor explicitly packs both H and L and binds the continuation to recipient input. Full native transplant closure and restricted subspace transfer are separate tests.

**Sudoku group leakage and proxy.** In a follow-up audit of original fixture groups, 57/64 test puzzles for root 901 and 53/64 for root 902 had isomorphic training siblings, even though exact token-string overlaps were zero. The new canonical 4x4 identifier accounts for legal band/stack permutations, transpose and digit relabeling. The original uniform loss rewards copying ten known clues out of sixteen cells: 62.5% cell accuracy is achievable from clues alone. The successor scores blank-only cross entropy, clue preservation and whole-board exactness.

**ARC input specification.** The old pilot only encoded a test input grid. Demonstration input-output pairs were omitted even when distinct program families require different transformations of the same query. The successor encodes three public demonstration pairs and a marked test input as 127 tokens for bounded 4x4 synthetic ARC. Host answer stays outside the model view. A model trained only with transformations producing colors 0..2 must also have an explicit color-support audit before interpreting heldout rules producing color 3.

**Storyworld teacher/utility mismatch.** The old expert's extra advance actions can destroy terminal stability, and independent public-state label accuracy overweights listen. The successor uses an exact finite-horizon public-state policy satisfying all three possible hidden-gate predicates, with an explicit unreachable-deadline result. Because one public plan solves every gate, this is a control-world test rather than proof of hidden-cause inference.

**Depth and reproducibility.** DT02 has no same-case depth-6 competence readout. Its aggregated JSON and checkpoint hashes omit full per-item predictions and neural checkpoint bytes. New scientific claims require same-case depth 6/8/12 predictions, disjoint source groups, complete sham and physical dose ledger, and independent EvalMath root-level preflight.

## Additive changes

* dt03_causal.py: all-output Jacobians, physical Gramian, whole-case scoring, signed full-state sham checks, correct output-side SVD, full H/L packing and native recipient continuation.
* dt03_tasks.py: Sudoku isomorphism contract and blank loss, ARC demo serialization and terminal-reachable story policy.
* test_dt03_contract.py: deterministic regression tests.
* bpd-dt03-source.yml: source-only Python 3.11/3.13 CI with pinned CPU Torch and original Stage0/DT02 source-lock verification.

This branch changes no archived DT01/DT02 source file and draws no registered experimental roots. It must remain an engineering amendment until a newly source-registered model-training and mechanism protocol produces real competence and independent causal evidence. The ARG-03-N 168-case protected dataset remains sealed.
