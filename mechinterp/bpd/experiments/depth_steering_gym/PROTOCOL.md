# BPD-DT01: recurrent task steering under depth transport

**Stage 0 source and contract registration, 2026-11-18.** Motivated by the [CGC-02 first trained-model failure](https://github.com/MoralityLabAI/Compactification-Theory/pull/247): protected rank-two task blocks that recovered donor behavior at supervised depth 4 fell to ~14.9% root-mean recovery at held-out depth 8, and the signed two-task Jacobian-matched control explained their performance. This stage establishes the task, data, training and causal-evaluation API. It is **not a trained neural result**.

## Tasks and split contract

* **Sudoku:** exact validated, uniquely solvable 4x4 development examples; 9x9 pretrained [PureSudokuTRM](../../adapters/trm.py) bridge supported when the caller supplies a verified checkpoint and an official/legal puzzle corpus. Score full board exactness, per-cell accuracy, solver validity and later-loop oscillation. Group puzzles by generating root and exclude isomorphic sibling puzzles from opposite splits.
* **ARC-1:** accept user-provided public ARC-AGI-1 JSON, validate rectangular colors, use demonstration input/output pairs and test **input only**. The task's gold may reside in the *host evaluator*, never the model view. Synthetic ARC-style programs (`flip_h`, `recolor_1_2`) are development; held-out synthetic program families (`rotate_180`, `fill_zero`) test program transfer. Synthetic scores are never labeled official ARC-AGI-1 scores. One model/task endpoint may be measured only against an externally supplied held-out host gold.
* **Simple storyworlds:** exact short-world transition semantics (`listen`, `research`, `stabilize`, `share`, `advance`) with visible trust/evidence/stability, a private root-dependent gate and a terminal-only **hidden ending**. The model's observations omit `secret_cause`, target predicate and ending. A hash is a host provenance commitment, not a cryptographic hiding proof. Hold out all episodes sharing the same world root or story program from development.

## Training and interpretability contract

Every model adapter exposes `output_at_depth`, `protected_readout_at_depth`, and host-owned `step`/state intervention hooks. The original [7M TRM adapter](../../adapters/trm.py) already distinguishes unheaded inner-L states from digit readouts at H commits. Supervised stage-1 training fits **only** depths 2,4,6, and the held-out scorer queries 8,12. Freeze source/program group IDs, checkpoints, training budgets and rank selection before any model evaluation; the source-only harness enforces the split.

Let `P(h)` be the **actual complete native normalized output** at the nominated site (all Sudoku 81x9 digit logits/probabilities, every observable ARC cell label, or all public storyworld action/ending probabilities). Its current protected kernel is derived from a declared physical Jacobian, with an exact finite readout-equality test for every proposed intervention. The top-r protected task direction is built from the train-only averaged multi-depth Gramian

\[
G_{\rm fit}=\sum_{k\in\{2,4,6\}} w_k Q^\top J_k^\top J_k Q,
\quad J_k=D(T\circ F^k)(h),\quad Q=\operatorname{basis}(\ker DP).
\]

This is a local **linear sensitivity** optimum. It is not a closed state quotient. Record the one-step commutator/closure defect `||Pi_E DF (I-Pi_E)||`, principal-angle drift with `k=8,12` *after freezing development selection*, and the measured nonlinear finite patch behavior.

Compare the task-gramian rank-r block to a gauge-stable physical BPD `V@U` output-weight SVD, independent BPD refits, true full donor write, exact donor hidden-state patch, random protected subspaces, same-physical-dose Haar/random edits, and **privileged full signed task Jacobian and norm matched shams**. Every null is evaluated at the same donor pairs, token sites and depths. Original `BPDModel/TiedComponentLinear` execution and its native 7M checkpoint path remain independent engineering gates. Current readout equality and future public stability are separate metrics.

For each task, retain both model-vs-independent **gold correctness** and candidate-vs-clean **donor-model recovery**. A wrong donor model must not be counted as task progress. Protect the full normalized visible channel; a scalar-only head projection is insufficient when other outputs are observed. A full-rank BPD physical matrix reproducing the teacher is a reconstruction positive control, not sparse mechanism recovery.

## Training and steering readouts

Stage 1 records checkpoint IDs and hashes, train-only exact program roots, independent development/heldout root families, model budgets, action/answer masks, every generated/eligible pair and any failure to solve the base task. Measure depth-8/12 exact success, hidden-ending reachability, action-validity, factual leakage into the model observation, finite current and future public readout change, selected block dimension, physical dose, gauge drift and BPD component usage. Carry a comparable unmodified-model and randomly routed alternative across the same seeds. Any steering edit must be applied to a **frozen** model first; trained BPD/adapter edits are separately logged.

**Scientific gate to register in Stage 1:** a block must improve verified held-out task success over both the native unedited teacher and gradient-matched controls, remain stable under factor gauge/restart, and maintain protected readout/future trajectory fidelity within a fixed task-specific tolerance over unseen loop depths. A fixed-rank reconstruction without held-out advantage is a negative result. Do not mine held-out ARC-1 official solutions or hidden endings for training/tuning. New independent root calibration and multiplicity-adjusted confirmation are required for inferential claims.

## Current Stage 0 evidence

This code supplies exact task oracles, the official ARC-1 loader, concealed storyworld endpoints, supervised-depth loss and depth-split adapter, protected observer/kernel calculus, signed gradient matched shams, and a synthetic rotating-subspace regression demonstrating why depth extrapolation is necessary. Tests and `SOURCE_RECEIPT.json` are **source-only**, with no live 7M checkpoint, trained TRM, trained ARC solver, or detected hidden-ending mechanism. The CGC-02 4-root negative and earlier BPD SF03 0/32 strong incremental and SF04 3/8 public-safe results remain unchanged; keep ARG-03-N's 168 sealed cases untouched.
