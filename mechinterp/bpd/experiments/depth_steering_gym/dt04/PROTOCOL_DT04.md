# BPD-DT04: competent depth-transport and protected causal steering across 59 calibrated TRM roots

**Prospective protocol, 19 November 2026.** Commit this protocol and its execution lock (`EXECUTE_LOCK.json`) before initializing any of the 59 registered development/calibration roots (`2604000..2604058`). All archived DT01 (`LOCK_STAGE0.json`), DT02-A (`LOCK_CI.json`, `results/dt02a_native_excluded.json`), DT03 (`dt03_causal.py`, `dt03_tasks.py`) and UI (`ui_server.py`, `ui/index.html`) files remain byte-identical. The ARG-03-N 168-case dataset and all prior confirmation cohorts remain strictly sealed.

## Scientific motivation and hypothesis

BPD-DT02-A trained six 14,082-parameter mini `PureSudokuTRM` checkpoints for 64 CPU updates and failed the prerequisite task-competence gate (`0/64` exact Sudoku4, `0/64` exact synthetic ARC, `0/16` closed-loop terminal Storyworld), while suffering canonical puzzle leakage (`57/64` and `53/64` isomorphic test siblings), clue-copying uniform loss, omitted ARC demonstrations, and an incomplete H-only steering probe lacking both fitted BPD `(V, U)` factors and a signed full-task-Jacobian sham. BPD-DT03 repaired the source contracts (`dt03_causal.py`, `dt03_tasks.py`) without training competent models.

BPD-DT04 executes the prospective Stage-1 development and 59-root sham-calibration study required by `PROTOCOL.md`, `DT03_REVIEW.md`, and `RESULTS_CI.md`:

1. **Competence and depth-transport admission (`fit = {2, 4, 6}` -> unseen `test = {8, 12}`):** Raise native `PureSudokuTRM` competence above `85%` exact whole-case accuracy on canonical-disjoint 4×4 Sudoku, demo-conditioned synthetic ARC in-distribution grids, and 12-step closed-loop Storyworld terminal endings, recording same-case predictions at depths `6, 8, 12`.
2. **ARC specification vs induction dissection:** Pair the strict `worlds.arc_analogue` two-rule split (`flip_h, recolor_1_2` -> `rotate_180, fill_zero`) with an explicit DT03 color-support audit (`color 3` absent in training targets) alongside a 4-color-balanced `192 -> 64` lookup-program split serialized through the exact 127-token `dt03_tasks.arc_demo_tokens` format (`t - 1` onto `0..10`).
3. **Complete-state protected causal steering vs privileged signed-Jacobian shams:** On each root's frozen Sudoku4 `PureSudokuTRM`, pack the complete recurrent state `DualState(z_h, z_l)` after ACT step `t0 = 1`, lock the puzzle prefix, and restrict edits to the exact orthogonal kernel `Q = basis(ker DP)` of the complete 11-vocabulary normalized cell softmax across all 16 board cells (`max_abs <= 1e-6`). Select the rank-8 physical task-Gramian block strictly over supervised continuation offsets `delta_d = (1, 3, 5)` (total depths `2, 4, 6`) via `dt03_causal.select_full_coordinate_block`. Compare against:
   - `no_edit` (corrupted dark-fiber recipient baseline),
   - `full_dark_donor` (complete `ker(DP)` transplant positive closure control),
   - `bpd_vu_r8_dose` (actual `BPDModel / TiedComponentLinear` physical `(V, U)` factors refit on `layers.{0,1}.{o_proj, down_proj}` strictly at depths `<= 6` with verified invertible `GL(C)` factor-gauge covariance, L2-dose matched per case),
   - `weight_svd_left_r8_dose` (`dt03_causal.output_weight_basis` left singular vectors, L2-dose matched),
   - `random_protected_r8_dose` (Haar random rank-8 subspace in `ker(DP)`, L2-dose matched), and
   - **eight independent privileged signed full-task-Jacobian and L2-dose matched shams** (`signed_jac_sham_0..7`), matched to `task_gramian_r8` across all `64` task output coordinates at supervised boundary depth `6` (`delta_d = 5`) and certified casewise via `dt03_causal.validate_delivered_sham` (`<= 1e-4`).
4. **Empirical hardware envelope and local SHA-verified 7M TRM bridge (Track B):** Measure actual RTX 5080 VRAM and host RSS across `14K`, `242K`, and `7M` TRM configurations (`BF16` and `FP32`, batches `1, 4, 16`) against the `UI_README.md` arithmetic lower bounds, and audit `NativeSudokuTRMBridge` offline (`HF_HUB_OFFLINE=1`) on locally cached SHA-verified `yagizdevre` and `gaoxin492` checkpoints over 16 fresh isomorphism-screened 9×9 Sudoku boards (`8 S4_stalled + 8 WB-Prop`, seed `2604700`).

## Registered population and cohort seals

- **Excluded engineering smoke:** `2603999` (excluded from all scientific summaries).
- **Development and 95/95 sham-calibration cohort:** **59 independent model roots `2604000..2604058`** (`1 - 0.95^59 = 0.9515 >= 0.95`, the minimum sample size for a nonparametric 95th-percentile / 95%-confidence sham upper tolerance bound in `EvalMath`). Every root is attempted once; failures remain in the archive without replacement.
- **Prospective confirmation cohort:** `2604100..2604355` (`256` roots) — **strictly sealed, ungenerated, and forbidden** absent a separate post-preflight human release manifest.
- **Compute and memory envelope:** Executed on NVIDIA GeForce RTX 5080 Laptop GPU (16 GB VRAM), Python 3.12, Torch `2.11.0+cu128` (`allow_tf32 = False`), NumPy `2.5.3`, with hard safety ceilings at `12.0 GiB` host RSS (within ~22 GiB free DRAM) and `12.5 GiB` reserved CUDA VRAM.

## Registered 6-endpoint family and decision rule

The independent unit is one trained model root (`n = 59`). All six directional endpoints are evaluated via `16,384` whole-root percentile bootstrap resamples (`PCG64`, seed `26040019 + 101*j`) at two-sided familywise Bonferroni coverage `1 - 0.05/6` (tails `0.05/12` and `1 - 0.05/12`):

| Endpoint | Root contrast | Passing lower bound |
|---|---|---:|
| `E1_gramian_vs_no_edit_d8_exact` | `exact(gramian_r8, d=8) - exact(no_edit, d=8)` | `> +0.10` |
| `E2_gramian_vs_signed_sham_d8_exact` | `exact(gramian_r8, d=8) - mean_8 exact(signed_sham, d=8)` | `> +0.10` |
| `E3_gramian_vs_signed_sham_d12_exact` | `exact(gramian_r8, d=12) - mean_8 exact(signed_sham, d=12)` | `> +0.10` |
| `E4_gramian_vs_signed_sham_d8_blank_gain` | `net_blank_gain(gramian_r8, d=8) - mean_8 net_blank_gain(signed_sham, d=8)` | `> +0.05` |
| `E5_bpd_vu_vs_signed_sham_d8_exact` | `exact(bpd_vu_r8, d=8) - mean_8 exact(signed_sham, d=8)` | `> +0.10` |
| `E6_gramian_vs_signed_sham_d6_fit_exact` | `exact(gramian_r8, d=6) - mean_8 exact(signed_sham, d=6)` | `> +0.10` |

`DEVELOPMENT_MECHANISM_GATE_PASSED_PENDING_HUMAN_RELEASE` requires all `59/59` roots admitted, `E1`, `E2`, and `E3` passing their familywise lower bounds, and pinned `EvalMath` returning `READY_FOR_CONFIRMATION` on `E2`. If competence gates pass while the signed-Jacobian sham explains the rank-8 subspace's unseen-depth recovery (replicating CGC-02's mechanism failure mode on competent native TRMs), the study records `COMPETENCE_ADMITTED_STEERING_PILOT_ONLY_OR_NULL` and quantifies principal-angle subspace rotation and one-step commutator leakage across depths `6, 8, 12`.
