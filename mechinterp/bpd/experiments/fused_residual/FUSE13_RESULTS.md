# BPD FUSE-13: Precision-Aware Retained Residuals — Results

- **Campaign commit:** `a2b43953f2c042435541423f3f6a665666b902a8`
- **Hardware & lease:** NVIDIA GeForce RTX 5080 Laptop GPU (16 GB VRAM, Windows WDDM), VRAMorama strict FIFO (`12 GiB` lease, `2048 MiB` headroom, `>=60 W` power guard, exit code `75` pause).
- **Production defaults unchanged:** `allow_cuda_residual_cache=False`, `residual_cache_storage_mode="fp32"`, all residual caches default off.
- **Tolerances preserved:** `ATOL=5e-5`, `RTOL=5e-4`; finite byte equality (`view(uint8)`) required for mixed-precision admission; nonfinite overflow parity reported separately and never counted as finite pass.

## Executive Gate Summary

| Stage | Scope | Status | Evidence Receipt |
|---|---|---|---|
| Stage 0 | FUSE-12 receipt reconciliation & audit | **PASS** (50 unique cells, BF16 C/F/G=50/52/52, FP16=42/42/42, FP16 384 nonfinite outcomes) | `receipt_fuse13_audit.json` |
| Stage A | Native `observe_clean_pass` mixed-precision EMA repair (CUDA FP32/TF32/BF16/FP16) | **PASS** (48/48 trainer cells, 4/4 unit suites) | `receipt_fuse13_stage_a_cuda.json` |
| Stage B/C Smoke | Prepared 16-cell CUDA microprobe smoke (`run_fuse13_gym --tier smoke`) | **16/16 completed** (errors=0) | `receipt_fuse13_cuda_smoke.json` |
| Stage B/C Gate | 6-strategy ablation + 96-case reg + 48-case holdout + FP16 GradScaler + matched trainer A vs G | **FP32=PASS, TF32=PASS, BF16=PASS, FP16=HOLD/OVERFLOW**; trainer `15/15` finite bitwise | `receipt_fuse13_stage_bc_cuda.json` |
| Stage C Stress | Frozen 2,304-cell microprobe stress matrix (`4 modes x 3 widths x 4 depths x 4 roots x 4 scales x 3 targets`) | **2304/2304 completed** (errors=0) | `receipt_fuse13_cuda_stress_summary.json` |
| Stage D | Width-4096 full trainer parity (4 cells) + preflighted frontier scaling (10 cells) + checkpoint audit | **COMPLETE** (zero forced WDDM spill) | `receipt_fuse13_frontier.jsonl` |
| Stage E | DataGoblin physical cost model + learned correction (`correctness_path=False`) | **COMPLETE** (303 rows, 706 site samples) | `receipt_fuse13e_predictor.json` |

## Stage 0: FUSE-12 Evidence Reconciliation

- Audit pass: `True` (`50` distinct cells: `38` FP32 ok, `4` TF32 ok, `2` BF16 ok, `2` FP16 ok, `2` BF16 `native_autocast_error`, `2` FP16 `native_autocast_error`).
- Width-512 FUSE-12B trainer failures (reconciled in `FUSE12_RESULTS.md`): BF16 `C=50, F=52, G=52` (sum `154`); FP16 `C=42, F=42, G=42` (sum `126`).
- FP16 FUSE-12B stress `nonfinite_parity_tensors`: `384` (matched nonfinite outcomes, explicitly excluded from finite passes).

## Stage A: Native Mixed-Precision Calibration (`observe_clean_pass`)

Repaired `TiedComponentLinear.observe_clean_pass` (`mechinterp/bpd/components.py`) casts completed `acts_mean`, `delta_mean`, and `write_rms` statistics to each FP32 calibration buffer's dtype before `lerp_`/`copy_`, preserving native autocast reduction semantics and exact FP32/TF32 zero-copy invariance.

| Mode | Raw `acts_mean` dtype | Raw `delta_mean` dtype | Raw `write_rms` dtype | Buffer dtypes | Expansion `max_passes` | Unit Status |
|---|---|---|---|---|---|---|
| `fp32` | `torch.float32` | `torch.float32` | `torch.float32` | `float32` | `6` | **PASS** |
| `tf32` | `torch.float32` | `torch.float32` | `torch.float32` | `float32` | `6` | **PASS** |
| `bf16` | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `float32` | `6` | **PASS** |
| `fp16` | `torch.float16` | `torch.float16` | `torch.float32` | `float32` | `6` | **PASS** |

- Native Arm A trainer matrix (CUDA, widths `128, 512`, `closed_loop` & `clean_window`, cadences `off/1/50`, 3 steps each): **48/48 passed**, **48/48 finite losses**, 100% FP32 finite calibration buffers.

## Stage B: Precision-Aware Residual Storage & Autograd Identity Ablation

Measured inside `TiedComponentLinear` (`width=128, passes=12`) on CUDA across all 6 Stage B strategies:

| Mode | Strategy | Recons | Unique Sq Storages | Unique Sq Bytes | Saved Dtypes | Peak Alloc (MiB) | Finite Bitwise | `g_V` max_abs | `g_U` max_abs |
|---|---|---:|---:|---:|---|---:|---|---:|---:|
| `fp32` | `native` | 12 | 12 | 786,432 | `torch.float32` | 17.86 | ref | 0.0 | 0.0 |
| `fp32` | `shared_fp32` | 1 | 1 | 65,536 | `torch.float32` | 17.35 | **YES** | 0 | 0 |
| `fp32` | `shared_compute_per_use_cast` | 1 | 1 | 65,536 | `torch.float32` | 17.51 | **YES** | 0 | 0 |
| `fp32` | `shared_compute_shared_cast` | 1 | 1 | 65,536 | `torch.float32` | 17.37 | **YES** | 0 | 0 |
| `fp32` | `shared_factor_casts_only` | 12 | 12 | 786,432 | `torch.float32` | 18.02 | **YES** | 0 | 0 |
| `fp32` | `native_graph_replay` | 1 | 1 | 65,536 | `torch.float32` | 17.55 | **YES** | 0 | 0 |
| `tf32` | `native` | 12 | 12 | 786,432 | `torch.float32` | 18.04 | ref | 0.0 | 0.0 |
| `tf32` | `shared_fp32` | 1 | 1 | 65,536 | `torch.float32` | 17.42 | **YES** | 0 | 0 |
| `tf32` | `shared_compute_per_use_cast` | 1 | 1 | 65,536 | `torch.float32` | 17.59 | **YES** | 0 | 0 |
| `tf32` | `shared_compute_shared_cast` | 1 | 1 | 65,536 | `torch.float32` | 17.75 | **YES** | 0 | 0 |
| `tf32` | `shared_factor_casts_only` | 12 | 12 | 786,432 | `torch.float32` | 18.53 | **YES** | 0 | 0 |
| `tf32` | `native_graph_replay` | 1 | 1 | 65,536 | `torch.float32` | 17.75 | **YES** | 0 | 0 |
| `bf16` | `native` | 12 | 12 | 393,216 | `torch.bfloat16` | 18.10 | ref | 0.0 | 0.0 |
| `bf16` | `shared_fp32` | 1 | 12 | 393,216 | `torch.bfloat16` | 18.25 | **NO** | 0.818306 | 0.340853 |
| `bf16` | `shared_compute_per_use_cast` | 1 | 1 | 32,768 | `torch.bfloat16` | 18.15 | **NO** | 0.804688 | 0.351562 |
| `bf16` | `shared_compute_shared_cast` | 1 | 1 | 32,768 | `torch.bfloat16` | 17.96 | **YES** | 0 | 0 |
| `bf16` | `shared_factor_casts_only` | 12 | 12 | 393,216 | `torch.bfloat16` | 18.23 | **YES** | 0 | 0 |
| `bf16` | `native_graph_replay` | 1 | 1 | 32,768 | `torch.bfloat16` | 18.02 | **YES** | 0 | 0 |
| `fp16` | `native` | 12 | 12 | 393,216 | `torch.float16` | 18.48 | ref | 0.0 | 0.0 |
| `fp16` | `shared_fp32` | 1 | 12 | 393,216 | `torch.float16` | 18.63 | **NO** | 0.0907097 | 0.0379553 |
| `fp16` | `shared_compute_per_use_cast` | 1 | 1 | 32,768 | `torch.float16` | 18.53 | **NO** | 0.101562 | 0.0358887 |
| `fp16` | `shared_compute_shared_cast` | 1 | 1 | 32,768 | `torch.float16` | 18.13 | **YES** | 0 | 0 |
| `fp16` | `shared_factor_casts_only` | 12 | 12 | 393,216 | `torch.float16` | 18.38 | **YES** | 0 | 0 |
| `fp16` | `native_graph_replay` | 1 | 1 | 32,768 | `torch.float16` | 18.17 | **YES** | 0 | 0 |

## Stage C: Factor-Gradient Parity, Stress Suites, GradScaler & Matched Trainer Gates

### 1. `StepResidualPool(storage_mode="compute_dtype")` Stress Suites (Regression 96 cases + Holdout 48 cases)

| Suite | Mode | Cases | Tensors | Finite Bitwise | Within Tol | Overflow Parity (nonfinite) | Failures | Max Abs (finite) | Admission Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Regression (4012001..8) | `fp32` | 96 | 1440 | 1440 | 1440 | 0 | 0 | 0.000e+00 | **PASS** |
| Regression (4012001..8) | `tf32` | 96 | 1440 | 1440 | 1440 | 0 | 0 | 0.000e+00 | **PASS** |
| Regression (4012001..8) | `bf16` | 96 | 1440 | 1440 | 1440 | 0 | 0 | 0.000e+00 | **PASS** |
| Regression (4012001..8) | `fp16` | 96 | 1440 | 1080 | 1080 | 360 | 0 | 0.000e+00 | **OVERFLOW_ONLY_HOLD** |
| Holdout (4013001..4) | `fp32` | 48 | 720 | 720 | 720 | 0 | 0 | 0.000e+00 | **PASS** |
| Holdout (4013001..4) | `tf32` | 48 | 720 | 720 | 720 | 0 | 0 | 0.000e+00 | **PASS** |
| Holdout (4013001..4) | `bf16` | 48 | 720 | 720 | 720 | 0 | 0 | 0.000e+00 | **PASS** |
| Holdout (4013001..4) | `fp16` | 48 | 720 | 540 | 540 | 180 | 0 | 0.000e+00 | **OVERFLOW_ONLY_HOLD** |

### 2. FP16 Dynamic `GradScaler` Skip/Update Parity

| Case | Ref Skipped | Cand Skipped | Skip Match | Scale After Match | `V` Bitwise | `U` Bitwise |
|---|---|---|---|---|---|---|
| `safe_update` | `False` | `False` | `True` | `True` | `True` | `True` |
| `overflow_skip` | `True` | `True` | `True` | `True` | `True` | `True` |

### 3. Full Matched Native Arm A vs Arm G (`storage_mode="compute_dtype"`) Trainer Parity

| Mode | Width | Passes | Regime | Cal | Budget | Tensors Finite Bitwise | Overflow Parity | Failures | Max Abs | Arm A Step (ms) | Arm G Step (ms) | Speedup | Arm A Peak Alloc (MiB) | Arm G Peak Alloc (MiB) | VRAM Saved (MiB) |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fp32` | 512 | 12 | `closed_loop` | `1` | `full` | 186/186 | 0 | 0 | 0.000e+00 | 3073.37 | 3132.46 | 0.981x | 363.5 | 195.6 | +167.9 |
| `tf32` | 512 | 12 | `closed_loop` | `1` | `full` | 186/186 | 0 | 0 | 0.000e+00 | 3365.59 | 3179.03 | 1.059x | 363.5 | 195.6 | +167.9 |
| `bf16` | 128 | 6 | `closed_loop` | `1` | `full` | 182/182 | 0 | 0 | 0.000e+00 | 1390.06 | 1416.90 | 0.981x | 41.7 | 39.9 | +1.8 |
| `bf16` | 128 | 6 | `clean_window` | `50` | `full` | 154/154 | 0 | 0 | 0.000e+00 | 197.74 | 229.11 | 0.863x | 31.0 | 30.7 | +0.3 |
| `bf16` | 512 | 12 | `closed_loop` | `1` | `binding` | 186/186 | 0 | 0 | 0.000e+00 | 3980.30 | 3816.52 | 1.043x | 505.9 | 468.6 | +37.3 |
| `bf16` | 512 | 12 | `closed_loop` | `off` | `full` | 186/186 | 0 | 0 | 0.000e+00 | 983.39 | 1045.81 | 0.940x | 498.2 | 414.4 | +83.8 |
| `bf16` | 1024 | 12 | `closed_loop` | `1` | `binding` | 186/186 | 0 | 0 | 0.000e+00 | 4000.88 | 4430.99 | 0.903x | 1742.8 | 1563.5 | +179.3 |
| `bf16` | 2048 | 12 | `closed_loop` | `off` | `full` | 186/186 | 0 | 0 | 0.000e+00 | 872.24 | 1062.50 | 0.821x | 6301.3 | 4985.0 | +1316.3 |
| `bf16` | 2048 | 24 | `closed_loop` | `off` | `full` | 194/194 | 0 | 0 | 0.000e+00 | 1914.20 | 1997.52 | 0.958x | 11995.4 | 9263.8 | +2731.6 |
| `fp16` | 128 | 6 | `closed_loop` | `1` | `full` | 182/182 | 0 | 0 | 0.000e+00 | 1495.21 | 1448.65 | 1.032x | 41.7 | 39.9 | +1.8 |
| `fp16` | 128 | 6 | `clean_window` | `50` | `full` | 154/154 | 0 | 0 | 0.000e+00 | 216.18 | 220.47 | 0.981x | 31.0 | 30.7 | +0.3 |
| `fp16` | 512 | 12 | `closed_loop` | `1` | `binding` | 186/186 | 0 | 0 | 0.000e+00 | 3942.91 | 4120.78 | 0.957x | 505.9 | 468.6 | +37.3 |
| `fp16` | 512 | 12 | `closed_loop` | `off` | `full` | 186/186 | 0 | 0 | 0.000e+00 | 1010.96 | 1086.23 | 0.931x | 498.2 | 414.4 | +83.8 |
| `fp16` | 1024 | 12 | `closed_loop` | `1` | `binding` | 186/186 | 0 | 0 | 0.000e+00 | 4100.23 | 4127.42 | 0.993x | 1742.8 | 1563.5 | +179.3 |
| `fp16` | 2048 | 12 | `closed_loop` | `off` | `full` | 186/186 | 0 | 0 | 0.000e+00 | 938.42 | 1102.55 | 0.851x | 6301.3 | 4985.0 | +1316.3 |

### 4. Frozen Microprobe Stress Matrix (`2,304` Cells)

| Mode | Cells | Per-Use Cast Finite Bitwise | Shared Cast Finite Bitwise | Shared Nonfinite Cells | Shared Overflow Parity | Shared Finite Mismatches | Storage Collapsed (1 uniq) | Shared Max Abs (finite) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bf16` | 576 | 144/576 | 576/576 | 0 | 0 | 0 | 576/576 | 0.000e+00 |
| `fp16` | 576 | 123/576 | 447/576 | 129 | 129 | 0 | 576/576 | 0.000e+00 |
| `fp32` | 576 | 576/576 | 576/576 | 0 | 0 | 0 | 576/576 | 0.000e+00 |
| `tf32` | 576 | 576/576 | 576/576 | 0 | 0 | 0 | 576/576 | 0.000e+00 |

## Stage D: Width-4096 Parity, Frontier Scaling (Preflighted, Zero Forced Spill) & Checkpoint Audit

- **Pretrained TRM checkpoint audit:** `blocked_no_local_checkpoint` — Neither gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att (step_21700) nor yagizdevre/trm-sudoku-extreme-att (model.pt) is present in local caches; offline/no-download policy respected.

### Width-4096 Full Trainer Parity

| Mode | Width | Passes | Status | Finite Bitwise Tensors | Failures | Arm A Peak Alloc (MiB) | Arm G Peak Alloc (MiB) |
|---|---:|---:|---|---:|---:|---:|---:|
| `fp32` | 4096 | 2 | **PASS** | 178/178 | 0 | 4291.9 | 3256.9 |
| `fp32` | 4096 | 6 | **PASS** | 182/182 | 0 | 8330.1 | 3544.4 |
| `bf16` | 4096 | 2 | **PASS** | 178/178 | 0 | 5966.1 | 5418.1 |
| `bf16` | 4096 | 6 | **PASS** | 182/182 | 0 | 13261.2 | 10841.7 |

### Frontier Scaling Matrix (Synthetic Mini-TRM, Independent Per-Arm Preflight)

| Mode | Width | Passes | Arm A Status / Median (ms) | Arm G Status / Median (ms) | Speedup (95% CI) | Arm A Alloc / Res / NVML (MiB) | Arm G Alloc / Res / NVML (MiB) | Arm G Energy (mJ/step) |
|---|---:|---:|---|---|---|---|---|---|
| `fp32` | 2048 | 48 | `memory_blocked_preflight` (est 13382 MiB) | `ok` (3851.2 ms) | n/a (A blocked) | blocked | 2415 / 2554 / 2806 | 276895.2 |
| `fp32` | 4096 | 24 | `memory_blocked_preflight` (est 25837 MiB) | `ok` (3156.0 ms) | n/a (A blocked) | blocked | 4703 / 4828 / 5080 | 232584.1 |
| `fp32` | 4096 | 48 | `memory_blocked_preflight` (est 49289 MiB) | `ok` (6246.5 ms) | n/a (A blocked) | blocked | 6305 / 6680 / 6932 | 475114.9 |
| `fp32` | 6144 | 24 | `memory_blocked_preflight` (est 56472 MiB) | `ok` (8026.0 ms) | n/a (A blocked) | blocked | 9316 / 9780 / 10032 | 615240.5 |
| `fp32` | 8192 | 12 | `memory_blocked_preflight` (est 53848 MiB) | `memory_blocked_runtime_headroom` (est 11112 MiB) | n/a (A blocked) | blocked | blocked | n/a |
| `bf16` | 2048 | 48 | `memory_blocked_preflight` (est 7622 MiB) | `memory_blocked_runtime_headroom` (est 1966 MiB) | n/a (A blocked) | blocked | blocked | n/a |
| `bf16` | 4096 | 24 | `memory_blocked_preflight` (est 14509 MiB) | `memory_blocked_runtime_headroom` (est 3589 MiB) | n/a (A blocked) | blocked | blocked | n/a |
| `bf16` | 4096 | 48 | `memory_blocked_preflight` (est 78089 MiB) | `memory_blocked_preflight` (est 55841 MiB) | n/a (A blocked) | blocked | blocked | n/a |
| `bf16` | 6144 | 24 | `memory_blocked_preflight` (est 88728 MiB) | `memory_blocked_preflight` (est 64296 MiB) | n/a (A blocked) | blocked | blocked | n/a |
| `bf16` | 8192 | 12 | `memory_blocked_preflight` (est 82648 MiB) | `memory_blocked_preflight` (est 61624 MiB) | n/a (A blocked) | blocked | blocked | n/a |

**Key Stage D memory finding (FP32 vs autocast BF16 at frontier scale):** In FP32, Arm G collapses live residual storages from `2 * T` to `1`, fitting `(2048, 48)`, `(4096, 24)`, `(4096, 48)`, and `(6144, 24)` cleanly inside the `12 GiB` lease (`2,806–10,032 MiB` NVML) while native Arm A is preflight-blocked (`13.4–56.5 GiB`). Under `autocast(bf16)` on an FP32 base model, Arm G still saves `2.62–3.27 GiB` NVML on the wrapped residual sites (`w=2048, T=24`: `13,102 -> 9,832 MiB`; `w=4096, T=6`: `14,336 -> 11,716 MiB`), but PyTorch `autocast` retains per-pass BF16 cast tensors for the *unwrapped* frozen layers (`qkv_proj`, `gate_up_proj`, ~`285 MiB/pass` at `w=2048`, ~`1,136 MiB/pass` at `w=4096`), causing all 5 BF16 frontier cells (`(2048, 48)` through `(8192, 12)`) to exceed the `12 GiB` lease without spilling.

## Stage E: DataGoblin Physical Cost Model + Learned Admission Correction

- **Status:** `experimental_non_correctness_path` (`correctness_path=False` — strictly outside numerical correctness path).
- **Dataset:** `303` rows, `706` site samples (`fuse13e_datagoblin_dataset.jsonl`).
- **Holdout family status:** hardware=`blocked_single_gpu_rtx5080`, checkpoint=`blocked_single_family_synthetic_only`.

| Holdout | Policy | Oracle Step Fraction | Oracle Recon Fraction | Worst-Case Step Regret (ms) | Unsafe Admission Rate |
|---|---|---:|---:|---:|---:|
| Leave-one-width-out | `physical_plus_learned_gated` | 0.9591 | 0.9560 | 234.343 | 0.0000 |
| Leave-one-width-out | `physical_plus_learned` | 0.9668 | 0.9724 | 234.343 | 0.0162 |
| Leave-one-width-out | `physical_prior` | 0.9780 | 0.9801 | 234.343 | 0.0162 |
| Leave-one-width-out | `fuse12c_log_ridge` | 0.9194 | 0.9242 | 26275.660 | 0.0162 |
| Leave-one-width-out | `exact_saved_mac` | 0.9364 | 0.9226 | 234.343 | 0.0162 |
| Leave-one-width-out | `density` | 0.9379 | 0.9284 | 26510.003 | 0.0162 |
| Leave-one-width-out | `first_fit` | 0.9331 | 0.9360 | 26275.660 | 0.0162 |
| Leave-one-depth-out | `physical_plus_learned_gated` | 0.9329 | 0.9234 | 26510.003 | 0.0000 |
| Leave-one-depth-out | `physical_plus_learned` | 0.9425 | 0.9406 | 26510.003 | 0.0169 |
| Leave-one-depth-out | `physical_prior` | 0.9850 | 0.9839 | 234.343 | 0.0169 |
| Leave-one-depth-out | `fuse12c_log_ridge` | 0.9209 | 0.9256 | 26275.660 | 0.0169 |
| Leave-one-depth-out | `exact_saved_mac` | 0.9505 | 0.9354 | 234.343 | 0.0169 |
| Leave-one-depth-out | `density` | 0.9439 | 0.9310 | 26510.003 | 0.0169 |
| Leave-one-depth-out | `first_fit` | 0.8969 | 0.9053 | 26275.660 | 0.0169 |
| Leave-one-precision-out | `physical_plus_learned_gated` | 0.6440 | 0.7165 | 234.343 | 0.0000 |
| Leave-one-precision-out | `physical_plus_learned` | 0.9479 | 0.9554 | 234.343 | 0.2500 |
| Leave-one-precision-out | `physical_prior` | 0.9485 | 0.9556 | 234.343 | 0.2500 |
| Leave-one-precision-out | `fuse12c_log_ridge` | 0.9483 | 0.9134 | 26275.660 | 0.2500 |
| Leave-one-precision-out | `exact_saved_mac` | 0.9352 | 0.8970 | 234.343 | 0.2500 |
| Leave-one-precision-out | `density` | 0.9560 | 0.9167 | 26510.003 | 0.2500 |
| Leave-one-precision-out | `first_fit` | 0.9222 | 0.9253 | 26275.660 | 0.2500 |

### Safety-Gated Worth-Caching Classifier Calibration

| Holdout | Eval N | Accuracy | False Rejection Rate | Unsafe Admission Rate | Brier Score | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Width | 225 | 0.7244 | 0.3696 | 0.0000 | 0.2011 | 0.0591 |
| Depth | 225 | 0.6533 | 0.3478 | 0.0000 | 0.2427 | 0.1999 |
| Precision | 225 | 0.6178 | 0.9022 | 0.0000 | 0.2871 | 0.3031 |

## Admission Decisions & Production Defaults

- **FP32 (`allow_cuda=True`, `storage_mode="fp32"` or `"compute_dtype"`):** **PASS** (100% finite byte-identical across 96-case reg, 48-case holdout, 576 microprobe cells, matched trainer up to `w=4096, T=6`).
- **TF32 (`allow_cuda=True`):** **PASS** (100% finite byte-identical across 96-case reg, 48-case holdout, 576 microprobe cells, matched trainer).
- **BF16 (`allow_cuda=True`, `storage_mode="compute_dtype"`):** **EXPERIMENTAL OPT-IN PASS** (repaired native Stage A calibration + `_SharedCastPrimal` + 0-FLOP `_resolve_autocast_factors` sharing native `ToCopyBackward0`; `1440/1440` reg + `720/720` holdout + `576/576` microprobe + `9/9` matched trainer cells including `w=2048, T=24` and `w=4096, T=6` are 100% finite byte-identical with `max_abs=0.0` and `24 -> 1` storage collapse).
- **FP16 (`allow_cuda=True`, `storage_mode="compute_dtype"`):** **HOLD (overflow-bounded)** (`1080/1440` reg, `540/720` holdout, `447/576` microprobe, `6/6` matched trainer cells, and `2/2` dynamic `GradScaler` cases are 100% finite byte-identical with `0` mismatches; all `scale=10,000` nonfinite cells match native signed-Inf/NaN masks 100%, but because native FP16 itself overflows at `scale=10,000`, FP16 remains on hold outside dynamic-scaled regimes).
- **Production defaults unchanged:** `allow_cuda_residual_cache=False`, `residual_cache_storage_mode="fp32"`, all residual caches default off; Stage E DataGoblin predictor remains strictly experimental (`correctness_path=False`).

