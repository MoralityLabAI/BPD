# BPD FUSE-13: Precision-Aware Retained Residuals — Results

- **Campaign commit:** `d2eaa0c0186aae91ef7a7ff8be35d7b8c2ac52b0`
- **Hardware & lease:** NVIDIA GeForce RTX 5080 Laptop GPU (16 GB VRAM, Windows WDDM), VRAMorama strict FIFO (`12 GiB` lease, `2048 MiB` headroom, `>=60 W` power guard, exit code `75` pause).
- **Production defaults unchanged:** `allow_cuda_residual_cache=False`, `residual_cache_storage_mode="fp32"`, all residual caches default off.
- **Tolerances preserved:** `ATOL=5e-5`, `RTOL=5e-4`; finite byte equality (`view(uint8)`) required for mixed-precision admission; nonfinite overflow parity reported separately and never counted as finite pass.

## Executive Gate Summary

| Stage | Scope | Status | Evidence Receipt |
|---|---|---|---|
| Stage 0 | FUSE-12 receipt reconciliation & audit | **PASS** (50 unique cells, BF16 C/F/G=50/52/52, FP16=42/42/42, FP16 384 nonfinite outcomes) | `receipt_fuse13_audit.json` |
| Stage E | DataGoblin physical cost model + learned correction (`correctness_path=False`) | **COMPLETE** (283 rows, 706 site samples) | `receipt_fuse13e_predictor.json` |

## Stage 0: FUSE-12 Evidence Reconciliation

- Audit pass: `True` (`50` distinct cells: `38` FP32 ok, `4` TF32 ok, `2` BF16 ok, `2` FP16 ok, `2` BF16 `native_autocast_error`, `2` FP16 `native_autocast_error`).
- Width-512 FUSE-12B trainer failures (reconciled in `FUSE12_RESULTS.md`): BF16 `C=50, F=52, G=52` (sum `154`); FP16 `C=42, F=42, G=42` (sum `126`).
- FP16 FUSE-12B stress `nonfinite_parity_tensors`: `384` (matched nonfinite outcomes, explicitly excluded from finite passes).

## Stage E: DataGoblin Physical Cost Model + Learned Admission Correction

- **Status:** `experimental_non_correctness_path` (`correctness_path=False` — strictly outside numerical correctness path).
- **Dataset:** `283` rows, `706` site samples (`fuse13e_datagoblin_dataset.jsonl`).
- **Holdout family status:** hardware=`blocked_single_gpu_rtx5080`, checkpoint=`blocked_single_family_synthetic_only`.

| Holdout | Policy | Oracle Step Fraction | Oracle Recon Fraction | Worst-Case Step Regret (ms) | Unsafe Admission Rate |
|---|---|---:|---:|---:|---:|
| Leave-one-width-out | `physical_plus_learned_gated` | 0.9591 | 0.9403 | 234.343 | 0.0000 |
| Leave-one-width-out | `physical_plus_learned` | 0.9668 | 0.9724 | 234.343 | 0.0324 |
| Leave-one-width-out | `physical_prior` | 0.9780 | 0.9801 | 234.343 | 0.0324 |
| Leave-one-width-out | `fuse12c_log_ridge` | 0.9194 | 0.9242 | 26275.660 | 0.0324 |
| Leave-one-width-out | `exact_saved_mac` | 0.9364 | 0.9226 | 234.343 | 0.0324 |
| Leave-one-width-out | `density` | 0.9379 | 0.9284 | 26510.003 | 0.0324 |
| Leave-one-width-out | `first_fit` | 0.9331 | 0.9360 | 26275.660 | 0.0324 |
| Leave-one-depth-out | `physical_plus_learned_gated` | 0.9329 | 0.9067 | 26510.003 | 0.0000 |
| Leave-one-depth-out | `physical_plus_learned` | 0.9425 | 0.9406 | 26510.003 | 0.0339 |
| Leave-one-depth-out | `physical_prior` | 0.9850 | 0.9839 | 234.343 | 0.0339 |
| Leave-one-depth-out | `fuse12c_log_ridge` | 0.9209 | 0.9256 | 26275.660 | 0.0339 |
| Leave-one-depth-out | `exact_saved_mac` | 0.9505 | 0.9354 | 234.343 | 0.0339 |
| Leave-one-depth-out | `density` | 0.9439 | 0.9310 | 26510.003 | 0.0339 |
| Leave-one-depth-out | `first_fit` | 0.8969 | 0.9053 | 26275.660 | 0.0339 |
| Leave-one-precision-out | `physical_plus_learned_gated` | 0.6440 | 0.4842 | 234.343 | 0.0000 |
| Leave-one-precision-out | `physical_plus_learned` | 0.9479 | 0.9554 | 234.343 | 0.5000 |
| Leave-one-precision-out | `physical_prior` | 0.9485 | 0.9556 | 234.343 | 0.5000 |
| Leave-one-precision-out | `fuse12c_log_ridge` | 0.9483 | 0.9134 | 26275.660 | 0.5000 |
| Leave-one-precision-out | `exact_saved_mac` | 0.9352 | 0.8970 | 234.343 | 0.5000 |
| Leave-one-precision-out | `density` | 0.9560 | 0.9167 | 26510.003 | 0.5000 |
| Leave-one-precision-out | `first_fit` | 0.9222 | 0.9253 | 26275.660 | 0.5000 |

### Safety-Gated Worth-Caching Classifier Calibration

| Holdout | Eval N | Accuracy | False Rejection Rate | Unsafe Admission Rate | Brier Score | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Width | 225 | 0.7244 | 0.3696 | 0.0000 | 0.2010 | 0.0603 |
| Depth | 225 | 0.6533 | 0.3478 | 0.0000 | 0.2403 | 0.1997 |
| Precision | 225 | 0.6311 | 0.9022 | 0.0000 | 0.2785 | 0.3032 |

