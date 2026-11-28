# BPD FUSE-14: Whole-Block Recurrent Cast Compactification — Results

- **Campaign commit:** `0dfb171b76d2b1468040125f0dc8445d79ec6a94`
- **Hardware & lease:** NVIDIA GeForce RTX 5080 Laptop GPU (16 GiB VRAM, Windows WDDM), VRAMorama strict FIFO (`12 GiB` lease, `2048 MiB` headroom, `>=60 W` power guard, exit code `75` pause; zero forced WDDM spill).
- **Production defaults unchanged:** `allow_cuda_residual_cache=False`, `residual_cache_storage_mode="fp32"`, `cache_frozen_projection_casts=False`, `projection_cache_allow_recompute=False`. FP16 remains **HOLD**. Trainable projection cast sharing remains excluded.

## Executive Gate Summary (Stages A–H)

| Stage | Scope | Verdict | Evidence Receipt |
|---|---|---|---|
| Preflight CPU Probe | Standalone CPU BF16 probe replay (`18` cells) | **PASS** (18/18 finite bitwise; depth-12 storages `48 -> 2`) | `receipt_fuse14_cpu_probe_summary.json` |
| Stage A | CUDA standalone probe (`18/18`) + trainer per-layer saved-cast audit | **PASS** (saved cast storages `94 -> 4`, `max_abs=0.000e+00`) | `receipt_fuse14_suite_cuda.json` |
| Stage B | 6-arm isolation ablation (`qkv_only`, `gate_up_only`, `both`, autocast cache off, per-call cast) | **PASS** (4/4 valid arms finite bitwise; `autocast_cache_disabled` falsified) | `receipt_fuse14_suite_cuda.json` |
| Stage C | Step-local trainer integration + fail-closed lifecycle guards | **PASS** (17/17 guards pass; defaults off) | `receipt_fuse14_suite_cuda.json` |
| Stage D | Frozen BF16 multi-step trainer matrix (`w=128..2048, T=2..24`, 2 roots, `closed_loop`/`clean_window`, `off/1/50`) | **PASS** (13/13 finite bitwise; FP16 `HOLD_CONFIRMED`) | `receipt_fuse14_suite_cuda.json` |
| Stage E | Independent activation checkpointing + recompute-safe frozen cast reuse | **PASS** (2/2 finite bitwise) | `receipt_fuse14_suite_cuda.json` |
| Stage F | All 5 previously blocked BF16 frontier cells (+ `(8192,24)` preflight) | **COMPLETE** (Arm H fits `2/6`; Arm H+Ckpt fits `2` additional; zero forced spill) | `receipt_fuse14_frontier.jsonl` |
| Stage G | Pretrained checkpoint transfer (TRM/DG `blocked_no_local_checkpoint`; local SHA-256 `Qwen3-4B` `2560/9728`) | **PASS (Qwen3-4B)** (2/2 finite bitwise; TRM/DG honestly blocked) | `receipt_fuse14_suite_cuda.json` |
| Stage H | DataGoblin hard-evidence-gated routing (`correctness_path=False`) | **COMPLETE** (`0.0000` unsafe admissions; router: `safe_physical_prior (hard-evidence-gated)`) | `receipt_fuse14h_router.json` |

## Stage A: CUDA Standalone Probe & Per-Layer Saved-Cast Audit

- Standalone CUDA probe (`fuse14_projection_probe --device cuda`): **18/18 finite bitwise**; depth-12 saved projection cast storages collapsed **`48 -> 2`**.
- Native `BPDTrainer` audit (`w=512, T=12`, BF16, `closed_loop`, `cal=1`): finite bitwise **`186/186`** (`max_abs=0.000e+00`).

| Arm | Total Pack Calls | Unique Cast Storages | Unique Cast Bytes | `qkv_proj` Unique (Bytes) | `gate_up_proj` Unique (Bytes) | Peak Alloc (MiB) | NVML (MiB) |
|---|---:|---:|---:|---|---|---:|---:|
| `FUSE-13 Arm G` | 188 | **94** | 223,346,688 | 46 (72,351,744 B) | 48 (150,994,944 B) | 414.4 | 700 |
| `FUSE-14 Arm H` | 188 | **4** | 9,437,184 | 2 (3,145,728 B) | 2 (6,291,456 B) | 205.4 | 480 |

## Stage B: 6-Arm Isolation Ablation (`BPDTrainer`, BF16 CUDA)

| Arm | Finite Bitwise vs Arm G | Max Abs | Unique Cast Storages | Unique Cast Bytes | Peak Alloc (MiB) | NVML (MiB) | Step Time (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `fuse13_arm_g` | ref | 0.000e+00 | 94 | 223,346,688 | 414.4 | 700 | 3609.79 |
| `autocast_cache_disabled` | **NO** | 4.688e-02 | 94 | 223,346,688 | 438.1 | 724 | 3571.98 |
| `explicit_per_call_cast` | **YES** | 0.000e+00 | 94 | 223,346,688 | 414.4 | 700 | 3805.17 |
| `qkv_only` | **YES** | 0.000e+00 | 50 | 154,140,672 | 345.4 | 620 | 3927.36 |
| `gate_up_only` | **YES** | 0.000e+00 | 48 | 78,643,200 | 273.4 | 540 | 3673.57 |
| `both_block_projs` | **YES** | 0.000e+00 | 4 | 9,437,184 | 205.4 | 480 | 3849.87 |

## Stage C: Fail-Closed Lifecycle & Guard Audit

All **17/17** Stage C checks passed: `default_proj_cache_off`, `default_proj_recompute_off`, `default_cuda_cache_off`, `reject_biased`, `reject_subclass`, `reject_parametrized`, `reject_prepatched`, `trainable_fallback_and_grad`, `noncontiguous_fallback`, `mutation_rejected`, `weight_replacement_rejected`, `nested_rejected`, `closed_no_reentry`, `restored_after_close`, `byte_cap_enforced`, `fp32_and_fp16_hold_fallback`, `cuda_cross_stream_rejected`.

## Stage D: Frozen Multi-Step BF16 Parity Matrix (Arm G vs Arm H)

| Width | Visits | Root | Regime | Cal | Budget | Finite Bitwise Tensors | Failures | Max Abs | Arm G Step (ms) | Arm H Step (ms) | Arm G Alloc (MiB) | Arm H Alloc (MiB) | Alloc Saved (MiB) | NVML Saved (MiB) |
|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2 | 14001 | `closed_loop` | `1` | `full` | 178/178 | 0 | 0.000e+00 | 377.1 | 383.9 | 26.8 | 25.3 | +1.5 | +2 |
| 128 | 6 | 14001 | `closed_loop` | `1` | `full` | 182/182 | 0 | 0.000e+00 | 1326.4 | 1389.5 | 39.1 | 33.1 | +6.0 | +6 |
| 128 | 6 | 14002 | `clean_window` | `50` | `binding` | 154/154 | 0 | 0.000e+00 | 192.3 | 185.1 | 30.5 | 27.8 | +2.6 | +4 |
| 512 | 6 | 14001 | `closed_loop` | `1` | `binding` | 182/182 | 0 | 0.000e+00 | 1114.9 | 1245.5 | 261.6 | 162.6 | +99.0 | +100 |
| 512 | 12 | 14001 | `closed_loop` | `off` | `full` | 186/186 | 0 | 0.000e+00 | 759.6 | 763.8 | 406.7 | 197.7 | +209.0 | +220 |
| 512 | 12 | 14002 | `clean_window` | `50` | `full` | 158/158 | 0 | 0.000e+00 | 319.6 | 341.4 | 240.2 | 141.7 | +98.5 | +100 |
| 1024 | 6 | 14001 | `closed_loop` | `1` | `binding` | 182/182 | 0 | 0.000e+00 | 1127.1 | 1241.4 | 866.6 | 492.9 | +373.7 | +392 |
| 1024 | 12 | 14001 | `closed_loop` | `off` | `full` | 186/186 | 0 | 0.000e+00 | 928.0 | 866.3 | 1354.7 | 563.7 | +791.0 | +864 |
| 1024 | 24 | 14002 | `closed_loop` | `off` | `full` | 194/194 | 0 | 0.000e+00 | 1985.9 | 2352.6 | 2518.4 | 882.6 | +1635.8 | +1800 |
| 2048 | 6 | 14001 | `closed_loop` | `1` | `binding` | 182/182 | 0 | 0.000e+00 | 1152.8 | 1011.9 | 3163.0 | 1716.9 | +1446.1 | +1640 |
| 2048 | 12 | 14001 | `closed_loop` | `off` | `full` | 186/186 | 0 | 0.000e+00 | 666.6 | 727.1 | 4960.3 | 1880.8 | +3079.4 | +3172 |
| 2048 | 12 | 14002 | `clean_window` | `50` | `full` | 158/158 | 0 | 0.000e+00 | 280.1 | 313.6 | 2848.2 | 1400.8 | +1447.4 | +1562 |
| 2048 | 24 | 14001 | `closed_loop` | `off` | `full` | 194/194 | 0 | 0.000e+00 | 1403.8 | 2020.6 | 9212.4 | 2870.8 | +6341.6 | +6474 |

- **FP16 HOLD confirmation:** `HOLD_CONFIRMED` (`hits=0, misses=0, fallbacks=96`).

## Stage E: Independent Activation Checkpointing Evaluation

| Width | Visits | Finite Bitwise (`ckpt_native` vs `ckpt+proj`) | Native Ckpt Alloc / NVML (MiB) | Ckpt+Proj Alloc / NVML (MiB) | Alloc Saved (MiB) | Native Ckpt Step (ms) | Ckpt+Proj Step (ms) |
|---:|---:|---|---|---|---:|---:|---:|
| 512 | 12 | **YES** (186/186) | 99.4 / 384 | 103.9 / 384 | -4.5 | 1554.3 | 1577.1 |
| 2048 | 24 | **YES** (194/194) | 1136.7 / 1498 | 1207.3 / 1540 | -70.6 | 3140.2 | 3372.4 |

## Stage F: Reattempting All Five Blocked BF16 Frontier Configurations

Note: When FUSE-13 Arm G is preflight-blocked, comparison is explicitly **censored** (not a speedup). Hardware measurements on `(2048, 48)` (`4,848 MiB`) and `(4096, 24)` (`10,173 MiB`) revealed an additional per-pass `W_target` BF16 cast in `TiedComponentLinear.forward` (`components.py:382`, stochastic `routing_mask is not None`, `T * site_elems * 2` bytes); refitting Arm H preflight with this measured term prevents WDDM spill on `(4096, 48)+` while `Arm H + Checkpointing` (`H_ckpt`) eliminates it cleanly.

| Width | Visits | Baseline Comparison | FUSE-13 Arm G | FUSE-14 Arm H (p50 / p95 ms, tok/s) | Arm H Alloc / Res / NVML (MiB) | Arm H Energy (mJ/step) | Arm H+Ckpt (if run) |
|---:|---:|---|---|---|---|---:|---|
| 2048 | 48 | `censored_baseline_blocked` | `memory_blocked_preflight` (est 15022 MiB) | `ok` (4101.9 / 4443.6 ms, 936 tok/s) | 4848 / 5186 / 5436 | 297208.3 | not needed |
| 4096 | 24 | `censored_baseline_blocked` | `memory_blocked_preflight` (est 29317 MiB) | `ok` (2478.0 / 2552.5 ms, 775 tok/s) | 10173 / 10314 / 10564 | 181908.7 | not needed |
| 4096 | 48 | `censored_baseline_blocked` | `memory_blocked_preflight` (est 55841 MiB) | `memory_blocked_preflight` (est 16544 MiB) | blocked | n/a | `ok` (10633.2 ms, 5406/5930 MiB) |
| 6144 | 24 | `censored_baseline_blocked` | `memory_blocked_preflight` (est 64296 MiB) | `memory_blocked_preflight` (est 20826 MiB) | blocked | n/a | `ok` (10473.5 ms, 9436/9952 MiB) |
| 8192 | 12 | `censored_baseline_blocked` | `memory_blocked_preflight` (est 61624 MiB) | `memory_blocked_preflight` (est 23862 MiB) | blocked | n/a | `memory_blocked_runtime_headroom` (est 10154 MiB) |
| 8192 | 24 | `censored_baseline_blocked` | `memory_blocked_preflight` (est 113876 MiB) | `memory_blocked_preflight` (est 36059 MiB) | blocked | n/a | `memory_blocked_preflight` (est 10184 MiB) |

## Stage G: Pretrained Checkpoint Transfer

- **TRM / DataGoblin checkpoint audit:** `blocked_no_local_checkpoint` — No local gaoxin492/step_21700, yagizdevre/model.pt, complete HRM-Text-1B, or DataGoblin PyTorch checkpoint present; offline/no-download policy respected.
- **Real Pretrained Transfer (`pretrained_qwen3_4b_block`, kept separate from synthetic mini-TRM):** `Qwen/Qwen3-4B (local cached shard 2/3, layers 16-17, no download)`, verified SHA-256 `6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5`.

| Visits | Hidden / Inter | Finite Bitwise | Saved Cast Storages (G -> H) | Arm G Alloc / NVML (MiB) | Arm H Alloc / NVML (MiB) | VRAM Saved Alloc / NVML (MiB) | Arm G -> H Step (ms) |
|---:|---|---|---|---|---|---:|---|
| 6 | `2560 / 9728` | **YES** (71/71) | `120 -> 10` | 4352.4 / 4724 | 1616.3 / 2016 | +2736.1 / +2708 | 114.8 -> 112.3 |
| 12 | `2560 / 9728` | **YES** (71/71) | `240 -> 10` | 7534.9 / 7944 | 1740.3 / 2162 | +5794.6 / +5782 | 275.4 -> 246.7 |

## Stage H: DataGoblin Hard-Evidence-Gated Routing (`correctness_path=False`)

- **Hard evidence gate modes:** `bf16, fp32, tf32` (FP16: `HOLD`).
- **Checkpoint families exported:** `pretrained_qwen3_4b_block, synthetic_mini_trm` (`46` FUSE-14 rows in `fuse14h_datagoblin_dataset.jsonl`, `706` FUSE-12/13 site samples).
- **Residual predictor promotion check:** `beats_safe_physical_on_all_holdouts=False` -> recommended experimental router: **`safe_physical_prior (hard-evidence-gated)`**.

| Holdout | Policy | Safe Oracle Fraction | p95 Regret (ms) | Max Regret (ms) | Unsafe Admission Rate | False Rejection Rate |
|---|---|---:|---:|---:|---:|---:|
| Width | `safe_physical_prior` | 0.9786 | 234.343 | 234.343 | 0.0000 | 0.0000 |
| Width | `safe_physical_plus_learned` | 0.9674 | 234.343 | 234.343 | 0.0000 | 0.0000 |
| Width | `safe_exact_saved_mac` | 0.9356 | 234.343 | 234.343 | 0.0000 | 0.0000 |
| Width | `safe_density` | 0.9375 | 26510.003 | 26510.003 | 0.0000 | 0.0000 |
| Width | `ungated_physical_prior_ref` | 0.9786 | 234.343 | 234.343 | 0.0162 | 0.0000 |
| Depth | `safe_physical_prior` | 0.9857 | 234.257 | 234.343 | 0.0000 | 0.0000 |
| Depth | `safe_physical_plus_learned` | 0.9431 | 26500.292 | 26510.003 | 0.0000 | 0.0000 |
| Depth | `safe_exact_saved_mac` | 0.9497 | 234.257 | 234.343 | 0.0000 | 0.0000 |
| Depth | `safe_density` | 0.9433 | 26500.292 | 26510.003 | 0.0000 | 0.0000 |
| Depth | `ungated_physical_prior_ref` | 0.9857 | 234.257 | 234.343 | 0.0169 | 0.0000 |
| Precision | `safe_physical_prior` | 0.9669 | 3.838 | 234.343 | 0.0000 | 0.0000 |
| Precision | `safe_physical_plus_learned` | 0.9661 | 6.052 | 234.343 | 0.0000 | 0.0000 |
| Precision | `safe_exact_saved_mac` | 0.9232 | 16.144 | 234.343 | 0.0000 | 0.0000 |
| Precision | `safe_density` | 0.9457 | 11.327 | 26510.003 | 0.0000 | 0.0000 |
| Precision | `ungated_physical_prior_ref` | 0.9669 | 3.838 | 234.343 | 0.2500 | 0.0000 |

