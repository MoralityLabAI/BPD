# BPD-DT05: RTX 5080 GPU Optimization & Native Causal Research Sprint Report

**Repository:** `https://github.com/MoralityLabAI/BPD` · **Branch:** `gym/TRM-Sodoku` · **Baseline Commit:** `8ccdcb2ed7ed937805efbd4276c524932321f95a`

## 1. Hardware & Memory Envelope (`benchmarks/01_hardware_probe.json`)

- **GPU:** NVIDIA GeForce RTX 5080 Laptop GPU (`16,303 MiB` / `15.92 GiB` GDDR7, Compute Capability `12.0` / `sm_120`, 60 SMs, Driver `577.05`, PyTorch `2.11.0+cu128`, `bf16_supported=true`, 80W enforced TGP).
- **CPU & System DRAM:** AMD Ryzen AI 9 365 (10 physical cores / 20 logical threads), `47.12 GiB` physical RAM (`11.93–18.28 GiB` available under normal load; dipped `<0.5 GiB` during external multi-agent peaks), `67.36 GiB` pagefile.
- **Measured Memory Bandwidths:**
  - Host DRAM float32 copy / STREAM triad: **`11.38 GB/s`** / **`9.90 GB/s`**
  - PCIe H2D pageable (64 MiB): **`3.68 GB/s`** → **Pinned DMA (`non_blocking=True`): `12.50 GB/s` (`3.39x` faster)**; Pinned D2H: **`12.56 GB/s`**
  - RTX 5080 VRAM D2D copy (512 MiB): **`594.16 GB/s`**

---

## 2. Diagnosed Bottlenecks & Before/After GPU Benchmarks (`benchmarks/02..06`)

### Root Bottleneck Diagnosis
Baseline profiling (`03_baseline_execution_profile.json`) showed that `dt04` (`hidden_size=96, total_len=18`, 36 `TRMBlock` calls across 6 ACT steps) spent **`96.5%` of step time in Windows WDDM eager kernel-launch overhead** (~2,700 tiny launches/microbatch; GPU utilization `7%`), making eager `bf16` slower than `fp32` (`210.5 ms` vs `196.5 ms` at `mb=128`) due to autocast cast launches. Conversely, `full_7m` (`hidden_size=512, total_len=97`, 252 `TRMBlock` calls across 6 fit steps) hit **Tensor Core compute** in inference (`2.77x` BF16 speedup) and **activation-memory exhaustion** in training (`mb=32` `BUDGET_REJECTED` above `12.5 GiB` VRAM). Causal Jacobians (`dt03_causal.flat_observer_jacobian`) executed `B × M` serial backward passes (`112.31s` for 16 boards), and `load_replay_cases` used set-comprehension Python backtracking (`>12 min` on 64 hard 9x9 puzzles).

### Before vs. After Benchmark Summary (`cuda:0`, `allow_tf32=False`)

| Workload & Measurement Scope | Frozen Baseline (`8ccdcb2`) | Optimized DT05 | Speedup / Outcome | Peak Host RSS | Peak CUDA Reserved |
|---|---:|---:|---:|---:|---:|
| **`dt04` Calibrator** (isolated fw/bw, `lb=128`, BF16) | `441.7 ex/s` (`mb=128`) | **`3,282.8 ex/s`** (`CUDAGraph`) | **`7.43x`** | `1.65 GiB` | `0.81 GiB` |
| **`dt04` Calibrator** (isolated fw/bw, `lb=128`, FP32) | `505.5 ex/s` (`mb=128`) | **`5,030.8 ex/s`** (`CUDAGraph`) | **`9.95x`** | `1.65 GiB` | `0.81 GiB` |
| **`dt04` Full Optimizer Step** (`calibrate-step`, prep+H2D+fw/bw+AdamW+EMA, `lb=128`, BF16) | `389.3 ex/s` (`3.04 upd/s`) | **`4,585.7 ex/s`** (`35.83 upd/s`) | **`11.78x`** | `1.68 GiB` | `0.83 GiB` |
| **`dt04` Full Optimizer Step** (`calibrate-step`, `lb=128`, FP32) | `400.5 ex/s` (`3.13 upd/s`) | **`4,276.0 ex/s`** (`33.41 upd/s`) | **`10.68x`** | `1.68 GiB` | `0.83 GiB` |
| **`dt04` 120-Step Training** (default `mb=16` FP32 ref vs auto-tuned `mb=128` BF16 graphed) | `245.83 s` (`62.5 ex/s`) | **`2.61 s`** (`5,885.1 ex/s`) | **`94.23x`** | `1.74 GiB` | `0.85 GiB` |
| **`dt04` 480-Step Training** (same `mb=128` FP32 ref vs graphed+prefetched opt) | `146.30 s` (`420.0 ex/s`) | **`15.12 s`** (`4,063.5 ex/s`) | **`9.67x`** (`max Δloss=6.4e-6`) | `1.74 GiB` | `0.85 GiB` |
| **`dt04` 480-Step Training** (same `mb=128` BF16 ref vs graphed+prefetched opt) | `259.68 s` (`236.6 ex/s`) | **`10.06 s`** (`6,107.4 ex/s`) | **`25.83x`** (BF16 Lyapunov flagged) | `1.74 GiB` | `0.85 GiB` |
| **`full_7m` Inference Calibrator** (8 ACT steps, `lb=256`) | `32.47 ex/s` (FP32) / `90.02 ex/s` (BF16, `mb=64`) | **`98.60 ex/s`** (BF16 `CUDAGraph`) | **`3.04x` vs FP32** (`1.10x` vs BF16) | `1.72 GiB` | `2.64 GiB` |
| **`full_7m` Training Calibrator & Full-Step** (`lb=32`, BF16) | `mb=16` (`8.25 ex/s`), **`mb=32` REJECTED (`>12.5 GiB`)** | **`mb=32` ADMITTED (`8.19 ex/s` full-step)** via `checkpoint_act` | **Unlocks `mb=32` in `3.61 GiB` VRAM** (`-71%` VRAM) | `1.76 GiB` | **`3.61 GiB`** |
| **Factor Distillation** (`80` steps, `dt04`, `lb=128`) | `67.69 s` (uncached) / `50.74 s` (ref cached) | **`27.44 s`** (pinned staging + `BoundedTensorCache`) | **`2.47x`** (`10,240` teacher rollouts avoided, `100%` hit rate, `10.5 MiB` cache) | `1.75 GiB` | `0.85 GiB` |
| **Causal Observer Jacobian** (16 cases, `dt04` depth 6) | `112.31 s` (`dt03_causal` loop) | **`1.259 s`** (`chunked vmap(vjp)`) | **`89.2x`** (`max Δ = 5.83e-16`) | `1.75 GiB` | `0.85 GiB` |
| **7M Thin JVP Transport** (`rank=8`, `gaoxin492`) | `6.41 s` (serial JVP loop) | **`1.120 s`** (`chunked vmap(jvp)`) | **`5.72x`** (`overlap = 0.99999903`) | `1.80 GiB` | `1.95 GiB` |
| **9x9 Replay Uniqueness Gate** (64 hard boards) | `>720 s` (`worlds.count_solutions`) | **`0.04 s`** (`_count_solutions_bitmask`) | **`>15,000x`** (exact parity verified) | — | — |

### Numerical Parity & Host Contention Telemetry
- **FP32 Optimization Parity:** `GraphedSupervisedExecutor` + `SudokuSymmetryPipeline` (bitwise-identical RNG batch sequence, double-buffered pinned H2D prefetch, `torch._foreach_lerp_` EMA) matched 480-step reference FP32 loss trajectories within **`max_abs_diff = 6.44e-6`**. Single-step `CUDAGraph` gradient difference is **`0.00e+00`**.
- **Flagged BF16 Trajectory Divergence:** Comparing `AdamW(fused=False)` vs `AdamW(fused=True)` in BF16 starts at `0` step-1 difference and bifurcates Chaotically over 480 recurrent steps (`max_trace_loss_abs_diff = 0.661`, automatically flagged by `ab_benchmark_training`), while preserving downstream Sudoku competence (`up to 98.4%–100%` on competent roots).
- **Live RAM Admission & Fail-Closed Guard (`07a_host_reserve_fail_closed_receipt.json`):** When concurrent host processes pushed system RAM utilization to `91%–97%` (`<0.5 GiB` free), `assert_budget` failed closed cleanly, and `ReusableTransferBuffer` automatically disabled WDDM `pin_memory` page-locking below `1.5 GiB` free RAM to prevent `cudaHostAlloc` driver errors.

---

## 3. Mechanistic Instrument Findings (`benchmarks/07..10`)

### A. Dynamic Causal Subspace Transport & `EvalMath` Preflight (`07_dynamic_transport_and_evalmath.json`)
- **Multi-step JVP Geometry (`depths 1 → 8`):** Confirmed DT04's static-subspace failure mechanism without allocating dense `[D, D]` matrices (`36.75 GiB` avoided on 7M): 1-step closure defect is `~0.98+` with rapid principal-angle rotation (`>75°` from initial $Q_0$) and non-zero projection escape out of $\ker(DP)$ into `rowspace(lm_head)`. Finite-perturbation nonlinear responses converge tightly to multi-step JVP predictions at $\varepsilon = 10^{-3}$ (`cosine > 0.999`) and depart nonlinearly at $\varepsilon = 0.5$.
- **8-Root Excluded Pilot (`2605900..2605907`, `patch_depth=3 → eval_depth=8`):** Comparing JVP-transported task-Gramian bases against rank- and L2-dose-matched static bases, 8 Haar protected shams, and 8 privileged signed-Jacobian shams (`signed_err < 1e-4`, `observer_err < 1e-5`, exact IEEE-754 zero prefix lock) yielded mean paired effect **`-0.0078125`** exact board.
- **EvalMath Preflight Verdict (`§5`):** `evalmath_preflight.py` returned **`REDESIGN_OR_PILOT_ONLY`** (`pilot_effect_not_above_mechanistic_variation_floor`, `sham_quantile_uncalibrated` at $N=8 < 59$). **0 of 59** reserved DT05 development roots (`2605000..2605058`) and **0** confirmation roots were consumed.

### B. Autonomous Sparse Factor Routing vs. Teacher-Scheduled Fidelity (`08_autonomous_routing_study.json`)
- Evaluated `online`, `teacher_scheduled`, `fixed`, and `random` routing on **identical fitted `(V, U)` matrices** (`C=24, top_k=12`) across all 8 pilot roots (`512` board evaluations/arm):
  - **Competent roots (`dense teacher = 89.06%` at depth 8):** `online` achieved **`58.85%`** exact board (up to `81.25%` on root `2605900`) vs **`4.17%` for `fixed` (`+54.69 pp` gain)** and **`0.00%` for `random`**.
  - **Deployment Gap vs. Teacher-Scheduled:** Privileged `teacher_scheduled` masks on the same `(V, U)` reached **`81.77%`** (`KL = 0.046 nats` vs `0.120 nats` online vs `0.610 nats` fixed), isolating a **`22.92 pp` compounding mask-drift gap** as online top-$k$ set overlap with teacher masks decays across recurrent steps. `online` dynamically recruits **`23.59 / 24`** unique components vs `12.0 / 24` for `fixed`.

### C. Gold-Free First-Valid Stopping on Local 7M Checkpoints (`09_first_valid_stopping_7m_and_dt04.json`)
- Executed offline SHA-verified `dt05.runner replay` across 64 solver-verified 9x9 `test_sudoku_bench.csv` boards (`depths 1..16`, `microbatch=32`, BF16):
  - **`yagizdevre` (`SHA-256: d6772339...`):** Raw fixed-depth accuracy peaked at **`96.88%`** (depth 4), dipped to **`85.94%`** (depths 8–9), and finished at **`93.75%`** (depth 16), **destroying 4 previously valid & correct solutions**. Gold-free compacted `FirstValidStopper` achieved **`100.0%` (`64/64`)**, recovering all 4 destroyed boards, **saving `883 / 1,024` ACT case-steps (`case_step_fraction = 13.77%`)**, cutting wall time **`16.74 s → 8.20 s` (`2.04x`)**, and flagging **`1` BF16 batch-compaction trajectory shift** when active batch size shrank (`32 → 30 → 5 → 2 → 1`).
  - **`gaoxin492` (`SHA-256: 932b5e0d...`):** Reached **`100.0%` (`64/64`)**, **saving `893 / 1,024` ACT case-steps (`case_step_fraction = 12.79%`)** and cutting wall time **`16.15 s → 5.53 s` (`2.92x`)** (`1` BF16 compaction depth shift recorded; `0` shifts in FP32 `dt04` sweeps).

### D. Depth-Indexed Causal Rank (`10_causal_rank_study.json`, BPD-ECA-03/04/05)
- Across recurrent depths `d ∈ {1, 2, 4, 6, 8}` (`horizon = 12`):
  - **Statistical participation ratio:** High and nearly depth-invariant (`21.4 → 19.7` full state; `23.2 → 20.1` dark $\ker(DP)$).
  - **Linearly decodable dimension:** Rises steadily (`operator PR: 14.3 → 18.3`; dark-state rank-8 blank-cell probe accuracy climbs from **`38.7%` at $d=1$ to `52.9%` at $d=6$** inside $\ker(\text{lm\_head})$).
  - **Native future-response causal dimension:** Strongly compressed (**`Gramian PR: 2.95 → 4.71`**, **`entropy rank: 3.66 → 6.06`**), expanding monotonically as remaining horizon steps decrease, while corrupted-state behavioral oracle headroom stays tight (`~0.01–0.02` exact board, `~0.016 nats` KL), confirming BPD-ECA-03/04's separation between wide observational decodability and low-rank/saturated causal headroom.

---

## 4. Tesseract Manifest & Remaining Scientific Limitations

- **Tesseract Manifest:** [`tesseract_manifest.json`](tesseract_manifest.json) and adapter [`tesseract.py`](tesseract.py) (mapped in [`TESSERACT_INTEGRATION.md`](TESSERACT_INTEGRATION.md)) bind exact Git revision, calibrated RTX 5080 microbatches (`128`, `64`, `32`), RAM/VRAM ceilings, `SIGINT`/`SIGTERM` receipt flushing, `max_attempts=1` root leases, and `protected` confirmation blocks.
- **Limitations:**
  1. All new neural runs used excluded engineering roots (`2605900..2605907`, `2605999`) or exposed benchmark boards; the 59 reserved DT05 development roots (`2605000..2605058`) remain unexecuted (`0/59`) for prospective Tesseract launch, and confirmation cohorts (`2604100..2604355`, `ARG-03-N`, `NC09`, `AMM-03..06`) remain sealed.
  2. Single-step JVP-transported subspace patching at depth 3 did not beat equal-dose static/random controls (`EvalMath: REDESIGN_OR_PILOT_ONLY`) due to strong native self-repair saturation (`oracle headroom ~ 0.016 nats`).
  3. Online sparse routing (`C=24, k=12`) recovers `+54.7 pp` over fixed SVD factors on competent models without teacher masks, but incurs `2.6x` eager Python/masking overhead relative to pre-sliced fixed factors and retains a `22.9 pp` fidelity gap to privileged teacher scheduling.
