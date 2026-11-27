# FUSE-12 | CUDA Validation, Cumulative Residual Caching Benchmark, Mixed Precision & DataGoblin Policy Curriculum

**Status:** Prospective registered protocol (registered prior to running any benchmarks).  
**Branch:** `research/fuse12-cuda-validation-20261127`  
**Base Commit:** `a4cc3d861df531f41d91000d8dd0922a5ef953ec` (FUSE-11 PR #185)  
**Production Defaults:** Unchanged (`cache_single_backward_residuals=False`, `cache_calibration_residuals=False`, `cache_pgd_mask_residuals=False`, `cache_pgd_final_residuals=False`, `allow_cuda_residual_cache=False`, `residual_cache_precision_mode="fp32"`).

---

## 1. Preflight Audit

### 1.1 Upstream PR & Hosted CI Audit (as of 2026-11-27)
| PR | Branch | Head SHA | State | `native-cpu` | `cpu-math-and-donor-contrast` | `wide-cpu` | Notes |
|---:|---|---|---|---|---|---|---|
| #178 | `research/crd28-fused-residual-20261127` | `1ee5a870c25e88ba6cafa07f349297498daf945f` | Draft open | `success` | `success` | `skipped` | `Analyze (python)` & `Analyze (actions)` `success`; `CodeQL` `neutral`; repo-level `Analyze (c-cpp)` `failure` (unrelated to Python BPD). |
| #184 | `research/fuse10-calibration-20261127` | `da81dab14226c413cd5d0a71f61c4fe8143bba6a` | Draft open | `success` | `success` | `skipped` | All FUSE-06..10 native CPU checks green. |
| #185 | `research/fuse11-retained-pgd-20261127` | `a4cc3d861df531f41d91000d8dd0922a5ef953ec` | Draft open | `success` | `success` | `skipped` | Zero FUSE-11 failures; ready as base. |

### 1.2 Local Hardware & Software Environment
- **GPU:** NVIDIA GeForce RTX 5080 Laptop GPU (Blackwell, 16,303 MiB NVML / 15,915 MiB WDDM dedicated VRAM)
- **Driver Model:** WDDM (`nvidia-smi` driver `577.05`, CUDA driver API `12.9`)
- **CUDA Python Environment:** Python `3.12.10`, PyTorch `2.11.0+cu128` (CUDA runtime `12.8`), NumPy `2.5.3`, SciPy `1.18.1`, scikit-learn `1.9.1`, psutil `7.2.2`, pytest `9.1.1`
- **Power Limit:** 80 W enforced / 80 W default (`sw_power_cap` monitored via NVML; Windows logged 2 GPU resets in prior 24 h)
- **GPU Queue Policy:** Competing workloads observed at preflight (`vramorama` leases `51dfcc41` `gym-b25-rft4k-logic` and `bb691bf6` `squid5080-serve` / `vmwp.exe` at 15,161 MiB dedicated + 958 MiB shared spill). No competing process is terminated; GPU benchmarks are queued strictly behind active leases via `vramorama run` with `--strict-fifo`, `--headroom 2048`, `--guard-power`, and `--pause-exit 75`.

---

## 2. CUDA Implementation Contract

1. **Explicit experimental opt-in:**
   - `allow_cuda_residual_cache: bool = False` (on `BPDTrainer`, `StepResidualPool`, `planned_sites`, `supported`, `independent_residuals`, `calibration_residuals`).
   - `residual_cache_precision_mode: str = "fp32"` (valid: `"fp32"`, `"tf32"`, `"bf16"`, `"fp16"`; non-`"fp32"` modes require `allow_cuda_residual_cache=True`).
   - When `allow_cuda=False`, CPU behavior and all production defaults are byte-identical to FUSE-11, and CUDA tensors bypass to the native reference path.
2. **Native materialized residual preservation:**
   - `D = W_target.float() - (V @ U).T` via `original()`, materialized in `float32` (`4 * d_out * d_in` bytes). No algebraic reassociation.
3. **FP32 baseline campaign guards (`precision_mode == "fp32"`):**
   - `torch.backends.cuda.matmul.allow_tf32 == False`, `torch.backends.cudnn.allow_tf32 == False`, `torch.get_float32_matmul_precision() == "highest"`.
   - `torch.is_autocast_enabled("cuda") == False` and `torch.is_autocast_enabled("cpu") == False`.
   - All `(W_target, V, U)` on same CUDA device, `dtype == torch.float32`, `ndim == 2`, contiguous, non-inference, `not W_target.requires_grad`.
4. **CUDA stream ownership, lifetime & allocator audit:**
   - Restricted to a **single sequential CUDA stream**. On first CUDA admission, the pool/scope records `(device.index, torch.cuda.current_stream(device).cuda_stream)`. Every subsequent `weight_delta()` call, scope entry, and `seal_forward()` verifies the current stream matches the owning stream; any stream switch raises `RuntimeError` and poisons the pool.
   - CUDA Graph capture (`torch.cuda.is_current_stream_capturing()`) is rejected (falls back to native reference).
   - Asynchronous tensor lifetime: all forward scopes, retained source query, and joint backward enqueue kernels on the same FIFO CUDA stream. `entries` holds strong references throughout forward scopes; autograd saved tensors and `_Primal` outputs retain storage references until backward completes; `pool.close()` releases `entries` after `total_loss.backward()` so PyTorch's stream-ordered `CUDACachingAllocator` cannot reuse memory before pending kernels finish.
   - Cross-site storage aliasing is rejected via live `data_ptr()` uniqueness across `entries`.
5. **Preserved FUSE-08..11 invariants:**
   - Independent first-order `_Primal` adjoints per request.
   - Exact saved-MAC admission planner (`admission.select(..., "exact")`) with fail-closed frontier overflow.
   - Parameter identity, `_version`, shape/stride/dtype/device/requires_grad, and runtime precision-state verification on every hit, scope entry, and `seal_forward()`.
   - Guaranteed `finally` restoration of `weight_delta` and `_residual_cache_active` on normal exit and exceptions.
   - Retained PGD source-gradient guard: explicit `seal_forward()` after final PGD rollout, zero factor-adjoint executions during `autograd.grad(loss_adv, leaf_sources, retain_graph=True)`, and expected independent factor adjoints during joint `total_loss.backward()`.

---

## 3. Matched Benchmark Arms

| Arm | Configuration | Trainer Flags |
|---|---|---|
| **A** | Native BPD | All residual caches disabled (reference) |
| **B** | FUSE-08 stochastic step cache | `cache_single_backward_residuals=True, policy="step_exact"` |
| **C** | FUSE-09 inner PGD cache | `B + cache_pgd_mask_residuals=True` |
| **D** | FUSE-10 calibration cache only | `cache_calibration_residuals=True` |
| **E** | FUSE-09 + FUSE-10 | `C + D` |
| **F** | FUSE-09 + FUSE-11 | `C + cache_pgd_final_residuals=True` |
| **G** | FUSE-09 + FUSE-10 + FUSE-11 | `C + D + F` |

All arms within a matched cell share identical initial weights, batches, ambient + trainer RNG states, optimizer state, calibration schedules, and PGD initialization. Execution order alternates across paired steps/roots. Warmup iterations precede timed regions, and `torch.cuda.synchronize()` brackets every timed region.

---

## 4. Progressive Benchmark Matrix (Staged Fractional Factorial)

Executed in strictly ascending computational cost with WDDM memory-headroom enforcement (>= 2 GiB free NVML VRAM before each escalation):

| Tier | Widths (`D`) | Passes (`T`) | Paired Roots | PGD Inner (`K`) | Regimes | Cal Cadence | Budgets | Arms |
|---|---:|---:|---:|---|---|---|---|---|
| **Smoke** | 64, 128 | 6 | 2 (fresh) | 0, 2, 4 (staged) | `closed_loop`, `clean_window` | 1, off | `binding`, `full` | A–G |
| **Medium** | 256, 512 | 6, 12 | 4 (fresh) | 0, 2, 4 (fractional) | `closed_loop`, `clean_window` | 1, 50, off | `binding`, `full` | A–G |
| **Wide** | 1024, 2048 | 6, 12 | 4 (fresh) | 0, 2, 4 (fractional) | `closed_loop`, `clean_window` | 1, 50, off | `binding`, `full` | A–G |
| **Frontier** | 2048, 4096 | 24 | 2 (headroom-gated) | 2, 4 | `closed_loop` | 1, off | `binding`, `full` | A, C, F, G |

Note: Synthetic mini-TRM native trainer configurations (4 sites: `layers.{0,1}.{o_proj,down_proj}`, heterogeneous ranks `D/32, D/16, 3D/32, D/8`, `batch_size=2`, `seq_len=16`, `puzzle_emb_len=4`, `l_layers=2`, `h_cycles=1`, `l_cycles=2`). Results are explicitly distinguished from pretrained model checkpoints.

---

## 5. Numerical Acceptance Gates

1. **96 CUDA FP32 Stress Fixtures:**
   - Fresh roots `4012001..4012008` × factor scales `{1, 10, 100, 10000}` × targets `{"ordinary", "exact", "near"}` (with `1e-4` perturbation for near-target).
   - Exercises stochastic + discarded inner PGD (`mask_only=True`) + retained final PGD (`seal_forward()`, `retain_graph=True`) + joint backward.
   - Asserts zero factor-adjoint executions during retained source-only query and exactly 6 joint backward nodes per case (3 stochastic + 3 final PGD).
2. **Full-Trainer Paired Parity (Arms B–G vs Arm A):**
   - Compares native residual values, forward outputs, every PGD source gradient (including retained final query), component (`V, U`) and CI gradients **before clipping**, losses/metrics, optimizer moments (`exp_avg`, `exp_avg_sq`), updated parameters, mean calibration buffers (`mean_comp_acts`, `mean_delta_out`, `mean_write_rms`, `calibrated_passes`), `pass_tolerances`, greedy argmax outputs, evaluation trajectories, ambient + trainer RNG states, and exact reconstruction counts.
3. **Reporting Discipline:**
   - Bitwise equality (`torch.equal`) is reported **separately** from numerical tolerance (`atol=5e-5, rtol=5e-4`).
   - Maximum absolute error, maximum relative error, and near-zero residual (`|ref| < 1e-4`) max absolute error are logged for every tensor group.
   - Every mismatch beyond tolerance triggers investigation and fails the gate.

---

## 6. Performance, Memory & Power Accounting

Measured separately per arm and cell:
- Full training step wall time (`perf_counter`) and CUDA event elapsed time (`torch.cuda.Event`)
- Phase breakdown: residual reconstruction, clean calibration, inner PGD, final PGD, joint backward, optimizer step, and cache planner overhead
- Allocator VRAM (`max_memory_allocated`, `max_memory_reserved`) AND externally reported NVML GPU memory (`memory.used`, `memory.free`) separately (WDDM accounting)
- Cached residual payload bytes (step pool + calibration cache)
- Host RSS (`psutil`)
- GPU board power (W) and estimated energy per training step (mJ) via NVML
- Summary statistics: raw paired timings, medians, p95, paired bootstrap 95% CIs (2,000 resamples when sample count >= 4), and **speedup per additional MiB of cached residual payload**
- Hard WDDM safety floor: maintain >= 2,048 MiB measured free NVML VRAM; stop escalating configurations if memory headroom drops below threshold.

---

## 7. FUSE-12B: Mixed Precision (Gated on FP32 Pass)

Proceeds **only** if all FP32 stress, lifecycle, and Smoke/Medium parity gates pass.
- Registered arms: `tf32` (FP32 tensors with `allow_tf32=True`), `bf16` (`torch.amp.autocast("cuda", dtype=torch.bfloat16)`), and `fp16` (`torch.amp.autocast("cuda", dtype=torch.float16)`).
- `_Primal` captures forward autocast context and executes backward inside the matching autocast context, preserving native casting and accumulation semantics (`V.dtype`, `U.dtype`).
- Cache identity key binds `(precision_mode, allow_tf32, fp32_matmul_precision, autocast_enabled, autocast_dtype)`; mismatched precision/autocast states fail closed or bypass to native reference.
- Unsupported configurations remain on the reference path. Tolerances are never silently loosened to obtain a pass; exact bitwise parity, max abs/rel errors, and overflow parity (e.g. FP16 at scale `10000`) are reported transparently.

---

## 8. FUSE-12C: DataGoblin Admission-Policy Curriculum

- Export `fuse12c_datagoblin_dataset.jsonl` containing architecture dimensions, recurrence depth, component ranks, cache budget, precision, admitted sites, reconstruction savings, measured latency, energy, and memory costs (tagged `source_dataset_kind="synthetic_native_trainer"`).
- Fit a compact, interpretable experimental predictor (regularized regression on physical/architectural features) estimating GPU latency savings per site/configuration.
- Compare **Learned-Value Knapsack** against:
  1. Exact saved-MAC planner (`admission.select(..., "exact")`)
  2. First-fit (`admission.select(..., "first")`)
  3. Density-based (`admission.select(..., "density")`)
  under identical byte budgets.
- Evaluate under strict **group holdouts**: leave-one-width-family-out and leave-one-recurrence-depth-out.
- Kept strictly experimental and outside the numerical correctness path.
