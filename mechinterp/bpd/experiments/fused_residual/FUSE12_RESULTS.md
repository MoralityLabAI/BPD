# FUSE-12 | CUDA Validation, Cumulative Residual Caching Benchmark & DataGoblin Policy Curriculum

- **Status:** Implementation, CPU regression suite (FUSE-06..12), CUDA validation harness, resumable 7-arm benchmark runner, and FUSE-12C DataGoblin pipeline complete; **overnight RTX 5080 CUDA campaign queued in VRAMorama behind active GPU workloads**.
- **Branch:** `research/fuse12-cuda-validation-20261127`
- **Base Commit:** `a4cc3d861df531f41d91000d8dd0922a5ef953ec` (FUSE-11 PR #185, all hosted CPU gates passing)
- **Prospective Protocol:** [FUSE12_PROTOCOL.md](./FUSE12_PROTOCOL.md) (registered prior to code edits and benchmarking, commit `1af76590f5043c06884faf2aeab6c68f35c9f6f9`)
- **Production Defaults:** Unchanged (`allow_cuda_residual_cache=False`, `residual_cache_precision_mode="fp32"`, all residual caches default off).
- **Scope Distinction:** All synthetic trainer benchmarks evaluate the 4-site mini-TRM native `BPDTrainer` and are explicitly distinguished from pretrained 7M TRM or LLM checkpoints.

---

## 1. Preflight & Hardware Audit

- **Upstream PRs Verified:**
  - PR #178 (`1ee5a870c25e88ba6cafa07f349297498daf945f`): `native-cpu` PASS, `cpu-math-and-donor-contrast` PASS, `wide-cpu` skipped.
  - PR #184 (`da81dab14226c413cd5d0a71f61c4fe8143bba6a`): `native-cpu` PASS, `cpu-math-and-donor-contrast` PASS, `wide-cpu` skipped.
  - PR #185 (`a4cc3d861df531f41d91000d8dd0922a5ef953ec`): `native-cpu` PASS, `cpu-math-and-donor-contrast` PASS, `wide-cpu` skipped.
- **Target GPU:** NVIDIA GeForce RTX 5080 Laptop GPU (16,303 MiB NVML / 15,915 MiB WDDM dedicated, WDDM driver model)
- **Driver / CUDA / PyTorch:** Driver `577.05`, CUDA driver `12.9`, PyTorch `2.11.0+cu128` (CUDA runtime `12.8`), NumPy `2.5.3`, SciPy `1.18.1`, scikit-learn `1.9.1`, psutil `7.2.2`
- **Power Limit:** 80 W enforced / 80 W default (Windows logged 2 GPU resets in prior 24 h)
- **GPU Occupancy at Preflight:** Occupied by existing VRAMorama leases (`51dfcc41` `gym-b25-rft4k-logic`, `bb691bf6` `squid5080-serve` / `vmwp.exe` at 15,161 MiB dedicated + 958 MiB shared spill). Competing workloads were respected and never terminated.

---

## 2. Mandatory CPU Regression & Dry-Run Receipts (Verified Locally, `CUDA_VISIBLE_DEVICES=""`)

All upstream FUSE-06..11 CPU gates and FUSE-12 CPU guards passed with zero regressions and exact bitwise parity:

| Gate | Exact Tensor Checks | Lifecycle / Stress Summary | Failures |
|---|---:|---|---:|
| `validate_fuse06_native` | `3,696 / 3,696` | 4 campaigns (widths 64/128, `closed_loop` & `clean_window`), 32 greedy checks | `0` |
| `validate_fuse07` | Oracle exact | Knapsack admission oracle verified | `0` |
| `validate_fuse08_pool` | `672 / 672` | 26/26 lifecycle checks, 96 stress fixtures, 0.0 max abs | `0` |
| `validate_fuse08_native` | `11,136 / 11,136` | 12 campaigns (widths 64/128/256, `exact` & `step_exact`) | `0` |
| `validate_fuse09_pool` | `1,152 / 1,152` | 14/14 lifecycle checks, 96 stress fixtures, 0.0 max abs | `0` |
| `validate_fuse09_native --stage small` | `3,000 / 3,000` | 3 campaigns (widths 64/128/256, `pgd_step`), 24 greedy checks | `0` |
| `validate_fuse10_native` | `2,168 / 2,168` | 4 campaigns, 8 paired steps, calibration lifecycle pass | `0` |
| `validate_fuse11_native` | `5,398 + 1,440` | 9 campaigns (18 paired steps) + 96 stress fixtures, 0 source-query factor adjoints | `0` |
| `validate_fuse12_cuda --cpu-guard-only` | Exact | Option validation, production defaults off, CPU Arm A vs Arm G bitwise parity | `0` |
| `bench_fuse12_cuda --device cpu --tiers smoke --max-cells 2` | Exact across Arms B–G | 2 smoke cells × 7 matched arms (A–G), `fp32_parity_failed=false` | `0` |
| `fuse12c_datagoblin` (dry run) | 14 dataset rows / 32 site samples | End-to-end JSONL export & predictor pipeline verified | `0` |

---

## 3. CUDA FP32, FUSE-12B Mixed Precision & FUSE-12C Results

> [!NOTE]
> Queued overnight in VRAMorama (`owner: bpd-fuse12-cuda`, `--vram 12G --headroom 2048 --strict-fifo --guard-power 60 --pause-exit 75 --restart 6`). Upon completion, `run_fuse12_overnight.py` automatically writes:
> - `receipt_fuse12_cuda_validate.json` (CUDA FP32 lifecycle, stream ownership, zero-leak allocator audit, 96 FP32 stress fixtures, and FUSE-12B TF32/BF16/FP16 identity + stress gates)
> - `fuse12_cuda_receipts.jsonl` (resumable raw JSONL receipts for Smoke, Medium, Wide, Frontier, and Mixed Precision cells)
> - `fuse12c_datagoblin_dataset.jsonl` & `receipt_fuse12c_predictor.json` (DataGoblin curriculum dataset & leave-one-width-out / leave-one-depth-out predictor comparison)
> - Updated `FUSE12_RESULTS.md` with full measured tables, bootstrap CIs, VRAM/NVML/power/energy accounting, and speedup-per-MiB metrics.

---

## 4. Reproduction Instructions

```bash
# 1. Mandatory CPU regression & CUDA default-off guard (no GPU required)
CUDA_VISIBLE_DEVICES="" python -m mechinterp.bpd.experiments.fused_residual.validate_fuse12_cuda \
  --cpu-guard-only --output /tmp/fuse12_cpu_guard.json

# 2. Overnight CUDA validation + resumable progressive benchmark + FUSE-12C (queued via VRAMorama)
vramorama run --vram 12G --headroom 2048 --strict-fifo --owner bpd-fuse12-cuda \
  --guard-power 60 --pause-exit 75 --restart 6 -- \
  python -m mechinterp.bpd.experiments.fused_residual.run_fuse12_overnight
```
