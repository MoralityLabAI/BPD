# FUSE-12 | CUDA Validation, Cumulative Residual Caching Benchmark, Mixed Precision & DataGoblin Curriculum

- **Generated (UTC):** 2026-11-28 08:07:11Z
- **Branch / Head:** `research/fuse12-cuda-validation-20261127` (`4e84ffa5d489d1769c5229385a29f9db08863d5a`)
- **Base Commit:** `a4cc3d861df531f41d91000d8dd0922a5ef953ec` (FUSE-11 PR #185; PRs #178, #184, #185 all green on `native-cpu` & `cpu-math-and-donor-contrast`)
- **Prospective Protocol:** [FUSE12_PROTOCOL.md](./FUSE12_PROTOCOL.md) (`1af76590f5043c06884faf2aeab6c68f35c9f6f9`)
- **Production Defaults:** Unchanged (`allow_cuda_residual_cache=False`, `residual_cache_precision_mode="fp32"`, all residual caches default off).
- **Scope Distinction:** All benchmark results are measured on the **synthetic 4-site mini-TRM native `BPDTrainer`** (heterogeneous ranks `D/32, D/16, 3D/32, D/8`, `batch_size=2`, `seq_len=16`, `puzzle_emb_len=4`) and are explicitly distinct from pretrained 7M TRM or LLM checkpoints.

## Executive Findings (RTX 5080 Laptop GPU, 16 GB WDDM)
1. **100% Bitwise Exact CUDA FP32 & TF32 Parity:** `30/30` CUDA lifecycle/stream-ownership/zero-leak checks pass (`after_close == base_bytes` to the byte); `1,536/1,536` CUDA FP32 stress tensors and `1,536/1,536` TF32 stress tensors match native BPD bitwise (`max_abs=0.0`, `failures=0`, `0` source-query factor adjoints, `576` joint factor nodes). All 16 parity-audited full-trainer cells (widths 64–2048, passes 6–24, Arms B–G vs Arm A) achieve `100%` bitwise equality across residuals, forward outputs, PGD ascent gradients, pre-clip component/CI gradients, optimizer moments, parameters, calibration buffers, tolerances, greedy outputs, evaluation trajectories, and CPU+CUDA+trainer RNG states.
2. **Primary GPU Win Is `O(T)` Autograd Residual-VRAM Collapse, Not Small-Width FLOP Reduction:** Native BPD saves a distinct materialized `delta_w` tensor in autograd for every pass of every backward-retaining rollout (`O(T * M * d_in * d_out)`). Sharing one detached primal storage per site collapses autograd residual storage to `O(M * d_in * d_out)`:
   - Width `1024`, `T=12`: peak allocated VRAM drops from `1,160.1 MiB` (Arm A) -> `446.1 MiB` (Arm G, **2.60x reduction**).
   - Width `2048`, `T=24`: peak allocated VRAM drops from `7,189.8 MiB` (Arm A) -> `1,638.1 MiB` (Arm G, **4.39x reduction**, `-5,552 MiB`), with `1.082x` step speedup (`1,453.3 ms -> 1,343.6 ms`).
   - Width `4096`, `T=24`: native Arm A allocates **`26,573.9 MiB`** (`28,120 MiB` reserved), spilling over PCIe under WDDM (`56.57 s/step`, `4.39 kJ/step`). Arm F/G holds peak allocated VRAM at **`4,757.6 MiB`** (`5,122 MiB` NVML, **5.59x reduction**, zero spill), cutting step latency to **`3.14 s/step`** (**`18.00x` speedup**, 95% CI `[17.63, 18.12]`) and step energy by **18.2x** (`4,388 J -> 242 J/step`).
   - At small/medium widths (`64–1024`), GPU step time is dominated by Python/autograd dispatch (`site_recon_ms = 0.03–0.22 ms`), so full-step speedups remain near `0.90–1.08x` (with isolated calibration wins like `w=512, p=6, cal=1`: `1.412x`). Cache planner overhead is negligible (`0.15–0.59 ms`).
3. **FUSE-12B Mixed Precision Verdict (No Silent Tolerance Loosening — Counts Computed Directly from Receipts):**
   - **TF32 (`PASS`):** `1536/1536` stress tensors bitwise exact (`max_abs=0.0`, `failures=0`), `2/2` parity-audited trainer cells bitwise exact (`182/182`, `192/192`), and **3.20x peak VRAM reduction** at `w=2048, T=12` (`3,892 -> 1,218 MiB`).
   - **BF16 & FP16 CUDA Autocast (`HOLD — keep on reference path`):** Three independent mechanisms block admission without loosening tolerances: (a) native `TiedComponentLinear.observe_clean_pass` (`components.py:226`) raises `RuntimeError` in `lerp_` under CUDA BF16/FP16 autocast with `pass_mean` (`2` BF16 + `2` FP16 cells recorded as `native_autocast_error`); (b) PyTorch's `AutoDispatchBelowAutocast` thread-local guard inside plain `Function.backward()` causes `_Primal.backward` to execute `g_V, g_U` in FP32 while native `MmBackward0` executes in BF16/FP16 (stress: BF16 `1344/1536` finite bitwise, `192` failures, `0` nonfinite; FP16 `1032/1536` bitwise, `1008/1536` finite within tol, `384` matching nonfinite outcomes [not finite passes], `144` failures; w=512 trainer parity failures: BF16 `C=50, F=52, G=52` [total `154`], FP16 `C=42, F=42, G=42` [total `126`]); (c) under autocast, `F.linear(xf, delta_w)` casts FP32 `delta_w` into a fresh per-pass BF16/FP16 copy saved by autograd, negating the `O(T)` VRAM collapse (`w=2048, T=12`: `6,317 -> 6,445 MiB`, `0.85–0.97x` slower).
   - **Frontier Parity Gap (Tracked in FUSE-13 Stage D):** Width `4096` (`T=24`) benchmark cells did not run full trainer parity validation (`parity_arms=[]`); full parity validation at width `4096` remains open and is scheduled in FUSE-13.
4. **FUSE-12C DataGoblin Curriculum (50-cell / 283-row / 706-site dataset):** Under strict group holdouts across widths `64–4096` and depths `6–24`, a naive raw-ms Ridge model fails (`0.331` width / `0.349` depth oracle fraction) due to 5-order-of-magnitude WDDM-spill leverage, whereas a log-space physical Ridge predictor achieves **`0.9194`** (width holdout) / **`0.9209`** (depth holdout) of oracle step savings (`0.9242` / `0.9256` on direct reconstruction savings), closely matching `exact_saved_mac` (`0.9364` / `0.9505`) and `density` (`0.9379` / `0.9439`). A compact logistic worth-caching classifier (`speedup >= 1.03x OR vram_saved >= 256 MiB`) achieves `71.6%` width-holdout and `66.3%` depth-holdout accuracy.

## 1. Hardware, Preflight & CUDA FP32 Validation Gate
- **GPU:** `NVIDIA GeForce RTX 5080 Laptop GPU` (SM `12.0`, `60` SMs, `16303.0` MiB NVML / `15,915` MiB WDDM)
- **Software Stack:** Driver `577.05`, CUDA driver `12.9`, CUDA runtime `12.8`, PyTorch `2.11.0+cu128`
- **Power / Headroom at Admission:** Enforced `80.0` W / Default `80.0` W, `15915.0` MiB free (`0%` util)
- **CPU Regression Gate (Mandatory):** All FUSE-06..12 CPU gates PASS (`0` failures; PR #186 hosted `native-cpu` & `cpu-math-and-donor-contrast` green).
- **CUDA FP32 Lifecycle & Stream Ownership:** `pass` (`30/30` checks). Zero allocator leak verified: `base=17666560 B`, `peak_in_scope=17928704 B` (`+262144 B` exact), `after_close=17666560 B`.
- **CUDA FP32 96-Case Stress Gate (roots `4012001..8` x scales `1/10/100/10000` x `ordinary/exact/near`):** `PASS` | bitwise exact `1536/1536`, within tol `1536/1536`, failures `0`, `max_abs=0.000e+00`, `max_rel=0.000e+00`, `near_zero_max_abs=0.000e+00`, `joint_factor_nodes=576`, `hits=768`.

## 2. FP32 Matched Benchmark Matrix (38 Synchronized CUDA Cells)
| Tier | W | T | Regime | K | Cal | Cap | Arm A (ms) | Arm C (ms / x) | Arm F (ms / x) | Arm G (ms / x) | G CI95 | Payload (MiB) | Spd/MiB | Alloc A->G (MiB) | NVML A->G (MiB) | Energy A->G (mJ) | Parity (Bitwise / Fail) |
|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---|---|---|---:|
| smoke | 64 | 6 | closed_loop | 2 | 1 | binding | 525.1 | 500.6 (1.049x) | 489.2 (1.073x) | 487.2 (1.078x) | [1.057,1.175] | 0.06 | 1.2464 | 29->28 | 282->282 | 28169->26838 | 1092/1092 / 0 |
| smoke | 64 | 6 | closed_loop | 2 | off | full | 172.0 | 190.8 (0.902x) | 171.3 (1.004x) | 175.6 (0.979x) | [0.942,1.027] | 0.12 | -0.1650 | 28->27 | 282->280 | 6844->7630 | 1092/1092 / 0 |
| smoke | 128 | 6 | closed_loop | 2 | 1 | binding | 496.6 | 527.1 (0.942x) | 502.0 (0.989x) | 490.6 (1.012x) | [0.974,1.057] | 0.25 | 0.0493 | 43->41 | 296->294 | 27423->27128 | 1092/1092 / 0 |
| smoke | 128 | 6 | closed_loop | 2 | off | full | 181.5 | 181.0 (1.003x) | 186.6 (0.973x) | 194.5 (0.933x) | [0.920,1.012] | 0.50 | -0.1337 | 42->37 | 296->292 | 7565->8381 | 1092/1092 / 0 |
| smoke | 128 | 6 | closed_loop | 0 | 1 | full | 384.8 | 405.1 (0.950x) | 433.4 (0.888x) | 403.2 (0.954x) | [0.925,1.009] | 0.50 | -0.0916 | 43->38 | 296->292 | 19459->20590 | 1044/1044 / 0 |
| smoke | 128 | 6 | closed_loop | 4 | off | binding | 275.7 | 263.9 (1.045x) | 268.8 (1.025x) | 259.2 (1.064x) | [1.028,1.065] | 0.25 | 0.2548 | 42->40 | 296->294 | 12333->12284 | 1140/1140 / 0 |
| smoke | 128 | 6 | clean_window | 2 | 1 | binding | 307.2 | 291.0 (1.055x) | 299.6 (1.025x) | 305.7 (1.005x) | [0.954,1.014] | 0.25 | 0.0191 | 33->32 | 286->286 | 14348->14280 | 996/996 / 0 |
| medium | 256 | 6 | closed_loop | 2 | 1 | binding | 520.9 | 505.2 (1.031x) | 523.7 (0.995x) | 519.9 (1.002x) | [0.956,1.016] | 1.00 | 0.0020 | 86->76 | 356->346 | 29637->29202 | 1092/1092 / 0 |
| medium | 256 | 6 | closed_loop | 2 | off | full | 199.4 | 205.0 (0.972x) | 209.1 (0.953x) | 215.0 (0.927x) | [0.901,1.011] | 2.00 | -0.0363 | 84->63 | 354->332 | 8498->9191 | 1092/1092 / 0 |
| medium | 256 | 12 | closed_loop | 0 | 50 | binding | 357.6 | 360.8 (0.991x) | 383.3 (0.933x) | 365.2 (0.979x) | [0.924,0.983] | 1.00 | -0.0208 | 143->121 | 406->382 | 20581->20528 | 1104/1104 / 0 |
| medium | 256 | 12 | closed_loop | 4 | 1 | full | 1558.1 | 1583.9 (0.984x) | 1604.2 (0.971x) | 1601.2 (0.973x) | [0.955,0.993] | 2.00 | -0.0135 | 143->98 | 406->360 | 103970->106284 | 1200/1200 / 0 |
| medium | 256 | 12 | clean_window | 2 | 1 | binding | 1794.9 | 1977.5 (0.908x) | 1764.9 (1.017x) | 1862.6 (0.964x) | [0.706,1.014] | 1.00 | -0.0364 | 89->79 | 356->346 | 114160->119574 | 1056/1056 / 0 |
| medium | 512 | 6 | closed_loop | 2 | 1 | binding | 1189.5 | 1108.1 (1.073x) | 1079.6 (1.102x) | 842.5 (1.412x) | [1.084,1.714] | 4.00 | 0.1030 | 221->180 | 502->452 | 77945->53310 | 1092/1092 / 0 |
| medium | 512 | 6 | closed_loop | 2 | off | full | 369.0 | 418.8 (0.881x) | 402.4 (0.917x) | 432.4 (0.853x) | [0.822,1.033] | 8.00 | -0.0183 | 217->134 | 498->398 | 19814->23023 | 1092/1092 / 0 |
| medium | 512 | 6 | closed_loop | 4 | 50 | full | 787.8 | 1064.3 (0.740x) | 1024.4 (0.769x) | 997.5 (0.790x) | [0.664,1.110] | 8.00 | -0.0263 | 214->131 | 478->398 | 48556->60199 | — |
| medium | 512 | 12 | closed_loop | 0 | 1 | full | 2310.6 | 2479.6 (0.932x) | 2806.1 (0.823x) | 2570.2 (0.899x) | [0.688,0.887] | 8.00 | -0.0126 | 369->190 | 658->452 | 156743->169626 | — |
| medium | 512 | 12 | closed_loop | 2 | 1 | binding | 2805.1 | 3237.8 (0.866x) | 3560.0 (0.788x) | 3621.6 (0.775x) | [0.757,0.914] | 4.00 | -0.0564 | 369->281 | 658->554 | 188928->236453 | — |
| medium | 512 | 12 | closed_loop | 2 | off | full | 861.7 | 973.8 (0.885x) | 978.5 (0.881x) | 862.2 (0.999x) | [0.855,1.112] | 8.00 | -0.0001 | 361->182 | 650->444 | 53347->52289 | — |
| medium | 512 | 12 | closed_loop | 4 | off | binding | 1214.4 | 1166.9 (1.041x) | 1222.0 (0.994x) | 1182.0 (1.027x) | [0.954,1.103] | 4.00 | 0.0069 | 361->273 | 650->546 | 80055->78197 | — |
| medium | 512 | 12 | clean_window | 2 | 1 | full | 2521.0 | 2774.2 (0.909x) | 2383.4 (1.058x) | 2360.7 (1.068x) | [1.022,1.176] | 8.00 | 0.0085 | 216->133 | 480->400 | 167173->158937 | — |
| wide | 1024 | 6 | closed_loop | 2 | 1 | binding | 1523.7 | 1511.5 (1.008x) | 1526.0 (0.998x) | 1655.9 (0.920x) | [0.795,1.210] | 15.00 | -0.0053 | 675->513 | 950->786 | 105812->114961 | 1092/1092 / 0 |
| wide | 1024 | 6 | closed_loop | 2 | off | full | 463.5 | 457.5 (1.013x) | 457.5 (1.013x) | 485.1 (0.955x) | [0.945,1.107] | 30.00 | -0.0015 | 655->325 | 936->612 | 27697->28293 | — |
| wide | 1024 | 12 | closed_loop | 0 | 50 | full | 843.7 | 953.7 (0.885x) | 910.0 (0.927x) | 879.3 (0.960x) | [0.904,0.983] | 30.00 | -0.0013 | 1160->446 | 1432->736 | 55750->57773 | — |
| wide | 1024 | 12 | closed_loop | 2 | 1 | binding | 3598.0 | 3623.2 (0.993x) | 3588.6 (1.003x) | 3584.8 (1.004x) | [0.933,1.076] | 15.00 | 0.0002 | 1160->806 | 1432->1084 | 258627->252540 | — |
| wide | 1024 | 12 | closed_loop | 2 | off | full | 862.5 | 908.5 (0.949x) | 982.0 (0.878x) | 971.6 (0.888x) | [0.816,0.939] | 30.00 | -0.0037 | 1145->431 | 1416->720 | 60617->67627 | — |
| wide | 1024 | 12 | closed_loop | 4 | off | full | 1180.9 | 1110.9 (1.063x) | 1214.4 (0.972x) | 1221.8 (0.966x) | [0.921,1.135] | 30.00 | -0.0011 | 1145->431 | 1416->720 | 88498->88097 | — |
| wide | 1024 | 12 | clean_window | 2 | 1 | binding | 2046.2 | 2124.6 (0.963x) | 2233.1 (0.916x) | 2176.5 (0.940x) | [0.910,1.003] | 15.00 | -0.0040 | 665->503 | 944->776 | 153620->165669 | — |
| wide | 2048 | 6 | closed_loop | 2 | 1 | binding | 1113.4 | 1016.6 (1.095x) | 1085.7 (1.025x) | 1064.9 (1.046x) | [0.980,1.089] | 60.00 | 0.0008 | 2283->1675 | 2704->2000 | 85565->81488 | 1092/1092 / 0 |
| wide | 2048 | 6 | closed_loop | 2 | off | full | 332.3 | 325.7 (1.020x) | 319.8 (1.039x) | 325.9 (1.020x) | [0.962,1.092] | 120.00 | 0.0002 | 2242->1010 | 2666->1346 | 21706->20254 | — |
| wide | 2048 | 12 | closed_loop | 2 | 1 | binding | 2525.0 | 2542.7 (0.993x) | 2558.1 (0.987x) | 2533.4 (0.997x) | [0.994,1.050] | 60.00 | -0.0001 | 3922->2594 | 4476->2980 | 192205->191635 | — |
| wide | 2048 | 12 | closed_loop | 2 | off | full | 720.4 | 719.5 (1.001x) | 752.0 (0.958x) | 744.8 (0.967x) | [0.891,1.020] | 120.00 | -0.0003 | 3892->1218 | 4444->1584 | 53464->55257 | — |
| wide | 2048 | 12 | closed_loop | 4 | 50 | full | 1432.2 | 1461.9 (0.980x) | 1421.9 (1.007x) | 1404.5 (1.020x) | [0.974,1.114] | 120.00 | 0.0002 | 3922->1247 | 4478->1618 | 110301->104911 | — |
| wide | 2048 | 12 | clean_window | 2 | 1 | full | 2090.1 | 2130.8 (0.981x) | 1961.3 (1.066x) | 1949.1 (1.072x) | [1.012,1.115] | 120.00 | 0.0006 | 2262->1029 | 2682->1362 | 160880->148755 | — |
| frontier | 2048 | 24 | closed_loop | 2 | off | binding | 1427.0 | 1395.3 (1.023x) | 1414.8 (1.009x) | 1391.2 (1.026x) | [0.976,1.033] | 60.00 | 0.0004 | 7273->4505 | 8038->4958 | 107633->105966 | 636/636 / 0 |
| frontier | 2048 | 24 | closed_loop | 2 | off | full | 1453.3 | 1366.0 (1.064x) | 1434.5 (1.013x) | 1343.6 (1.082x) | [1.011,1.105] | 120.00 | 0.0007 | 7190->1638 | 7976->1984 | 111569->101889 | — |
| frontier | 2048 | 24 | closed_loop | 4 | 1 | full | 9910.2 | 10025.8 (0.988x) | 10032.7 (0.988x) | 9773.6 (1.014x) | [0.975,1.020] | 120.00 | 0.0001 | 7250->1697 | 8040->2048 | 757031->734873 | — |
| frontier | 4096 | 24 | closed_loop | 2 | off | binding | 58694.8 | 18085.5 (3.245x) | 5231.0 (11.221x) | 5326.8 (11.019x) | [10.760,11.465] | 236.00 | 0.0425 | 26574->15683 | 15884->15906 | 4457749->408589 | — |
| frontier | 4096 | 24 | closed_loop | 2 | off | full | 56574.8 | 4876.7 (11.601x) | 3162.3 (17.890x) | 3142.8 (18.002x) | [17.626,18.119] | 472.00 | 0.0360 | 26574->4758 | 15884->5122 | 4388171->241603 | — |

### 2.1 Phase Timing & Reconstruction Breakdown (Representative FP32 Cells, ms)
| Cell | Arm | Clean/Cal | Inner PGD | Final PGD | Joint Bwd | Opt | Planner | Est Recon | Recons (Cached/Native) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `smoke__fp32__w128_p6_closed_loop_k2_cal1_binding` | A | 31.76 | 110.58 | 51.34 | 74.94 | 0.55 | 0.01 | 5.94 | 144/144 |
| `smoke__fp32__w128_p6_closed_loop_k2_cal1_binding` | C | 29.24 | 119.18 | 53.80 | 72.38 | 0.57 | 0.19 | 4.54 | 110/144 |
| `smoke__fp32__w128_p6_closed_loop_k2_cal1_binding` | F | 30.68 | 111.18 | 53.64 | 71.73 | 0.57 | 0.17 | 4.04 | 98/144 |
| `smoke__fp32__w128_p6_closed_loop_k2_cal1_binding` | G | 29.43 | 116.57 | 55.77 | 70.78 | 0.57 | 0.15 | 3.22 | 78/144 |
| `medium__fp32__w512_p6_closed_loop_k2_cal1_binding` | A | 53.62 | 236.78 | 86.39 | 175.27 | 1.04 | 0.01 | 6.85 | 144/144 |
| `medium__fp32__w512_p6_closed_loop_k2_cal1_binding` | C | 57.55 | 210.65 | 98.25 | 221.97 | 1.16 | 0.30 | 5.23 | 110/144 |
| `medium__fp32__w512_p6_closed_loop_k2_cal1_binding` | F | 84.64 | 252.88 | 122.08 | 146.39 | 1.86 | 0.31 | 4.66 | 98/144 |
| `medium__fp32__w512_p6_closed_loop_k2_cal1_binding` | G | 93.19 | 187.59 | 182.37 | 129.71 | 0.96 | 0.34 | 3.71 | 78/144 |
| `wide__fp32__w1024_p12_closed_loop_k2_caloff_full` | A | 35.33 | 304.92 | 119.29 | 217.37 | 1.47 | 0.02 | 29.11 | 192/192 |
| `wide__fp32__w1024_p12_closed_loop_k2_caloff_full` | C | 43.60 | 415.55 | 160.46 | 254.44 | 1.91 | 0.35 | 7.88 | 52/192 |
| `wide__fp32__w1024_p12_closed_loop_k2_caloff_full` | F | 53.55 | 423.26 | 217.49 | 257.67 | 2.00 | 0.59 | 0.61 | 4/192 |
| `wide__fp32__w1024_p12_closed_loop_k2_caloff_full` | G | 46.15 | 381.52 | 206.00 | 316.74 | 0.77 | 0.42 | 0.61 | 4/192 |
| `wide__fp32__w2048_p12_closed_loop_k2_caloff_full` | A | 33.06 | 268.87 | 130.94 | 160.30 | 1.23 | 0.01 | 89.47 | 192/192 |
| `wide__fp32__w2048_p12_closed_loop_k2_caloff_full` | C | 31.78 | 331.79 | 157.57 | 189.77 | 1.38 | 0.36 | 24.23 | 52/192 |
| `wide__fp32__w2048_p12_closed_loop_k2_caloff_full` | F | 39.74 | 296.41 | 156.56 | 213.21 | 0.65 | 0.42 | 1.86 | 4/192 |
| `wide__fp32__w2048_p12_closed_loop_k2_caloff_full` | G | 36.55 | 277.67 | 130.23 | 198.44 | 0.67 | 0.38 | 1.86 | 4/192 |
| `frontier__fp32__w2048_p24_closed_loop_k2_caloff_full` | A | 62.47 | 583.61 | 284.42 | 329.88 | 0.81 | 0.02 | 179.06 | 384/384 |
| `frontier__fp32__w2048_p24_closed_loop_k2_caloff_full` | C | 68.93 | 545.23 | 290.84 | 386.79 | 0.89 | 0.37 | 46.63 | 100/384 |
| `frontier__fp32__w2048_p24_closed_loop_k2_caloff_full` | F | 63.94 | 579.29 | 298.11 | 360.93 | 1.21 | 0.36 | 1.87 | 4/384 |
| `frontier__fp32__w2048_p24_closed_loop_k2_caloff_full` | G | 67.81 | 550.38 | 272.86 | 355.48 | 0.94 | 0.39 | 1.87 | 4/384 |
| `frontier__fp32__w4096_p24_closed_loop_k2_caloff_full` | A | 1648.87 | 18651.81 | 9821.87 | 13645.39 | 12.94 | 0.02 | 39889.83 | 384/384 |
| `frontier__fp32__w4096_p24_closed_loop_k2_caloff_full` | C | 168.23 | 848.35 | 1434.96 | 2146.39 | 3.01 | 0.39 | 10387.98 | 100/384 |
| `frontier__fp32__w4096_p24_closed_loop_k2_caloff_full` | F | 168.81 | 850.64 | 417.71 | 1391.91 | 2.94 | 0.39 | 415.52 | 4/384 |
| `frontier__fp32__w4096_p24_closed_loop_k2_caloff_full` | G | 168.25 | 863.95 | 395.91 | 1426.30 | 3.13 | 0.34 | 415.52 | 4/384 |

## 3. FUSE-12B: Mixed Precision (TF32, BF16, FP16)
| Mode | Cases | Bitwise Exact | Within Tol | Non-Finite Parity | Failures | Max Abs Err | Max Rel Err | Near-Zero Max Abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tf32` | 96 | 1536/1536 | 1536/1536 | 0 | 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `bf16` | 96 | 1344/1536 | 1344/1536 | 0 | 192 | 6.134e+04 | 4.360e+10 | 4.360e+02 |
| `fp16` | 96 | 1032/1536 | 1008/1536 | 384 | 144 | 1.812e+00 | 3.047e+07 | 3.047e-01 |

| Precision | W | T | Cal | Cap | Arm A (ms) | Arm G (ms) | Speedup | Payload (MiB) | Alloc A->G (MiB) | Parity (Bitwise / Fail) |
|---|---:|---:|---|---|---:|---:|---:|---:|---|---:|
| `tf32` | 128 | 6 | 1 | full | 883.6 | 886.3 | 0.997x | 0.50 | 43->38 | 546/546 / 0 |
| `tf32` | 512 | 12 | off | full | 623.6 | 572.9 | 1.088x | 8.00 | 371->193 | 576/576 / 0 |
| `tf32` | 1024 | 12 | 1 | binding | 2198.0 | 2281.3 | 0.963x | 15.00 | 1160->806 | — |
| `tf32` | 2048 | 12 | off | full | 651.3 | 651.8 | 0.999x | 120.00 | 3892->1218 | — |
| `bf16` | 512 | 12 | off | full | 2304.1 | 2724.3 | 0.846x | 8.00 | 509->520 | 391/576 / 154 |
| `bf16` | 2048 | 12 | off | full | 2348.6 | 2658.7 | 0.883x | 120.00 | 6317->6445 | — |
| `fp16` | 512 | 12 | off | full | 2420.5 | 2587.3 | 0.936x | 8.00 | 509->520 | 396/576 / 126 |
| `fp16` | 2048 | 12 | off | full | 2663.6 | 2748.2 | 0.969x | 120.00 | 6317->6445 | — |

### FUSE-12B Native Autocast Incompatibilities Recorded
- `mixed_precision__bf16__w128_p6_closed_loop_k2_cal1_full`: `native_autocast_error` — `expected dtype float for `end` but got dtype struct c10::BFloat16`
- `mixed_precision__bf16__w1024_p12_closed_loop_k2_cal1_binding`: `native_autocast_error` — `expected dtype float for `end` but got dtype struct c10::BFloat16`
- `mixed_precision__fp16__w128_p6_closed_loop_k2_cal1_full`: `native_autocast_error` — `expected dtype float for `end` but got dtype struct c10::Half`
- `mixed_precision__fp16__w1024_p12_closed_loop_k2_cal1_binding`: `native_autocast_error` — `expected dtype float for `end` but got dtype struct c10::Half`

## 4. FUSE-12C: DataGoblin Admission-Policy Curriculum & Held-Out Predictor
- **Dataset:** [`fuse12c_datagoblin_dataset.jsonl`](./fuse12c_datagoblin_dataset.jsonl) (`283` rows, `706` site samples, `source_dataset_kind="synthetic_native_trainer"`).
- **Widths Observed:** `[64, 128, 256, 512, 1024, 2048, 4096]` | **Recurrence Depths Observed:** `[6, 12, 24]`
- **Contract:** Strictly experimental and outside the numerical correctness path.

| Holdout Scheme | Metric | Learned Log-Ridge | Learned Raw-Ridge | Exact Saved-MAC | First-Fit | Density |
|---|---|---:|---:|---:|---:|---:|
| Leave-One-Width-Family-Out | Oracle Frac (Step ms) | 0.9194 | 0.3308 | 0.9364 | 0.9331 | 0.9379 |
| Leave-One-Width-Family-Out | Oracle Frac (Direct Recon ms) | 0.9242 | 0.3745 | 0.9226 | 0.9360 | 0.9284 |
| Leave-One-Recurrence-Depth-Out | Oracle Frac (Step ms) | 0.9209 | 0.3494 | 0.9505 | 0.8969 | 0.9439 |
| Leave-One-Recurrence-Depth-Out | Oracle Frac (Direct Recon ms) | 0.9256 | 0.3379 | 0.9354 | 0.9053 | 0.9310 |

- **Worth-Caching Classifier** (`speedup_cuda >= 1.03 OR vram_saved_mib >= 256`): leave-one-width-out accuracy = `0.716`, leave-one-depth-out accuracy = `0.663`.

## 5. Artifacts & Reproduction
- Protocol: [FUSE12_PROTOCOL.md](./FUSE12_PROTOCOL.md)
- CUDA Validation Receipt: [receipt_fuse12_cuda_validate.json](./receipt_fuse12_cuda_validate.json)
- Raw JSONL Benchmark Receipts: [fuse12_cuda_receipts.jsonl](./fuse12_cuda_receipts.jsonl)
- DataGoblin Curriculum Dataset: [fuse12c_datagoblin_dataset.jsonl](./fuse12c_datagoblin_dataset.jsonl)
- FUSE-12C Predictor Report: [receipt_fuse12c_predictor.json](./receipt_fuse12c_predictor.json)

```bash
# 1. Mandatory CPU regression & CUDA default-off guard (no GPU required)
CUDA_VISIBLE_DEVICES="" python -m mechinterp.bpd.experiments.fused_residual.validate_fuse12_cuda \
  --cpu-guard-only --output /tmp/fuse12_cpu_guard.json

# 2. Full CUDA validation + resumable progressive 7-arm benchmark + FUSE-12C (VRAMorama)
vramorama run --vram 12G --headroom 2048 --strict-fifo --owner bpd-fuse12-cuda \
  --guard-power 60 --pause-exit 75 --restart 6 -- \
  python -m mechinterp.bpd.experiments.fused_residual.run_fuse12_overnight
```

