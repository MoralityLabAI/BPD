# ER-27/28: tighter signed bounds and partial-probe search

## ER-27 mathematical bound

Use the four-pass tanh recurrence and SVD component ablations of ER-23/25. Write h3, hi, hj, hij for the clean, first-only, second-only and joint third-pass states, respectively. Define mixed state m=hij-hi-hj+h3. At the final preactivation, let z=W h3+b, a=W(hi-h3), c=W(hj-h3), d=Wm, so the exact final response is r^T tanh(z+a+c+d).

Let f0=r^T tanh(z+a+c). The signed Taylor center using the first-only trajectory is mi=f0+g_i^T m, with g_i=W^T diag(tanh'(z+a))r. The error is bounded by beta [sum_l |r_l||c_l d_l| + (1/2) sum_l |r_l|d_l^2], beta=4/(3 sqrt(3)). This follows by integrating the derivative difference along the segment from z+a+c to z+a+c+d and bounding the derivative's Lipschitz constant by beta. Cauchy-Schwarz bounds the first term by sqrt(q) sqrt(sum_l |r_l|c_l^2) where q >= m^T Q m and Q=W^T diag(|r|) W. Use q=min(sum_k m_k^2 sum_l |Q_kl|, ||m||^2 min(||Q||_infinity, max_l |r_l| ||W||_2^2)); the row-sum majorizer follows from symmetry and 2|m_k m_l|<=m_k^2+m_l^2. Exchange a and c to obtain a second signed interval. A third interval centered at f0 follows from the global Lipschitz bound |f-f0| <= sqrt(sum_l |r_l| q). Intersect all three intervals. This avoids pair-specific Wm, using exact third-pass mixed states and precomputed derivatives/weighted Gram matrices.

For each probe observation y, the distance from y to the response interval squared is an admissible cost floor. The maximum of the old and new total cost floors is admissible. All bounds are real-arithmetic derivations; float64 implementation adds EPS=1e-10 and uses numerical SVD and Gram operations, not directed rounding.

## ER-27 executed pilot

Widths 32,64,128; roots 52,53,54 at each width (nine fresh trained models). ER-25 generator, three independent 12-probe episodes per model, noise multipliers .1,1,3 times calibration RMS. Four arms: prefix-reused eager exact batch64; prior streamed bounds plus sorted exact batch64; cascade computes signed intervals for candidates surviving initial incumbent (including incumbent); signed-only computes signed intervals for all candidates and selects initial incumbent by lowest upper cost. Signed stage batch256. All timings include prefix construction, screens, exact evaluation and search, exclude training/environment generation and exhaustive audit. Two rotated timing repetitions per arm, single BLAS thread. Nine independent models, 27 correlated episodes. Early cascade bookkeeping omitted the initial incumbent from the intersection audit; corrected before final reruns. Width128 root54 rerun in full after command timeout; partial rows discarded.

All 27 MAP indices and costs match the full eager oracle within 1e-10. Hidden fault recoveries 9/27 (all at .1 RMS noise). Exhaustive audit checks 193536 candidate costs, 2322432 scalar response interval containments, old and new admissible cost bounds, and equality of final competitive-set count between cascade and full intersection.

| width | previous S* total | signed S* total | eager median ms | previous bound median ms | cascade median ms | signed-only median ms |
|---:|---:|---:|---:|---:|---:|---:|
|32|2186|11|4.360|2.324|5.148|9.442|
|64|9124|9|32.592|15.322|45.049|91.580|
|128|32556|9|457.997|139.889|304.398|737.985|
|total|43866|29|||||

Intersection S* counts equal signed-only counts on this roster. Total full joint calls eager 193536, previous bound 43947, cascade 461, signed-only 31. The 99.93% S* reduction does not establish an end-to-end speedup: cascade wins 0/27 timing comparisons against previous bounds. Screening itself performs exact third-pass activation and extensive vector work; full joint call count excludes that work. Negative runtime result retained.

## ER-28: partial-probe exact bounds

Instead of constructing tighter response intervals, replace old analytical floors by exact squared residual terms for a subset of probes. Given valid nonnegative floors l_gp <= e_gp, define lower bound L_g(T)=sum_(p in T) e_gp + sum_(p not in T) l_gp. This is admissible, nondecreasing as T grows, and becomes the exact energy when all probes are evaluated. Prune only when L_g(T)>exact incumbent+EPS; ties remain eligible. Start with the same incumbent and old bound ordering as the prior search. Rank probes by aggregate old upper-minus-lower cost gap; this uses observations and candidate bounds, no hidden labels. Evaluate candidates in batches of 64; stage sizes 4 or 6. Compute exact responses only on demanded columns by selecting shared prefix states before final dense recurrence. At each stage prune candidates with revised costs above the incumbent. Once all columns have been computed update exact incumbent and lowest-index tie break. This changes the definition of work: count each exact candidate-probe computation, not just fully evaluated joint candidates. Floors materialized as float64 candidate-by-probe table (1.5 MiB at width128); no measured RSS or hard memory cap.

Fresh roots 55,56 per width (six models); three noise regimes, 18 episodes, two rotated timing repetitions. All four methods return same MAP and score within 1e-10 as eager, evaluated residual floors checked online. True fault recovered 7/18 (.1:6/6, 1:1/6, 3:0/6). All pairwise timings descriptive, not a registered confirmation.

|width|eager median ms|previous bound median ms|partial4 median ms|partial6 median ms|median paired previous/partial4|median paired eager/partial4|
|---:|---:|---:|---:|---:|---:|---:|
|32|4.769|2.618|1.990|2.287|1.195|2.692|
|64|34.918|12.045|8.381|9.417|1.289|4.075|
|128|456.746|176.694|75.658|90.068|2.337|6.138|

Partial4 wins 17/18 vs previous bound, median paired speedup 1.374; partial6 wins 16/18, median 1.290. Exact candidate-probe computations: eager 1548288, previous bound 374724, partial4 133036, partial6 191706. Partial4 reduction 64.5% vs previous bound; 262 fully evaluated joint candidates (including 18 incumbents), stage survivor counts 31209/1752/244. Partial6 724 fully evaluated joint candidates. Do not interpret exact probe counts as FLOP counts, and do not compare full-call counts to signed screen work. Result supports partial4 as next fixed baseline, pending a larger frozen-root replication.

## Reproduce

From experiment directory (using merged er23.py and er25.py):
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for d in 32 64 128; do
  python er27.py --width "$d" > "er27_$d.jsonl"
  python er28.py --width "$d" > "er28_$d.jsonl"
done
```
For original width128 timing grouping run ER-27 roots 52,53,54 separately if necessary. NumPy 2.3.5, PyTorch 2.10.0+cpu. Source SHA256 er27 68a315afec551d32a53e268795de1145c2fe8c53d2bd6651b9bc2d1c4f3f85f7, er28 49952908794c2fa1c8f4074bf99f83531032574e0d887b02d3cafd27ca61a0ca. Remote Git blobs verified against local: 75909892dd5bb03fb141ce30dbbc3f14abcd1abb and c89e2698891799f414d0ccbd1a7d8b490abaa56e. Combined raw receipt SHA256 er27 4687bd0452fda8ab86d0672b943023736f6662c5fbb00d3130fe4941b0a0e322, er28 b869f269cc5ace5727314cb5168bec5a09d983545878b5d31c63cb2a2551082c.

## Next gate

Freeze partial4 as incumbent. On fresh roots compare probe-order nulls (fixed and shuffled), partial stages 2/4/6, and a fused/compiled exact-column kernel with matched budgets. Test the predicted floor-tightening contribution against actual improvement, register root-level timing and peak RSS criteria, and compare materialized floors against streamed/recomputed floors under a strict memory budget. Extend beyond the four-pass SVD proxy separately. No formal floating-point certificate, native learned-factor benchmark, or conclusion about high-noise true-fault identifiability.
