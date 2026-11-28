# ER-25: survivor-conditioned refinement kernel selection

## Scope and method

Fresh trained four-pass tanh teacher/student models with SVD-initialized components. Widths 32,64,128; three calibration roots (42..44) and three test roots (45..47) per width, 18 independently trained instances. Seed includes width. Each root yields three episodes on distinct 12-probe input/readout batches, with Gaussian noise scale equal to .1, 1, or 3 times RMS joint intervention response (separate 24 calibration probes and up to 256 calibration pairs). Hidden fault pair sampled uniformly; true response generated using direct weight edits and checked against prefix-reused exact response at tolerance 1e-10. Same observations for all arms. The task is uniform-prior common-Gaussian MAP. No learned decomposition, native larger-width parity, hard RAM limit, or preregistration.

All bounded arms share streamed bounds (block 8), minimum-upper-bound incumbent and one exact evaluation of that incumbent. The survival fraction is the count of other candidates with lower cost <= incumbent exact cost + EPS, divided by width squared. The finish kernels visit this initial eligible set sorted by lower bound (batch 16,64,256) or by index (batch 64), and discard candidates whose lower cost exceeds the updated incumbent. Only sort arms can terminate on first ineligible lower bound. Every evaluated cost checked against lower bound with EPS=1e-10. Ties select smallest index. Eager enumerates all pairs with batch 64 using the same prefix-reused exact kernel. Floating-point checking is not machine-certified interval arithmetic.

Adaptive selects the kernel with lowest median finish time among the three calibration episodes of the same width nearest in log(1+survivor count); static selects the lowest median finish time across all nine calibration episodes of that width. Width and observed bounds are the only features; test observations, labels and runtimes do not fit the choice. All timing includes prefix construction, screening (when used), policy selection and exact evaluations; excludes training, SVD and environment generation. Two rotated timing repetitions per arm/episode, single BLAS thread. Static also computes its calibration lookup at runtime, so a precompiled static selector could be faster still.

## Held-out results

27 calibration and 27 test episodes. All 27 held-out episodes have identical MAP indices across every arm and repetition, with costs agreeing within 1e-10. The true fault is recovered in 10/27: 9/9 at .1 RMS noise, 1/9 at 1 RMS, 0/9 at 3 RMS. This is evidence of agreement with exact optimization under noise, not recovery of an unidentifiable graph.

| width | eager median ms | sorted64 median ms | static median ms | adaptive median ms | adaptive median joint calls | median paired eager/adaptive | median paired static/adaptive |
|---:|---:|---:|---:|---:|---:|---:|---:|
|32|4.4967|1.7101|1.8520|2.5296|257|1.670|0.960|
|64|35.4865|10.5438|11.4786|12.5252|813|2.937|0.971|
|128|491.1106|111.6737|109.7554|109.6908|2784|4.658|1.004|

Static selects sort64 at widths 32/64 and sort16 at 128. Adaptive choices: width32 sort64 six, filter64 two, sort256 one; width64 sort64 nine; width128 sort64 five, sort16 four. Adaptive beats static in 8/27 episodes. Median paired static/adaptive speed ratio 0.97086 (adaptive ~3% slower); eager/adaptive 2.71867. Median runtime relative to fastest measured hindsight arm 1.0670 for adaptive, 1.0762 for static (descriptive, timing noise can favor a hindsight arm). No positive incremental speed claim. Across test episodes, Pearson correlation between initial eligible fraction and exact evaluated fraction under sort64 is .80893 (pooled descriptive statistic, correlated within roots). Median evaluated fractions: adaptive .19849, static .16992. No statistical confidence interval or independent hardware replication.

| noise RMS multiplier | hits /9 | median initial eligible fraction | median adaptive evaluated fraction |
|---:|---:|---:|---:|
|.1|9|.2793|.1045|
|1|1|.3579|.1699|
|3|0|.5214|.2959|

Calibration width128 run hit the tool timeout after eight rows; all three episodes for calibration root44 were rerun together and replace its partial rows. Every reported root is complete; no partial results included. The two repetitions and small root count limit performance conclusions. Process RSS not measured here; batch choices are bounded in count but not by a hard process memory limit.

## A sharper cost feature for sorted refinement

In real arithmetic with valid lower bounds L_g, true minimum exact cost E*, tolerance delta >= 0, and a sorted eligible list, define S*={g : L_g <= E*+delta}. All members of S* are evaluated (except the already evaluated incumbent, if it belongs to S*), because the current exact incumbent threshold never drops below E*+delta. Before the first evaluation of any optimum, all candidates earlier in the sorted list than that optimum belong to S*, since L_opt <= E*. At most b-1 extraneous candidates can be evaluated in the batch containing that first optimum, where b is batch size. After that batch all evaluations lie within S*. Thus the exact-call count is at least |S*| and at most |S*|+b (including a possibly extraneous initial incumbent). If the initial incumbent is already optimal, there are no extraneous evaluations. This argument assumes no arbitrary early stopping and excludes numerical failures of the bounds. It provides an offline diagnostic of batching overhead, and suggests estimating the final survivor count and per-batch kernel throughput separately. Initial survival alone does not predict how fast the incumbent improves; measure a cheap early batch, lower-bound quantiles and exact score contraction next. This bound is derived here, not yet independently tested or formally verified.

## Reproduction
From experiment directory:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for d in 32 64 128; do
  python er25.py cal "$d" > "cal$d.jsonl"
  python er25.py test "$d" --cal "cal$d.jsonl" > "test$d.jsonl"
done
cat cal32.jsonl cal64.jsonl cal128.jsonl > er25_cal.jsonl
cat test32.jsonl test64.jsonl test128.jsonl > er25_test.jsonl
```
Local environment NumPy 2.3.5, PyTorch 2.10.0+cpu. Source SHA256 482274c339335e3aea739e7990ff9c08a2b6333332cffcdcbc1d6aab4056396e; Git blob 3f40349a20ccbb60ab98ba7bbe04f4887c241b36 verified against remote. Raw receipts SHA256 cal 08abf616dabc71ac85093365ea3adea75c6a73c2184406b549d3135d838c4609; test b1d39dc4afc89ac2eb9529f60e83ba2a160afdda4297b85616cc619e89568cc6. Timings are hardware dependent.

## Next registered gate
Compare precompiled static sort64, survivor-conditioned selection, and an online incumbent-contraction controller at matched exact kernels, separate hardware calibration, fresh roots, and memory caps. Record S* and batch overhead directly. Keep noise-level MAP agreement separate from true-fault recovery. 
