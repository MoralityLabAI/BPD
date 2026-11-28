# ER-30: compiled probe-stage screening

## Design

Fresh ER-25 trained teacher/student tanh recurrence instances, roots 60,61,62 at widths 32,64,128; nine independent models, three distinct 12-probe episodes/model at noise multipliers .1,1,3 times calibration RMS. Six methods: prefix-reused eager batch64, ER-29 NumPy gap4/gap2/gap1 and Numba compiled gap1/gap2 (fused1/fused2). Same floor construction, exact initial incumbent, stable lower-cost candidate order, gap-based probe order, batch64 and stage pruning. The frozen prior policy is gap4 at width32 and gap2 at widths64/128. Compilation moves gathering, first activation, BLAS multiplication, final activation/readout, floor replacement and stage pruning into a single nopython function; fastmath disabled. No GPU or trained native BPD factors. Float64 EPS=1e-10, not outward-rounded interval arithmetic. All methods execute two rotated timing repetitions per episode. Widths sequential, one BLAS thread. Timings include prefix construction, screening, sorting, array copies, execution and floor checks, exclude training/SVD/environment generation and cold compilation. Each width runs in a fresh process with one separate synthetic width6 warmup of the compiled function. No disk compilation cache. No preregistration or measured peak RSS.

## Results

27/27 exact MAP and cost agreements within 1e-10 across all arms and repetitions. True fault recovery 9/27, all at .1 RMS noise (9/9), zero at 1 and 3 RMS. The compiled finish functions checked 432032 evaluated lower floors across both timing repetitions; incumbent excluded from this count. Identical exact candidate-probe counts for NumPy and compiled variants at the same stage size, on every episode. All work counts include the exact initial incumbent.

|width|eager ms|gap4 ms|gap2 ms|gap1 ms|fused1 ms|fused2 ms|
|---:|---:|---:|---:|---:|---:|---:|
|32|4.819|2.097|2.160|2.611|1.930|1.778|
|64|35.672|8.239|7.811|9.451|7.344|7.849|
|128|460.545|69.318|57.511|61.015|52.517|64.584|

|method|total exact candidate-probe evaluations|median paired frozen-policy/method ratio|wins vs frozen /27|
|---|---:|---:|---:|
|eager|2322432|.227|0|
|gap4|186272|1.000|7|
|gap2|122652|1.000|3|
|gap1|94012|.864|2|
|fused1|94012|1.089|18|
|fused2|122652|.949|12|

Single-probe stages reduce work 23.35% vs gap2 and 49.53% vs gap4. Fused1 vs gap1 median paired speedup 1.199, 26/27 wins. Fused2 vs gap2 median paired ratio .949, 12/27 wins. Fused1 vs frozen prior policy median paired speedup by width 1.125,1.069,1.032; wins 7/9,6/9,5/9 episodes. Median ratios over the three episodes within each independent model root favor fused1 on 7/9 roots. Median of these nine root medians 1.095; exploratory stratified model-root bootstrap percentile range [1.032,1.125] (10000 resamples with replacement within each width, seed 261143). Nine roots and two timing repetitions provide limited precision; this interval is descriptive and does not establish hardware transfer or a registered confirmatory win.

## Cold start and workload tradeoff

Compilation plus first tiny synthetic call takes 2275,2112,2053 ms in the three fresh processes (excludes imports). Mean paired frozen-policy minus fused1 time savings .1964,.3776,-3.1387 ms respectively at widths32/64/128. At the first two widths a simple cold-cost/mean-saving estimate gives about 11583 and 5592 same-mix episodes to amortize compilation; at width128 the measured mean is negative despite a positive median and no positive mean-based break-even is observed. These are extrapolations from small samples; persistent workers, cached or ahead-of-time kernels may change the tradeoff. Warm median improvement does not support universal deployment. Retain NumPy fallback; fused1 candidate for persistent processes. Fused2 adds no overall advantage over NumPy gap2.

## Reproduce
From experiment directory with er23.py, er25.py, er29.py:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for d in 32 64 128; do
  python er30.py --width "$d" > "er30_$d.jsonl"
done
```
NumPy 2.3.5, PyTorch 2.10.0+cpu, Numba 0.65.1. Source SHA256 c841c4731b48c0d91442217b84e1123b61cfeef6aaeed634bf42d3830af0475c; Git blob 91d3c278c5e04c614ab8ab8241647965ec418a9f verified remote. Combined local receipt (warmup rows then episode rows, lexical width-file order) SHA256 21fd8398d388c0bbe1961b9113004b323dd5026d49c7b05fc03ab15d8c8cde3c.

## Next gate

Register persistent-worker timing against precompiled static gap4/gap2, larger fresh roots and workload-dependent cold-start accounting. Compare vectorized activation approximations only if their error can be enclosed and their decisions verified. Directly measure memory overhead and operating-system containment. The main mathematical direction is reducing analytical floor construction and exact arithmetic per candidate, with a portable certificate; no learned BPD factors, native larger-width parity or Lean proof in this experiment.
