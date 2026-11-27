# ER-31: analytical envelope cost reduction

## Algebra and scope

In ER-20/23 the global interaction envelope for scalar readout r is
E_ij = |v_j^T d_i| (q^T |u_j|) + beta sum_l q_l |a_il b_jl| + beta sum_l |r_l| |a4_il b4_jl|,
where d_i=h2_i-h2_clean, q=|W|^T |r| (column convention), and beta=4/(3 sqrt(3)). The prior kernel computed v_j^T d_i with a separate matrix multiplication per first-component block. By linearity it equals v_j^T h2_i - v_j^T h2_clean. The first projection is already stored by Bank for exact joint rollout, so only the clean projection and subtraction are needed. This is a real-arithmetic identity, not guaranteed bitwise equivalence in floating point.

For nonnegative weight vector w and partition G of hidden coordinates, Cauchy-Schwarz gives
sum_l w_l |a_l b_l| <= sum_{g in G} sqrt(sum_{l in g} w_l a_l^2) sqrt(sum_{l in g} w_l b_l^2).
Refinement of a partition weakly decreases this upper bound (apply Cauchy-Schwarz to the sub-group norms). Singletons recover the original term. One group gives the global weighted norm product. Replace both curvature contractions with this expression, leaving the direct cross-write term exact using projection reuse. Groups of contiguous coordinates, padded with zero coordinates when necessary. Pair contraction work decreases from O(P C^2 D) to O(P C^2 ceil(D/g)) after group-norm computation, where g is group size, P probe count, C component count, D hidden width. Base prefix and single-ablation trajectories, norm preprocessing, storage, lower-bound scoring, sorting and exact response evaluation still cost work. Admissibility follows from the original real-arithmetic envelope; numeric implementation uses EPS=1e-10 and is not a verified floating-point interval method.

## Method

ER-25 teacher/student world generator, roots 63,64,65 at widths 32,64,128 (nine independent trained models). Three distinct 12-probe episodes per model, Gaussian noise multipliers .1,1,3. All seven arms use common observations. Eager uses batch64 exact enumeration. Fused1 is unmodified ER-30 compiled single-probe implementation; reuse changes only analytical projection calculation, with group size1. Group4, group16 and global replace curvature contractions as above; all use the same compiled finish kernel and gap-based probe ranking. Cascade4 first computes group4 floors, evaluates minimum-upper-bound incumbent exactly, then tightens each initially eligible candidate with the original contraction using indexed gathers (batch128), and recomputes probe gap ranking over those candidates only. Thus its ranking and incumbent may differ from baseline. All end-to-end timing includes prefix setup, floor computation, any second screen, sorting and exact finish; excludes SVD/training, world generation, independent audit, and cold compilation. Two rotated timing repetitions per episode, separate processes per width, one BLAS thread. Width128 command returned a timeout after writing all nine complete episode rows and warmup; no missing rows or partial JSON lines, all embedded audit assertions passed before those rows were emitted. Clean serial process-exit replication still desirable. No preregistration.

## Results

27/27 episode MAP and score agreements across all seven arms and both timing repetitions, checked against independently materialized exact response bank. 9,289,728 scalar response-envelope containment checks (four group choices times 2,322,432 distinct candidate-probe terms); each group's envelope dominates original within tolerance, singleton matches original within tolerance, each cost lower bound admissible. Additional 2,322,432 selective fine-pair terms match original predictions and envelopes within tolerance. All executed floor replacements checked by compiled finish. True latent fault recovery 10/27 (.1 RMS 9/9, 1 RMS 1/9, 3 RMS 0/9). No claim of high-noise identifiability.

|width|eager ms|fused1 ms|reuse ms|group4 ms|group16 ms|global ms|cascade4 ms|
|---:|---:|---:|---:|---:|---:|---:|---:|
|32|4.280|1.547|1.457|2.208|2.046|1.837|2.997|
|64|32.947|6.886|6.283|7.617|7.277|6.807|18.626|
|128|517.893|56.578|53.392|57.272|53.691|48.090|161.992|

|method|total exact candidate-probe evaluations|median paired fused1/method ratio|wins /27|model-root wins /9|
|---|---:|---:|---:|---:|
|fused1|98861|1|reference|reference|
|reuse|98861|1.04835|22|8|
|group4|101805|.88247|5|2|
|group16|103127|.94562|10|3|
|global|103403|1.00843|14|5|
|cascade4|103297|.49998|0|0|

Root wins based on median paired episode ratio within each root. Exact candidate-probe count identical between reuse and fused1 in every episode. Reuse is a small positive exploratory timing result. At width128 global norm screen median 26.591 ms versus fused1 34.269 ms, with 67489 vs 64823 exact candidate-probe evaluations (+4.11%); median paired end-to-end speedup 1.126, wins 8/9. Group16 1.100 and group4 1.017 at width128. At smaller widths grouping often increases preprocessing overhead, so no universal grouping win. Cascade4 fails timing comparison on all episodes; median initial eligible fraction .41357, irregular gathers and recomputation overwhelm the savings. Cascade's changed probe ranking means even tighter floors do not guarantee lower work.

|width|original final competitive set total|group4|group16|global|
|---:|---:|---:|---:|---:|
|32|2127|2182|2202|2206|
|64|9858|10127|10212|10233|
|128|31616|32560|32903|33019|

Global norm approximation increases mean envelope by about 3.5-4.0% at these widths. Coarsening increases the competitive set as predicted. All arms materialize a candidate-by-probe floor table, group preprocessing allocates additional norm tables, and RSS was not measured; no memory reduction claim. Speedups are descriptive, based on only three independent model instances per width.

## Reproduce

From experiment directory:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for d in 32 64 128; do
  python er31.py --width "$d" > "er31_$d.jsonl"
done
cat er31_32.jsonl er31_64.jsonl er31_128.jsonl > er31.jsonl
```
NumPy 2.3.5, PyTorch 2.10.0+cpu, Numba 0.65.1. Local SHA256 source 6b613129ec485c7e855d6efb7257c337c3036c0a60323f505bee81c282a4c757; Git blob 260a4ba94d64bea4800705bec4cee58ce106f09c verified remotely. Combined raw receipt SHA256 345cba20ec33c686d0f0961b6014419c3b6a7762a1365367fb91e35c2394b359.

## Next gate

Retain projection reuse as candidate exact algebraic optimization; register it against fused1 and global norm at widths 128 and 256, with fresh roots, warm/cold timings and isolated peak RSS. Freeze the selection rule before evaluating. Avoid irregular coarse-to-fine contraction until it has a credible cost advantage. Consider exploiting b_j rank-one factor structure for the first curvature term, or computing the floor table on demand over blocks of probes. Native learned-factor validation, hard memory limits and formal numerical certificates remain outstanding.
