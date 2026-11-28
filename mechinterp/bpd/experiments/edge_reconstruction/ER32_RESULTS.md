# ER-32: exact rank-one curvature contraction

## Identity

In the four-pass SVD component fault model of ER-23 through ER-31, the second-pass single-component preactivation perturbation is b_j=-u_j(v_j^T h2). Thus the first weighted curvature contraction in the global envelope factorizes:

sum_l q_l |a_il b_jl| = |v_j^T h2| sum_l q_l |a_il| |u_jl|.

The last expression uses a shared write matrix, with the clean projection factored outside the contraction. The direct cross-write term continues to use the difference of already-stored prefix projections. The last curvature contraction still uses the full downstream single-intervention displacement, which is generally not rank one. The rank implementation preweights the two first-component displacement arrays and uses a flattened matrix multiplication for the first contraction, a batched contraction for the second. It drops retained b, br and b4 arrays and retains only the arrays needed by screening. Both algebraic identities preserve the global bound in real arithmetic; float64 reassociation can change rounding. EPS=1e-10, no directed rounding or formal floating-point guarantee.

## Design

Fresh widths 128 and 256, roots 66 and 67 per width, trained with the ER-25 teacher/student generator (four trained models), three distinct 12-probe episodes per root at noise multipliers .1,1,3. Prepare worlds once, then benchmark each configuration/root in a fresh process with a Numba warmup. Modes reuse (ER-31 group1), global (ER-31 global weighted norms), rank (factorized contraction and layout/preweight changes). Same compiled single-probe exact finish, floor table, batch64, first-component block8, gap ranking and exact incumbent rule across modes. Two repeated timings per episode, median within episode; root66 mode order reuse/global/rank, root67 reversed. Training, SVD, loading, cold compilation and independent exhaustive audit excluded from solve timing. One BLAS thread. Each process executes three episodes, so final recorded Linux ru_maxrss is the high-water mark for that root/mode, including interpreter, compiler and input loading. Baseline high-water RSS recorded after warmup before input loading. Subtracting baseline gives an observed high-water increment, not an allocation attribution. Named array payload sums ndarray.nbytes over Bank attributes, including input references and excluding temporary arrays, object overhead and separately allocated floors. This is a retained-state accounting measure, not actual RSS. No preregistration.

## Results

All 36 mode-episode results (two repetitions each) agree with exact enumeration's MAP index and score within 1e-10 on all 12 distinct episodes. Hidden fault recovered in 5/12. The independent audit compares rank and global envelopes to exact responses on every candidate-probe term: 11,796,480 containment checks across both methods; maximum absolute difference between factorized and original reuse envelopes 1.7763568394002505e-15. Rank and reuse have identical exact candidate-probe counts for every episode.

| width | mode | median end-to-end ms | median screen ms | total exact candidate-probe work | retained array bytes |
|---:|---|---:|---:|---:|---:|
|128|reuse|57.419|23.510|61035|14659680|
|128|global|61.034|25.338|64157|14708832|
|128|rank|58.467|24.184|61035|10072160|
|256|reuse|338.330|130.544|216266|58417248|
|256|global|342.985|115.214|227572|58515552|
|256|rank|340.706|127.585|216266|40067168|

Median paired reuse/rank speedup .9975 (3/6 wins) at 128 and .9932 (3/6 wins) at 256. Reuse/global paired ratios .9089 and 1.0054, wins 1/6 and 4/6 respectively. No incremental runtime win for rank. The global-norm contraction incurs about 5% more exact work at both widths. Rank reduces named array payload by 31.29% at width128 and 31.41% at width256, saving 4.38 and 17.50 MiB respectively. The separate float64 floor table remains 1572864/6291456 bytes (1.5/6 MiB) at both widths for every mode.

Whole-process peak RSS in MiB (root66/root67):
|width|reuse|global|rank|
|---:|---|---|---|
|128|231.49/231.29|231.34/231.34|231.56/231.41|
|256|289.20/288.64|289.41/289.23|289.61/289.56|

No peak RSS improvement observed. At width256 post-warmup high-water increments are 76.14/75.54 MiB for reuse and 76.25/76.47 MiB for rank. Setup temporaries, compiled execution temporaries and allocator behavior are potential contributors; this benchmark does not isolate their respective contributions. Retain reuse as reference runtime kernel, rank as a candidate retained-state optimization. Widths, roots and factorized layout selected for exploratory pilot; two roots per width, no hardware transfer or confidence claim.

## Reproduction

From experiment directory:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for d in 128 256; do
  python er32.py prepare worlds --width "$d" --roots 66 67
  for root in 66 67; do
    python er32.py audit worlds/${d}_${root}_*.npz
    modes='reuse global rank'
    [ "$root" = 67 ] && modes='rank global reuse'
    for m in $modes; do
      python er32.py bench worlds/${d}_${root}_*.npz --mode "$m" --reps 2
    done
  done
done
```
The original benchmark ran all audits for each width before the corresponding benchmark processes. NumPy 2.3.5, PyTorch 2.10.0+cpu, Numba 0.65.1. Executed source SHA256 994b42ade47749f828b75db2c31633420d02268e48bfc2384b9851e370e613df; Git blob f1c403ce7566a5903c7339d5818f198d2d348d78 verified remotely. Raw audit SHA256 0ae697adbd590daf3cacfa01122c6adfdb58db7d8af2cff4620bad36d72050f2; benchmark SHA256 70a4ce9e9b1413810dec2c3da741e7ed0f356f9a24486dd38a0992848817d078.

## Next gate

Reduce peak allocation by building first/second single-ablation trajectories in component blocks, freeing or reusing temporaries as soon as their final consumers complete. Record allocation-stage measurements and isolated process RSS; separate startup compiler effects from solve-stage work. Investigate recomputing only eligible candidate-probe floors after retaining a scalar lower cost, with no oracle labels and an explicit recomputation budget. Compare memory under equal exact-kernel work and hold model roots fixed only for development; freeze design before fresh confirmation. Native learned BPD factors, pass-mean and sphere-rescale semantics, hard memory caps and formal numerical certificates remain outstanding.
