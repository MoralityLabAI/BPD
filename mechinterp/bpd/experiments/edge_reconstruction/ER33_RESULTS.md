# ER-33: Blockwise construction and early release

## Design

Four newly trained ER-25 model instances at widths 128 and 256, roots 68,69 at each width, three distinct 12-probe episodes per root (noise multipliers .1,1,3 times calibration RMS). Compare ER-32 rank-one setup, blockwise setup (8 components per block), and blockwise setup with release of all screening-only Bank attributes before compiled exact finish. The block path constructs first-ablation prefixes/projections in preallocated arrays, does not retain first2, and computes single-ablation third/fourth-pass displacements in blocks with in-place elementwise operations. The same ER-32 envelope formula and ER-30 exact finish are used throughout; floors, candidate sort, initial incumbent, probe ordering, and exact batch64/single-probe settings are unchanged. All arms use two repetitions per episode in an isolated process per root/mode, modes rank/block/block_release for root68, reversed for root69. One BLAS thread. SVD, model training, input loading, independent audit and Numba warmup excluded from solve timings. Linux VmRSS sampled at named stages; ru_maxrss is cumulative process high-water RSS. Baseline sampled after warmup but before loading worlds. Stage measurements introduce small common overhead; cumulative high-water differences are not allocations attributable to particular routines. Retained payload is sum of named Bank ndarray nbytes (includes inputs by reference, excludes external references, temporary arrays, floor table and compiled input copies). No hard memory cap, preregistration or cross-machine replication.

## Exactness audit

All 5,898,240 scalar response-envelope containment checks pass; rank and block envelopes have max absolute difference 0.0 on these inputs. Predictions and exact candidate responses also agree within 1e-10. All 36 mode-episode results (72 timed calls) return the same MAP index and cost as full exact enumeration within 1e-10. True injected fault recovered in 5/12 episodes. Exact candidate-probe work counts identical in every episode and every arm: total 32,114 at width128 and 133,915 at width256. No bitwise guarantee for other BLAS implementations; EPS tolerance is not outward-rounded arithmetic.

## Performance

|width|mode|median total ms|median screen ms|median paired rank/mode|timing wins /6|setup retained bytes|finish retained bytes|
|---:|---|---:|---:|---:|---:|---:|---:|
|128|rank|46.802|23.732|1|reference|10072160|10072160|
|128|block|31.659|21.157|1.101|5|8499296|8499296|
|128|block_release|36.425|20.845|1.086|6|8499296|3432448|
|256|rank|295.423|128.338|1|reference|40067168|40067168|
|256|block|279.748|116.258|1.073|6|33775712|33775712|
|256|block_release|295.376|121.568|1.019|4|33775712|13680640|

Timing ratios are medians of paired per-episode ratios, not ratios of aggregate medians. Two model roots per width and timing repetitions are exploratory. Separate floor tables remain 1,572,864 and 6,291,456 bytes at 128 and 256 in all arms.

Final process peak RSS (MiB, median over two model roots):
|width|rank|block|block_release|
|---:|---:|---:|---:|
|128|231.54|222.56|222.61|
|256|289.62|254.53|254.62|

Post-warmup high-water increment at width256 (median of paired root process records): rank 76.47 MiB, block 41.39 MiB, block_release 41.41 MiB. Block reduces total peak by 35.09 MiB and the increment by 45.9%. This does not isolate compiler memory or individual allocations. For the first episode and repetition, median cumulative high-water increment at width256 is 74.74 MiB after rank setup and 32.71 MiB after block setup, before allocating the floor table. Blocking therefore addresses the earlier setup peak. Early release saves another 18.01 MiB of observed current RSS immediately before the first exact finish at width256 (254.26 to 236.25 MiB for block_release); at width128 no current-RSS reduction is observed at that phase. Releasing after screening cannot lower an already established peak. It removes 20,095,072 named bytes at width256, though not all referenced buffers necessarily return pages to the OS. Block mode is the candidate default; early release merits testing in persistent or concurrent workers rather than claiming a further peak improvement.

## Reproduce

From the experiment directory:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for d in 128 256; do
  python er32.py prepare worlds --width "$d" --roots 68 69
  for root in 68 69; do
    python er33.py audit worlds/${d}_${root}_*.npz
    modes='rank block block_release'
    [ "$root" = 69 ] && modes='block_release block rank'
    for m in $modes; do
      python er33.py bench worlds/${d}_${root}_*.npz --mode "$m" --reps 2
    done
  done
done
```
Original run prepared and audited worlds before the serial benchmark processes. Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu, Numba 0.65.1. Executed source SHA256 2067fc5e6e512573872fd0f4aa88068a2ab2bcf84b2a24a85e40ab8faa1e823f, verified remote Git blob 346e38dc8d212e0009577b03f8f46c1957c24ac5. Combined audit SHA256 3eecbc264226ece6b0f70f9eaf7f86ee76ea46f8a5b600e17b8a556f5ffb6170; benchmark SHA256 eae79e2ceffa7b4858169c4a5b4938d812029cb65124683858525b1b2c86d182.

## Next gate

The next bottleneck is retaining the full candidate-by-probe floor table. Compare dense floors with an initial scalar-cost-only screen followed by indexed floor recomputation for the eligible candidates. Count recomputation, retain exactly the same gap rank and incumbent rule, and report whether reduced memory survives whole-process RSS and wall-clock accounting. Larger fresh model rosters, learned native BPD factors, pass-mean/sphere semantics, OS containment, and formally verified arithmetic are separate pending gates.
