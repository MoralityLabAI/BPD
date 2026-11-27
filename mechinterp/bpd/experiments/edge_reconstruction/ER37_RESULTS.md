# ER-37: equal-byte prefix-only cache

## Design

Four fresh trained ER-25 models, roots 76 and 77 at widths 128 and 256; three 12-probe episodes per root, noise multipliers .1, 1, 3 times calibration RMS. Compare the full projection FIFO with 512 entries, a prefix-only FIFO of 512 entries, and prefix-only FIFO at the same array-byte budget as the full cache. All modes use the same compiled finish function, ER-35 streamed screen, exact incumbent, fixed candidate order, batch-local deduplication, batch64, and single-probe stages. Cache is empty at the start of each episode.

For a key (i,p), the full cache stores z3 and all C second-component projections. Prefix-only stores z3, reconstructs h2 by elementwise tanh of the clean preactivation minus the first-component write, then evaluates only the demanded scalar projections. On a cache miss z3 is computed from h2 by a dense matrix multiplication. The prefix-only mode must reconstruct h2 even on a hit; it shares h2 within the batch. For D-dimensional states and P probes the named cache payload includes float64 entries, int64 FIFO keys, and int64 slot map: 8[K(D+C+1)+CP] bytes full, 8[K(D+1)+CP] prefix-only. Equal-byte capacity is floor(512*(D+C+1)/(D+1)). This excludes temporaries, model arrays and shared inputs.

All benchmark processes run the same warmup, which executes all three modes, and use a common Numba disk cache compiled by the preliminary smoke run. Single BLAS thread. Fresh process per width/mode, two timed repetitions per episode, median of the two times. Width128 order full512/prefix512/prefix_equal; width256 reverse. The width256 combined command timed out during full512, so that configuration was rerun completely. Eager audit was split after a timeout, retaining only complete rows and auditing the remaining four episodes. The warmup initially exposed an invalid scalar index on the incumbent cost, corrected before all benchmark measurements. Training, world loading, warmup and exhaustive audit excluded from solve timing. No preregistration.

## Results

All 12 episode-level independent eager MAP audits pass, spanning 491,520 exact candidate pairs. All 36 mode-episode records (72 timed calls) have the same MAP index, incumbent, deduplication count, and candidate-probe work on each episode. Maximum absolute difference in score versus eager 3.58046925441613e-15. Total candidate-probe work per mode: 52,542 at width128 and 209,752 at width256; total 262,294. Every evaluated lower floor checked with EPS=1e-10; float64 arithmetic, not certified outward rounding. True injected fault recovery 4/12; high-noise MAP agreement is not latent recovery.

|width|mode|entries|cache bytes|misses|hits|h2 reconstructions|scalar projections computed|median solve ms|median finish ms|median paired full/mode ratio|wins /6|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|128|full512|512|1064960|4334|29234|4334|554752|87.339|53.074|1|ref|
|128|prefix512|512|540672|4334|29234|33568|52470|102.123|63.812|.863|2|
|128|prefix_equal|1020|1064928|3160|30408|33568|52470|91.985|57.859|.965|3|
|256|full512|512|2125824|29771|101649|29771|7621376|808.923|602.364|1|ref|
|256|prefix512|512|1077248|29771|101649|131420|209680|853.355|646.446|.911|1|
|256|prefix_equal|1022|2125808|10334|121086|131420|209680|813.945|625.605|.914|2|

Projection counts count scalar outputs, including full-vector entries on misses, not FLOPs or time equivalents. Cache hits, misses and dedup exclude the initial incumbent's twelve probes. Batch-local dedup avoids 18,902 and 78,260 requests at widths128 and256 respectively. Equal-byte prefix-only cuts misses by 27.1% and 65.3%; same-capacity prefix-only reduces named payload by 49.2% and 49.3%. Both prefix-only alternatives lose median paired runtime at both widths. The equal-byte alternative computes substantially fewer dense prefixes and scalar projections but reconstructs h2 on every unique batch request; scalar projection and tanh overhead matter.

|width|mode|post-warmup high-water MiB|final process peak RSS MiB|
|---:|---|---:|---:|
|128|full512|181.000|186.598|
|128|prefix512|181.000|186.672|
|128|prefix_equal|180.992|186.590|
|256|full512|180.914|199.504|
|256|prefix512|180.996|199.898|
|256|prefix_equal|180.813|199.875|

RSS is process high-water, measured in isolated mode processes, not an allocation-specific guarantee. No observed peak RSS benefit from the reduced entry size; first-stage screening and other shared allocations may dominate. Keep full512 as runtime reference. Prefix512 remains an optional bounded-array layout; prefix_equal is not promoted. Two independent models per width, exploratory timing only.

## Reproduce

From experiment directory with prior ER modules:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for d in 128 256; do
  python er32.py prepare worlds --width "$d" --roots 76 77
done
python er37.py full512 worlds/128_76_0.npz --reps 1 > /dev/null
for d in 128 256; do
  modes='full512 prefix512 prefix_equal'
  [ "$d" = 256 ] && modes='prefix_equal prefix512 full512'
  for m in $modes; do
    python er37.py "$m" worlds/${d}_*.npz > bench${d}_${m}.jsonl
  done
done
cat bench128_full512.jsonl bench128_prefix512.jsonl bench128_prefix_equal.jsonl bench256_full512.jsonl bench256_prefix512.jsonl bench256_prefix_equal.jsonl > er37_bench.jsonl
python er37.py audit worlds/*.npz --receipt er37_bench.jsonl > er37_audit.jsonl
```
Executed source SHA256 13279aef0d55bce791217b71b745d5e74fc4d8d4cb2e75a55ea1f0eedb7f0a59; remote Git blob cf479169bf06eee4d35828484f4579af9d8d4881 verified. Benchmark SHA256 28a03c61a80bc6cd97efd14bc563901167bab908223335d57f4fa0d2c4dc727d; audit SHA256 216934a12b77061e4e5c80cc130671522d896a27b6e4b8cf281ca90e621aa6aa.

## Next gate

Separate reuse policy from storage representation: keep the full projection entry fixed, compare FIFO versus LRU and per-probe segmented replacement under identical capacity, batch ordering, warmup and exact work. Use held-out models, report misses and timing with a root-level aggregate. Test a cold no-cache fallback and memory guard separately. A hybrid h2/z3 entry could also avoid tanh on hits but no benefit is claimed without testing. Native learned BPD factors, hard OS memory enforcement and formally certified numerical intervals remain outstanding.
