# ER-38: global and probe-segmented cache replacement

## Method

Four fresh ER-25 trained synthetic recurrent models, roots 78 and 79 at each of widths 128 and 256, three 12-probe episodes per model at noise multipliers .1, 1 and 3. All policies use full z3 and second-component projection vectors for the (first component, probe) cache key, the same 512-entry capacity, linked-list metadata arrays, ER-35 streamed screen, exact incumbent, stable candidate order, probe gap order, and shared compiled single-probe finish. Compare global FIFO, global LRU, per-probe FIFO, and per-probe LRU. Segments divide capacity equally (42 or 43 slots per probe at P=12); global policies share all slots. Hits are snapshotted for the whole batch before miss insertion, and LRU updates on hits in first-use order. Batch-local deduplication is identical in all modes. Exit assertions check list links, occupied counts, reverse mappings and segment ownership. The same accounting asserts hits+misses+dedup=exact candidate-probe count minus the initial incumbent probes.

Named cache payload includes float64 entries, int64 slot map, keys, list links, heads, tails and used counts: 8[K(D+C+3)+CP+3P] bytes. This excludes temporary batch buffers and diagnostic arrays. All policies have the same payload, 1,073,440 bytes at 128 and 2,134,304 at 256. Warmup and JIT cache shared; one fresh process per width/mode; one BLAS thread; two timed repetitions per episode. Timings exclude training, loading, warmup and exhaustive audit. Mode order FIFO/LRU/segFIFO/segLRU at width128, reversed at width256. Timings are sequential across modes, not randomized within episode; timing comparisons remain exploratory. The width256 LRU configuration timed out and was rerun completely; eager audit split after a timeout, retaining completed distinct rows. No preregistration.

## Results

12 independent episode-level exhaustive MAP audits, 491,520 exact pairs. All 48 mode-episode records (96 timed calls) match reference MAP and energy to EPS=1e-10; maximum observed energy discrepancy 3.191891195797325e-15. Every mode performs 42,750 exact candidate-probe evaluations at width128 and 227,219 at width256, including the initial incumbent. Five of twelve injected faults recovered. Agreement is with the optimum under observed noise, not guaranteed true-fault recovery. These are float64 audits, not certified interval arithmetic.

|width|mode|misses|hits|median solve ms|median finish ms|median paired FIFO/mode ratio|wins /6|
|---:|---|---:|---:|---:|---:|---:|---:|
|128|fifo|2887|23202|63.838|34.036|1|ref|
|128|lru|2717|23372|58.737|33.208|1.060|6|
|128|segfifo|13183|12906|83.144|51.109|.757|0|
|128|seglru|12199|13890|65.644|38.084|.973|2|
|256|fifo|43517|111986|510.711|353.335|1|ref|
|256|lru|32553|122950|595.860|402.444|.824|0|
|256|segfifo|106658|48845|606.192|448.344|.893|1|
|256|seglru|103136|52367|831.903|633.720|.634|0|

Batch dedup avoids 16,589 and 71,644 additional requests at widths128 and256 respectively. Global LRU reduces misses 5.9% and 25.2%; width128 shows a modest timing win while width256 shows a loss. Final process peak RSS MiB in FIFO/LRU/segFIFO/segLRU order: width128 186.977/186.980/186.676/186.844, width256 200.242/200.039/200.164/199.988. No meaningful RSS difference at matched payload.

Per-episode median share of unique batch cache requests attributable to the busiest probe is .534 at width128 and .603 at width256; busiest three probes account for .882 and .894. Static equal segmentation substantially increases misses. Observed runtime does not establish a causal relationship between replacement bookkeeping and the measured slowdown; BLAS batch shapes and process timing variation also matter. Retain global FIFO as incumbent and treat LRU as an optional exploratory arm. Two independent models per width; do not generalize the width-dependent timing pattern to deployment.

## Reproduce

From experiment directory with previous ER modules:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for d in 128 256; do
  python er32.py prepare worlds --width "$d" --roots 78 79
done
python er38.py fifo worlds/128_78_0.npz --reps 1 > /dev/null
for d in 128 256; do
  modes='fifo lru segfifo seglru'
  [ "$d" = 256 ] && modes='seglru segfifo lru fifo'
  for m in $modes; do
    python er38.py "$m" worlds/${d}_*.npz > bench${d}_${m}.jsonl
  done
done
cat bench128_fifo.jsonl bench128_lru.jsonl bench128_segfifo.jsonl bench128_seglru.jsonl bench256_fifo.jsonl bench256_lru.jsonl bench256_segfifo.jsonl bench256_seglru.jsonl > er38_bench.jsonl
python er38.py audit worlds/128_*.npz --receipt er38_bench.jsonl > audit128.jsonl
python er38.py audit worlds/256_78_*.npz --receipt er38_bench.jsonl > audit256_78.jsonl
python er38.py audit worlds/256_79_*.npz --receipt er38_bench.jsonl > audit256_79.jsonl
cat audit128.jsonl audit256_78.jsonl audit256_79.jsonl > er38_audit.jsonl
```

Source SHA256 34f6c5f5f10693cc8da30d0e852d77fc17e8f7c6ca90811f5b90d8cefcc4de26; verified Git blob b4cee376e901ef9034ff8835851267dee2031c8e. Bench SHA256 ebdcb6f507763e2598fe8cb64ca44047081710770ec45776cd00597ff92067cd; audit SHA256 8701b05c3a072a64fdaa69f3b06ffeb1d8a8a9e0c5e40e257fbc8a9d840a7c6d.

## Next gate

Capture the actual deduplicated batch access trace, compare fixed-capacity admission/replacement policies with an offline future-aware reference, and evaluate a demand-weighted or shared-reserve segment allocation using only information available at runtime. Charge trace construction and policy overhead in subsequent online timing experiments; separate miss batch-size distribution from total misses. Freeze calibration and fresh evaluation roots. Native learned BPD factors, operating-system memory enforcement and formal numerical certificates remain pending.
