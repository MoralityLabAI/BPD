# ER-36: batch-local prefix deduplication and bounded FIFO cache

## Method

Fresh ER-25 synthetic trained four-pass recurrent teacher/student worlds at widths 128 and 256, model roots 74 and 75 at each width, with three 12-probe episodes per root at noise .1, 1 and 3 times calibration RMS. All five modes use the ER-35 streamed screen, identical analytical floor computation, incumbent selection, probe gap ranking, stable lower-bound ordering, and exact candidate batch 64 with single-probe pruning. Stream retains the full prefix and projection banks; replay uses the ER-35 uncached exact finish. The new cached finish starts with the replay screen (no retained prefix bank). For each demanded probe within a batch, group by first-component intervention i; compute one prefix per missing (i,p) key, retaining z3 and the vector of all second-component projections. Cache hits are snapshotted before writes can evict them. A bounded FIFO slot array with key-to-slot map supports 0,128,512 entries. All cached modes use this batch-local deduplication. Capacity 0 is the batch-only control. Cache lifetime is one episode, with empty initialization. Cache bytes = 8*[capacity*(width+component_count+1)+component_count*probe_count]; excludes transient batch buffers, input arrays and a contiguous transpose of the read vectors. The full two-prefix-bank payload at widths128 and256 is 3 and12 MiB respectively. Cache miss computes a dense vector of all projections rather than just one scalar projection as in uncached replay, so the miss counts are not FLOP-equivalent measures.

Single BLAS thread. One process per width/mode, containing all six episodes, two timed solve repetitions per episode, median within episode. Fixed mode order stream/replay/cache0/cache128/cache512 at width128, reverse at width256; the final two controls at width256 were rerun separately after the combined command timeout. No partial rows included. Warmup excluded from solve timings, but the new Numba kernel uses a disk cache (populated before benchmark); old control uses a different compilation path. Compilation/loading costs and baseline process high-water memory differ. Avoid interpreting absolute cross-kernel RSS or baseline-subtracted high-water differences as an isolated measurement of prefix-bank savings. Named cache payload is the reliable capacity measure. Training, SVD, input loading, and independent eager enumeration excluded from solve timing. No preregistration.

## Results

All 12 episodes and all 60 mode-episode results agree with full eager ER-23 enumeration (absolute score difference <1e-10). The eager audit enumerates 491,520 candidate pairs total. Every mode uses the same initial incumbent and exact candidate-probe count on every episode; 33,810 at width128 and195,018 at width256, including the initial 12-probe incumbent evaluations. All online floor replacement assertions and cache accounting assertions passed. The audit verifies the reference MAP and score, not a formal floating-point certificate or universal admissibility result. Four of twelve injected hidden pairs recovered, all at .1 noise; exact MAP is different from recovering the actual latent pair at high noise.

|width|mode|median total ms|median finish ms|median paired replay/mode speedup|prefix misses|cache hits|batch dedup|cache bytes|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|128|stream|49.951|20.510|1.376|0|0|0|0|
|128|replay|77.003|41.521|1|33738|0|0|0|
|128|cache0|59.930|32.603|1.172|20492|0|13246|12288|
|128|cache128|62.058|31.416|1.263|11335|9157|13246|275456|
|128|cache512|54.621|26.491|1.345|3670|16822|13246|1064960|
|256|stream|645.986|457.837|1.412|0|0|0|0|
|256|replay|963.686|771.057|1|194946|0|0|0|
|256|cache0|958.231|778.095|1.074|124645|0|70301|24576|
|256|cache128|905.528|721.489|1.071|91826|32819|70301|549888|
|256|cache512|789.642|601.691|1.138|27545|97100|70301|2125824|

The 512-entry cache eliminates 89.1% of width128 prefix recomputations and85.9% at width256 relative to uncached replay (excluding initial incumbent). At width256 the exact prefix cache holds approximately 2.027 MiB rather than a 12 MiB full two-bank array payload. Cache512 is a faster replay alternative on this roster; full retention is still faster at both widths. No inference about optimal replacement policy or across-machine performance. Only two independent model instances at each width.

Final process peak RSS / baseline MiB: width128 stream 224.711/217.992, replay221.180/217.805, cache0 186.086/180.488, cache128 186.078/180.367, cache512 186.086/180.363; width256 stream247.078/217.988, replay234.914/217.750, cache0 199.465/180.367, cache128 199.402/180.488, cache512 199.340/180.488. The lower absolute peaks for cached modes are confounded by disk-cached versus fresh compiler initialization, and peak increments include allocator behavior. Do not claim an isolated 48 MiB RSS reduction from the cache. This experiment demonstrates bounded array storage and a recomputation/runtime tradeoff, not a hard process memory guarantee.

## Reproduce

From experiment directory with earlier ER scripts:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for d in 128 256; do
  python er32.py prepare worlds --width "$d" --roots 74 75
done
python er36.py cache128 worlds/128_74_0.npz --reps 1 > /dev/null
for d in 128 256; do
  modes='stream replay cache0 cache128 cache512'
  [ "$d" = 256 ] && modes='cache512 cache128 cache0 replay stream'
  for m in $modes; do
    python er36.py "$m" worlds/${d}_*.npz > bench${d}_${m}.jsonl
  done
done
cat bench*.jsonl > er36_bench.jsonl
python - <<'PY'
import json,glob
from er23 import Bank,decision,EPS
from er32 import load
rows=[json.loads(s) for s in open('er36_bench.jsonl')]
assert len(rows)==60
for path in sorted(glob.glob('worlds/*.npz')):
    args,y,hidden,noise=load(path)
    k,s,n,_=decision(Bank(*args),y,'eager',batch=64)
    rr=[r for r in rows if r['path']==path.split('/')[-1]]
    assert len(rr)==5 and all(r['best']==k and abs(r['score']-s)<EPS for r in rr)
    assert len({r['pairprobes'] for r in rr})==len({r['incumbent'] for r in rr})==1
    print(json.dumps(dict(path=path.split('/')[-1],best=k,score=s,hidden=hidden,exact_pairs=n,comparisons=len(rr))))
PY
```

Executed source SHA256 897740bddc3068a29ea2fb67b39f6af82bb660263ef9c323bf8772212c4bcc1e; Git blob 89a12e24f73d48b7e3c024a10184596ed1f680ae. The roster and cache sizes were selected for this exploratory pilot. Next: separate cache hit locality by probe, compare per-probe segmented FIFO or frequency-aware admission with equal payload and identical cache warmup, and test a prefix-only cache that computes one demanded projection instead of every projection. Hard OS memory containment and learned native BPD factors remain pending.
