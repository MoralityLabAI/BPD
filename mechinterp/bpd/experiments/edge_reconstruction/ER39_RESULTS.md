# ER-39: optimal batched cache reuse and demand-ranked allocation

## Model and theorem

Cache keys are (first component i, probe p); entries contain the full prefix and projection vector. A request batch consists of distinct keys in first-use order after the ER-38 stage's batch-local deduplication. Hits are measured against the cache at batch start. Newly computed values can be used from transient batch buffers, then admitted or bypassed. Persistent capacity K limits the set of retained entries between batches. The initial cache is empty; the initial exact incumbent is outside the cache trace. All misses have unit objective cost; this is not a latency model.

If a key occurs at successive batch indices t and u, a hit at u is possible exactly by keeping that key throughout [t,u) (no prefetch/recomputation between requests). Each such consecutive-request interval yields one saved miss when selected. Feasible selected intervals have overlap at most K at every inter-batch boundary. Thus minimum misses = number of requests minus the maximum number of intervals retained.

Process interval starts in chronological order, removing intervals that end at the current boundary, and tentatively retain every newly starting interval. Whenever more than K intervals overlap, discard the interval with latest ending time. This produces an optimal maximum-cardinality interval subset: at the first overflow, any optimal subset avoiding prior deletions must omit some active interval I. If it contains the latest-ending interval J, exchange J for I. Before the overflow boundary, all not-yet-deleted intervals were feasible, so the exchange cannot overload that earlier region. At and after the boundary, I ends no later than J and occupies only a subset of J's remaining span. The exchange preserves cardinality and feasibility. Induct through deletions; the final retained intervals admit all hits represented by that subset. Ties in ending times can be broken arbitrarily. The implementation tracks current retained keys and next-use indices using a heap. This is a batch-aware, optional-admission (bypass allowed) future-aware oracle, not ordinary mandatory-admission sequential Belady. It is a lower bound on misses for the ER-38 online policies; the oracle uses future knowledge and is not a deployable policy. Transient buffers, varying miss batch sizes, cache metadata, and actual compute costs are not optimized.

Independent dynamic program enumerates all reachable post-batch cache subsets, charging each request absent at batch start and permitting any subset of resident plus just-requested keys of size <=K. Exhaustively tested all 4-batch traces over the eight subsets of three keys, capacities 0..3: 16,384/16,384 agreement with the greedy oracle.

## Trace and calibration design

Trace generation replays ER-35/38 streamed envelopes, the same minimum-upper-cost incumbent, stable lower-bound candidate sort, gap-ranked probe order, exact batch64, single-probe pruning, and first-use deduplication. A NumPy retained-prefix implementation records each batch of distinct keys, checks every evaluated floor, and independently compares the final MAP and energy with eager enumeration (EPS=1e-10). It is a trace reconstruction, not instrumentation of the existing compiled cache kernel. Replay exactly matches ER-38 work, deduplication, hits and misses for all four policies on every one of its 12 spent episodes (48 counter comparisons). This validates the simulator on those traces but is not a bitwise guarantee across all floating-point workloads.

Use ER-38 model roots 78,79 per width solely to estimate the mean normalized request share per gap-ranked probe position (separate table per width). Weighted FIFO/LRU allocate K slots by largest-remainder integer apportionment of these weights, retaining one slot per probe when possible. On each fresh episode the already available gap-ranked probe order maps rank quotas to actual probe identities, without using the evaluation access trace to allocate capacity. Equal segmented policies use uniform weights. Global FIFO/LRU use the same batch snapshot, first-use insertion and hit-order semantics as ER-38. Test models roots 80,81 per width (four new independently trained models, twelve episodes with .1,1,3 RMS noise). Capacities 128/512/1024. Every policy at a given capacity uses the same trace and complete projection entry representation. No online timing, RSS, cache-policy implementation speed or hard memory guarantee measured. This is an exploratory held-out trace replay; roots, widths, capacity and allocation rule selected for this pilot.

## Fresh-roster results

All 12 fresh traces reproduce eager MAP and score within tolerance across 491,520 exact candidate pairs. Four of twelve latent faults recovered, so MAP agreement is distinct from recovering the injected fault. Per width 128/256: 56,604/97,677 deduplicated requests, 3,782/7,006 compulsory misses (distinct keys summed per episode), 3,261/7,410 batches; exact candidate-probe evaluations 84,750/163,747 and within-batch dedup 28,074/65,998 (72 initial incumbent probes per width excluded from the traces).

|width|K|global FIFO misses|global LRU|equal seg FIFO|equal seg LRU|weighted FIFO|weighted LRU|offline oracle|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|128|128|44472|43364|48818|48705|38169|37285|23481|
|128|512|14997|12873|30380|28785|19116|18543|6651|
|128|1024|4817|4005|12965|10985|14028|13482|3782|
|256|128|64516|60158|87237|86911|64962|62616|33146|
|256|512|22536|17048|66424|64128|21784|20050|9722|
|256|1024|9795|7966|46187|42041|14066|12824|7006|

At 512 entries: total FIFO misses 37,533, LRU 29,921, oracle 16,373. Oracle avoids 56.4% of FIFO misses, or 79.1% of FIFO misses beyond the 10,788 compulsory distinct-key misses. LRU closes 7,612 of the 21,160 miss gap from FIFO to oracle. The oracle at 1024 achieves compulsory-only misses on all twelve episodes. This does not imply 1024 entries will suffice on other traces. Weighted allocation helps at tight 128-entry capacity at width128 (weighted LRU beats FIFO 6/6 episodes); at width256/128 global LRU is better in aggregate. At 512 entries weighted LRU loses to global LRU at both widths. At 1024 weighted segmentation strands capacity and performs poorly. Do not promote static weighted segmentation.

Miss batch shape matters: at width128/512 FIFO, LRU, oracle have 2618,2435,2245 nonempty miss batches; at width256/512 6036,5711,4784. The corresponding total miss counts differ more sharply. Each replay receipt retains a miss-batch-size histogram per policy, suitable for a calibrated batched-kernel cost model. No speedup claim follows from miss counts alone.

## Reproduce

From the ER experiment directory with previous modules, using separate clean output directories for calibration/test worlds:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for d in 128 256; do
  python er32.py prepare cal_worlds --width "$d" --roots 78 79
  python er32.py prepare test_worlds --width "$d" --roots 80 81
done
python er39.py audit > audit.json
python er39.py trace cal_worlds/*.npz > er39_cal_traces.jsonl
python er39.py replay er39_cal_traces.jsonl --cal er39_cal_traces.jsonl --caps 512 > er39_cal_replay.jsonl
python er39.py trace test_worlds/128_*.npz > test128.jsonl
python er39.py trace test_worlds/256_80_*.npz > test256_80.jsonl
python er39.py trace test_worlds/256_81_*.npz > test256_81.jsonl
cat test128.jsonl test256_80.jsonl test256_81.jsonl > er39_test_traces.jsonl
python er39.py replay er39_test_traces.jsonl --cal er39_cal_traces.jsonl > er39_test_replay.jsonl
```

Original calibration reused ER-38 saved world arrays. The initial calibration trace command timed out after ten complete records; remaining two were run separately and appended, with all 12 distinct records verified. Fresh trace commands completed. Source SHA256 74fa35547d1671887a660c0b68ae1edef7d97381a388e82ce039455808ab158d; Git blob d8fbeb58622956eee177c9957fb334849ea1c7f9 verified remotely. Receipt SHA256 cal traces 4d481e55c8bd1879fdb07334005c4cf129b9f8a26b60089d5b16995edec74746; cal replay 888336b153128ad8882e6c35e50dcfa46a0da01ac28ef71ee7c533c2b1ebe0e8; fresh traces 5a1f37917613ce8b33629983a6a2d647fe23957e7f28b1ca86ff23240d15561e; fresh replay 8ad0f8bcebc9f6a05c62a8b5665f4dbb73f78b2cc7a62795617a9cc3227b3b6b; DP audit 64d65e26a52b7c63e102d7833c693f8926579ef34d8206ebba27662a52bbb1ce.

## Next gate

Use the oracle's retained-interval decisions as offline supervisory labels on development traces only. Predict next-use distance and admission benefit from causal features of the current search (gap rank, candidate lower-bound order, current incumbent and previous cache access), test a shared-capacity admission policy against FIFO/LRU with optional bypass controls. Compare calibrated miss-batch cost and actual compiled runtime; freeze independent model roots before evaluation. Verify the oracle interval proof formally if useful; current proof and finite DP audit are not Lean formalization. Learned native BPD factors, formally certified numerical bounds and OS memory enforcement remain pending.
