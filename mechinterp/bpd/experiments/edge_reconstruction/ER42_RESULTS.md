# ER42: compact history and partitioned exact top-K admission

Fresh roots 86 and 87, widths 128 and 256, three episodes per root. Frozen ER41 model and identical score/recency/key tie order. Compact int32 history (bounded episode) and a bounded partition selector with stable-sort fallback. Accounting includes persistent cache and history payload, excludes runtime allocator overhead. All 43 completed mode-episode comparisons matched the independent eager oracle at 1e-10; 12 learned/learned_sort pairs had identical misses, bypass and pairprobe counts. Width-256 frequency arm timed out after one episode and is excluded from aggregate comparisons.

| width | arm | entries | bytes | misses (6 episodes) | median ms |
|---:|---|---:|---:|---:|---:|
|128|LRU|148|318968|32265|106.38|
|128|Frequency|143|318584|25832|121.23|
|128|Learned sort|128|319816|26417|129.55|
|128|Learned partition|128|319816|26417|96.72|
|256|LRU|148|634360|74407|746.81|
|256|Learned sort|128|638280|63433|729.13|
|256|Learned partition|128|638280|63433|707.91|

Median paired sort/partition ratios 1.3381 and 1.0293; LRU/partition 1.0957 and 1.1743. Miss reductions versus equal-payload LRU 18.12% and 14.75%. Timing is exploratory; four independent roots, no cross-hardware validation. The width-128 frequency arm wins on misses. The selector audit passed 3592 cases; 2048-case warmed microbenchmark sort 67.80/73.66 ms, partition 10.73/13.50 ms. This microbenchmark is not end-to-end throughput. CPU only.

Reproduce: prepare fresh worlds via er32.py --width 128/256 --roots 86 87, use er42.py --model er41_frozen_model.json --mode learned / learned_sort / lru_budget / freq_budget / audit. Run er42_select.py for selection audit. Warm each Numba path, use one BLAS thread, separate process for each arm. The width-256 frequency timeout is a blocker for its paired aggregate. No settlement or financial conclusions follow from these synthetic worlds.
