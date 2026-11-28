# HAD-04 / HAD-04B: cached coordinated repair and learned tabu guidance

Evidence class: DEVELOPMENT_ONLY. November 26, 2026. The experiment works with odd-length Goethals-Seidel complementary sequences. Every reported construction is at a previously known matrix order; no unresolved order is solved.

## Exact cache

For m odd, r[t] is the sum of the four periodic autocorrelations at t=1,...,(m-1)/2. E=r dot r. Cache d[i,t] = -2 x[i](x[i+t]+x[i-t]), with indices inside each sequence. Single-flip energy is E+2 r dot d[i]+d[i] dot d[i]. For two distinct moves i,j, the joint displacement is d[i]+d[j]+c, where c is zero across sequences; within a sequence c[t]=4x[i]x[j] at the unique inverse-shift class. After flip i, update r+=d[i], negate d[i], update one coordinate of each other displacement in the same sequence by c, and update the sequence sum. Cache maintenance costs O(m) rather than rebuilding O(m^2) entries. Scoring all single moves still costs O(m^2). Row-sum squared total and FFT sum are checked on successful classical certificates; neither is used as a hard search filter.

## Classical search

Engineering root 84001: eight random starts per length. Greedy solves m=17:7,19:3,21:0,23:2,25:0,27:0,29:0. Select 19 and 23 for headroom and 29 as a stress stratum. Fresh seeds 84101..84104, 12 cases per length per root, 16,384 candidate-energy scores maximum per case. All arms share the random initial states. Tabu tenure 7 with best-energy aspiration. Pair scans only on non-improving steps. Shortlist 4; full pair enumeration sampled without replacement when the remaining budget is insufficient. Every distinct scored candidate counts once, including tabu-ineligible candidates; comparisons are candidate-budget limited, not FLOP matched. No repeated-seed hyperparameter selection on the evaluation roots.

| Arm | m=19 /48 | m=23 /48 | m=29 /48 | total /144 | sum search seconds | scored candidates |
|---|---:|---:|---:|---:|---:|---:|
| greedy |17|3|0|20|0.796|2193044|
| tabu |48|22|1|71|0.531|1565860|
| pair |20|3|2|25|0.554|2062596|
| pair_tabu |47|14|1|62|0.459|1734064|
| full_pair |16|3|1|20|0.261|2161888|

The search source performed 16,444 formula/deduplication/certificate checks. Classical search run time 2.845 seconds excluding that audit; Linux process peak RSS 105108 KiB (includes interpreter/NumPy and other overhead). Timings are single-run measurements.

## Learned stalls

Fresh independent model roots 84201..84204; same m=19,23,29 evaluation strata, 12 starts per stratum per root, 16,384 candidate-score ceiling. Train lengths 11,13,15,17. 192 cached teacher states per length, sampled from tabu stalls. Bank payload 1,247,232 bytes per root. Seven candidate features: sign, exact one-flip energy change, squared displacement, energy, sequence sum, tabu remaining duration, admissibility. Exact two-flip teacher minimizes one- or two-flip energy among initially admissible positions; it is a restricted lookahead label, not an exact multi-step tabu policy. Three shared-factor passes with global-mean context, width 24, 1945 parameters. No attention. Common 100-update warmup, 60 paired updates, batch 16, AdamW lr .003. Six groups of four output factors are evaluated on 64 separately generated training-length-17 calibration states. A: task only. B: fixed projector. C: signed-permutation sham. D: live projector. Label-informed negative loss gradient is projected at the first-pass state, physical dose .75, auxiliary gradient normalized to .25 of task norm, global clip 1. All four arms compute all auxiliary gradients. No steering at evaluation; at improving steps all use ordinary tabu; at non-improving steps neural arms rank the admissible candidate moves. This is a direct shared-factor proxy, not native BPDTrainer evidence or semantic transport. Optional --native uses the repository's TiedComponentLinear with open gates and zero delta; that path was inspected from the API but was not executed locally.

| Arm | m=19 /48 | m=23 /48 | m=29 /48 | total /144 | sum search seconds |
|---|---:|---:|---:|---:|---:|
| tabu |48|26|4|78|0.550|
| task |48|28|0|76|3.301|
| fixed |48|21|3|72|3.360|
| sham |47|28|1|76|3.236|
| live |48|25|1|74|3.244|

Fixed minus sham -4/144, fixed minus task -4/144. No learned advantage over tabu; no causal steering benefit established. Four roots are development evidence. Learned evaluation has additional uncharged neural compute, so this candidate-matched comparison favors the learned arms on computational accounting. Full run 21.681 seconds. NumPy 2.3.5, PyTorch 2.10.0+cpu, one Torch thread and OPENBLAS_NUM_THREADS=1.

## Cache microbenchmark

128 identical random flips in cached and rebuilt states; final residual, displacement matrix and sums equal. Table reports maintenance only (not end-to-end search).

| m | persistent bytes | cached seconds | rebuild seconds | speedup |
|---:|---:|---:|---:|---:|
|19|6256|.001363|.004707|3.45|
|29|14176|.001359|.006566|4.83|
|167|450256|.001554|.107324|69.05|
|599|5755216|.002551|1.511870|592.65|

At m=599, 19,132 identical shortlisted pairs under workspace caps 8,16,64 MiB yielded 2.384,3.405,0.780 million scored pairs/sec respectively. Integer sum checksum 13793912976 and minimum 701824 match in all three. The cap controls scoring block width and tracked arrays/estimated scratch; it is not a process-RSS bound. Full pair indexing currently materializes candidate indices before blocking, and the bank cap is checked after construction. Do not run full enumeration at large m without streaming indices and preflight memory accounting. Timing needs repeated randomized-order runs. The current resource module import is Unix-specific.

## Independent verification

audit.py independently reconstructs each saved block row as a bit string, checks all scalar cyclic sums and verifies every pair of rows by integer Hamming distance. 586 serialized successful certificates: engineering 9, engineering_extra 3, results 198, learn_results 376. 1,964,750 integer assertions, all pass. Raw receipts and certificates retained in the local downloadable experiment package; reproduce them with the commands below. Training checkpoint weights were not persisted; native adapter and CI are pending.

## Reproduce from repository root on Linux/WSL

```bash
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.
D=mechinterp/bpd/experiments/hadamard_coordinated_repair
python $D/search.py --lengths 17 23 29 --root 84001 --roots 1 --cases 8 --arms greedy --out engineering.json
python $D/search.py --lengths 19 21 25 27 --root 84001 --roots 1 --cases 8 --arms greedy --out engineering_extra.json
python $D/search.py --lengths 19 23 29 --root 84101 --roots 4 --cases 12 --out results.json
python $D/learn.py --out learn_results.json
python $D/cache_bench.py --out cache_results.json
python $D/audit.py --dir .
# Pending native smoke (separate evidence):
python $D/learn.py --native --roots 1 --cases 2 --warm 10 --paired 4 --bank 32 --lengths 19 --out native_smoke.json
```

## HAD-05 prospective direction

Use a memory-bounded population of independently initialized cached trajectories, tabu as co-primary baseline, variable tenure and restart comparisons, diversity-based archive eviction, LDT row-sum and PSD feasibility for initialization/restricted search. Stream pair indices; charge candidate, CPU time and training/inference costs; enforce actual allocated bank bytes before generation. Use a fixed wall-time budget and repeated trials. A learned intervention must improve solve rate beyond tuned tabu and a matched sham on independent roots. Investigate whether latent state remembers trajectory-level stagnation rather than distilling a short-horizon greedy label. Include feature-only MLP, unrolled model, native BPD discovery and specificity controls. No large-order training or unresolved-order search claim until independently checked exact certificates exist.
