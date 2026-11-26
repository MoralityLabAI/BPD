# HAD-05: population repair, LDT constraint joins and recurrent compute allocation

The [2026-11-26 follow-up](REVISION_20261126.md) adds native BPD component
execution, a completed-start budget fix, 77 focused tests and source-locked
development replay receipts. The next study is specified in
[FABLE_HAD06.txt](FABLE_HAD06.txt). The historical report below retains the
original source versions and experimental scope.

**DEVELOPMENT_ONLY**, 2026-11-26. Standalone NumPy/PyTorch CPU gym for odd-length Goethals–Seidel complementary systems. All completed orders are previously known. The recurrent model is a directly trained tied-factor proxy, without native TRM attention, native LDT runtime or native BPDTrainer. Four history observations are replayed rather than retaining latent state across all macro decisions.

## Exact LDT state and memory

For row sums u enumerate positive odd magnitude assignments a satisfying sum(a_j²)=4m and a_j<=m. Distance to signed sector a is sum |a_j-|u_j||/2. For 3|m, the exact cube-root power of a row with residue-class sums A,B,C is Q=A²+B²+C²-AB-BC-CA. A valid row has Q<=4m. Since a bit flip changes Fourier amplitude by at most two, the necessary condition after k flips is Q<=(2sqrt(m)+2k)². Reject k exactly if D=Q-4m-4k²>0 and D²>64k²m. Combine per-row lower bound b_j with row-sum distance via min_a sum_j max(b_j, |a_j-|u_j||/2), then take max with ceil(max_t|r_t|/4). Count magnitude sectors surviving a finite horizon; intermediate infeasibility remains allowed. Other-frequency floating FFT excess is only a heuristic.

Cache maintenance O(m), all single scores O(m²). Preflight tracked persistent payload, teacher arrays per bank and encoded archive payload; caps exclude Python objects, temporary workspace, coexisting banks and RSS. At m=599 a trajectory tracks 5,774,624 bytes. Worst-case exact archive payload at 128/512/1024/2048 rows is 5.0/77.5/308.5/1230.9 MiB; near archive 5.7/89.5/356.5/1423.0 MiB. These are modeled capacities, not large-order experiments.

## A: recurrent allocation

Development roots 85101–85104 after exposed smoke 85001. Eight trajectories, 24 rounds, 16 tabu flips per allocation, total candidate-score ceiling 384*4m; tenure seven, best-energy aspiration. Train lengths 19,21,25, unseen evaluation lengths 23,27,29. Six cases/root/order, 72 paired populations per arm. Teacher labels exact next-chunk solve bonus (4), global incumbent improvement and 0.15 relative own-best improvement; only past/current features enter model. 699 informative snapshots, 2,242,368 teacher scores excluding calibration/diagnostic banks. Four retained observations, 15 features, width 24, 2,137 parameters; MLP 2,337. Common initialization/batches for LDT feature ablation. 100 warmup, 60 paired AdamW updates at lr .003, batch 16. Training-only calibration selects one of six groups of four output factors. Fixed/live projector edits use detached negative task gradient at penultimate history boundary, dose .75; signed-permutation sham, auxiliary gradient norm .25 of task norm. All paired arms compute four parameter gradients, clip at 1. No steering at scored inference.

| arm | m23 | m27 | m29 | total/72 |
|---|---:|---:|---:|---:|
| serial |16|3|3|22|
| round_robin |14|4|1|19|
| best |19|13|4|36|
| ldt_best |19|13|4|36|
| ucb |19|13|4|36|
| ucb_transfer |18|12|3|33|
| mlp |18|4|2|24|
| no_ldt |15|4|0|19|
| trm |15|3|1|19|
| fixed |14|4|0|18|
| sham |15|6|1|22|
| live |14|5|0|19|
| trm_transfer |14|3|1|18|

936 arm-episodes, run 25.75 seconds. Best and UCB identical decisions; LDT 16-flip horizon rules out zero sectors. Best concentrates 17.99 blocks in its most-allocated trajectory on average vs TRM 5.83. Teacher top-set hit on separate unseen-order bank 10.39% TRM, 8.75% fixed, 10.54% sham, 9.12% live (uniform reference depends on ties). Fixed-arm selected-subspace removal changes 54.13% of rankings vs rotated 21.08%, but teacher loss decreases from 2.1534 to 2.1253 (sham 2.1548); directional sensitivity, not useful selectivity. No steering benefit; no semantic transport or native decomposition claim. No improving guarded row swap among 3148 UCB/1277 TRM offers, but constructed same-sum same-power positive control succeeds. Model checkpoints not saved.

## B: exact archive join

Store unique individual autocorrelation signatures c(x); squared DC and Fourier power are determined by c and m. Admit row and pair only if row sum sector and order-3 power necessary conditions hold. Hash pair residual c(x)+c(y) and seek exact negative. Repeated rows allowed; keep one representative per pair signature. Complete within fully indexed admitted rows; no guarantee for omitted rows. O(K²m) worst-case index work and storage, avoids enumerating all K^4 quadruples.

Roots 85201–85204, 96 paired populations/arm. Charge m candidate units per offered row and one per attempted pair, a scaling proxy rather than equal FLOPs/wall time. Randomized control permutes observed signs within each row, preserving signed sum. 11.98 seconds total.

| arm | solved/96 | archive solutions | mean encoded bytes |
|---|---:|---:|---:|
| best |39|0|0|
| archive16 |39|0|5042|
| archive64 |36|0|63095|
| archive128 |34|1|138108|
| random64 |36|0|62809|

One exact archive solution from different points in one slot. No overall advantage. Disjoint-source order-12 positive fixture passes.

## C: partial joins and injection

Complement keys enforce exact DC, order-3 power if defined, and residual shifts 1 and floor(m/2). Score full residual after lookup of up to four recent pair references per bucket. Admit child below worst incumbent and within best+32, avoid current row-sign/order duplicates. Protect best; replace worst. Save child and source slot/block/row/generation; cross-slot does not imply independent ancestry. Charge 4m candidate units for rebuild. Commitments hold at injection; ordinary tabu may leave them. Engineering roots 85301–85304 (uncharged rebuild, no serialized children) solved best 31, exact 29, noinject 27, near 28, random 24. Fresh corrected development roots 85401–85404, 72 paired populations per arm, 10.69 seconds total:

| arm | m23 | m27 | m29 | total/72 | injections | solves from injected | search seconds summed |
|---|---:|---:|---:|---:|---:|---:|---:|
| best |19|4|1|24|0|0|0.880|
| exact64 |18|4|1|23|0|0|1.874|
| near64_noinject |17|4|1|22|0|0|2.361|
| near64 |22|5|1|28|133|25|2.478|
| random_near64 |17|6|5|28|145|25|2.430|

124/133 real and 144/145 randomized injections use distinct population slots; each arm's 25 injected-descendant solves trace to a cross-slot child. Randomized control ties real archive; earlier run did not beat best. No replicated advantage or learned archive policy claim. Archive policies use classical allocation.

## Verification

NumPy 2.3.5, PyTorch 2.10.0+cpu, one Torch thread, OPENBLAS_NUM_THREADS=1. Exhaustive 4096 m=3 quadruples vs exact Hamming distances to 1728 solutions; 512 m=9 rows vs actual order-3 feasibility distance; cache and transfer checks. Independent scalar autocorrelation and bit-packed Gram audit of 630 full certificates (321+184+125) and 278 corrected-run partial injected candidates. 3,026,330 counted assertions including receipts and positive controls, not independent samples. Earlier engineering C excluded from counted audit. Raw receipts, engineering source, SHA256 manifest and extended protocol are in local experiment package.

Five executed/audited source blobs:
- had05.py 3a06be89b3a502b1de02a4a34d8182c97377be6f
- archive.py 563cdb30ffcc4f0facd341c72bc413d36baf8632
- near.py b74568f2a18f4cdcc13c3e3bb167d46842156d27
- audit.py e726e437cf6dc917878c54761abbd504dd235564
- archive_audit.py 5aa264f8b61350668582562204d7af867681f2e5

## Reproduce

From this experiment directory, with NumPy and Torch:

```bash
export OPENBLAS_NUM_THREADS=1
python had05.py --out results.json
python archive.py --out archive_results.json
python near.py --out near_results.json
python audit.py --results results.json --out audit.json
python archive_audit.py --dir .
```

PowerShell: set `$env:OPENBLAS_NUM_THREADS="1"`, then run Python commands. Configurable population/chunk/rounds, cache-mib, bank-mib, archive-mib, near row-cap. Candidate ceilings do not equal wall time; actual CPU times retained. Remote CI pending.

## HAD-06 prospective

Use classical best as default. Learn residual decision over continue, allocate another basin, admit archive child, evict/restart. Include abstention; train multi-chunk solve hazard and marginal budget regret against classical best, not next-block gain. Record all proposals and future outcomes, lineage and costs. Compare same-energy same-constraint randomized children, tuned classical portfolio, current-feature MLP, history shuffle, tied recurrence and native TRM/LDT. Native BPD discovery on matched counterfactual histories with same current energy/constraint sector but different future utility, targeted vs unaffected action, dose-matched sham, fixed/live projectors, steering-disabled ordinary inference. Freeze independent evaluation seeds/order strata after engineering; use root-level paired gains over best and randomized transfer and intervention specificity. Measure repeated wall-time matched search, training cost, actual memory/RSS and parallelism. No large-order or conjecture claim.
