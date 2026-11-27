# ER-35: streamed curvature and replayable prefix layouts

## Method

Fresh ER-25 teacher/student trained four-pass tanh networks with SVD rank-one component faults, roots 72 and 73 at widths 128 and 256. Three distinct 12-probe episodes per model at noise multipliers .1,1,3 times calibration RMS. Four trained instances, 12 episodes. Fixed block size 8, exact batch64, one-probe gap-ranked refinement, dense per-probe floors in all arms.

The dense control uses ER-33 BlockBank setup, with a common compiled ER-35 finish and early release of screening state. In StreamBank the right-side curvature table b4r is built once. For each first-component block, compute h2_i, z3_i, the second-component projection table, first-only response and both weighted left curvature terms, contract immediately against the retained right table and write the floors. Discard left curvature tensors after each block rather than materializing them across all first components. In stream mode retain z3_i and projections for the exact finish; in project mode retain only z3_i and reconstruct the demanded scalar projection using h2_i=tanh(z2-read_i*u_i); in replay mode retain neither and reconstruct the demanded prefix with one additional batched dense multiplication. These are real-arithmetic equivalences, with changed evaluation ordering in floating point. The first screen retains the right table for all modes. Streamed modes recompute a small block for the exact initial incumbent. All modes release screening-only Bank attributes before refinement. The dense comparator has a small common-interface overhead to construct clean z2 and reads, so this is not a direct timing replay of ER-34.

Each root/mode executed in a fresh process; two timed repetitions per episode; modes dense/stream/project/replay for root72 and reverse for root73. One BLAS thread. Warmup uses a small synthetic case; compilation plus warmup excluded from solve timings (1.65–1.92 s across processes). Training, SVD, input loading and independent audit excluded. RSS measured using Linux ru_maxrss. Final process peak per root/mode (all three episodes) summarized over two roots; post-warmup increment subtracts baseline before inputs are loaded. Named ndarray payload sums Bank attribute nbytes after screening, including input references, excluding floor table, external references, object overhead, temporaries and copies. The dense control additionally retains small clean prefix inputs outside Bank, so its named array payload excludes these; this metric is not total process memory.

## Exactness

All 5,898,240 scalar response envelope containment checks passed against exact enumeration. Streamed and dense envelope maximum absolute difference 0.0. All four layouts had identical full floors, sorted candidate order, probe order, exact incumbent and candidate-probe counts on all 12 episodes. All 48 mode-episode results (96 timed repetitions) match independent exact MAP index and energy within EPS=1e-10. Identical work counts: 55,860 at width128 and 65,476 at width256 per arm, total 121,336. Project and replay incur additional prefix replay work for 55,788 and 65,404 candidate-probe terms respectively (excludes initial incumbent; not FLOP-equivalent to final response counts). Four of 12 injected faults recovered; exact MAP is distinct from latent fault recovery. Float64 checks are numerical, not machine-certified intervals.

## Results

|width|mode|median solve ms|median screen ms|median finish ms|median paired dense/mode|wins /6|named array MiB|process peak MiB|post-warmup peak increment MiB|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|128|dense|48.428|20.886|27.630|1|reference|8.106|228.402|9.342|
|128|stream|48.811|20.429|28.204|1.012|4|5.141|225.471|6.406|
|128|project|55.420|19.666|36.446|.874|2|3.641|223.861|4.846|
|128|replay|62.578|18.664|43.913|.774|2|2.141|222.318|3.250|
|256|dense|187.086|117.081|66.984|1|reference|32.211|259.299|40.246|
|256|stream|192.001|119.148|72.853|.967|2|20.281|247.869|28.807|
|256|project|207.564|119.819|89.664|.926|1|14.281|242.078|23.078|
|256|replay|242.036|123.572|121.458|.754|0|8.281|236.125|17.129|

Separate float64 floors remain 1.5 MiB and 6 MiB. At width256 streaming reduces named array payload by 11.93 MiB, peak process RSS by 11.43 MiB, and the post-warmup peak increment by about 28.4%. Median paired runtime ratio is .967 (about 3.4% paired slowdown), so stream is a memory-prioritizing candidate rather than a speed promotion. Project and replay cut peak RSS by 17.22 and 23.17 MiB respectively at larger compute cost. Full replay reduces named array payload by about 74.3%, and post-warmup peak increment by about 57.4%. These are observations on two independent models per width, not hardware-independent guarantees.

## Reproduce

From the experiment directory with previous ER scripts available:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for d in 128 256; do
  python er32.py prepare worlds --width "$d" --roots 72 73
done
for d in 128 256; do
  for root in 72 73; do
    python er35.py audit worlds/${d}_${root}_*.npz > audit_${d}_${root}.jsonl
    modes='dense stream project replay'
    [ "$root" = 73 ] && modes='replay project stream dense'
    for m in $modes; do
      python er35.py bench worlds/${d}_${root}_*.npz --mode "$m" --reps 2 > bench_${d}_${root}_${m}.jsonl
    done
  done
done
cat audit_*.jsonl > er35_audit.jsonl
cat bench_*.jsonl > er35_bench.jsonl
```
The width256 root72 benchmark command timed out after the first three configurations completed; the unfinished replay configuration was rerun in its entirety. All rows complete and separately validated against audit. Source SHA256 da2e2e57911c1ed1f74b0138308b733766c42fddbcb7eb4787fc5101db3c38cb; Git blob 02a384c488238937e413ddf291e3cdab6b1f2bc9. Audit SHA256 0cfdbf07a76ff83ffee25c0626c2a07dc8e1f0ab9d233f1cd671a57ed400f01e; benchmark SHA256 a679eb4bef7ebfd34c057c850a6c47c3a02525339a2188f7e631f5904e2720fd.

## Next gate

Replay currently repeats h2 for each demanded (i,j,p) term. A bounded cache keyed by (i,p) can reuse h2 and z3, or use batch-local deduplication to compute the prefix once for every distinct (i,p) within a refinement stage. Preserve probe and candidate ordering, count cache hits/misses and bytes, and compare fresh workloads under explicit memory budgets. The right curvature table and floor table remain materialized; consider their lifetimes only after measuring prefix-cache tradeoffs. Independent registered model rosters, learned native BPD factors, hard memory limits and formal numerical certificates remain pending.
