# ER-34: exact compact per-probe floors

## Design and mathematical justification

Fresh ER-25 trained tanh recurrence instances at widths 128 and 256, roots 70 and 71 at each width, three independent 12-probe episodes per model at noise multipliers .1,1,3 times calibration RMS. Common fixed SVD factors and observations. Compare dense floors on the ER-33 blocked bank, a scalar-only first pass followed by compact floor recomputation, and provisional one-pass retention. All methods use the same modified compiled single-probe finish with a candidate-to-floor-row index, the same block size 8, exact batch64, initial incumbent, sorted lower costs, and gap-ranked probes. Dense is a matched-layout control, not an unmodified ER-33 runtime replay.

Let L_g be a sum of admissible per-probe lower costs, U_t the running minimum candidate upper cost over completed blocks, and E0 the exact cost of the candidate minimizing the final upper cost. E0 <= min_t U_t in real arithmetic. A candidate with L_g > U_t cannot subsequently be eligible for exact refinement. With tolerance delta, retain floors if L_g <= U_t + delta; the implementation uses 2*EPS to cover the checked incumbent slack. Provisional storage keeps each qualifying row and its candidate ID when the block is visited; after evaluating the selected incumbent, it copies only initially eligible non-incumbent rows to the final compact floor table. It does not retroactively prune earlier provisional chunks during the first pass. Recomputation first stores scalar lower costs, aggregates the identical probe gaps, and selects the same incumbent. It re-evaluates each block with at least one eligible candidate and keeps only eligible rows. Both compact modes preserve the original floors exactly on this roster. All modes retain scalar costs and a dense int64 row-index map. Float64 EPS=1e-10 and numerical checks are not directed rounding or machine-certified interval arithmetic.

## Audit

12 distinct episodes, 5,898,240 scalar response-envelope containment checks against exact enumeration. SHA256 fingerprints of scalar lower costs, candidate order, probe order, initially eligible IDs, and each eligible floor table agree across all three modes in every audited episode. All 36 timed mode-episode results, each with two repetitions, match the independent enumerator MAP index and cost within 1e-10 and identical exact candidate-probe work. Total exact candidate-probe evaluations per mode: 54,738 at width128 and 183,147 at width256, including initial incumbent. Four of 12 injected faults recovered. Exact MAP agreement is distinct from true latent fault recovery.

## Measurements

Isolated fresh process per model-root/mode, two warm timed repetitions per episode. Root70 order dense/recompute/provisional, root71 reversed. Single BLAS thread. Timings include prefix setup, screening, any floor recomputation/compaction, sorting, index construction and compiled finish. Training, SVD, loading, Numba warmup and independent exhaustive audits excluded. RSS is Linux process high-water; final per-root maximum across its three episodes and two repetitions is used, then median over the two roots. Post-warmup high-water increment subtracts a baseline measured before loading worlds. No hard memory cap or preregistration.

| width | mode | median ms | median paired dense/mode | wins /6 | median screen ms | median floor MiB | peak RSS MiB | peak increment MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|128|dense|47.139|1|reference|20.889|1.5000|226.633|9.342|
|128|recompute|53.446|.882|1|26.574|.6410|225.748|8.389|
|128|provisional|49.916|.955|0|26.257|.6410|225.840|8.510|
|256|dense|337.197|1|reference|121.477|6.0000|258.008|40.680|
|256|recompute|348.579|.967|1|145.096|2.1731|255.230|37.867|
|256|provisional|330.110|1.027|4|118.994|2.1731|256.396|39.121|

Median initially eligible candidate counts 7001.5 / 16384 and 23736.5 / 65536 at widths128/256. Dense int64 row index adds 0.125/0.5 MiB to every mode (not included in floor column). Provisional retains 54,049 and 235,359 temporary rows summed over six episodes at respective widths, plus IDs, before compaction; transient storage is not represented by final compact payload. Recomputation revisits all 98,304 and 393,216 candidate rows at respective widths over six episodes (all blocks occupied), not just eligible rows. Thus selective recomputation does not reduce analytical contraction work here. Compact floor payload reductions of 57.3%/63.8% translate to modest process peak savings. At width256 recompute saves 2.778 MiB and provisional saves 1.612 MiB of measured total peak relative to matched dense. Provisional wins four of six timings at 256 but none at 128. No robust incremental runtime claim, two independent roots per width. Retain dense default and provisional optional memory layout pending confirmation. Recomputation is a memory-prioritizing fallback only when appropriate.

## Reproduce

From experiment directory with er23, er25, er29, er30, er31, er32, er33:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for d in 128 256; do
  python er32.py prepare worlds --width "$d" --roots 70 71
  for root in 70 71; do
    python er34.py audit worlds/${d}_${root}_*.npz > audit_${d}_${root}.jsonl
    modes='dense recompute provisional'
    [ "$root" = 71 ] && modes='provisional recompute dense'
    for m in $modes; do
      python er34.py bench worlds/${d}_${root}_*.npz --mode "$m" --reps 2 > bench_${d}_${root}_${m}.jsonl
    done
  done
done
cat audit_*.jsonl > er34_audit.jsonl
cat bench_*.jsonl > er34_bench.jsonl
```
Source SHA256 8944cdd3553a9699a8ae6158ad5c5496d5a788a2a885c77d9b0795d5c7dbea55; remote Git blob 7b0d453ebd6cebde28720d3d25116576f4d91ac3 verified against local source. Raw audit SHA256 f22b7b29419094b0b8a7ffc2904b8862cbc8fb5db345deb5ac89fa78693b17f3; benchmark SHA256 f0fffa9dae59659c53f173ef0a68a8191f712069864004e8e7fea313e9e22f7c.

## Next gate

Profile prefix and curvature banks. Investigate streaming the retained first-component prefixes and recomputing exact response states on demand, or blockwise storage of weighted displacement arrays. Avoid a second full analytical screen when the first-stage eligible set touches every block. Freeze an independent model-root roster and measure end-to-end RSS and time under explicit memory budgets before promoting. Learned native BPD factors, pass-mean and sphere-rescale intervention semantics, verified floating-point bounds and operating-system memory containment remain open.
