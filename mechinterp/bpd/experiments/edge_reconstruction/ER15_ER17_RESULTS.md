# ER-15/16/17: native parity, sparse response storage and scaling

Native ER-13 parity workflow succeeded: https://github.com/MoralityLabAI/BPD/actions/runs/45857674796 . Fable pre-run source regression checks also succeeded: https://github.com/MoralityLabAI/BPD/actions/runs/45857674481 . Scope is eight synthetic SVD-initialized models, zero-ablation/no sphere rescaling, tolerance 1e-6. No learned BPD decomposition parity claim.

## ER-16 fresh root replication (script er15.py)
CPU PyTorch 2.10.0+cpu, NumPy 2.3.5. Roots 16..23, same training, calibration, noise and episode generator as ER-14, 48 episodes per root, 12 probes per episode, 48 available evaluation probes. Selection uses independent calibration inputs. The oracle generates ground truth and sigma separately. Candidate bank stores clean and first/second single-response vectors plus exact selected joint rows and indices; query constructs only requested columns. Store exact rows rather than subtracting and adding a residual, to preserve float32 rounding. Dense and sparse paths use identical arithmetic order. All 1536 query comparisons and all posterior metrics identical. Dense materialization in audit is not part of the intended sparse production path.

| policy | hits/384 at 12 | 95% coverage/384 | mean truth NLL bits | persistent sparse bytes | dense bytes |
|---|---:|---:|---:|---:|---:|
| additive |330|334|22.57993|2496|6912|
| random k16 |352|357|17.26885|5696|6912|
| tangent k16 |379|382|0.08676|5696|6912|
| full k36 |382|384|0.01848|9696|6912|

At k16, tangent vs random hits by model root: 48/42,47/44,46/40,47/43,48/46,48/47,48/45,47/45. Eight fresh root wins, combined with eight earlier exploratory wins. Episodes within roots are correlated. Median tangent posterior times dense 1.9914 ms, sparse 2.5566 ms (audit retains both stores; no neural-process RSS inference). Full k36 should use dense storage. Test rollouts 13+k; environment oracle/calibration separate. No preregistration.

## ER-17 synthetic response-array scaling

For C components each pass, P available probes, k exact overrides, float32 and int64 indices: dense 4PC^2 bytes; sparse 4P(2C+1+k)+8k bytes. Savings iff k < 4P(C^2-2C-1)/(4P+8). Sparse queries temporarily materialize C^2 times requested probe count, not the entire bank. Complexity savings depend on access patterns and override density.

Three fresh processes per configuration, seeds 1,2,3; each queries 48 batches of 12 distinct probes, P=1024, k=C^2/16; alternating order by seed. Linux ru_maxrss reports process high-water RSS including interpreter and preexisting inputs; baseline before bank build retained in raw receipt. Timing covers query and float64 sum, excluding input generation; dense construction separately charged. Nine paired checksums exactly equal; checksum is not an elementwise equivalence proof at scale. Neural equivalence checked separately above.

| C | dense payload bytes | sparse bytes | median peak RSS dense/sparse KiB | median dense/sparse build ms | median dense/sparse query ms |
|---|---:|---:|---|---|---|
|32|4194304|528896|100564/96532|2.394/0.007|3.548/1.684|
|64|16777216|1579008|113892/97688|11.929/0.008|15.772/5.707|
|128|67108864|5255168|166712/101932|45.085/0.008|58.466/23.792|

At C=128 payload ratio 12.77 and query median ratio 2.46. Sparse build timing mostly measures retaining existing input arrays; dense build materializes the full matrix. This is a synthetic layout microbenchmark, no trained large-model or end-to-end rollout speed claim.

Reproduce from experiment directory:
```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python er15.py > er15.json
for n in 32 64 128; do
  for s in 1 2 3; do
    if [ $((s % 2)) = 1 ]; then modes='dense sparse'; else modes='sparse dense'; fi
    for m in $modes; do
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sparse_store.py "$n" 1024 "$((n*n/16))" "$m" "$s"
    done
  done
done
```
Local SHA256: sparse_store.py 7cb6bbd83561356e6af502f24aab81200d7c0ec3cddb6150a0190f2fecea6a49; er15.py c2f1623ee7389437b6ce98f9d037184c18913a0c54740d519d59b2c2c4be0ec9; er15.json 4106bc7656e7d69672f07ef6e0489a07e488f4ebe9b3e8915ff412b60fd472b7; er15_scale.json 5835452318611035c2198d6349ca05e5187d0bd9953fc156ab188f626113a983.

Next: native sparse parity, trained BPD factors, pass-mean/sphere calibration, fresh registered roots and shuffled tangent controls; test error bounds and intervention uncertainty for omitted joint pairs. Full paper audit and Lean proof remain outstanding.
