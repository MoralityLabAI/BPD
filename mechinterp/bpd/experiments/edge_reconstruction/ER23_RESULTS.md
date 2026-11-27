# ER-23: streamed bound scoring at widths 128 and 256

Four fresh independent trained models: widths 128 and 256, two roots each (36,37; ER-20 training with width included in seed). Three independent 12-probe episodes per model, paired across policies. Same four-pass tanh teacher/student recurrence, fixed SVD components. Separate 24-probe calibration and 256 sampled joint pairs determine Gaussian noise scale; ground truth generated with independent direct weight edits, checked against prefix rollout within 1e-10. No preregistration.

All methods use exactly 12 demanded probes, shared prefix construction and the same batched exact-response kernel (64 pairs per batch). SVD precomputed and excluded from timing in all arms. Eager scores every exact pair in batches, retaining only incumbent. Dense bounded arm constructs pair-by-probe predictions and envelopes before reducing to lower costs; streamed arm processes eight first-component rows at a time and stores only scalar lower costs, minimum upper incumbent, and prefix/single-component tensors. Sorted candidates refined in batches of up to 64 while lower cost <= best exact cost plus numerical slack. Batching can over-evaluate candidates relative to serial best-first. Real-arithmetic validity follows from ER-18/20; floating-point tolerance, not outward-rounded arithmetic. Checked exact costs of evaluated candidates against lower bounds. No exhaustive new scalar envelope audit. All three methods matched MAP index and cost within 1e-10 on 12/12 episodes; hidden fault recovered 11/12. Two independent roots per width, exploratory pilot.

|width|candidate pairs|median exact rows (both refinements)|eager median ms|dense median ms|stream median ms|median paired eager/stream|
|---:|---:|---:|---:|---:|---:|---:|
|128|16384|2654|526.953|131.045|121.616|4.660|
|256|65536|16741.5|6333.053|1689.773|1844.083|3.339|

Median paired eager/dense speedup: 4.314 and 3.657. Two timing repetitions at width 128; one at width 256 (reduced after a command timeout; no incomplete rows included). Separate serial process per model and policy; order eager/dense/stream for root 36, stream/dense/eager for root 37. Time includes prefix construction, bounds, sorting, exact batches and scoring; excludes training, SVD, loading and environment generation.

Peak RSS (Linux ru_maxrss, medians across two processes per width, MiB):
|width|eager|dense|stream|
|---:|---:|---:|---:|
|128|102.180|117.818|114.896|
|256|119.617|187.568|175.092|

Increase over pre-benchmark high-water RSS: 7.283/22.932/20.000 MiB at width 128; 22.979/90.932/78.572 MiB at width 256. Streaming improves memory relative to dense bound scoring but uses more memory than eager streaming exact evaluation. The shared prefix and single-trajectory tensors dominate retained bound-state memory at these widths. No general RAM reduction claim.

Reproduce from experiment directory:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python er23.py prepare er23_worlds > er23_prepare.jsonl
for d in 128 256; do
  reps=2; [ "$d" = 256 ] && reps=1
  for root in 36 37; do
    modes='eager dense stream'; [ "$root" = 37 ] && modes='stream dense eager'
    for m in $modes; do
      python er23.py bench "er23_worlds/${d}_${root}.npz" --mode "$m" --repeats "$reps"
    done
  done
done > er23.jsonl
```
NumPy 2.3.5, PyTorch 2.10.0+cpu. Local source SHA256 51cf4af5ea24f0fbb4babd91207844173fa189baf960774a7abee401264d0b47, raw receipt SHA256 c394d501f3aff33aac5e462ee00a05972b85d4e0dacfa530daa399203a015749, preparation receipt SHA256 8ed6f68e39f7d06510f1cd1624409a09c88dc7bb5ad4c9ed29eea3b44ad83550.

Next: register more independent roots; vary batch sizes and block sizes; profile prefix-state lifetime, use a memory budget and empirically calibrated cost model for adaptive eager/stream switching. Evaluate native learned BPD factors and pass-mean ablation, and formally verify arithmetic. No learned factor result or dynamic policy comparison here.
