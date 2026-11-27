# ER-20/21: contracted bounds and demand-matched MAP refinement

## Algebra

In ER-18 let row vector r be the scalar readout c^T O, d_i the change of hidden state after the first ablation, a_i=W d_i, b_j=-M_j h2, M_j=u_j v_j^T, and a4_i,b4_j the single-ablation preactivation changes at the final pass. The global coordinatewise envelope is

r3_ij <= |v_j^T d_i| |u_j| + beta |a_i| |b_j|,
r4_ij <= |W| r3_ij + beta |a4_i| |b4_j|,

where products of vectors are elementwise, beta=4/(3 sqrt(3)). With q=|r| |W|, exact reassociation of nonnegative sums gives the same global scalar error envelope

E_ij = |v_j^T d_i| (q |u_j|) + beta q (|a_i| |b_j|) + beta |r| (|a4_i| |b4_j|).

The bound can therefore be computed with batched weighted Gram-like contractions, without multiplying a separate hidden-state vector by |W| for each pair. With C components, hidden width D, P probes, and dense W, the pair-scoring term costs O(P C^2 D) rather than O(P C^2 D^2); base single-trajectory and projection work, SVD, storage, sorting, and exact queries are additional. Eager comparator also reuses first-ablation prefixes and rank-one downdates; it batches all second ablations per prefix. The implementation materializes C^2 by P prediction and bound tables, so no persistent RAM reduction claimed. Local bound vectorized per prefix remains much slower.

Sorted refinement evaluates minimum-upper-bound incumbent first, then visits candidates in ascending lower-bound order until strict separation. It is equivalent to ER-18's repeated minimum scan in real arithmetic with valid intervals; ties choose the lowest index. It returns the exact uniform-prior common-Gaussian MAP decision, not the full posterior or the true latent fault with certainty. Numerical float64 checks use EPS=1e-10, not directed rounding.

## ER-20 executed (er20.py)

16 independently trained teacher/student roots (widths 6,16,32,64, four roots per width 32..35, width included in seed); four tanh passes, 350 Adam updates, 128 batch, separate calibration 24 and eval 48 probes. Known SVD factors and common Gaussian sigma=0.1 RMS calibration joint effect (floor 1e-5); 24 episodes/model, 12 distinct probes each. Oracle used for calibration noise and simulated observations, not candidate selection. 16 randomly selected full independent weight-ablation rollouts per root match prefix-reused exact within 1e-10 (256 batch checks). All 2,078,208 scalar bound comparisons passed (both global and local against exact). Local <= global within tolerance. All 384 episode MAP decisions agree across both refinements and eager, with 372/384 true-fault recoveries.

Timings include building prefix states, SVD, bounds or eager exact bank and decisions. Full 48-probe bank in all arms; each lazy arm retains exact rows across episodes within the workload. Excludes training and environment generation. Three rotated timing repetitions per model/workload, median within root then across roots. ER-21 ran concurrently with part of the width-64 timing series; shared machine resource contention may affect measurements. Exploratory, no registration or independent hardware replication.

|width|pairs|episodes|eager ms|global ms|local ms|median global exact rows|paired eager/global|
|---:|---:|---:|---:|---:|---:|---:|---:|
|6|36|1|.330|.314|1.237|2.5|1.040|
|6|36|8|.411|.965|2.381|17.5|.452|
|6|36|24|.503|1.810|3.076|28.5|.277|
|16|256|1|2.769|1.378|10.754|16|2.032|
|16|256|8|2.864|3.815|13.343|75.5|.758|
|16|256|24|3.061|7.754|16.767|117.5|.394|
|32|1024|1|18.530|6.231|85.153|85.5|3.074|
|32|1024|8|18.838|18.937|94.425|301.5|.977|
|32|1024|24|19.818|38.585|119.516|467.5|.487|
|64|4096|1|198.703|45.645|945.483|342.5|4.425|
|64|4096|8|199.817|134.424|1036.773|1244.5|1.477|
|64|4096|24|205.673|264.755|1156.487|1759.5|.757|

Global wins at larger widths for sparse workloads; full-bank eager wins at 24 episodes for all widths. Local bound has no runtime win at tested sizes.

## ER-21 stronger demand-matched baseline (er21.py)

Retrains same model seeds as ER-20, different probe/episode seed 261138; not independent model replication. Both methods receive exactly the 12 demanded probes and build only their responses. No shared cache across episodes. Sorted refinement computes global contracted bound, evaluates exact candidate rows on demand. Three alternating repetitions per episode, 8 episodes/root, 32 per width. 128/128 agreement with independent full bank MAP, 125/128 hidden recoveries. Timing includes SVD, prefixes, bounds, sorting, inference. Some runs overlapped ER-20 width-64 series. Medians over episodes, not independent root-level confidence intervals.

|width|eager ms|sorted ms|median exact rows|paired speedup|
|---:|---:|---:|---:|---:|
|6|.173|.208|3|.840|
|16|.859|.567|14|1.520|
|32|5.170|2.189|69.5|2.308|
|64|39.407|10.591|322.5|3.798|

## Reproduce
From experiment directory:
```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python er20.py > er20.jsonl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python er21.py > er21.jsonl
```
Run sequentially for clean timing. NumPy 2.3.5, PyTorch 2.10.0+cpu. Local SHA256: er20.py 17178a243b7885c85a1691cc261137169f8f181f81f3b1d8fdf61d75116000f2; er21.py b8560a359efaf17a483144baab0643c44292c26a9a8f2cc18cb447ace60473ef; er20.jsonl d3c236d405fbe4888a44db59c1ddbd6234d1414b828c85c040c7f9e8e6c644c5; er21.jsonl 564cb139be7c4c91291a08081c9cb0ed4b7eba15b822d41fb52ceba0d26afa07.

Next: register independent roots and clean serial timing; compare adaptive streaming/eager switching and candidate blocks at widths 128+; test learned native BPD factors and mean/sphere ablation separately; formalize algebra and MAP certificate. No actual learned BPD decomposition, native parity at larger widths, Lean proof, or formally verified numerical intervals in this pilot.
