# ER-26: sorted-batch overhead and online incumbent contraction

## Mathematical scope

Let the exact candidate energy be E_g and valid lower bound L_g <= E_g. Start from one exactly evaluated incumbent, and form the eligible set of other candidates with L_g <= incumbent + delta (delta >= 0). Visit this set in nondecreasing lower-bound order with batches of size at most b; discard remaining candidates only when their lower bound exceeds current exact incumbent + delta. Stop on strict separation. Return the smallest index among exact minimizers, with ties evaluated. Define E*=min_g E_g and S*={g:L_g<=E*+delta}.

All S* are either the initial incumbent or in the initial eligible set and cannot be pruned. Before the first optimum is evaluated, candidates preceding that optimum in sorted order have lower bounds at most L_opt<=E*, and belong to S*. At most b-1 extraneous calls occur in the batch that first evaluates an optimum. Afterwards every evaluated candidate belongs to S*. The initial incumbent may contribute one additional extraneous call, so |S*| <= N_exact <= |S*|+b. If the initial incumbent is optimal, N_exact=|S*|. This is a property of this specific sorted search, not an information-theoretic lower bound on all possible algorithms. The proof is real arithmetic; float64 implementation uses EPS=1e-10 and does not implement outward rounding.

Exhaustive finite audit: n=1..4, integer energies 0..2, every integer lower bound in [0,E_g], every incumbent, sorted batch sizes 1,2,3 and online schedule. All 23,640 runs agree with eager argmin including ties, and satisfy the call-count inequality. No Lean formalization.

## Fresh-model pilot

Widths 32,64,128; roots 48..51 for each width (12 newly trained teacher/student instances, four tied tanh passes, SVD factor ablations). ER-25 generator and noise regimes (.1,1,3 times calibration RMS). Three separate 12-probe episodes per model, 36 total. Compare eager batch64 (stores scalar costs for independent audit), fixed sorted batches 16,64,256, online batch controller. Static width-specific comparator was selected in ER-25 calibration and frozen here: 64,64,16 respectively. Online starts at 16, resets to 16 after >=25% contraction of the competitive suffix in a batch, otherwise doubles up to 256. No new fitting. Threshold and root roster chosen for this exploratory run, not preregistered. All bounded arms use same screening and exact kernel. Timing includes prefix construction, screening, sorting, dispatch and trace logging; training and environment generation excluded. Two rotated timing repetitions per episode, single BLAS thread; no concurrent width runs. Width128 roots executed in two sequential groups. All comparisons are episode-level descriptive statistics; episodes within model roots correlated. Time includes a full scalar-cost audit array in eager, which a deployment-only eager comparator could omit. Peak RSS not measured.

All 36 distinct episodes match exact eager MAP, including every method/repetition; costs within 1e-10. 1,032,192 scalar lower-bound comparisons (four bounded arms, 258,048 distinct candidate-episode combinations). All 144 bounded episode-method call counts satisfy the inequality. Hidden fault recoveries 12/36, all at .1 noise (12/12); 0/12 each at 1 and 3 noise. This is optimization equivalence, not a claim of accurate inference at high noise.

|width|eager median ms|sort16|sort64|sort256|online|frozen static|online wins /12|median paired static/online|
|---:|---:|---:|---:|---:|---:|---|---:|---:|
|32|4.445|1.965|1.874|2.181|1.977|sort64|4|.970|
|64|36.927|15.312|14.874|15.150|15.936|sort64|1|.863|
|128|471.629|169.237|163.755|190.207|179.146|sort16|1|.932|

Overall median paired eager/online 2.486, static/online .93191 (online approximately 7.3% slower), 6/36 online wins. FAIL incremental timing objective, retain fixed-batch baseline. Do not infer online policy benefit from eager comparison.

|method|total exact calls|excess above S*|median excess|max excess|episodes with excess >1|
|---|---:|---:|---:|---:|---:|
|sort16|69178|33|1|1|0|
|sort64|69182|37|1|5|1|
|sort256|69916|771|1|170|9|
|online|69229|84|1|27|2|

Sum of |S*|=69145. Fixed64 overhead 37/69145=0.0535%; almost all remaining calls arise from the competitive set under current lower bounds, not dispatch strategy. Median S*/C^2 fraction by noise .1/1/3: .06897/.27966/.36841. This is the relevant next target. Improve the lower bounds or exact-kernel cost before further online batch tuning.

## Reproduce

From experiment directory:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python er26.py --audit > audit.json
python er26.py --width 32 > er26_32.jsonl
python er26.py --width 64 > er26_64.jsonl
python er26.py --width 128 --roots 48 49 > er26_128a.jsonl
python er26.py --width 128 --roots 50 51 > er26_128b.jsonl
cat er26_32.jsonl er26_64.jsonl er26_128a.jsonl er26_128b.jsonl > er26.jsonl
```
NumPy 2.3.5, PyTorch 2.10.0+cpu. Source SHA256 7b95056a6d8d69d3d04eb0ad50338bfed4afc280cd52be8b02258d519c6087a1; local Git blob 47f468d2782a7e74e9771ab84c667319c416f3bc. Results SHA256 5679159d9e2d20e603f24530b71a102285079c29318e26fcdf89421172f2473b; audit SHA256 87658e170a9ed81ce677e11a65cb3cca1970d87d251c1eb6ca059d0cb860a15b. Raw receipt includes first-repetition screen/finish component timings and traces, while total ms is the median across two repetitions.

## Next gate: tighter lower bounds, not further batch-policy tuning

For a second-stage envelope use exact pair-specific state after the second ablation, followed by a signed first-order approximation of the final readout. For z final preactivation and displacement d, let fhat=r tanh(z)+r diag(tanh'(z)) d. Taylor's theorem yields |f-fhat| <= beta/2 sum_l |r_l| d_l^2; replace beta by a coordinate-local curvature supremum where feasible. Bound validity is exact in real arithmetic for this recurrence. Whether calculating d and the signed approximation saves time over direct final tanh evaluation is an open engineering question. A more promising implementation should exploit partial hidden-coordinate contractions or use a cheap earlier bound cascade, charge all stages, and compare against the strong prefix-reused eager kernel. Test candidate-set reduction at matched noise and fresh roots; use actual learned BPD factors and hard memory enforcement separately. No implementation or speed claim for this proposed extension.
