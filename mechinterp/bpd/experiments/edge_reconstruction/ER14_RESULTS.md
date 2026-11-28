# ER-14: sparse cross-pass selection under a joint-rollout budget

## Method

Eight fresh synthetic trained-model roots 8..15, teacher/student setup from ER-13. 24 calibration probe/readout pairs, 48 disjoint evaluation pairs, 48 episodes per root; 12 distinct randomly ordered noisy probes per episode. All arms receive the same hidden faults and observations. 36 candidate pairs (one SVD component removed at pass 1 and one at pass 2). Root-level correlation must be respected (eight independent models, 384 episodes).

For a first-pass ablation i, obtain the resulting change delta_h2_i at the input of the second ablation. With component j read vector v_j and write vector u_j, the additional preactivation change due to removing j is exactly cross_ij = -(delta_h2_i dot v_j) u_j. Approximate the final scalar interaction by propagating cross_ij through the downstream clean tangent map. Rank pairs by calibration RMS of this scalar. This proxy omits nonlinear superposition terms; no claim of exact final-output residual. Also compare RMS cross-write overlap, product of marginal effect RMS, random ordering, and calibration joint-residual oracle ranking. All non-oracle policies have access to the same clean and 12 single-component calibration rollouts, without calibration joint outcomes (random does not need those inputs). Oracle ranking uses 36 additional calibration joint rollouts. The environment separately computes the full calibration oracle to set a common noise scale and test oracle to simulate ground truth; these are not candidate selection inputs except for oracle ranking. All policies are given the same known noise scale.

For each selected pair, compute its exact joint effect on the evaluation input batch, with additive predictions elsewhere. 13+k full evaluation batch rollouts per arm, explicitly asserted, excluding environment generation. All arms currently materialize the full 36 x 48 predicted response matrix, so this is a rollout reduction, not a measured RAM reduction. No trained selector, timing-matched policy or native-model claim.

## Executed results

`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python sparse_crosspass.py > sparse_crosspass.json` (PyTorch 2.10.0+cpu, NumPy 2.3.5). Source SHA256 04f8578e324c38947340e9c7e8df456dcbbd2441221aed700baf633137d78a7c; local receipt SHA256 f95c4d3792108558f65a9f41bf272a1158841ca51e5ecbd7186ab205b5de7197. Native path optional; repository-side native parity gate requested separately.

| k | eval rollouts | random MAP /384 | marginal | overlap | tangent | calibration oracle (extra cost) |
|---:|---:|---:|---:|---:|---:|---:|
|0|13|309|309|309|309|309|
|4|17|321|324|333|334|334|
|8|21|326|339|347|348|348|
|16|29|341|351|364|368|368|
|36|49|370|370|370|370|370|

At k=16 guided vs random per-root hits (48 episodes each): 47/44, 47/40, 47/44, 48/43, 35/34, 48/46, 48/45, 48/45. At k=8 the tangent arm lost one of eight root comparisons. At k=16 the tangent method achieved 384/384 empirical 95% credible-set coverage, mean truth NLL 0.152 bits and median prediction RMSE 0.00140; random k=16 achieved 359/384, 19.201 bits, 0.00876 respectively. Full exact achieved 384/384, 0.152 bits, zero RMSE. Gaussian likelihood uses environment sigma and a numerical probability floor of 1e-300; approximate predictions are misspecified. Coverage is empirical, not a calibration guarantee. All k=36 banks matched the exact oracle within 1e-6.

Median test build times (exploratory CPU, no calibration/episode inference charged): tangent k=16 3.749 ms vs full tangent k=36 6.048 ms. All arms' base routine also computes selection statistics on evaluation inputs but never uses those scores to select pairs; timing includes this redundant work. No RSS measured. Budgets and roots chosen for exploratory pilot, not preregistered. No full learned BPD factors, real LLM ablations, or external generalization tested.

## Next gate

Native BPD wrapper parity, then learned decomposition and pass-mean/sphere-rescale semantics. Freeze new roots and selection budgets, include shuffled tangent score control and interaction uncertainty. Replace materialized prediction matrix with sparse residual storage and measure end-to-end time/RSS at matched rollout budgets. For certification, bound omitted interaction errors via downstream tangent and curvature terms.
