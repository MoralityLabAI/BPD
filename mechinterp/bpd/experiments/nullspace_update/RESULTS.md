# Protected low-rank update solvers: synthetic study (synthetic_cpu)

Module: `nullspace.py` (NumPy only). Tests: `test_nullspace.py` (13 tests). Campaign and controller: `campaign.py`.
Evidence class `synthetic_cpu`. Nothing here is evidence about any hosted model.

## Reproduce

```
python -m unittest mechinterp.bpd.experiments.nullspace_update.test_nullspace
python -W ignore -m mechinterp.bpd.experiments.nullspace_update.campaign --out mechinterp/bpd/experiments/nullspace_update/results/campaign.json --n 800
```

Run on one CPU thread (`OMP/MKL/OPENBLAS_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=-1` set inside `campaign.py`). The 800-scenario run took 819 s on a loaded host; peak working set 270 MiB. Raw numbers: `results/campaign.json`.

## Numerical tests (13 pass)

* Stable form `D = N (X_new N)^+ R`: scale-free leak `||X_keep D|| / (||X_keep||_2 ||D||_F)` below 1e-13 in float64 and below 5e-6 in float32 (campaign median 2.2e-16 float64, 8e-9 float32).
* Literal `P = I - X_keep^+ X_keep`, `U = pinv(X_new P) R`: in the campaign `||U|| / ||D_stable||` has median 1.7e9 (float64 and float32 pooled); leak median 0.088, i.e. the literal form is not protected.
* Rank-deficient, cond up to 1e11, underdetermined X_new (m < d_in), empty null space (returns zeros), rank truncation inside null(X_keep): all stay protected.
* `ridge_rel` (Tikhonov in null coordinates, mu = ridge_rel * sigma_max(X_new)^2) keeps the leak at 1e-12 and bounds `||D||` when X_new is nearly inside span(X_keep); plain pinv there is more than 1e3 times larger.

## Campaign design

800 scenarios, parameters sampled jointly at random per scenario (seeded): d_in 48-256, d_out 16-48, protected rank 5%-98% of d_in (8% have an empty null space, 48 of 800), X_keep condition 1 to 1e8 (log-uniform), conflict (fraction of X_new energy inside span(X_keep)) 0-1, new samples 0.2x-5x d_in (underdetermined common), noise 1e-3 to 0.5 of target RMS, target update rank 1-16, off-span protected test energy 0-0.5, float32 in 40%. New-task inputs are drawn with the keep spectrum inside the span, so the distributions overlap and are not separable. Utility = held-out new-task R2 - 2 x held-out protected relative error (`||X_keep_test D|| / ||X_keep_test W^T||`).

`ns_*` and `tsvd_*` use `ridge_rel = 1e-2`; `ns_pinv` is the unregularised stable form. `ridge_lam10/1000` are soft protected ridge, `approx_e90` protects the top directions holding 90% of X_keep energy.

## Method table (800 scenarios, means; leak is the in-sample scale-free leak)

| method | utility | new R2 | held-out prot. err | leak median | leak max | oracle wins |
|---|---|---|---|---|---|---|
| ordinary ridge | -1.297 | 0.852 | 1.074 | 0.220 | 0.992 | 0 |
| soft ridge lam=10 | -1.684 | -0.045 | 0.820 | 3.7e-3 | 0.992 | 17 |
| soft ridge lam=1000 | -1.712 | -0.163 | 0.774 | 2.0e-4 | 0.992 | 1 |
| null-space, pinv | -15.500 | -13.200 | 1.151 | 2.2e-16 | 2.6e-5 | 48 |
| null-space, full (ridge) | -0.651 | 0.260 | 0.456 | 2.2e-16 | 3.1e-5 | 17 |
| null-space rank 2 | **-0.566** | 0.136 | 0.351 | 2.2e-16 | 3.1e-5 | 329 |
| null-space rank 4 | -0.621 | 0.202 | 0.412 | 2.2e-16 | 3.1e-5 | 39 |
| null-space rank 8 | -0.643 | 0.250 | 0.447 | 2.2e-16 | 3.1e-5 | 72 |
| null-space rank 16 | -0.648 | 0.262 | 0.455 | 2.2e-16 | 3.1e-5 | 84 |
| truncated SVD rank 4 | -1.030 | 0.681 | 0.856 | 0.292 | 0.993 | 153 |
| truncated SVD rank 16 | -1.095 | 0.850 | 0.972 | 0.292 | 0.992 | 40 |
| approx protect (90% energy) | -4.931 | -1.577 | 1.677 | 0.030 | 0.985 | 0 |
| random rank 8 (matched Frobenius to null-space r8) | -1.670 | -0.446 | 0.612 | 0.177 | 0.872 | 0 |
| literal pinv diagnostic | -3.333 | -0.564 | 1.384 | 0.088 | 0.993 | 0 |

Null-space leak max 3e-5 comes from float32 scenarios with cond above the float32 rank cut; float64-only max is 4e-15. Float32 only: null-space median leak 8e-9, in line with the float32 rounding scale.

Reading: the null-space methods have the best mean utility and exact in-sample protection, but they give up new-task R2 (0.14-0.26 against 0.85 for unprotected) because conflict is high in many scenarios and held-out protected inputs carry off-span energy that no in-sample projector covers (protected error 0.35-0.46, not zero). Rank 2 wins most often because small norm limits damage on off-span protected inputs, a property of this utility weighting. Unprotected truncated SVD wins 153 + 40 scenarios (low conflict, low off-span energy). Without the ridge, pinv null-space is far worse than ridge: noise is amplified when X_new N is small.

## Controller

Features computed before solving: in-span energy fraction of X_new, log10(m/d_in), log10 condition of X_keep over its numerical rank, rank/d_in, log10(m / null dim), is_float32. Label = argmax-utility candidate among the 11 non-control methods above. Models written in NumPy: nearest centroid on standardised features, depth-3 gini tree, and a per-method quadratic ridge utility regressor (argmax of predicted utility). Mean utility on test scenarios (higher is better):

| policy | (a) held-out seeds, same distribution (400 train, 200 test) | (b) held-out property combination (201 train, 195 test) |
|---|---|---|
| oracle | -0.483 | -0.541 |
| tree (depth 3) | **-0.549** | -0.688 |
| best fixed on train = null-space rank 2 | -0.613 | **-0.681** |
| nearest centroid | -0.632 | -0.780 |
| fixed null-space rank 4 | -0.676 | -0.768 |
| fixed null-space full | -0.709 | -0.821 |
| utility regressor | -0.808 | -2.630 |
| fixed truncated SVD rank 4 | -0.899 | -0.860 |
| fixed ordinary ridge | -1.192 | -1.125 |
| random choice | -1.553 | -3.647 |

(b) trains on cond < 1e3 and conflict < 0.6 and tests on cond >= 1e3 and conflict >= 0.6.

Findings: on held-out seeds the tree recovers 0.064 of the 0.130 gap between the best fixed method and the oracle and beats every fixed method. Under the property shift no controller beats always-null-space-rank-2: the tree (-0.688) ties it within noise and loses by 0.007, the centroid and regressor lose clearly (the regressor extrapolates badly to cond >= 1e3, utility -2.63, worse than ordinary ridge). With 195 test scenarios and per-scenario utility sd near 1, differences below about 0.1 are not resolved. Mean-utility claims here rest on one pool of 800 scenarios and one train/test split per setting.

## Adaptive rank allocation

100 groups of 4 modules (null dim >= 16), total rank budget 16, pooled held-out R2 over the group: adaptive 0.1863, uniform (4 each) 0.1899; adaptive wins 54 groups, uniform 45. Greedy allocation maximises captured singular energy on the training coordinates but does not improve held-out R2 here; the difference is within noise.

## Activation archive

Skipped. The only archive under 50 MB, `notes/data/bpd_b4_v2_stage4_weights.pt` (24 MiB), holds float32 `(V, U)` weight factors for layers `o_proj` (512 x 512) and `down_proj` (1536 x 512), not activations, so there are no keep or new-task inputs to build `X_keep` and `X_new` from. Its matrices also exceed the d_in <= 256 memory cap used here. No `.npz` activation files exist in the repository.
