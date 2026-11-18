# BPD-SUB-02: future-observability grain versus state-PCA grain

**Prospective registration:** November 18, 2026. This is the first SUB-02 commit, prior to writing its experimental implementation or training any SUB-02 development models. Scientific classification is *fresh synthetic engineering only*. SUB-01 original failure remains frozen in PR #84.

## Scientific question

SUB-01 clean prediction KL at rank 10 averaged only 0.0027 nats, but its rank-10 discarded-fiber impulses moved final logits by 0.4113 RMS on the selection split, above the registered 0.050 gate in all eight model roots. PCA preserved ordinary predictions while leaving counterfactual sensitivity. SUB-02 tests whether choosing the retained state subspace by *future-output observability* instead of state variance reduces this finite-dose sensitivity at a fixed rank.

This examines a functional output-specific causal quotient of a trained recurrent digital model. A future-logit Jacobian criterion does not establish preservation of all hidden-state dynamics, biological physics, consciousness or subjective continuity.

## Registered implementation

Reproduce the **same** SUB-01 Looper architecture, 12-dimensional state, four nonlinear tanh recurrent steps, independent external binary task law, data-generation schedule and source-training loop, using fresh SUB-02 seeds. Preserve the original SUB-01 imported source at commit 38dc50de5b7faf2e8dde54862c9666de65832de4, and its frozen protocol fe46c382fa69f9d7a2390bad0cff64245861ff1e. Do not modify the source, results, execution lock or audit.

- **Eight independent engineering model roots:** 1300..1307. **Excluded smoke:** 1299. Reject other seeds in the runner.
- Each root: 2048 train, 512 discovery, 512 calibration and 1024 independent heldout inputs, exactly as in SUB-01. The calibration split records source-only spectra and physical intervention dose; it cannot adapt architecture, rank, measured target or data generation.
- Source training: PyTorch float32 CPU, Adam lr 0.01, minibatch 128, 256 steps, four passes, same task function, seeded training samples and initialization. Holdouts are read only after every projection, sham and impulse distribution is defined.
- **Rank 8 is the single primary comparison. Rank 10 is secondary.** No adaptive rank selection and no reuse of SUB-01 development roots.
- On the discovery split only, record post-first-pass state h1. Compute a future-output Jacobian for each source input: J(x) = dR(F_x^3(h1(x)))/dh1 in *physical source hidden coordinates*, with source model parameters fixed. The global future-observability Gramian is C_obs = mean_x J(x)^T J(x). Let P_obs,r be its top-r spectral orthogonal projector, float64 Eigh.
- Comparator P_pca,r is the top-r projector of the **SUB-01 exact uncentered second moment of all clean h1..h4 states** on the same discovery inputs. This method difference is fixed prospectively.
- For numerical parity, report P symmetry/idempotence, rank trace, spectrum, eigengap and the difference between a finite-difference Jacobian check and autograd on the excluded smoke root. Any failure of numerical integrity blocks interpretation.
- At heldout input x, use the **same raw standard-normal impulse vector** for P_obs,r and P_pca,r, project onto ker(P), then scale each to identical physical L2 norm alpha = 0.25 * sqrt(E_discovery ||h1||^2). Inject once, immediately after pass 1; run the remaining three nonlinear tied passes on the altered states. Max per-example dose error <= 5e-6 and max retained-projection error <= 5e-6.
- For each rank, pre-generate eight seeded Haar-random rank-r control projectors, with the same raw impulse vectors and norm. Score all eight on every root regardless of outcome; matched-dose sham baseline is the median terminal-logit RMS over them.
- Identity/full-state closure and exact orthogonal gauge transport under q(h)=Qh must satisfy max clean and correctly transported counterfactual logit error <=5e-5. The deliberately untransported impulse must change outputs (RMS>1e-5). Failing any admission guard makes a root ineligible but never removes it from the denominator.

## Fixed metrics and engineering decision

**Unit of inference:** the independently trained model root. The 1024 heldout examples, 8 sham projectors, two ranks, and four passes are repeated observations.

Define per-root terminal intervention response

S_r(P)=sqrt(mean_x [R(F_x^3(h1+delta_{P,x}))-R(F_x^3(h1))]^2).

**Primary matched root effect at rank 8:** Δ_i = S_8(P_pca) − S_8(P_obs), in logit RMS units. Positive means the future-observability projector permits less counterfactual influence to travel through its discarded fiber at the same physical impulse norm. Root bootstrap with 2000 resamples, seed 130202, only over all eight independent trained roots; no alternative unit count.

An engineering-positive decision requires all eight models competent at >=0.80 separately within x3-positive and x3-negative heldout contexts; all eight gauge and dose admissions pass; all eight return numerically valid rank-8 projectors, **mean Δ >=0.15 logit RMS**, at least 7/8 roots Δ>0, and root-bootstrap 95% lower endpoint >0. Preserve failures as \`ENGINEERING_NULL_OR_INELIGIBLE\`. The threshold 0.15 is a predetermined practically large portion of the SUB-01 rank-8 response (~0.59) and exceeds a tiny descriptive difference, but future sham-tail calibration must still decide scientific resolution.

Secondary fixed-rank-10 Δ and each rank's sham-minus-obs S improvement are descriptive. A new explanatory family of statistical tests must apply Holm over those secondary comparisons. At each rank also report the source-to-projector clean predictive Bernoulli KL, external-label NLL and accuracy, next-pass projected-state commutator, response RMS at passes 2/3/4, source accuracy, spectra and physical impulse checks. The source model's behavior after an impulse is **not** replaced by a surrogate when computing S_r, ensuring the measured response is a counterfactual effect of a hidden-state intervention.

## EvalMath boundary and next hypothesis

Eight independent development roots provide a pilot with root-bootstrap uncertainty. They cannot supply the distribution-free 95th percentile / 95% confidence sham variation floor requiring at least 59 independent sham-calibration roots. Freeze results, compare the independent-root mean primary causal effect to the matched-sham variation floor with the EvalMath strict preflight, and separately preregister new confirmation units only if an independent development signal is resolvable. A positive rank-8 result is \`ENGINEERING_SIGNAL_ONLY\` and an endpoint-specific causal-compression result; it cannot be described as strict full-state quotient invariance. A future study can replace the source transition with a separately learned architecture and test transported interventions across all passes. Existing ARG-03-N, AMM-03/04/05, CQ-08 confirmation and NC-08 protected populations stay sealed.
