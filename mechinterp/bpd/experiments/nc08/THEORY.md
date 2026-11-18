# BPD-NC-08: finite rank budgets, local score loss and closed-loop causal sufficiency

## 1. The rank-budget problem is a finite multiple-choice knapsack

At four tied recurrent physical write sites, let r_t be the number of retained BPD factor read/write directions at pass t. The total retained dimensional expenditure is

B = r_0 + r_1 + r_2 + r_3.

NC-07 used a common rank r, so its total expenditure was 4r. That forced all four passes to rank six when a single pass might have required extra capacity. NC-08 registers allowed individual ranks {2,3,4,5,6}, candidate budgets {8,10,12,14}, and a fixed rank-four reference costing 16. The physical model is the same full-rank 20D shared nonlinear transition at every pass; r_t is a *retained intervention subspace rank* and not a change to the number of trained model parameters.

Given precomputed pass-rank losses L_t(r), the source-only combinatorial optimizer solves

minimize sum_{t=0}^3 L_t(r_t)
subject to r_t in {2,3,4,5,6}, sum r_t = B,
and all selected passwise eigenvalue/eigengap gates hold.

Enumerating at most 5^4=625 tuples exactly solves each budget's multiple-choice knapsack, with no training stochasticity, task labels or approximate search. Lexicographic tuple order fixes numerical ties within 1e-12. The budgets are predeclared before any new model root; the smallest budget passing independent clean-KL admission is chosen.

This removes a particular **allocation constraint** in NC-07. It does not guarantee that the variable-rank solution will beat a fixed-rank teacher when nonlinear downstream behavior matters.

## 2. Source-only sensitivity proxy

On a separate derivative-discovery stream, let z_{n,t} denote the clean full BPD read factors and g_{n,t} the gradient of the final model-self score with respect to those factors. Define the passwise uncentered second moment C_t=E[g_t g_t^T] and top-r Euclidean orthogonal projector P_t(r).

The NC-08 fixed omission proxy is

L_t(r) = E[(g_{n,t}^T (I-P_t(r)) z_{n,t})^2].

This estimates the squared first-order change of the model-self score from zeroing the complementary read coordinates at a clean state. It differs from the purely spectral tail sum of eigenvalues because it also accounts for the observed activation z at that pass. Unlike a full student-training loss, it has an explicit physical local-derivative meaning and can be computed without external true-label access.

The four local losses are added to choose a tuple, which neglects cross-pass interactions and recurrent Jacobian products. Subsequent hidden states change under the actual intervention, and the discarded-score linearization can miss this nonlinear effect. The separate model-self clean-vs-retained closed-loop KL validator is therefore essential.

The first-order proxy is not inherently monotone in r on finite correlated source observations, even though the subspaces are nested. A cross term in g^T (I-P) z may cancel or amplify when one direction is removed. The exact finite optimizer evaluates every eligible rank tuple in the registered grid and never assumes monotonic loss.

## 3. The KL selector is a teacher-model fidelity proxy

For a clean teacher Bernoulli distribution p_t(x) and the true four-pass retained model prediction q_{r,t}(x), define KL_t=E_x KL(p_t||q_{r,t}) on the source-only 512-case selection set. Pinsker and Jensen yield a bound on mean predictive probability mismatch E|p_t-q_{r,t}|<=sqrt(KL_t/2). This remains a statement about model-to-model prediction agreement. There is no general implication for the externally labeled true task NLL or sign-correctness if the teacher is wrong or retained probabilities are extreme.

NC-08 chooses the smallest budget passing the preregistered NC-07 selector thresholds (final mean KL <=.027, all-pass means <=.05, p95 final KL <=.25, model decision flips <=.04, and retained clean physical write RMS >=35%). The final independent true-label test checks the absolute external task loss cost and difference from the fixed rank-four model, with a separate physical sham specificity gate.

The unweighted model-final-score derivative covariance and its top-r eigenspaces are gauge covariant under orthogonal coordinate rotations, with an eigengap assumption. They are not guaranteed to be GL(d)-covariant under arbitrary nonorthogonal reparameterizations.

## 4. Matched variable-rank sham projectors

Each selected tuple defines one rank-r_t orthogonal subspace at each of four physical write sites. A local sham proposal rotates the retained rank-r_t basis through the same registered principal angle theta in each pass toward an independent orthogonal complement. The model-specific physical retained write RMS is calculated on the derivative-discovery source only, and exactly eight of the first 2,048 seeded proposals within +/-15% of this dose are admitted.

A rank-tuple matched null controls the number of retained physical degrees of freedom and their clean-source write dose. It does not match the intervention's downstream task derivative, post-edit state distribution or future-loop Jacobian. The actual source-locked test independently reports heldout task loss and physical displacement, with full-mask closure and an orthogonal native-vs-rotated factor basis audit.

The NC-07 high-rank sham failure motivated the new **prospectively registered** 2,048-proposal search. No NC-07 result is revised or rescued by this expansion, and sham eligibility is still a mandatory eight-root primary gate in NC-08.

## 5. Statistical unit and success criterion

The experimental unit is an independently trained recurrent model seed. For eight independent development roots, a deterministic 2,000-repetition whole-root percentile bootstrap summarizes true-label task NLL and matched-sham differences. Model-root bootstrap remains an engineering aid at n=8. It does not certify a distribution-wide 95th percentile with 95% confidence, which requires an independent calibration source of at least 59 roots.

The minimum-rank objective is not a sufficient statistic for task behavior. NC-08 registers a positive developmental signal only if all roots are fully admitted, mean budget <=12.5 with at least five roots selecting B<=12, the true-label NLL upper root interval <=.05 above clean, a matched-sham margin >=.02 with positive lower interval, and upper selected-minus-flat-rank-four true-label NLL <=.02.

The within-root **DP budget-12 versus flat rank-three** NLL contrast is prespecified as descriptive. Both use exactly 12 total retained physical write directions, so it isolates allocation at constant rank cost. Outcomes remain nested inside eight model roots and never count as a second independent hypothesis test.

## 6. Wider research interpretation

BPD-PD-15–17 established a source-conditional identification mechanism for rotated sparse latent signals, including a test showing norm weighting fails under radial camouflage even when latent axes are identifiable by angular rays. NC-06 showed that porting its norm tilt to a learned nonlinear recurrent model produced no meaningful incremental causal benefit over shuffled and unweighted rank-three projectors. NC-07 then found that clean model KL preserved terminal behavior in some roots but did not supply a sufficient common rank reduction and failed rank-six sham admission.

NC-08 isolates another possible explanation: retaining equal ranks at every pass may waste physical capacity where the native recurrence has uneven task sensitivity. It tests the passwise allocation geometry with native physical BPD, source-only teacher-output validation and independently labeled future task outcomes, while retaining all negative evidence.

The next admission after NC-08 is still an independent 59-root sham-calibration cohort and EvalMath root-level power/variation preflight before 32-root scientific confirmation. None of those protected sources are generated here.
