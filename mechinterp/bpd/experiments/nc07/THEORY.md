# NC-07 mathematics: source-only task distillation, causal rank and observational limits

## 1. Which rank does source-only clean KL select?

Let F_t be the physical shared-model update at recurrent pass t and h_t its clean hidden state. A selected pass-specific orthogonal factor projector P_{r,t} replaces the full physical shared write VU by V P_{r,t} U, and subsequent states are genuinely recomputed. The clean output distribution at pass t is p_t(x), while the closed-loop retained counterpart is q_{r,t}(x).

On an **independent model-only selection stream**, define

K_t(r) = E_x KL(Bern(p_t(x)) || Bern(q_{r,t}(x))).

This controls the similarity between the two *models' predictive distributions* on the selection distribution. The fixed selector chooses the smallest rank r in {2,3,4,6} satisfying K_4(r) <=.027, each K_t(r)<=.050, the 95th percentile of pointwise final KL <=.250, final model-decision disagreement <=.04, physical retained-write RMS >=.35 of the full clean write, and explicit eigengap conditions. All constants were frozen before NC-07 trained any model.

The selector is a discrete **minimum-capacity empirical constrained optimizer**, with a source-only proxy risk K_t(r). Candidate projectors are nested in *linear rank space* because they use eigenspace prefixes of C_t. Their closed-loop task-output KL need not be monotone with rank, since the recurrent update is nonlinear and later readouts depend on earlier edited states. Therefore evaluate every predeclared candidate on the selection source, and take the smallest eligible rank without assuming monotonicity.

## 2. What can KL certify, and what cannot it?

For a Bernoulli output, Pinsker's inequality gives total-variation distance |p-q| <= sqrt(KL(p||q)/2) pointwise. Consequently source-mean KL gives an upper bound on the source-mean probability difference by Jensen:

E|p-q| <= sqrt(E KL(p||q)/2).

This bound concerns **agreement with the teacher prediction**, not ground-truth task correctness or true-label NLL. A confident but wrong teacher can have tiny clean-to-retained KL and both predictions can fail task labels; matching a teacher near an incorrect decision boundary may also preserve incorrect probabilities. With no lower bound on retained true-label probabilities, clean-KL alone does not bound the worst-case external-label log loss.

NC-07 therefore measures held-out **external-label true NLL** after the selector is locked and independently checks its mean/upper root-bootstrap absolute cost <=.05 nats. The 95th-percentile KL and sign-flip gates are additional source-only tail guards, not a theorem that terminal behavior will pass.

## 3. Projector geometry and source-only gradient subspaces

From a clean derivative-discovery stream, g_t(x) is the gradient of the model's own last logit with respect to the full-rank BPD physical read at pass t. The pass-indexed positive-semidefinite second moment is

C_t = E[g_t g_t^T].

The rank-r orthogonal spectral projector onto its top r eigenvectors maximizes tr(P C_t) among rank-r orthogonal projectors. Its residual energy is sum_{j>r} lambda_j. This is a **model-score-gradient energy** criterion; neither that residual nor the clean-to-retained KL proxies alone proves causal sufficiency for the external true-label target.

The covariance is invariant to multiplying the final score's gradient by detached per-example signs, because (-g)(-g)^T=gg^T. NC-06 showed that weighting this covariance by an observed norm source tilt did not improve true task NLL over uniform and exact-ESS shuffled-rank-three controls, despite rank-three physical sham specificity. NC-07 consequently uses the exact unweighted covariance to concentrate on source-only *rank selection*.

## 4. Physical gauge and downstream task controls

For an orthogonal factor coordinate change R, choose V'=V R and U'=R^T U. The physical full write VU is unchanged; a projector transports as P'=R^T P R, preserving the retained write operator V P U. For a fixed physical input and output, the actual closed-loop recurrent logits should agree under both representations, up to floating numerical tolerance.

The source-locked rank selector and sham search use the native factor basis; an independent orthogonal-rotation audit tests the **same physical projector** and retained logits. This does not prove invariance under arbitrary nonorthogonal rescalings. The physical retained-write RMS is measured on clean discovery activations; exact dose matching on future edited states is a separate and potentially impossible constraint in nonlinear recurrences. Report actual test-side RMS and final per-pass KL to expose dynamics-induced dose changes.

## 5. Causal sham comparison and root-level inference

A selected rank-r projector is compared with eight same-rank local projectors whose retained clean physical write RMS is within 15% of its source-only dose. All shams are chosen from independent, fixed-seed local rotation proposals without task-label access. Their held-out true-label NLL forms a physical source-matched specificity comparator.

The selector may choose fewer physical dimensions than fixed rank four and still cause larger test NLL. The protected **mean + bootstrap upper cap** selected NLL minus clean <=.05, plus paired selected minus fixed-rank-four upper cap <=.02, guard against this. The sham median-minus-selected NLL mean must exceed .02 with a positive bootstrap lower bound.

Eight new independent trained model roots form eight sampling units for the 2000-draw root bootstrap. Prompts, recurrent passes, shams and projector variants are nested paired observations. The bootstrap is a developmental uncertainty summary and is not calibrated to a 95th percentile/95% confidence sham tolerance criterion. A nonparametric 95/95 bound at the 95th percentile requires at least 59 independent calibration roots, with a separate EvalMath root-level mechanism variation and power preflight before confirmation.

## 6. Relation to Compactification / LoopedTransformers

The protected causal compactification framework expresses the task-conditioned rank of DT restricted to ker DP for a chosen public observer P. NC-07 tests a special empirical source-only **rank budget** and task-fidelity proxy. Its C_t projector is a rank-constrained physical write operator whose causal competence is evaluated by actually removing complementary writes through later nonlinear loops. It does not identify the exact minimal task quotient, and its KL selector can be conservative.

LOOPSCOPE-08's nonconservative native hidden-state dynamics show why a decreasing scalar energy cannot be equated with the future task gradient. NC-07 never assumes such equivalence. It scores four-pass actual terminal loss, along with readout KL by pass, rather than claiming the model is an optimizer of the discovery potential.

## 7. Evidence classes

Exact mathematics: covariance eigenspace optimization, orthogonal gauge transport, Pinsker and Jensen bounds on predictive probability disagreement.

Prospective engineering: source-only selected rank, independently trained source roots, actual BPD recurrent keep-only effects, physically dose-matched shams, absolute true-label fidelity.

Future scientific confirmation: independent nonparametric sham-calibration roots, power/variation preflight and untouched confirmation model families, all unrun and sealed in NC-07.
