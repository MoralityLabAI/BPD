# BPD-NC-07: discovery-only KL and physical-write adaptive rank in a learned recurrence

**Prospective source registration, November 18, 2026.** This protocol is committed BEFORE NC-07 source code or any new NC-07 model training. Its eight independently trained development roots, ranks, source-only selection rule, absolute task constraints, sham controls and verdict are immutable after first execution. A separate future confirmation protocol, provisionally NC-08, needs fresh independent calibration and confirmation roots.

## Why this is the next test

NC-04 found a rank-two causal projector that beat physical-dose shams but incurred 0.123414 nats of extra final true-label NLL, above the frozen 0.05-nat absolute-fidelity cap. NC-05 showed rank four at gain 1.0 met fidelity/sham development gates on eight new trained 20D nonlinear recurrent roots, with mean final true-label keep cost 0.021219 nats and adjusted upper cost 0.045398. NC-06 registered a rank-three norm-conditioned versus uniform versus shuffled training comparison on eight further new roots. Rank three met absolute fidelity with mean keep-only NLL cost 0.021158 and upper 95% 0.033707; local physically dose-matched sham advantage was 0.074525 nats. Norm conditioning added only 0.000357 nats relative to uniform and 0.000336 relative to shuffled, failing its preregistered 0.01 practical-effect gate. The uniform rank-four reference had keep-only cost 0.005384 nats.

PD-15, PD-16 and PD-17 explained why norm conditioning successfully identifies the synthetic sparse-linear identity source but fails to transfer when the source norm lacks a meaningful cardinality signal. An additional lesson from LOOPSCOPE-08 is that an energy-like scalar and recurrent-gradient alignment need not make interventions improve future task behavior; NC-07 therefore uses actual closed-loop physical updates and held-out external labels.

**NC-07's new hypothesis**: a rank chosen by the *teacher's clean-vs-retained output KL* on an independent discovery-only validation split can retain external task fidelity while using fewer physical write directions than a fixed rank-four reference. Selection is source-only and model-score-based; external task labels are excluded until independent evaluation.

## Source, model and separations

- Exactly **eight NEW independent engineering model roots 700 through 707**. Excluded engineering smoke seed **699**. Historical NC-04 400–407, NC-05 500–507, NC-06 600–607, and all earlier source family test roots remain untouched.
- Exact inherited DenseLooper: fully trainable 20D shared physical mix matrix used across four nonlinear tied recurrence passes, learned encoder, preweight, scalar readout. One physical BPDModel/TiedComponentLinear wrapper with rank 20, ablation zero and sphere rescale disabled.
- Per root, train on 2,048 independent source inputs from sample_x(seed,1,N) and external_labels ONLY during supervised model training. Exactly 384 Adam updates, batch 128, learning rate 0.005, with the original seed +1,737,701 minibatch RNG.
- Set aside **1,024 derivative-discovery samples (split 2)**, **512 rank-selection samples (split 4)**, and **1,024 final independent task evaluation samples (split 3)**. The latter labels and logits are *never* consulted for projector discovery, selection, sham construction, or candidate rejection.
- Admit roots only when intact-model accuracy >=0.85 in BOTH external task contexts and absolute mix density >=0.90, four-pass native full-mask target/BPD output max logit error <=5e-5, all-pass mean binary KL <=1e-7 nats. Non-admission stays in the eight-root intent-to-treat denominator. Each root is ONE independent unit; no splitting test cases/passes/shams into pseudo-roots.

## Frozen estimator, candidate ranks and source-only selection

Let a_{n,t} be the clean physical preweight activation and g_{n,t} the *model-final-score* gradient of the pass-t BPD read factor, computed from the model's own last logit with detached sign. No true-label sign enters the covariance; the uncentered gradient outer product is invariant to per-example sign reorientation. Compute float64 C_t = mean_n(g_{n,t}g_{n,t}^T) from the 1024 derivative-discovery samples, with NO norm weighting (the incremental NC-06 norm conditioning failed).

Freeze the nested rank grid **R=(2,3,4,6)**. For each r, P_{r,t} is the top-r spectral projector of C_t for all 4 physical passes. Require eigenvalue lambda_r>1e-12 and eigengap lambda_r/max(lambda_(r+1),1e-12)>=1.001 on ALL four passes. Rank failure marks that candidate ineligible without replacement; report all spectral data. Factor gauge is the exact orthogonal physical basis from the existing BPD NC-05 instruments; projected physical write is V P_r U and is orthogonally gauge covariant.

On the 512 selection examples only, run each candidate r through all four genuinely closed-loop physical BPD override-read passes with retained physical gain alpha=1.0. Compare the teacher's *clean* pass-by-pass logits to each retained model using binary KL(clean||retained), including pass-specific KL, final mean KL and 95th-percentile per-example final KL. This is a model-output self-fidelity signal, *not external true-label NLL*. Also record sign-flip fraction of model final decisions and retained clean physical-write RMS relative to all-component clean write RMS on the 1024 derivative-discovery samples.

**Frozen source-only admission for candidate r**, with no tuning on the exposed eight new models:
- Mean final selector KL(clean||retained) <= **0.027 nats**.
- Every one of the four passwise mean selector KL values <= **0.050 nats**.
- Selector 95th-percentile per-example final KL <= **0.250 nats**.
- Clean-versus-retained model final-logit sign-flip fraction <= **0.040**.
- Discovery retained clean physical-write RMS / discovery full-write RMS >= **0.350**.
- Numerical rank eigengap and exact full-mask closure pass.

Choose the **smallest qualifying rank** in (2,3,4,6), ties predetermined by rank order. When none qualifies, abstain and count the root as non-admitted; never substitute the true-label-best or a higher rank after inspecting heldout outcomes. The historical NC-05/NC-06 ranks and this cutoff motivated the design but all new NC-07 source roots are independent.

Two fixed controls are evaluated on the same final heldout stream: **uniform rank three** and **uniform rank four** projected physical writes from the same nested C_t eigenbasis. These are descriptive/paired predeclared task comparators, never used to select the adaptive rank. The all-full target remains the clean oracle. Rank two and rank six are also evaluated on heldout only when the adaptive selector chooses them; do not score every candidate on test or collect hidden labels for exploratory choice.

## Causal shams and gauge controls

For the selected rank, preselect exactly **eight independent discovery-only, equal-rank local physical projector shams**, from **1024 seeded proposals** at principal-angle grid {0.45,0.65,0.85,1.05} radians relative to selected eigenspace, matched within +/-15% of its retained clean physical write RMS as measured on derivative-discovery activations. The proposal RNG is root+7,071,701+rank*139. Stop at the FIRST eight qualifying proposals; no test-side oracle or outcome filters. If fewer than eight qualify, mark that root ineligible and retain all original candidate/proposal diagnostic data.

The physical counterfactual is a full-loop native BPDModel.intervene(mode=override_read) at all four tied passes, with no residual bypass. Each edit changes the future hidden state. For every root, additionally compute the selected projector under one independent orthogonal factor rotation and transport P_rot=R^T P R; require max physical operator relative discrepancy <=1e-4 and retained final logit maximum absolute difference <=5e-5 on 64 selection inputs. The initial full-mask BPD target must agree with the original teacher at all four passes.

No extra residual or post-hoc per-case rescaling is permitted. The sham matching criterion is discovery clean *retained* write RMS, while heldout physical displacement and task-gradient dose mismatches are recorded as a limitation.

## Frozen eight-root development test

The selected adaptive projector is evaluated on the **1,024 independent final examples per root** using external_labels only after source selection and shams are frozen. Record actual external-label final NLL, accuracy, clean-vs-retained KL by all four passes, physical write RMS and NLL of each of the eight matched shams. Record paired root scores for fixed uniform-r3 and fixed uniform-r4 on exactly the same heldout examples.

A POSITIVE *engineering-development* signal requires all of:

1. **8/8 independently trained roots** pass task, source, closed-loop full-mask parity, rank selection, selected-rank eigengap, eight sham admission and rotated physical-map parity.
2. Selected ranks have **mean <=3.5**, with **>=4/8 roots selected at rank <=3**. This is a real dimensional saving against fixed rank four.
3. Mean selected kept-minus-clean external true-label NLL <=0.050 nats, and **95% upper root-bootstrap <=0.050 nats**.
4. Mean selected matched-median-sham minus selected heldout true-label NLL >=0.020 nats with **95% bootstrap lower >0**.
5. The **95% bootstrap upper** on selected minus fixed rank-four true-label NLL across the eight paired roots <= **0.020 nats**. This bounds the accuracy cost of saving at least one physical dimension.

Use a deterministic **2000-draw root-cluster percentile bootstrap**, RNG seed 707007, resampling exactly eight trained model roots. Do not treat shams, model heads, passes, or heldout examples as independent roots. Publish signed per-root outcomes, every sampled selected rank and the complete original source/model receipt. A failure is **ENGINEERING_NULL_OR_INELIGIBLE**, without opening any scientific confirmation or tuning this first-run threshold.

The rank-three fixed control establishes how the selected-rank policy behaves relative to NC-06's feasible rank-three source family. It has no additional confirmatory significance claim. If every selected rank is four, the compression gate is FAIL even if absolute fidelity and sham specificity pass. If selected ranks are below four but true-label NLL worsens, preserve the FAIL. This makes the experiment falsifiable.

## Scientific admission and future work

This is an 8-root DEVELOPMENT study using one synthetic dense 20D nonlinear recurrence family. Its root-bootstrap is an engineering interval, not independent scientific confirmation. Before an eventual confirmation: pre-register a separate final adaptive selection policy in a new prospective protocol, obtain at least **59 genuinely new independent matched-sham calibration model roots** for the 95th-percentile/95%-confidence nonparametric tolerance bound, run the EvalMath mechanistic-variation/power preflight, and draw at least **32 genuinely new confirmation roots** only after passing it. Those future populations and unrelated ARG-03-N/AMM-03 protected sets are sealed and ungenerated in NC-07.

A positive engineering result would establish only that clean-model KL can support smaller causal physical subspaces under this trained synthetic recurrence and heldout task. It is not a finding of general SAE/neuron identities, natural pretrained TRM causal circuits, GL(20)-gauge invariance, or terminal model-gradient descent.
