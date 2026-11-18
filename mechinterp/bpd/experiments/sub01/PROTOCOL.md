# BPD-SUB-01: causal substrate / quotient-grain engineering protocol

**Registration date:** 2026-11-18. **Evidence class:** prospective synthetic engineering development, pending independent sham calibration and scientific confirmation. This file is committed before writing the new experiment runner or training any SUB-01 development model.

## Scientific question and scope

For a freely trained four-pass nonlinear tied recurrence, how far can we coarse-grain its state while preserving counterfactual changes to subsequent state and terminal task output? Test whether matching clean behavior conceals causal influence along fibers of a state projection. An exact invertible physical-coordinate change is the positive control. These are computational and mechanistic claims about synthetic models, not tests of sentience or phenomenal continuity.

Reference: Daniel Burger, *Meditations on Synthetic Consciousness* 6/30, November 18, 2026, distinguishes computational from physical substrate independence and leaves the required grain of mechanistic replacement open. Connection to BPD: quotient closure, local nonlinear coupling, hidden causal fibers, and transfer of interventions.

## Formal objects

The trained source is \`F(h,x)=tanh(Wh+Ux+b)\`, width 12, input width 4, zero initial state, four weight-tied passes, scalar readout \`R(h)=v^T h+c\`. Define PCA rank-r source-only projection \`P_r=U_r U_r^T\`, \`q_r(h)=U_r^T h\`, and the projected autonomous surrogate

\`G_r(z,x)=U_r^T F(U_r z,x)\`, with \`R_r(z)=R(U_r z)\`.

For fixed x and states within the reachable/counterfactual support, an exact quotient requires \`q_r(F(h,x))=G_r(q_r(h),x)\`. The necessary and sufficient fiber constancy condition on this domain is \`q_r(h)=q_r(h') => q_r(F(h,x))=q_r(F(h',x))\`. The empirical null-fiber twin residual is the RMS of \`P_r[F(h+delta,x)-F(h,x)]\` for \`P_r delta=0\`. Report projected-student vs source clean Bernoulli KL, task NLL, and terminal source-output response RMS to matched null-fiber impulses; these measure different failures.

Exact orthogonal conjugacy control: fix seeded Q with \`Q^TQ=I\`, define \`F_Q(z,x)=Q F(Q^Tz,x)\`, \`R_Q(z)=R(Q^Tz)\`; paired \`delta\` must map to \`Qdelta\`. It should commute for all passes to floating-point precision. An intentionally *untransported* impulse \`delta\` on the Q model is a negative control. Counterfactual parity does not establish consciousness, physical equivalence of devices, or personal identity.

## Precommitted generation and phase separation

- **Unit of inference:** independently initialized and trained model root. Within-root prompts, four loop steps, ranks, null directions and shams are repeated measurements.
- **Training roots:** 1200..1207 (eight engineering development units). **Excluded smoke root:** 1199. No other model seeds admitted in the runner.
- **Independent per-root samples:** 2048 training examples, 512 discovery, 512 selection and 1024 heldout. Inputs are four iid N(0,1) features with independent RNG streams per split; external binary task law is \`y = [0.85*x0 + 0.6*x1 + 0.35*x2*x3 >= 0]\`.
- **Source training:** PyTorch float32 CPU; width 12, four tied tanh passes, Xavier-initialized input/hidden matrices; BCE-with-logits/softplus; Adam lr 0.01, minibatch 128, 256 updates; model seed and sample seeds fixed. No architecture adaptation after viewing heldout.
- **Competence gate:** heldout binary accuracy >= 0.80 separately for x3>=0 and x3<0. Ineligible roots stay in the archived denominator.
- **PCA:** float64 uncentered second moment of clean h1..h4 on discovery inputs. Top eigenspaces at ranks [2,4,6,8,10]. Strict nesting up to numerical tolerance; deterministic sign and tie-insensitive projectors. Full rank 12 is a positive reference, not selected.
- **Impulses:** at the *post-first-pass hidden state*, generate fresh fixed-seed standard-normal directions and project onto null(P). Normalize each to exact physical L2 norm alpha=0.25 times RMS norm of discovery h1. Recur the remaining three passes using the altered states. Record actual displacement; every score includes clean matched x.
- **Matched shams:** eight independent Haar-random rank-(12-r) directions per rank, using the same impulse seed construction and exactly matched L2 norm alpha. They are generated before outcome inspection. Sham output RMS and signed root-level contrasts are recorded. No selecting shams by outcome.
- **Primary prospective engineering selection:** on the 512 *selection* inputs, choose the smallest of [2,4,6,8,10] with clean source-vs-projector mean final Bernoulli KL <=0.025 nats and null-fiber impulse terminal source-logit RMS <=0.050. If none qualifies record NO_ELIGIBLE_GRAIN. Freeze rank before reading heldout effects.
- **Primary evaluation contrast:** median sham terminal-logit RMS minus PCA null-fiber terminal-logit RMS at selected rank, in logit units; positive means low-variance directions are more causally disposable than same-dose random directions. Report paired root differences with 2000 whole-root bootstrap resamples, seed 120101. Scientific minimum \`delta_mech=0.03\` logit RMS. Gate for engineering signal: >=5/8 eligible roots, mean gain >=0.03, lower bootstrap 95% endpoint >0; every admitted root must pass physical dose and gauge closure. Missing ranks/roots are failures and the result stays engineering.
- **Secondary diagnostics:** rank-by-rank null-fiber next-state commutator RMS, clean KL, external-label NLL, accuracy, late-pass 1/2/3-output sensitivity, and untransported-gauge counterfactual RMS. Rank grid is one predeclared descriptive family; any exploratory rank-wise claims require Holm over five comparisons. A fitted student or nonlinear alternative is a later separately registered arm, never relabelled a current result.
- **Exact gauge admission:** clean and correctly transported-impulse max terminal logit error <=5e-5; untransported impulse negative control must produce detectable RMS >1e-5. Failure blocks mechanism interpretation.
- **Label/selection separation:** labels train the source as specified; PCA, impulse construction, model-only selection and sham selection use no heldout labels. Heldout labels are used once for descriptive task fidelity and competence.
- **Seals:** ARG-03-N, AMM-03/04/05, CQ-08 confirmation, NC-08 prospective roots, and the future SUB-01 sham/confirmation cohorts are untouched. Root IDs 1200..1207 are development only.

## EvalMath resolution boundary

The eight independent roots are too few for a distribution-free 95th-percentile, 95%-confidence sham-variation bound (59 independent calibration roots minimum for the sample-maximum rule). Strict \`EvalMath/skills/evalmath-mechinterp-power/scripts/preflight.py\` runs only after new independent development-root paired effects and same-root matched shams are available; its output, population and gate must be archived. No 32-root confirmation set is generated or released on this branch. Any observed pass at eight roots has verdict \`ENGINEERING_SIGNAL_ONLY\`. Failed competence, closure, dose, preflight or effect gates give \`ENGINEERING_NULL_OR_INELIGIBLE\`. Neither a negative result nor a small KL establishes equivalence.

## Reproducibility and future extension

Record protocol SHA-256, source commit and source hashes, deterministic sample seed schedule, Python/PyTorch/NumPy versions, every per-root receipt, root-level paired effects, all five ranks, full sham roster and a summary including explicit missing/ineligible roots. Run source-only structural unit tests first, then the excluded smoke model, then eight development roots exactly once. Planned second study: independently trained architectural replacement with explicitly learned state transport, causal-transfer loss and matched clean-behavior control, registered before inspecting outcomes.
