# BPD-SUB-07: depth-necessary six-token recurrent causal quotient

**Prospective engineering registration. November 18, 2026.** Commit this protocol before SUB-07 source, source-only tests, or any SUB-07 development-root training. Pre-registration follows excluded oracle enumeration and offline engineering design pilots, disclosed here. Prior SUB-05 engineering PASS and SUB-06 engineering FAIL, and their prospective scientific NOT READY statuses, remain untouched.

## Exact oracle and depth-necessity hypothesis

Each input is a **length-six** sequence of tokens in {X,S,C,R}, one-hot in R^4, applied to a two-bit state (a,b)∈F₂² initialized at **(0,1)**. Token actions are:
- X(a,b)=(a⊕1,b): flip first bit;
- S(a,b)=(b,a): swap bits;
- C(a,b)=(a,b⊕a): first controls second;
- R(a,b)=(a⊕b,b): second controls first.

Every operation is invertible, and they do not commute. Execute them **left to right** for six steps; the externally defined label is the **final second bit b₆**. Enumerate all 4^6=4096 ordered sequences lexicographically and calculate exact labels with a pure-Python oracle independent of torch/model weights.

Before registering source-model experiments, independent oracle enumeration on the uniform full-sequence population yielded:
- positive label frequency **2291/4096 = 0.559326171875**, entropy **0.9898206326381731 bits**;
- Bayes-optimal accuracy using first two tokens only **0.585693359375**, residual conditional label entropy **0.9694865132073917 bits**;
- using first four tokens only **0.667724609375**, residual conditional entropy **0.8710851970360994 bits**;
- using only last token **0.559326171875**;
- using token-count histogram (permutation-invariant bag) **0.570068359375**, residual conditional entropy **0.9581228215762154 bits**.

The code must independently reproduce these enumerated values exactly (integer accuracies/denominators, entropy tolerance 1e-12) before training. A named counterexample to token-count sufficiency is sequence (X,X,S,S,R,C) versus (X,X,S,S,C,R), which share counts but have opposite oracle labels. The exact Bayes ceilings show that later tokens and temporal order contain essential label information for this distribution. Claims about model-specific internal depth are contingent on competence and causal interventions.

## Immutable dataset split and experimental unit

Use the complete 4096 unique sequences once. With Python/Torch-independent NumPy PCG64 \`np.random.default_rng(246810).permutation(4096)\`, allocate **2048 unique TRAIN**, **512 unique DISCOVERY**, **512 unique DOSE-CHECK**, and **1024 unique HELDOUT** sequences in this exact order. The partitions are immutable across all model roots, with zero shared sequences. Train samples are drawn with replacement only from the 2048 TRAIN sequences; the other sequences never train the teacher. Source-discovery tensors and source-fit optimizer use DISCOVERY only, frozen physical norm calibration uses DISCOVERY, DOSE-CHECK only verifies geometry and actual dose, and HELDOUT is queried once for frozen candidates and task endpoints.

The independent statistical unit is the model seed, not a sequence/prefix/fit/loop. Shared dataset across model roots is acknowledged; any generalization to new sequence distributions is an independent future experiment.

Prospective roots are **1800..1807**, 8 independent freshly initialized source models. Excluded **1799** is a source-only smoke model. Disallow every other seed in the runner. Historical SUB-01..06 roots and protected ARG-03-N/AMM/CQ-08 model/tests remain sealed.

## Trained source architecture and competence

This study deliberately **changes the physical source transition family** from prior 12D tanh to a six-pass **32D GRUCell with input width 4**, shared weights at each pass, hidden initial state h₀=0, and linear scalar final-logit readout at h₆. The same cell \`F(h_t,x_(t+1))\` is reused six times; each token input changes. Train every model from scratch with PyTorch CPU float32, \`torch.manual_seed(model_seed)\`, Adam lr **0.003**, minibatch **128**, exactly **1200** updates (no early stopping or per-root retry), deterministic minibatch generator seed \`model_seed+44\`, binary cross entropy with logits. Input and outputs are one-hot sequence and oracle bit labels. Do not train a separate learned position embedding or clock tag.

**Competence admission:** each fresh model must score **>=0.90 final heldout accuracy** on the fixed 1024 unseen sequences, **>=0.85 final accuracy within each last-token family** (four groups), and >=0.85 final accuracy on true-label-zero and true-label-one subsets. Preserve all root failures, with no replacement. The exact uniform oracle depth necessity must pass the enumerated bounds above; record the source head's prefix-two and prefix-four predictions *descriptively*. A prefix-only head cannot exceed the population oracle ceiling even with unlimited fitting. Learned bag/last-token baselines may be added later as separate diagnostics; the exact Bayes upper bounds are already stronger certificates than any model with those restricted inputs.

Excluded offline engineering-design probes (seeds 1797,1798,1799, not the eight registered roots) were used to choose six passes, GRU width 32 and 1200 updates after narrower 12D tanh recurrences failed to generalize well. They provide no prospective evidence or effect-size information. The protocol is frozen **before** 1800..1807 training.

## Source-only Jacobians and candidate projection families

Set rank **r=20** for six projectors \`P_1,...,P_6\` in the original physical 32D hidden-state coordinates, leaving 12 discarded dimensions. The source is a driven tied recurrence, so source Jacobians are \`D_t(x)=∂_h F(h_t,x_(t+1))\` for t=1..5. Use the **first 64 sequence rows of DISCOVERY** for computing \`D_t\` via \`torch.func.vmap(torch.func.jacrev(...,argnums=0))\` in float32, then cast to float64 and detach. Use all 512 DISCOVERY sequences to compute \`C_t=E[g_tg_tᵀ]\`, where \`g_t=∂(terminal logit)/∂h_t\`, t=1..6, using the frozen source GRU. The 64 Jacobian states are preselected by fixed input order, with no adaptive filtering. All output/projector training is source-only; no HELDOUT labels are used.

Fit a **stationary P_static** of rank 20 from the top eigenvectors of \`mean_t C_t\`, then exactly 80 float64 QR/Adam updates at lr .02 minimizing \`A_static/A_initial + .5 B_static/B_initial\`, where

\`A_static=mean_(x,t)||P D_t(x)(I-P)||_F²\`, and
\`B_static=mean_t tr((I-P)C_t)\`.

Initial normalizers are clamped below at 1e-12. Freeze P_static and its final float64 orthonormal basis. All aligned and wrong-clock bundles receive **exactly the same** six copies of this basis and common source-only normalizers \`a0,b0\` calculated from static P on correctly ordered D and C.

Each source-fit bundle has rank20 projector \`P_t=Q_tQ_tᵀ\` (QR basis). Fix for six slots:

\`A=mean_(x,t=1..5) ||P_(t+1)D_t(x)(I-P_t)||_F²\`,
\`B=(1/6)Σ_t tr((I-P_t)C_t)\`,
\`S=(1/(5*2r))Σ_t ||P_(t+1)-P_t||_F²\`,
\`L=A/a0 + .35 B/b0 + .50 S\`.

Run **120 float64 CPU Adam/QR updates at lr .02** on all six Q_t, no early stop or refit. All fitted arms use the same exact initialization, data and optimizer budget. Record complete loss trajectories, true-clock and fitted-clock source A/B/S, eigen/projector geometry and CPU dependency versions.

Correct arm keeps original source ordering. Seven fully independent wrong-clock fits mislabel the **entire input D_t and output C_t arrays** with these registered zero-based maps:

\`CLOCK_MAPS=[(1,2,3,4,5,0),(2,3,4,5,0,1),(3,4,5,0,1,2),(1,0,3,2,5,4),(2,1,0,3,4,5),(0,2,1,3,4,5),(5,4,3,2,1,0)]\`

\`STEP_MAPS=[(1,2,3,4,0),(2,3,4,0,1),(3,4,0,1,2),(1,0,3,2,4),(2,1,0,3,4),(0,2,1,3,4),(4,3,2,1,0)]\`.

Every clock and step map is a true permutation preserving its source tensor multiset. A wrong-clock bundle stays in its **native fitted order** during heldout evaluation on the original token sequence. Fixed post-fit permutations of the aligned bundle are secondary source-selection controls. No arm may be chosen using HELDOUT performance.

## Physical dose, causal metrics and controls

Physical amplitude \`alpha=0.25 sqrt(mean_discovery ||h_1||_2²)\`. At each post-pass t=1..5, sample independent preseeded raw Gaussian z for each HELDOUT sequence, using torch generator \`generator_seed=807000+source_seed*4001+split*17011+postpass*101\`. Use the **same raw z** for all source-fitted arms and secondary controls, project \`delta=alpha (I-P_t)z/||(I-P_t)z||\`, and inject after a clean recurrent step. Rerun **every remaining physical GRU step with its actual next distinct token**. The next-state closure endpoint is the Euclidean RMS of \`P_(t+1)[F(h_t+delta,x_(t+1))−F(h_t,x_(t+1))]\`. The pooled C is \`sqrt(mean_t C_t²)\`. Logit RMS, final-state change, true-label correct→wrong and wrong→correct counts, and state radii remain separately recorded.

The primary per-root contrast is

\`d_i=min(C_static, C_wrong0,...,C_wrong6) - C_aligned\`,

where all eight controls are source-fitted independently and identical hidden dimensionality is used. Exact rank32 projector identity is an algebraic zero-null-fiber control. An orthogonal physical coordinate gauge must preserve clean source logit and all five projected counterfactual next-state and terminal-logit responses under correctly transported state/projectors/delta to max 5e-5, while a deliberately untransported delta must have RMS logit discrepancy >1e-5. Actual dose and null projection error <=5e-6. Every fixed source arm's geometry must satisfy projector idempotence/symmetry max <=1e-5, rank trace abs <=1e-4, and finite source gradients.

Clean autonomous clocked students apply each fitted P_t immediately after GRU step and feed that state into the next physical GRU step with its own token. Report source-to-student Bernoulli KL, true-label NLL, external correctness and full-rank clean source-logit max parity <=5e-6. Fidelity gate on model-root means: aligned clean KL <=0.040; aligned minus stationary clean KL <=+0.015; aligned minus stationary true-label NLL <=+0.030; aligned minus stationary pooled terminal-logit intervention RMS <=+0.040. Competence admission and complete precommitted arm roster are mandatory.

## Engineering decision and statistical resolution

Prereigstered engineering success requires **all 8/8 source roots competent and fully admitted**, >=**7/8** positive root primary effects, **mean root gain >= +0.010 hidden-state Euclidean RMS**, and 2000-draw independent-root bootstrap (seed 180707) with 95% lower endpoint >0, *plus* every autonomous clean-task fidelity gate. Otherwise record \`ENGINEERING_NULL_OR_INELIGIBLE\`. Preserve every root, wrong-clock arm and negative outcome. Any discovery-only loss reduction is a source-fit diagnostic and cannot rescue a failed heldout causal gate.

Predeclare one primary comparison, all seven wrong fits included inside the strongest-null root statistic. Source/sequence examples, five passes, multiple readouts and shams are repeated within a trained model root. The exact full-sequence oracle depth certificate is an algebraic task property, not a model causal effect.

After original eight-root receipts, run the pinned standard-library EvalMath preflight script \`MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a\`, using bounded \`b(d)=d/(1+|d|)\`, minimum \`b(.010)=0.009900990099009901\`, target \`b(.025)=0.024390243902439025\`, measurement resolution .002, alpha .05, comparisons=1, planned confirmation capacity=64, and per-root matched nulls \`b(max(0,C_static−C_wrong_j))\`. Strict \`REDESIGN_OR_PILOT_ONLY\` is expected because **eight roots cannot calibrate the 95/95 sham-tail quantile (>=59 independent roots required)**. This engineering study must not generate a SUB-07 59-root null-calibration or future scientific confirmation cohort. Protected historical ARG-03-N, AMM-03..09, CQ-08 and SUB-01..06 sources remain untouched.

## Interpretation

The finite oracle certifies that the task depends on future tokens and sequence order. Passing the learned-model causal gate would support a source-specific time-indexed causal subspace advantage against equally source-fitted wrong-clock candidates in a six-pass driven recurrent computation. This is a synthetic model claim. It does not establish a global quotient for all states, a material physical substrate interchange, biological neuron replacement, phenomenal consciousness, or continuity of identity.
