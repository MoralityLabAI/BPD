# BPD projector transfer under a limited target gradient budget

**Research addendum, 18 November 2026.** Evidence class: synthetic engineering on eight independent trained model pairs. This note applies the original NC-12 result to BPD and the looped-transformer interpretation program. It accompanies the existing manuscript and preserves the original experiment records.

## 1. Claim added to the BPD argument

The registered NC-12 projector-transfer procedure failed every behavioral gate across eight technically admitted model pairs. At eight target gradient examples, estimating the projector directly from the target reduced shared-task negative log-likelihood by **0.072721 nats** relative to source-assisted transfer under the registered physical-dose calibration. The original verdict is **`ENGINEERING_NULL_OR_INELIGIBLE`**, with zero qualifying low-query budgets. [1, 2]

A claim that source-derived components reduce the target information needed to identify a useful intervention therefore needs a target-estimated comparator at the same available target budget. BPD's source-local fidelity is evidence about the measured model and intervention. Reuse in another model adds an empirical requirement: the transferred intervention must retain behavior on the receiving model under an independently evaluated continuation. NC-12 supplies a negative result for its fixed orthogonal transport estimator in a small recurrent task family.

The observed source-local final divergence, **mean KL(clean A || kept A) = 0.0058004434 nats**, is averaged over both task contexts. The primary transfer endpoint uses B's external task labels on the shared context. These quantities have different references and populations; they enter the argument as separate measurements. [1, 3]

## 2. Experiment and information access

Each root from **1200 through 1207** specifies a pair of separately initialized and trained 20-dimensional `DenseLooper` models. Each executes four nonlinear recurrent steps, with the same physical `mix` matrix used at every step. Their external task agrees when `x4 > 0`, where the answer is `sign(x0)`; the other context uses `sign(x1)` for A and `sign(x2)` for B. Each model receives its own 2,048-case training stream and 384 Adam updates. The independent inference unit is one trained A/B pair. [2, 3]

The intervention retains a rank-four read projector at every pass through the native BPD `override_read` interface, with zero residual bypass. Source A's projector is estimated from the uncentered second moment of its final-score gradients on 1,024 discovery inputs. At each target budget, B's direct estimate uses the top four directions of its own unit-normalized gradient second moment. Source discovery covers both contexts, while target calibration selects the shared context. That prospectively specified asymmetry belongs to the tested comparison. [2, 3]

For a pass-specific pair of unit-normalized gradient matrices, the frozen transfer rule is

$$
R_{q,t}=\operatorname{polar}\!\left(G_{A,q,t}^{\mathsf T}G_{B,q,t}/q+0.05R_{\mathrm{act},t}\right),
\qquad
P_{A\to B,q,t}=R_{q,t}^{\mathsf T}P_{A,t}R_{q,t}.
$$

The coefficient 0.05 was fixed before NC-12 data. With fewer than 20 gradient rows, the unregularized cross-moment has rank at most q. The activation prior supplies a rule for completing the alignment in unobserved directions; singular-value admission remains necessary for the regularized matrix. [2, 3]

| Information or operation | Source-assisted transfer | Target-estimated projector |
|---|---|---|
| B score-gradient calibration inputs per arm | q = 4, 8, 16 or 32 | The same q inputs |
| B activation calibration inputs | 128, used in the activation prior | The same 128 made available; unused by the projector estimator |
| Source information used | A's 1,024 discovery gradients, plus activation traces on 128 paired calibration inputs with q of those inputs queried for gradients | Projector estimation uses B gradients |
| Physical-dose calibration | 1,024 label-free B inputs; reference gain 1 | Same B inputs; scalar gain relative to the transfer reference |
| External labels for scoring | Frozen shared-context B task labels | Identical labels and evaluation inputs |

The q arms are nested prefixes of **32 unique B gradient calibration inputs**, each providing gradients at all four passes. Those 32 inputs are a subset of the 128 activation cases. The protocol makes the paired activation cases available to both methods; direct subspace estimation consumes B gradients alone. Discovery, calibration, dose and final evaluation use separate input streams. The target query budget describes the projector estimators: the full execution additionally computes a 1,024-gradient B reference after primary choices are fixed, and a 32-input zero-head gradient diagnostic on a modified B model. The 1,024-gradient reference is descriptive and carries a larger target information budget. [2, 3]

The original label **B-only** applies to subspace estimation. Its deployed scalar gain is

$$
\alpha_{B,q}=\frac{D_B(P_{A\to B,q})}{D_B(P_{B,q})}.
$$

Thus the reported arm is a **target-estimated projector under a common, source-transfer-referenced dose calibration**. The full comparison has access to target model activations and weights for that calibration. [3]

Each pair has 1,024 evaluation inputs. The shared-context counts, in root order, are **498, 544, 531, 519, 541, 523, 489 and 492**, totaling **4,137** out of 8,192 inputs. All reported means weight the eight roots equally. Cases, recurrent passes and nested budgets remain repeated observations within a root. [1, 5]

## 3. Original results and decision

Lower task NLL is preferable. Define the primary transfer advantage as

$$
\Delta_q=\operatorname{NLL}_B(P_{B,q},\alpha_{B,q})-
\operatorname{NLL}_B(P_{A\to B,q},1).
$$

A positive value favors source-assisted transfer. The registered gate requires the lower confidence endpoint to exceed **+0.01 nats**. All intervals below preserve the original 4,096-resample whole-pair bootstrap, using nominal **99.6875%** cell coverage for the declared 16-comparison Bonferroni family. [1, 2, 4]

| Target gradient inputs q | Target-estimated NLL | Transfer NLL | Delta q, nats | Original nominal interval |
|---|---:|---:|---:|---|
| 4 | 0.187580 | 0.249602 | -0.062022 | [-0.125637, +0.025912] |
| 8 | 0.165307 | 0.238029 | -0.072721 | [-0.130338, -0.015052] |
| 16 | 0.151345 | 0.230822 | -0.079477 | [-0.150495, -0.018228] |
| 32 | 0.148804 | 0.243815 | -0.095011 | [-0.165446, -0.050870] |

Direct target estimation has lower mean NLL at every budget. The intervals at q = 8, 16 and 32 lie wholly in its favor; the q = 4 interval crosses zero. The clean B shared-context mean NLL is **0.123181**. Its separate 1,024-gradient descriptive reference has mean NLL **0.144165**. [1, 5]

The remaining twelve registered contrasts are shown in full. The first two columns require lower endpoints above +0.01 and +0.02 nats respectively. Absolute fidelity requires an upper endpoint at most +0.05 nats.

| q | Activation-only minus transfer: mean [interval] | Median local sham minus transfer: mean [interval] | Transfer minus clean B: mean [interval] |
|---|---|---|---|
| 4 | +0.129893 [-0.074624, +0.333582] | +0.016237 [-0.065582, +0.118807] | +0.126421 [+0.030961, +0.243486] |
| 8 | +0.139448 [-0.044873, +0.328774] | +0.007514 [-0.057446, +0.098932] | +0.114848 [+0.047724, +0.201922] |
| 16 | +0.150839 [-0.044675, +0.353139] | +0.027123 [-0.016584, +0.075716] | +0.107641 [+0.041205, +0.205112] |
| 32 | +0.135291 [-0.072683, +0.338753] | +0.022960 [-0.018563, +0.091925] | +0.120634 [+0.056951, +0.212934] |

**All sixteen directional gates failed.** Each budget had eight technically admitted pairs, so the result arose after successful technical admission. The primary rule selected the smallest passing q in {4, 8}; its output was `null`. The larger q values belong to the same declared comparison family and have a descriptive role in the low-query decision. The cyclically permuted gradient alignment remains an additional descriptive control. [1, 2, 4]

Two earlier studies motivated this test. Their original reports supply context with distinct protocols and fresh root populations:

| Study | Test and recorded result |
|---|---|
| NC-10, roots 1000–1007 | Activation Procrustes transfer had a +0.177406-nat shared-task deficit against B's own projector, with nominal 98.75% interval [+0.050955, +0.338728]. Its registered verdict was `ENGINEERING_NULL_OR_INELIGIBLE`. |
| NC-11, roots 1100–1107 | At 128 target gradient inputs, Jacobian transfer had a +0.129189-nat deficit against the target-calibrated projector, interval [+0.006393, +0.310443]. All four registered gates failed. |

NC-12 addresses the lower-query regime with a regularized aligner. Changes in estimator and discovery rules make the cross-study sequence a record of separate tests. A pooled learning curve would require a common design. The NC-11 values above use its report's explicit contrast definitions; the first two column names in its original per-root CSV reverse the sign of their stored contrasts. [6, 7]

## 4. What the physical controls establish

Using row-vector prewrite activations z and U_B = W_B transpose, the registered dose is the retained physical write RMS along clean B trajectories:

$$
D_B(P)^2=\frac{1}{4\cdot20}\sum_t
\operatorname{tr}\!\left(M_tP_tU_BU_B^{\mathsf T}P_t\right),
\qquad M_t=\mathbb E[z_t^{\mathsf T}z_t].
$$

The direct, activation and permuted comparators receive one frozen gain across all four passes, admitted within [0.5, 1.5]. Eight local rank-four shams per budget are selected in fixed proposal order from 2,048 candidates, with retained write RMS within 15% of the transfer reference. Their repeated draws are clustered within the trained pair. [2, 3]

The receipts show main-comparator retained RMS matching within **1.12e-16**. Closed-loop displacement has a separate value: at q = 8, mean recorded clean-to-intervened write displacement RMS is **0.416206416** for transfer and **0.323368335** for the target estimate. Equal retained energy leaves the direction relative to the clean write and the future trajectory unconstrained. The experiment's matched-dose conclusion is limited to its specified retained-energy instrument. [3, 5]

A rotation of the factor coordinates supplies an implementation check. For a fixed physical factorization, the substitutions

$$
V'=VQ,\quad U'=Q^{\mathsf T}U,\quad P'=Q^{\mathsf T}PQ,
\qquad Q^{\mathsf T}Q=I
$$

give V'P'U' = VPU by direct multiplication. The q = 32 gauge check recorded maximum relative physical-operator error **6.39158e-8** and maximum closed-loop logit discrepancy **2.32458e-6**. Its scope is coordinate covariance within B. Cross-model transfer is assessed by the behavioral endpoints in Section 3. [1–3]

The negative result leaves the cause of the transfer deficit unresolved. Orthogonal alignment restricts the allowable correspondences. Discovery conditions also differ between estimators, and each intervention passes through nonlinear recurrence. Attributing the complete pipeline's failure to a particular explanation would require another controlled test.

## 5. Consequences for BPD and looped-transformer interpretation

The [optimal-basis theory](../../mechinterp/bpd/theory/OPTIMAL_BASIS_AND_MASKS.md) identifies a mathematical object under an exact factorization and a specified activation law. In the [LoopedTransformers manuscript, Proposition 10](https://github.com/MoralityLabAI/LoopedTransformers/blob/b490542bf81beb7e11c98c2e2fa32867f0ff77f7/papers/shared_weights_dark_fibers/PAPER.md), pooling across passes can contribute exact-support witnesses for basis identification. NC-12 concerns transferring a finite-sample rank-four projector between independently trained models. Its evidence restricts the tested reuse procedure, while the theory retains its stated hypotheses and identified object. [8, 9]

The looped-transformer component ledger assigns a visible share and a delayed gain to a physical write. Those descriptors depend on the receiving model's observer and continuation horizon. NC-12's final-score gradients refer to a fixed four-pass horizon. Applying the transfer argument to TRM or a pretrained transformer therefore calls for checkpoint-specific measurements at declared edit sites, with target-local estimation supplied the same target information. The present evidence population consists entirely of the synthetic DenseLooper pairs. [2, 9]

For future source-assisted reuse claims, this addendum adds the following reporting requirements:

| Claim | Evidence required at the receiving model |
|---|---|
| A source component saves target queries | A target-estimated comparator given the same target access, with every calibration and diagnostic channel disclosed |
| A transported subspace preserves behavior | Complete intervened rollouts scored against held-out external outcomes at the registered horizon |
| The transfer has causal specificity | Matched physical controls plus a registered response-sensitive null capable of distinguishing the proposed mechanism |
| A mechanism transfers across checkpoints or depths | Independent trained roots and a prospective family covering the tested checkpoints, depths and intervention choices |

A possible follow-up would fit correspondence using finite intervention responses over declared continuation horizons. That is a research proposal. Its source intervention budget and target access would need to be fixed before new data, together with a fresh sham calibration. The current receipts leave the usefulness of that alternative open.

## 6. Resolution, selection exposure and audit limits

The eight roots support an engineering uncertainty estimate. At the registered tail probability, each bootstrap tail contains about **6.4 of 4,096 draws**. Bonferroni allocates the nominal comparison error budget; reliable simultaneous coverage also depends on the underlying interval procedure. With eight pairs, that calibration remains a limitation. The result establishes the original gate outcome and the measured directional deficits within the tested family. Equivalence would require a separately specified margin and suitable precision.

The independent sham population has eight roots. For a prespecified per-root magnitude sampled independently from one fixed null distribution, the sample maximum reaches a distribution-free 95% confidence bound on the population's 95th percentile at **59 roots**, since coverage is at least 1 - 0.95^n. The scientific floors remain +0.01 nats for the target and activation contrasts, +0.02 for the sham contrast, and +0.05 for the fidelity allowance. A calibrated sham floor is unavailable from this cohort. [2, 10]

An illustrative [strict EvalMath planner check](NC12_ADDENDUM_RESOLUTION_AUDIT_20261118.json), performed after the outcomes were known, returned **`REDESIGN_OR_PILOT_ONLY`**. It examines q = 8, the largest primary low-query budget, and retains the original 16-comparison family. The input marks the development readout exposed and uses eight as the sample size in a hypothetical reuse check. Its sham summary takes the maximum magnitude among the 28 pairwise differences between the original eight shams within each root, leaving eight independent observations. This summary was chosen for the retrospective audit. [10]

That threshold-only call sets both target and scientific minimum to the registered 0.01 nats. Their zero difference mechanically triggers the planner's target-floor and power-failure flags. Those flags carry no empirical power estimate for eight pairs. Its approximate-resolution field uses the 0.01 fallback because the sham tail is uncalibrated, so that field supplies no calibrated minimum detectable effect. The negative observed transfer advantage and unavailable sham bound remain the substantive findings; a prospective confirmation design remains to be registered.

The original NC-12 protocol was committed before implementation or model generation. A missing import caused the excluded source smoke to fail before development; the documented repair preceded the first development training. The original complete run, **44283950085**, supplies the evidence used here. Later executions reuse the study's roots and add no independent pairs. The addendum's receipt review reproduced all sixteen bootstrap means and intervals with maximum absolute discrepancy **0.0** and ran the complete committed archive auditor. [1, 4, 5]

Two provenance limits remain visible. The execution lock pins four NC-12 files, while imported helpers and the full dependency environment fall outside that file lock. The receipts record sham counts and minimum regularized singular values; selected proposal indices and alignment condition numbers requested by the protocol are absent. The dated addendum preserves those original records and documents their scope. NC-12 registers development roots only. Protected confirmation populations in other studies retain their existing seals.

## 7. Sources and original evidence

The BPD source snapshot for this addendum is commit **`100033addcaa98cf3d0530541a3598ed40c170c9`**, which merged [PR #108](https://github.com/MoralityLabAI/BPD/pull/108).

1. [NC-12 original results](https://github.com/MoralityLabAI/BPD/blob/100033addcaa98cf3d0530541a3598ed40c170c9/mechinterp/bpd/experiments/nc12/RESULTS_20261118.md).
2. [NC-12 frozen protocol](https://github.com/MoralityLabAI/BPD/blob/100033addcaa98cf3d0530541a3598ed40c170c9/mechinterp/bpd/experiments/nc12/PROTOCOL.md), first committed at `6a4b5ebc16fae730f17b0e5dac5564eaf419c071`.
3. [NC-12 runner](https://github.com/MoralityLabAI/BPD/blob/100033addcaa98cf3d0530541a3598ed40c170c9/mechinterp/bpd/experiments/nc12/experiment.py). Runner SHA-256: `30aa76ef6b2150d96a37fa696d25d714ed11bfcb0359d6e6205fd97842d2f1b5`. Protocol SHA-256: `2b0bcb1e7f99536c7c1d5717c8802800671a6ea79d42c7efcd0ab854de62a152`.
4. [Original aggregate](https://github.com/MoralityLabAI/BPD/blob/100033addcaa98cf3d0530541a3598ed40c170c9/mechinterp/bpd/experiments/nc12/aggregate.py) and [archive auditor](https://github.com/MoralityLabAI/BPD/blob/100033addcaa98cf3d0530541a3598ed40c170c9/mechinterp/bpd/experiments/nc12/audit_original_receipts.py).
5. [Original nine-file receipt artifact 14418272512](https://github.com/MoralityLabAI/BPD/actions/runs/44283950085/artifacts/14418272512), containing roots 1200–1207 and `NC12_ORIGINAL_SUMMARY.json`. ZIP SHA-256: `3b97a25641caa553b12b67da992bc72e2a7c4c82ecc8712fd79ef498a76eabe5`. [Per-root CSV](https://github.com/MoralityLabAI/BPD/blob/100033addcaa98cf3d0530541a3598ed40c170c9/mechinterp/bpd/experiments/nc12/PER_ROOT_20261118.csv).
6. [NC-10 original report](https://github.com/MoralityLabAI/BPD/blob/100033addcaa98cf3d0530541a3598ed40c170c9/mechinterp/bpd/experiments/nc10/RESULTS_20261118.md).
7. [NC-11 original report](https://github.com/MoralityLabAI/BPD/blob/100033addcaa98cf3d0530541a3598ed40c170c9/mechinterp/bpd/experiments/nc11/RESULTS_20261118.md).
8. [BPD optimal-basis and invariant-mask theory](https://github.com/MoralityLabAI/BPD/blob/100033addcaa98cf3d0530541a3598ed40c170c9/mechinterp/bpd/theory/OPTIMAL_BASIS_AND_MASKS.md).
9. [LoopedTransformers manuscript, Sections 9–10](https://github.com/MoralityLabAI/LoopedTransformers/blob/b490542bf81beb7e11c98c2e2fa32867f0ff77f7/papers/shared_weights_dark_fibers/PAPER.md), snapshot `b490542bf81beb7e11c98c2e2fa32867f0ff77f7`.
10. [EvalMath mechanistic-resolution guidance](https://github.com/MoralityLabAI/EvalMath/blob/6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a/skills/evalmath-mechinterp-power/SKILL.md) and [planner](https://github.com/MoralityLabAI/EvalMath/blob/6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a/skills/evalmath-mechinterp-power/scripts/preflight.py). The accompanying resolution record includes exact inputs and output with source identifiers.
