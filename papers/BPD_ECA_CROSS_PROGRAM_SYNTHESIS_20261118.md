# BPD–ECA: Recurrent Evidence, Observable Quotients and the Economics of Inspection

**Research synthesis:** 18 November 2026. Companion to the ECA-01 through ECA-05 series in [Control-Harness](https://github.com/MoralityLabAI/Control-Harness) and the exact audit-value framework in [EvalMath](https://github.com/MoralityLabAI/EvalMath). This paper is a cross-repository research plan and interpretation, not a result from a new BPD intervention.

## 1. The transferable scientific question

The ECA sequence built a controlled agent with noisy signed evidence, hidden issuer reliability, explicit acquisition costs and a recurrent policy that decides when to inspect. The central question for compact, recurrent AI is:

> **When does additional recurrent computation justify the evidence needed to make it useful?**

The ECA-04 trainable 16-unit GRU used 12 gold-labeled sentinel reports to infer which audit channel governed an issuer and select a fixed number of audits. In 128 heldout blocks of previously unseen group/alias configurations, its cost-2 weighted loss was **777**, compared with **980** for the pooled-reliability baseline and **776** for a known-family exact Bayesian reference, across **3,072 decisions per arm**. The sham that reassigned issuer histories scored **1,141**. This was a synthetic model-family result, and the model was trained on the same *three noise families* later available in heldout worlds. Its public graph edges carried no channel effect.

The follow-up ECA-05 changed the problem. It introduced three out-of-support noise channels and host-authenticated but semantically rebound issuer calibration records. The registered semantic-gate gain against source-only GRU was **17/3,072 = 0.00553385 weighted-loss units per case**, below its frozen practical floor **0.01**, with a descriptive cluster-bootstrap interval spanning zero. Source-only GRU incurred all-in cost **1,082.4**, approximately the **1,082.0** public no-audit comparator, because its gains in task correctness were consumed by the **138.0** cost of sentinel acquisition and the **38.4** cost of case audits. A declared six-channel minimax policy reached **1,000.6** but escalated **1,398/3,072** decisions under a low 0.4 escalation cost.

The earlier ECA-02 and ECA-03 studies show how objective choice changes the optimum. ECA-02's symmetric fitted-value selector achieved 3,143/4,096 correct decisions versus 3,109/4,096 for uncertainty targeting, but incurred 118 more false authorizations and 152 fewer missed legitimate actions. The frozen-action cost crossover was **76/59**. ECA-03 preregistered a new unauthorized-approval cost ratio of **2:1**, sampled fresh training/heldout roots, and found cost-weighted fitted-value loss **1,121** against **1,308** for cost-aware uncertainty and **1,503** for the old symmetric fitted policy. These values come from synthetic channels and heldout seed blocks; they are not real-world source verification or evidence of frontier-model control.

## 2. BPD/VPD and looped-transformer mechanistic interpretability

The archived ECA-04 GRU presents a tractable mechanistic target. The model's heldout latent variables, verified twelve-step evidence histories, exact discrete Bayesian posterior and recorded audit actions provide observational targets and causal endpoints.

**Proposed BPD-ECA-01:** instrument the unchanged, SHA-256-pinned ECA-04 16-unit GRU and record hidden states after all twelve sentinel steps. Fit simple linear decoders of the exact regime posterior as a diagnostic; optionally fit sparse autoencoders or BPD subspaces using *training/discovery roots only*. Measure the first step at which a state encodes the three-regime posterior and whether its representation survives issuer alias changes. Compare with a true posterior obtained by exact rational enumeration of the finite channel likelihoods.

A high decoding score establishes *decodability*. A mechanism claim requires causal manipulation: overwrite an aligned hidden-state component at a fixed step, hold other conditions equal, propagate the native GRU to its final logits, and score changes to regime calibration, case-audit ranking and authorized decisions. Compare with rank-/dose-matched random directions and independently selected donor states. Random interventions that produce the same behavior rule out identifying a unique causal representation.

The separate dark-fiber work requires time-indexed causal effects, current-state versus trace-read observability, and full-head closure. Keep these restrictions. A reconstruction of an exact posterior in a single readout is an observational sufficiency demonstration; a sequence of readout and intervention tests is needed to argue that the same mechanism carries through repeated computation.

**Immediate test:** run a source-registered, development-only CPU experiment on new host-generated issuer histories, with a frozen model checkpoint, independently generated training/discovery/evaluation roots, and a matched causal sham. Predeclare intervention step, dose, outcome and family of tested subspaces. Do not touch BPD's ARG-03-N protected 168-case test or AMM-03 sealed confirmation population.

## 3. Compactification-Theory: minimal recurrent sufficient statistics

For the registered three-regime channel, a Bayes-optimal history decoder can summarize an arbitrary sequence of independent, gold-labeled sentinel reports by

\[
\pi_t(k) = P(\text{regime}=k\mid e_1,\ldots,e_t).
\]

The three posterior entries sum to one, so the sufficient-statistic manifold has intrinsic dimension at most **two** under this declared family. The GRU uses sixteen hidden coordinates. The compactification question becomes whether a low-dimensional quotient \(q(h_t)\in\Delta^2\) preserves the next Bayesian update and downstream audit decisions.

A prospective NOR-ECA study should compare observers with nominal ranks 1, 2, 4 and 8 on new issuer roots, using matched parameter budgets and proper scoring loss, and should test the update equation \(q(F(h_t,e_{t+1}))\approx T(q(h_t),e_{t+1})\) under a new evidence prefix. A small final-state readout is not evidence for a closed quotient unless repeated future updates preserve it.

Under the ECA-05 unseen channels, the exact three-regime posterior is *misspecified*. An extra latent unknown-regime mass or an enlarged channel family may be necessary. This makes the experiment a candidate depth–precision–information frontier: count representation dimensions, verification queries, recurrence depth and downstream decision loss jointly.

## 4. MathCan and OntologicalProofs: limits of observable authorization

Merged MathCan AUTHQ-01 shows that a deterministic authorization classifier factors through an observation if and only if authorization is constant over observation fibres (with surjectivity for a total classifier). Opposite authorization truths with identical public observations force at least one classification error. AUTHQ-02 establishes a finite inspection-budget ceiling in matched authorized/forbidden worlds:

\[
J=\mathrm{TPR}-\mathrm{FPR}\le B/N,
\]

where \(B\) root pairs can receive decision-relevant inspections out of \(N\) equally weighted pairs. The ECA-01 planted permission simulator saturated the ceiling when an omniscient host auditor exposed the planted truth.

The next formal project is a **sequential evidence-acquisition calculus** with observation filtrations \(O_0\subseteq O_1\subseteq\cdots\), finite inspection budget, costly reliable witnesses and stopping/escalation rules. A useful theorem would state how much decision separation is impossible from a given information partition, and what additional witness cost is necessary to refine it. Claims of an optimal learned controller require an empirical test and must remain separate from the combinatorial theorem.

The OntologicalProofs normalized gcd-ground kernel and its nonunit common-ground support could provide planted structures for distinguishing genuinely independent roots from many observations sourced through one underlying cause. This is an analogy and a proposed construction; the existing arithmetic theorem does not certify real-world issuer independence.

## 5. EvalMath: make evidence accounting standard across gyms

The ECA sequence reinforces [EvalMath FSI-01](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md). Its independent unit is a root seed, not each repeated candidate action, token, GRU step or intervention. Every score should carry independent-root IDs, a preregistered minimum effect, comparison family, matched control doses, acquisition cost and whether selection was exposed to any heldout readout.

ECA-05's below-floor positive observation is a model case: a small measured gain and a bootstrap interval crossing zero cannot support the registered mechanism success claim. The ECA-05 independent replay also found four near-tied audit slots in a shuffled control separated by \(5.55\times10^{-17}\), with identical actions and costs. Future studies must freeze deterministic quantization or a numerical-tie policy *before* sampling fresh heldout roots.

For BPD-ECA development, run the EvalMath mechanistic-resolution preflight before a confirmatory evaluation. Define a physical intervention dose, a matched sham-variation floor, the minimum improvement over the null, and a sufficient number of independent issuer roots. Exact algebraic theorems, trained-model observations and causal transfer belong to separate evidence classes.

## 6. Control-Harness, Hypergames and cryptographic verification

This is an actionable generalization of the issuer fixture. An agent can receive an origin-authentic signed statement whose subject was rebound or whose label was validly re-signed contrary to an independent gold reference. That distinction can be turned into a control gym for permission to execute actions under strategic evidence manipulation: independent capabilities, permissions, revocations, issuer lineage, provenance and verification budgets.

A plausible Hypergames extension lets adversaries affect evidence ordering and source attribution and measures whether a defending agent can maintain correct authority boundaries under constrained verification. ECA-05 suggests that escalation capacity, delay and its explicit cost must be modeled rather than assuming an unlimited trusted human reviewer. These are simulation/evaluation ideas, not claims about actual threat actors or deployment.

For Bitcoin/BitVM work, the analogous design principle is to bind a cryptographic witness to the exact action, transaction context, and authority state transition that it authorizes. The ECA semantic-rebinding fixtures may motivate protocol-specific transaction-context or stale-authorization tests. Any Bitcoin consensus or cryptographic security conclusion requires its own formal protocol proof and implementation tests.

## 7. Computation priorities and scientific scope

| Order | Campaign | Target | Main falsifier |
|---|---|---|---|
| 1 | **BPD-ECA-01** | Frozen ECA-04 GRU, CPU/RTX 3050 | Posterior-probe direction fails a dose-matched, donor-specific causal test |
| 2 | NOR-ECA | Low-dimensional recurrent quotient | Rank-two reconstruction fails future-step closure or task decisions |
| 3 | ECA-06 | Adaptive acquisition with reviewer capacity | Evidence-query gain vanishes when all costs are counted |
| 4 | Hypergames | Strategic manipulation of permissions | Defending agent authenticates origin while accepting stale/misbound authority |
| 5 | EvalMath/Lean | Sequential information bound | Formal assumption does not describe the policy's actual observation interface |

**Priority conclusion:** use the frozen ECA-04 model, exact finite Bayesian posterior and intervention interface to establish a causal representation before spending larger GPU time. Begin on development roots and require a heldout registration only if the primary causal intervention exceeds the matched sham floor. Probe fit alone is an observational result; causal interpretation requires native-state interventions with fixed dose, root-level uncertainty and task-level effects.

### Provenance and nonclaims

ECA-04 run: https://github.com/MoralityLabAI/Control-Harness/actions/runs/44145067899

ECA-04 independently checked checkpoint: https://github.com/MoralityLabAI/Control-Harness/actions/runs/44146877006

ECA-05 registered result: https://github.com/MoralityLabAI/Control-Harness/actions/runs/44219847153

ECA-05 independent audit with acknowledged floating-point tie variance: https://github.com/MoralityLabAI/Control-Harness/actions/runs/44221715744

MathCan authorization quotient: https://github.com/patrickdugan/MathCan/pull/140

All source identities, issuer regimes, audit gold labels and signing keys in ECA are synthetic. New BPD tests require a separate source/seed registration. No real institutional issuer authentication, natural language agent behavior, or transfer to larger models has been established.

## Implementation handoff (BPD-ECA-01)

The registered follow-up is implemented in [Control-Harness PR #55](https://github.com/MoralityLabAI/Control-Harness/pull/55) to run against the original ECA-04 model artifact without copying or retraining its parameters. The experiment uses six-pass native recurrent-state transfer, a training-only rank-two ridge projector to the exact Bayesian posterior log odds, eight rank-matched physical-dose-matched random projector shams per independent world, a separate 64-root 95/95 sham-variation calibration set, and 96 new development roots. The exact [EvalMath mechinterp-power preflight](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power) gates any protected confirmation. The intervention target is the frozen model's own donor-prefix/source-suffix counterfactual; source-root-level statistical evidence and task-action alignment are reported separately. This handoff is subject to the BPD AGENTS.md seal restrictions.

## BPD-ECA-01 retrospective result and exact sufficient-statistic correction

[Control-Harness PR #55](https://github.com/MoralityLabAI/Control-Harness/pull/55) completed a **96-independent-root development-only causal intervention pilot** on the unchanged ECA-04 16-unit GRU. The train-only rank-two ridge projector derived from the exact three-regime posterior achieved an initial candidate-minus-median-sham donor-prefix restoration of **0.11357**. A separate **64-root** 95th-quantile/95%-confidence sham-calibration envelope was **0.53357**, so the precise EvalMath mechanistic-resolution preflight returned **REDESIGN_OR_PILOT_ONLY**. No protected confirmation roots were generated.

A separate CPU replay reproduced the raw source/output hash provenance but found the fitted rank-two projector and its interventions numerically unstable: its candidate-minus-median-sham effect was **0.04983**, and its sham floor was **0.48509**. Both runs blocked confirmation. The [source-registered report](https://github.com/MoralityLabAI/Control-Harness/blob/research/bpd-eca01-causal-prefix-20261118/reports/bpd_eca01_development_result.md) records the negative result and replay variance.

The deeper finding is exact and model-independent. All ECA-04 noise channels, and the three later ECA-05 channels, satisfy `f_k=1-t_k`. For every sentinel, the likelihood of an audit response depends only on whether it matches the host-gold label. A fixed-length prefix of `n` events has sufficient statistic `m`, the **number of matches**, with likelihood `t_k^m(1-t_k)^(n-m)`. At fixed `n`, every posterior log odds `log(pi_k/pi_ref)` is affine in the *same single scalar* `m`. Therefore the 2-dimensional simplex of possible three-regime posteriors collapses to a **one-dimensional affine set in log-odds coordinates** for these channels. For the six-sentinel prefix, a seven-point exact likelihood enumeration yields centered target singular values about 27.7151 and 1.62e-15.

This is an important restriction on the proposed compactification experiment: the correct controlled target is **rank one for fixed recurrence depth**, with prefix length/clock supplied separately for variable depth. The rank-two probe attempted to identify a second direction that the supervision did not constrain. The next BPD-ECA experiment should preregister a rank-one match-count causal projector, archive its actual physical matrix and numerical stability checks, include equal-dose rank-one Haar shams, and use new independent roots.
