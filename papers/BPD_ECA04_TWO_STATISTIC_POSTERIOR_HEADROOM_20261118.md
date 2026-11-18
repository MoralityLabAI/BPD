# BPD-ECA-04: Two-dimensional Bayesian evidence versus behavioral saturation

**Research report, November 18, 2026.** The associated source-registered CPU feasibility study is [Control-Harness PR #60](https://github.com/MoralityLabAI/Control-Harness/pull/60), following merged [BPD-ECA-03 #58](https://github.com/MoralityLabAI/Control-Harness/pull/58). This note connects posterior geometry, recurrent representation and task-grounded intervention design for the wider BPD / Compactification-Theory program.

## A genuinely rank-two Bayesian target

The ECA-04 issuer model's old channels all satisfied `f_k=1-t_k`. With a fixed number of gold-labeled sentinel observations, both Bayesian log odds depend on the same scalar count of label/report matches, yielding one-dimensional affine sufficient-statistic geometry. BPD-ECA-01/02's rank-two and rank-one state-transfer experiments were below EvalMath's calibrated mechanism-variation floor; BPD-ECA-03's task-correct rank-one steering was also negative, with little oracle headroom.

BPD-ECA-04 instead uses channels with independent sensitivity and false-positive rate:

| Regime | `t_k` | `f_k` |
|---|---:|---:|
| A | 0.91 | 0.30 |
| B | 0.64 | 0.07 |
| C | 0.30 | 0.55 |

For six known-authorized and six known-forbidden sentinels, let `u` count positive reports on authorized gold labels and `v` count positive reports on forbidden gold labels. The exact likelihood is

```
L_k(u,v) = t_k^u (1-t_k)^(6-u) f_k^v (1-f_k)^(6-v).
```

The log odds of A/B against C are affine in `(u,v)` with coefficient matrix

```
[[3.160933, -1.047969],
 [1.422662, -2.787360]]
```

whose determinant is **−7.3197526982**. The exact 49-count-state log-odds grid has centered singular values **59.211576** and **24.229579**. With the complementary-channel control `f=1-t` and unchanged sensitivities, the determinant is zero. This mathematical calculation gives a controlled transition from one to two **affine log-odds dimensions**. Arbitrary finite nonlinear coding can still encode 49 states into a scalar, so the determinant is not a theorem that all recurrent models require a two-dimensional state representation.

## A new recurrent model actually captures the extra information

The source-frozen [first registered development readout](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44265687600) trained a new sixteen-unit GRU on **2,048 independently seeded issuer roots**, validated on **256** and fitted an independent linear two-log-odds decoder on **512** separate roots. Hyperparameters and twelve training epochs were source-locked before new OS entropy. Final model validation class accuracy was **85.9375%**.

The same fitted state-to-log-odds coefficient matrix was evaluated at rank one and rank two on **128 independent development roots**:

| Observational readout | Heldout result |
|---|---:|
| Rank-one posterior MSE | **3.7651843** |
| Rank-two posterior MSE | **0.8376914** |
| Relative error reduction from second dimension | **77.75165%** |
| Independent-half rank-two physical projector overlap | **0.965356** |

The registered probe improvement threshold was five percent and the half-fit overlap floor 0.90, so both observational representation conditions passed. The two physically executed 16×16 projector matrices are retained in [the original source-hashed artifact](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44265687600/artifacts/14410850044). No hidden-state interventions were performed in this feasibility test.

## Correct-action headroom remains small

Every issuer in an independent 64-root screen and a separate 128-root development cohort generated 32 host-private permission cases under the declared noise regime. Public gold sentinel reports were available to the GRU and to an exact Bayesian reference. The reward was true `2×false_authorizations+missed_authorizations`, scored on identical cases.

| Policy | Screen, 2,048 cases | Development, 4,096 cases |
|---|---:|---:|
| Public prior without an audit report | 695 | 1,467 |
| New GRU channel posterior | 574 | **1,141** |
| Exact Bayes from public gold sentinels | 571 | **1,128** |
| Privileged true-channel oracle | 537 | 1,091 |

On development cases, the trained GRU captured **326 of 339** weighted-loss units available between the public prior and the exact public-evidence Bayesian decision. That is **96.2%** of the realized public-information improvement on this fixed synthetic cohort. The GRU-to-Bayes behavioral headroom was **13/(2×4,096)=0.00158691** per case, compared with a source-frozen practical floor **0.020**. The separate screen produced **0.00073242**. The descriptive 128-root block-bootstrap interval for development headroom was **[−0.00012207, 0.00378418]**.

Thus **the Bayesian geometry, rank-two probe advantage and physical subspace stability conditions passed, while both independent behavioral-headroom gates failed**. The source-reported verdict is **`HEADROOM_OR_GEOMETRY_GATE_BLOCKED`**, and the protected causal-confirmation cohort was never generated.

The negative result extends the BPD-ECA-03 finding to a channel family where the exact posterior is genuinely rank two. The model's internal decoding error remains substantial in squared log-odds coordinates, yet much of it is *behaviorally inert* for these five public priors, binary audit reports and the fixed 2:1 error cost. Accuracy and proper probability scoring can therefore tell different stories about the same recurrent representation.

## Independent trained-model replay

The separate [Control-Harness independent CI](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44266899308) returned **PASS**, verifying seven source SHA-256 digests, seven output digests and all 3,008 independent roots. The GRU was retrained from its registered 2,048 training roots and 256 validation roots; maximum checkpoint parameter delta was **7.15×10⁻⁷**. Its original public-evidence inference was reproducible with maximum probability deviation **1.79×10⁻⁷**, unchanged private task-loss totals, and near-exact rank-one/rank-two physical projector overlaps **0.999999990** and **0.999999998**, respectively. [Machine-readable audit artifact](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44266899308/artifacts/14410532396).

A separate root-entropy-derived arithmetic audit regenerated 192 public issuer histories, **6,144 independently seeded private case labels/reports**, and all policy-cost totals directly from exact channel parameters and archived model posterior probabilities. The first-readout result and blocked behavioral-headroom verdict remained unchanged.

## Evaluation boundary and next mechanism experiment

A neural probe that decodes two posterior coordinates is an **observational** result. The fixed rank-two geometry is a mathematical result about the host-defined likelihood. The fact that the recurrent model gets most action decisions correct does not identify the physical internal subspace that causes them.

A suitable **BPD-ECA-05** should pre-register a *proper-scoring causal endpoint* such as heldout posterior KL or logarithmic loss on issuer reliability, with exact public-evidence Bayes as reference. It could patch a frozen rank-two physical posterior subspace at a designated recurrent step, propagate through the native suffix, and score improvement against equal-L2-dose rank-two Haar shams and time-indexed gradient controls. Any claim about correct authorization actions should separately require a new distribution with enough task headroom and a root-level sham-variation preflight.

This requires a new source registration, independently sampled calibration and pilot roots, a fixed physical intervention budget, a 95/95 independently rooted sham tolerance floor, and an archived physical projector for cross-CPU replay. It should explicitly measure future-step quotient closure: if `q(h_t)` claims to represent a sufficient posterior, the next update should be predictable from `q(h_t)` and the new public evidence, rather than only from a final probe.

**Current evidence classes:** exact finite likelihood geometry; trained 16-unit GRU observational posterior decoding; development-only host-true behavioral headroom. No BPD-ECA-04 native causal intervention, real external issuer credential verification or transfer to larger looped transformers has been established. Existing protected ARG-03-N and AMM-03 partitions remain sealed.

### Provenance

[First registered CI](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44265687600) · [Complete root/model artifact](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44265687600/artifacts/14410850044) · [Previous task-grounded negative result](https://github.com/MoralityLabAI/Control-Harness/blob/main/reports/bpd_eca03_task_grounded_development.md) · [BPD causal mechanism report](BPD_ECA01_02_CAUSAL_MECHANISM_REPORT_20261118.md).
