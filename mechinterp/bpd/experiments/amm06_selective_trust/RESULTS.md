# AMM-06: selective trust of noisy local calibration, prospective development

**2026-11-18. Evidence class: trained synthetic tied-linear recurrent BPD development-only null for the registered trust advantage.** The 128 AMM-06 confirmation model roots remain **strictly sealed**, as do the AMM03/04/05 and ARG-03-N protected cohorts.

## Protocol and chronology

- [PROTOCOL.md](PROTOCOL.md) registered before source and data at commit `4ede3b9290903dbf22a81c44811c9a00246ac2ee`.
- Trained generator, casewise causal patches, split-calibration likelihood gate, 15 engineering tests and staged CI frozen at `5fd82942ed9c30f20c15c4074b838d8ce6f6fd68` before first training.
- [Hosted source-only run](https://github.com/MoralityLabAI/BPD/actions/runs/44269463066) passed 15 engineering tests, including actual BPDModel/TiedComponentLinear PassMaskSchedule parity, and generated only TRAIN roots.
- The 64-root training-memory digest was committed at `28eb88bf32316f87bbfd20a98a85ad757d6b914d` **before** evaluating the new DEVELOPMENT population.
- [Hosted independent development run](https://github.com/MoralityLabAI/BPD/actions/runs/44269667331) reproduced the source-locked training memory, scored development exactly once, generated the corrected EvalMath preflight, and saved full raw case-level JSON.
- Full independent development results are locked in `results/DEVELOPMENT_LOCK.json` and the planner decision in `results/PREFLIGHT.json`. No source file changed after training started.

Exact source SHA256:
- `models.py`: `78b2b91532625c524eb9da0b22485e70dd43f98d9d47c7adb15ab675efc43126`
- `case.py`: `0fe333bba17289c7d9a288e70a77327878c2417f1890d8dac568aeaa5a2c802c`
- `gate.py`: `641a9e24ed707a1ec169bd3fb9fe0f37c17c3b41f71ff5ae05fa74b6210124e8`
- `experiment.py`: `6b883ccabaffb82c080dcc6dccd5929838cafdbf5211e1233f35530386882907`
- Training memory canonical SHA256: `fe39c6bb94885b6e968ac95dc6589892480b862e2060dd69c4162dff25b73c9c`.
- Complete raw development canonical SHA256: `b67a2e8836e441e5d05db8136c276fdc9fd344e598a06e0189408b3b93ab439d`.
- Development selected-action, all candidate, calibration and repeatability bits SHA256: `63963c059df3b29f9368320df06707d2d85bc7954efbb511aefcd745ba5f765b`.

## Hypothesis and equal-budget intervention

Each root separately fits a rank-two physical weight-tied 4D recurrent operator from source-defined two-pass tasks and a new rank-two BPD decomposition from teacher trajectories. A donor modifies the task-relevant source coordinate and an independent nuisance coordinate. The independent external donor source defines the behavioral target.

Every active candidate patch is SVD-canonical, projected current-dark and renormalized to the same within-pair physical L2 dose. Another *learned weight-tied pass* scores whether the chosen patch repairs the donor's behavioral target. The full donor intermediate replacement is an explicitly larger-dose ceiling, never an equal-dose control.

Each root receives eight disjoint calibration donor pairs, giving 16 noisy candidate labels. The hidden judge flips labels with probability 0.10 on half the roots and 0.45 on the other half, balanced across both source families. Both trust and always-update policies share **exactly the same 16 noisy labels, same causal geometry and three unlabeled candidate-readout probes per case**. Only the cross-root memory has an additional, disclosed, amortized training corpus: 64 independent model roots ×128 clean donor cases/model.

The fixed `selective_trust` policy trains a local residual from calibration pairs 1–4 and tests its noisy-label likelihood against the frozen memory predictor on pairs 5–8, marginalizing over the two possible root-level judge error rates. It accepts the full 16-label local update only if the preregistered **log Bayes factor exceeds log(2)**. The baseline `always_update` always uses the full local residual; `memory_only` never updates. All final actions are case-conditioned, selecting exactly one BPD component per heldout donor pair. The gate sees no true quality mode, independent donor target, clean calibration labels or future donor continuation.

## Frozen training cohort

Independent TRAIN models: 64, each with 128 donor pairs (8,192 recorded pairs per candidate action).

| Candidate component | Correct TRAIN donor cases | Training hit rate |
|---|---:|---:|
| SVD-canonical component 0 | 3,241/8,192 | 39.56% |
| SVD-canonical component 1 | 2,086/8,192 | 25.46% |

The frozen global ridge chooses component 0 as the pooled fixed policy. Its continuous casewise probabilities use 11 causal-response features and regularization λ=10; the precise coefficients are in the archived training receipt.

## Prospective DEVELOPMENT results

**64 independently trained model roots; 256 new heldout donor cases/root = 16,384 scored cases per arm**, plus 256 independent repeatability cases/root. Every case is retained, including zero-dose/failed-model observations. The root, rather than a donor case or a different pass, is the independent unit.

| Frozen strategy | Correct / 16,384 | Accuracy |
|---|---:|---:|
| **Training-only case-conditioned memory** | **6,995** | **42.694%** |
| **Selective trust, preregistered primary arm** | **6,922** | **42.249%** |
| **Always apply local feedback update, primary control** | **6,790** | **41.443%** |
| Equal-budget local ridge | 5,793 | 35.358% |
| Training-pooled fixed component | 5,682 | 34.680% |
| Casewise random component | 4,799 | 29.291% |
| Equal-dose random-dark A / B | 3,244 / 3,240 | 19.80% / 19.78% |
| Zero-patch sham | 12 | 0.073% |
| Full donor intermediate state (different dose) | 16,160 | 98.633% |

The registered primary effect is **+132 / 16,384 = +0.00805664** (+0.806 percentage points) for trust-gated memory against always updating. It has **23 positive, 17 negative and 24 tied independent model roots**, exact one-sided root-sign p=.214795, and Holm-adjusted p=.429591. The frozen 2,000-sample paired-root bootstrap 95% interval is **[-0.00097656,+0.01757813]** or **[-0.098,+1.758] percentage points**. It falls below the registered 8pp physical minimum.

| Registered one-sided root contrast | Root-mean effect | +/−/tie roots | Holm-adjusted sign p | Root bootstrap 95% CI |
|---|---:|---:|---:|---|
| Trust vs always-update (primary) | **+0.008057** | 23/17/24 | **.429591** | [-.000977,+.017578] |
| Trust vs memory-only | -.004456 | 0/3/61 | 1.0 | [-.011719,0] |
| Trust vs equal-budget local ridge | +.068909 | 40/13/11 | .000403 | [+.038452,+.101868] |

A local update was accepted on **7 of 64 independent roots**. Five were in the low-noise regime (5/32, 15.625%) and two in the high-noise regime (2/32, 6.25%). The acceptance-rate difference, **9.375pp**, missed the preregistered **10pp** discrimination veto. Actual calibration-label flips were **51/512=9.96%** in the low-noise roots and **239/512=46.68%** in high-noise roots, consistent with the two hidden generative error rates.

On the seven accepted roots, the gate produced *73 fewer successes* than frozen memory-only. Its modest advantage over always updating came mainly from rejecting weak or harmful local updates across the other 57 roots. It did not establish that its positive trust decisions identified useful corrections. On accepted roots, only three lost outcomes relative to frozen memory and four were tied; none improved.

The audit-only best-component-per-case oracle exceeds always-update by **1,314/16,384 = 8.020pp**, barely clearing the preregistered 8pp headroom **veto**. That oracle reads forbidden heldout outcome labels and is retained exclusively as an upper bound, never as a policy input.

## Physical and model diagnostics

- Maximum active matched-dose L2 discrepancy: **4.7684e-7**. Every active patch has zero immediate current-readout displacement.
- Minimum SVD singular-value gap across development roots: **0.078708**, above the registered .01 separated-spectrum diagnostic floor. Engineering GL2 and recanonicalized rank-one patches passed.
- Maximum physical teacher task MSE: **1.173e-7**. Two factor-fit outliers remain in the ITT denominator: development root 0001 has BPD factor behavioral MSE **.690934** and reconstructed-weight relative error **1.5485**; root 0063 has factor MSE **.150837** and relative error **1.1571**. These are genuine synthesis limitations and are reported rather than censored.
- Native BPDModel/TiedComponentLinear and independent factor masked forward parity passed in both hosted runs. All 15 excluded-root engineering tests passed.

## Corrected EvalMath decision and protected confirmation

The preflight uses **64 independent DEVELOPMENT roots** and the paired root-level primary effects. The matched sham is the *same frozen always-update policy* evaluated on each root's 256 primary and 256 separate repeatability donor cases, with identical physical intervention dose and calibration labels. Across 64 independently generated model roots, the sample maximum is an upper tolerance bound for the 95th population quantile at >=95% confidence.

- Registered mechanistic minimum: **+0.08** absolute accuracy. Registered target: **+0.15**.
- Independent-root 95th-quantile/95%-confidence repeatability floor: **0.125 (32/256 = 12.5pp)**. The variation floor is therefore **12.5pp**.
- Observed root-mean primary gain: **0.00805664**, independent-root sample standard deviation **0.03791005**, well below the physical variation floor.
- Corrected preflight from [EvalMath PR #51](https://github.com/MoralityLabAI/EvalMath/pull/51), pinned at `6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a`, returns **`REDESIGN_OR_PILOT_ONLY`**, with reason `pilot_effect_not_above_mechanistic_variation_floor`. Its normal 25-root forecast refers to a *hypothetical +15pp target effect*, not the observed effect.
- Two additional prospective *vetoes* fail: gate acceptance on 7 roots (minimum 8), and low-minus-high noise acceptance 9.375pp (minimum 10pp).
- The **128 independent CONFIRMATION model roots remain sealed and unscored**. A historical or fabricated READY record cannot authorize their execution. AMM03/04/05 and ARG03-N protected populations are also untouched.

## Interpretation and research direction

AMM-06 found that rejecting noisy local updates can protect a stronger model-memory prior in some synthetic recurrent worlds. The preregistered *trust calibration* claim failed, and frozen memory alone still outscored the trust-gated policy. The source-locked negative outcome suggests the hard eight-pair gate is too weak to distinguish the value of individual local updates at model-root resolution.

An informative follow-on would first improve causal intervention fidelity on difficult trained roots, assess robustness under factor-fit failures, and study label-efficient reliability calibration using genuinely independent judge signals. Any new method needs new model-root populations, a separately registered effect floor, and a cost-matched local comparator. Repeating the same small calibration with additional donor cases in already-seen roots cannot establish a cross-root effect.

This result is restricted to **synthetic trained rank-two linear recurrent BPD interventions with noisy feedback**, without demonstrating nonlinear/TRM language model reasoning, deployed agent-to-agent memory, or a Buzz/Nostr network.
