# BPD-SF05B Stage 2 donor generator: source-only CPU admission

**November 18, 2026.** This is a synthetic **data-generation and exact oracle test**, with no trained model or scientific root evaluated. The amended source lock passed in [hosted GitHub Actions run #44147545726](https://github.com/MoralityLabAI/BPD/actions/runs/44147545726). All **11/11 deterministic tests passed**, including exact source-tag-only swaps, recipient/donor oracle target recomputation, separate public-query population, name permutations, and **independent closed-form checks** of the source-aware conditional Bayes recipient and donor means.

The original [failed source-only run #44146966540](https://github.com/MoralityLabAI/BPD/actions/runs/44146966540) and the pre-model [math amendment](MATH_AMENDMENT_01.md) remain in the PR history. The generator's exact 24-configuration conditional enumeration was unchanged; only the analytic closed forms and tests were repaired before additional source verification.

## Frozen deterministic cohort admission

| Source-only cohort | Attempted | Qualified under exact source-label rule | Retained | Exact source-target label-change rate | Mean absolute recipient/donor target difference |
|---|---:|---:|---:|---:|---:|
| \`authorized_unconditional\` | 512 | 512 | 512 | 100% | 0.5339117 |
| \`authorized_high_contrast\` | 2,048 | 1,355 | 512 | 100% | 0.7362190 |
| \`public_U\` | 512 | 512 | 512 | 0% | 0.0000000 |

The high-contrast qualification rate \`1355/2048 = 66.16%\` comes solely from a 0.30 minimum difference in independently generated **source-oracle labels**. For two independent \`Uniform[-0.8,0.8]\` values, the theoretical probability of an absolute difference ≥0.30 is \`(1 - 0.30/1.60)^2 = 0.66015625\`, consistent with this particular deterministic sample. The high-contrast population conditions on **data oracle labels**, not on neural teacher correctness or successful interventions.

## Corrected recipient and donor blind baselines

Under the new source-aware query-selection law, the squared-error source-blind conditional means of the recipient and donor generally differ. Their exact [derivation](THEORY.md) includes the original final source's *previous write* when the donor tag is switched away. The hosted test enumerates all six source-name permutations, two possible donor-source switches and two role choices and independently matches the analytic expression, with an additional admission condition for the high-contrast population.

| Source-only cohort | Recipient conditional-mean predictor MSE | Donor conditional-mean predictor MSE |
|---|---:|---:|
| Authorized unconditional | 0.1369860 | 0.1117552 |
| Authorized high contrast | 0.1315173 | 0.1176077 |
| Public U | 0.0000000 | 0.0000000 |

These values measure the exact structured generator targets against its own correctly conditioned **squared-error Bayes predictions**. They are not neural model evaluation scores. The public U target is the independent public scalar on both sides of every source-only donor pair, so it has zero oracle-label difference.

## Scientific boundary

The new authorized cohort changes which source's value is requested in the original recipient context, so it will require a **new frozen model-level experimental protocol** before scoring any teachers. The source-label-only donor pair still preserves every numeric payload and prior event, and recipient/donor share the same query. The public cohort is generated independently and must not be selected for clean model accuracy before measuring public output effects.

This source-only benchmark improves target-contrast admission, while the previous Stage 1 recipient/donor *model-level eligibility* of 18/512 and 13/512 remains a separate, untested concern under the new generator. No calibration roots 6101–6159, confirmation roots 7101–7132 or ARG-03-N sealed cases were accessed. The experiment cannot substantiate LLM privacy or canonical BPD features.
