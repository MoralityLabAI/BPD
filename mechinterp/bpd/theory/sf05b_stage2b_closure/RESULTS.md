# BPD-SF05B Stage 2-B: frozen four-arm source-closure training results

**Executed November 18, 2026. Evidence: development-only, two previously examined teacher roots × four training objectives.** [Source-locked hosted run #44223294245](https://github.com/MoralityLabAI/BPD/actions/runs/44223294245) verified [LOCK.json](LOCK.json) before generating the eight excluded-root teacher variants, passed 6/6 training-only engineering tests and the native BPD full-mask/source-donor parity checks. Maximum native full-mask seven-call error: \`1.1920928955078125e-6\`; native donor-swap write error: \`7.748603820800781e-7\`. Full [raw eight-arm JSON artifact](https://github.com/MoralityLabAI/BPD/actions/runs/44223294245/artifacts/14391164759), SHA256 \`b7e2d14310b47e1a2c6b687f7c3a38f04f4264becebde698d5ffbb6dc9c3518a\`, and [in-repository compact receipt](results/excluded_8arm_compact.json).

The physical teacher architecture, original supervised task, F_train 2–4-write samples, 2,600 teacher updates, 450 full-rank BPD factor updates and 240 F_train-only oracle-label policy updates were frozen. Two new regularizers have fixed coefficient 1.0: **current** (recipient/donor prequery readout squared difference) and **public** (recipient/donor future role-U output squared difference). The original Stage 1 baseline is an exact unmodified teacher-training control. Each arm on a given root was evaluated on the **same source-aware 512-record unconditional, 512-record high-contrast and 512-record independent public U histories**. Only roots 901 and 902 were used.

## Headline: the regularizers move different failure modes

| Root | Training arm | F_train prequery visible difference | Heldout high-contrast eligible / 512 | Heldout clean public U source-tag effect | Public U deviation after learned correction | Clean public U target MAE |
|---:|---|---:|---:|---:|---:|---:|
| 901 | Baseline | 0.10098 | 43 | 0.03975 | 0.03411 | 0.04200 |
| 901 | Current only | **0.02477** | **110** | 0.06457 | 0.05797 | **0.06074** |
| 901 | Public only | 0.17555 | 32 | **0.03106** | 0.03229 | 0.03631 |
| 901 | Joint | **0.02422** | 71 | 0.03928 | **0.03120** | 0.03526 |
| 902 | Baseline | 0.13750 | 29 | 0.06966 | 0.06646 | **0.05509** |
| 902 | Current only | **0.02747** | **112** | 0.05775 | 0.05376 | **0.050017** |
| 902 | Public only | 0.10710 | 43 | 0.04239 | 0.03269 | 0.04663 |
| 902 | Joint | **0.02847** | 49 | **0.03520** | **0.03254** | 0.04681 |

The trained **current-only** penalty strongly decreases q-current source-tag dependence on F_train and raises heldout high-contrast eligible count from the baseline 43→110 (901) and 29→112 (902). It also degrades the independent public-role target task: 901 MAE is 0.06074 and 902 MAE is 0.050017, both **above** the frozen descriptive ≤0.05 ceiling. Public output still changes under a source-tag-only pair, with especially poor root 901 effect of 0.06457.

The **public-only** penalty lowers public source-tag dependence and corrected public output (especially root 902, 0.06646→0.03269), while generally preserving public target competence. Root 901's prequery visible source dependence worsens from 0.10098 to 0.17555, reducing high-contrast eligible donor pairs from 43 to 32. This reflects an **objective tradeoff** within the particular trained shared-state architecture.

The **joint** penalty improves the baseline's learned public error to 0.03120/0.03254 while keeping both roots' clean public-target MAE below 0.05. It achieves only **71/512 and 49/512** high-contrast model-eligible cases, with heldout overall MAE worsening to 0.10769/0.11632 against baseline 0.07448/0.09603. The teacher OOD competence ceiling overall MAE ≤0.12 is still met. Neither root's joint arm reaches ≥100 clean-and-current-close cases or ≤0.02 learned public deviation.

**No training arm satisfies the complete descriptive engineering targets on both excluded roots.** This is a genuine negative outcome under the frozen coefficients and architecture.

## Denominator audit: source label contrast versus clean/current closure

| Root | Training arm | Unconditional eligible / 512 | High-contrast eligible / 512 | High-contrast clean-correct and current-close / 512 |
|---:|---|---:|---:|---:|
| 901 | Baseline | 42 | 43 | 43 |
| 901 | Current only | 84 | **110** | **111** |
| 901 | Public only | 22 | 32 | 32 |
| 901 | Joint | 56 | 71 | 73 |
| 902 | Baseline | 20 | 29 | 29 |
| 902 | Current only | 98 | **112** | **114** |
| 902 | Public only | 38 | 43 | 44 |
| 902 | Joint | 48 | 49 | 51 |

All high-contrast cohorts were selected by the exact source oracle's ≥0.30 target-gap rule before any model scoring. The frozen clean recipient/donor correctness and prequery current-readout proximity gates then determine usable causal sample sizes. The **current penalty** can improve those counts, but higher eligible count alone is not evidence of a more private model.

The full physical 32-component BPD \`VU\` source-donor write and the bounded learned correction both recovered the **clean donor model answer in 100% of the eligible cases** across all eight trained arm/roots (897 total across both authorized strata). This is the expected positive control from a close physical reconstruction of the shared teacher matrix. All factor weight MSEs remained below the registered 1e-4 ceiling. Rank-24 output SVD behaved similarly; singleton BPD factors remain gauge dependent. The privileged Newton oracle commonly enforces the public scalar level set at ~1e-5 but can reduce authorized donor recovery, further showing that public equality alone is an insufficient mechanism criterion.

## Independent public U cohorts and objective competition

Every public U cohort contains 512 generated records **regardless of the clean model's correctness**. The ideal public scalar target is unchanged by a last-source-tag-only change. The measured source-conditioned output variation of the *clean* model is therefore an actual synthetic model imperfection. The full physical BPD donor write closely reproduces that clean source-tag effect, while the nonprivileged policy removes only part of it.

Within each root, the current penalty makes visible prequery states less sensitive to source tags; this has measurable effects on authorized clean-current intervention eligibility and sometimes on public-role accuracy. The public penalty suppresses role-U source-tag output sensitivity at the cost of a larger source-conditioned prequery scalar or weaker authorized performance. The joint objective offers a partial compromise while increasing OOD task error. The experiments do not imply these objectives are universally incompatible. They reveal a constraint under this 32D shared recurrent architecture and the particular fixed unit-strength penalties.

Matched controls were rerun per authorized cohort using 16 random-dark and 16 signed authorized-future-gradient-and-dose-matched interventions **per model/arm**, each with the same frozen learned correction applied after the initial matching. These are dependent within-root observations and cannot support a rank p-value or 95/95 independent-root tolerance. Full raw control distributions and eligible denominators are preserved in the hosted artifact.

## Scientific admission and next approach

- **Engineering source, tests and native parity: PASS.** All eight trained root/arm models completed with finite results, exact current readout conservation and physically gauge-stable full BPD writes.
- **Current-readout closure: improved** under current-only and joint penalties on F_train and in many heldout source-aware authorized pairs.
- **Public readout preservation and combined task utility: FAIL** against the frozen development targets. No arm passes both the target public-output error and the ≥100 clean/current-close challenge across both roots.
- **Independent scientific result: NONE.** These are only two previously exposed roots, with eight correlated training arms and one source-family pair. The 59 new root calibration \`6101..6159\` and 32 root confirmation \`7101..7132\` remain unexecuted. The prior SF03 0/32 strong-mechanism and SF04 3/8 public-safe partial-factor failures are preserved; all ARG-03-N sealed cases remain untouched.

The next **separately preregistered development experiment** should add explicit public target supervision on both source-tag variants and investigate separating the unauthorized public readout from the authorized query decoder, with a matched state-capacity/parameter baseline. Evaluate on excluded engineering seeds before using new roots. A direct observation of improved public invariance must be paired with independently correct public outputs to rule out trivial output collapse; the correct-source task remains a separate gate.
