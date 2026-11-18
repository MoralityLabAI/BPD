# SF05B Stage 2-A: source-aware model-level donor admission on excluded teachers

**November 18, 2026, development-only.** [Prospectively frozen source lock](LOCK.json) passed in [hosted GitHub Actions #44221039293](https://github.com/MoralityLabAI/BPD/actions/runs/44221039293) **before any model-level scoring**. The job passed 7 source/interface tests, real seven-call native \`BPDModel/TiedComponentLinear\` parity (max 9.54e-7) and native source-donor write parity (7.15e-7). Full two-root, three-cohort [raw JSON artifact](https://github.com/MoralityLabAI/BPD/actions/runs/44221039293/artifacts/14390902136), SHA-256 \`77dc1730bfab6341de24e8d36ed55f8710c3f7cd8ec00a498af309048cf52958\`. A [compact in-repository receipt](results/excluded_2roots_compact.json) retains key values and the raw artifact reference.

The only trained teacher/factor/policy roots were **901,902**, both previously used as Stage 1 development roots. Original teacher parameters were re-created from deterministic frozen training schedules (2,600 teacher, 450 BPD factor, 240 bounded correction updates), all trained on short F_train histories. No new 59-root calibration or 32-root confirmation was executed. The Stage 2 donor source family remains structurally heldout (5–6 writes), and the target-changing last-source tag, both authorized query roles and independent public role U cohorts were generated solely from the exact Stage 2 source oracle.

## Eligibility improvement and persistent bottleneck

The previous Stage 1 uniform-role heldout source-pair assay had only **18/512** eligible pairs on 901 and **13/512** on 902. The new data-oracle source-aware authorized selection increases the usable population:

| Excluded root | Earlier Stage 1 | Source-aware unconditional | Source-aware high-contrast |
|---:|---:|---:|---:|
| 901 | 18/512 (3.5%) | **43/512 (8.4%)** | **42/512 (8.2%)** |
| 902 | 13/512 (2.5%) | **20/512 (3.9%)** | **29/512 (5.7%)** |
| Pooled descriptive | 31/1024 (3.0%) | **63/1024 (6.2%)** | **71/1024 (6.9%)** |

This is a **cross-cohort comparison** on the same two trained roots, with changed query/source-admission laws and different deterministic history draws. It is neither a randomized intervention effect nor independent-root confirmation. The selected high-contrast cohort also has a distinct conditioned population, as specified before scoring.

All 512 pairs in each authorized cohort change the exact source-oracle target. Within the high-contrast candidate streams, 1,351/2,048 (901) and 1,344/2,048 (902) records met the independently defined ≥0.30 exact-label gap; the first 512 qualifying records were used. The high-contrast intervention did **not** deliver a large further gain in model-level eligible cases.

### Exact stepwise model gate

| Root, cohort | Exact source-target gap mean | Both teacher answers within ±0.10 | Both correct and current readout difference ≤0.05 | Final eligible after model future change ≥0.20 |
|---|---:|---:|---:|---:|
| 901, unconditional | 0.5231 | 237/512 | 58/512 | **43/512** |
| 901, high contrast | 0.7116 | 198/512 | 42/512 | **42/512** |
| 902, unconditional | 0.5397 | 153/512 | 32/512 | **20/512** |
| 902, high contrast | 0.7484 | 155/512 | 29/512 | **29/512** |

The primary bottleneck has shifted from the random query's missing source-label effect toward **clean teacher competence and prequery observable-state agreement**. On the high-contrast population, the model future-output gap test admits 491/512 (901) and 474/512 (902) independently; the clean/curr-readout intersection only admits 42 and 29. The prequery current readout \`q(h)=h[0]\` remains measurably source-sensitive. A successful dark intervention is therefore more difficult than an exact label-gap alone suggests.

## Physical BPD and matched interventions

The full 32-component BPD \`VU\` reconstruction and separately frozen source-tag donor edit recovered the **clean donor final output within ±0.10 in 100% of the 134 eligible source-aware authorized pairs** (43+42+20+29). The bounded nonprivileged learned correction retained those same donor outcomes. The complete reconstructed operator and output SVD rank-24 produce very similar effects; this is a **full physical reconstruction positive control**, not evidence of a sparse, gauge-canonical secret-bearing factor. In the new cohorts the privileged finite public-output Newton oracle sometimes perturbs otherwise successful donor transport, further separating public preservation from target recovery.

Each authorized cohort included 16 within-root norm-matched corrected random-dark shams and 16 **pre-correction signed authorized-future-gradient-and-norm matched** shams. For the high-contrast cohort, the signed-gradient corrected-sham donor recovery means were 23.2% (901) and 7.1% (902), compared with 100% for the full physical BPD candidate. These are source-dependent **within-root developmental comparisons**, with only two independent teacher roots; a root-level 95/95 tolerance or exchangeable null test has not been established.

## Public U cohort: no clean-label eligibility filtering

A separate 512-record public-role cohort was generated for each root, with the exact public source-oracle answer held unchanged under the last-write source-tag flip. Every record counts in the public-output error, whether its clean model answer is accurate or not.

| Independent public-role measure (512 records each) | Root 901 | Root 902 |
|---|---:|---:|
| Clean public source-tag-only donor effect | **0.03981** | **0.06966** |
| Raw physical BPD edit public absolute Δ | 0.04168 | 0.07285 |
| After frozen learned correction | **0.03409** | **0.06717** |
| After privileged oracle repair | 0.0000157 | 0.0000123 |
| Learned edit records with public Δ ≤0.02 | 38.3% | 19.7% |
| Clean recipient public-target MAE | 0.04199 | **0.05510** |

Neither root meets the provisional mean public-output-deviation target **≤0.02** after the nonprivileged correction. Moreover, root 902's **0.05510 clean public-target MAE fails** the descriptive public-role competence threshold ≤0.05 on this new independent U cohort, even though that same root passed the separately sampled Stage 1 overall heldout admission. This failure is retained.

The clean public output also changes when only the last *private source tag* changes (0.03981 / 0.06966), despite the public scalar target remaining fixed by construction. This measures **source-conditioned synthetic public-output dependence** in the trained teacher; no actual text secret, user information or unauthorized external disclosure is involved. The raw full physical edit follows that clean donor-state effect, while the present policy only partly corrects it. The privileged oracle forces the scalar public output back to baseline using true future-output and derivative access, which is a high-information ceiling rather than a deployable solution.

## Scientific and technical decision

- **Engineering: PASS.** SHA/source gates, seven exact tests, native wrapper parity, gauge-stable complete physical edits, current readout preservation, frozen correction budget and complete denominator accounting passed.
- **Source-aware donor admission: partial descriptive improvement.** Both new authorized cohorts approximately double pooled eligible counts compared with the original Stage 1 source sampling. The usable donor population is still only 3.9–8.4% per root/cohort, far below a robust mechanism evaluation target.
- **Public-safe learned policy: FAIL versus the proposed ≤0.02 public deviation reference.** Root 902 also fails public-role clean task competence on its separately sampled U cohort. All public cases remain counted.
- **Scientific inference: UNADMITTED.** Two previously used teacher roots, conditioned source populations and full physical reconstruction provide no independent calibrated statement about secrecy-bearing BPD components, real generated text or general privacy.

A future registered experiment should address the **current readout's source dependence and the public-query model's sensitivity to private source tags** before spending the reserved 59 calibration roots or 32 confirmation roots. Candidate development directions include stronger prequery visible-readout regularization, public source-invariance training, or a separately defined rank-stable physical subspace with explicit nonlinear public-level-set correction. Each would change the teacher objective/architecture and must have its own source lock and excluded-root engineering cohort before any fresh-root confirmatory study.

The 168-case ARG-03-N sealed holdout, prior SF03's 0/32 strong-mechanism failure and SF04's 3/8 partial public-safe-routing failure remain untouched.
