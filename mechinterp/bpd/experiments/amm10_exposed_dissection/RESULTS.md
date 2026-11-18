# AMM-10: what actually drives bounded BPD strategy memory?

**2026-11-18. Exploratory, post-selection mechanism audit.** Original AMM09R1 experiment and ALL of its protected cohorts remain unchanged. This analysis uses **64 already-exposed TRAINING model roots** with full causal-response feature records and **64 already-exposed DEVELOPMENT roots** containing original action/outcome bits, with no new model fitting. The purpose is to decompose the successful +13.623pp AMM09R1 *prospective development* result and select stronger controls for a separately registered future experiment.

**Full methodology and analysis order:** [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md), committed at \`9a8a322e30b02f0fe05008d059ce3fb735f06416\` before AMM10 ablation metrics were computed. The source-only [analysis.py](analysis.py) and [test_analysis.py](test_analysis.py) were committed and tested in [GitHub Actions #44305663753](https://github.com/MoralityLabAI/BPD/actions/runs/44305663753) on the **original immutable GitHub Actions archives**, before a numerical result lock was committed. No AMM09R1 source-policy, scoring, original training or development result was edited. **No confirmation root was accessed.**

## First finding: only nine candidate-varying terms affect the frozen linear ranking

The original trained 18-feature ridge produces a separate score for each of six admissible one- or two-component masks \`S\`, then clips the scores to [0,1] and takes \`argmax\`. Label the current-dark component response on the two future horizons \`Δ₂(S)\`, \`Δ₃(S)\`, and the cosine of the equal-L2 candidate write with the full first-pass donor displacement \`c(S)\`. The unclipped score satisfies

\`\`\`
score(S) = C(recipient,donor,model)
          + 0.1778 * |S|/3
          + 0.0018 * Δ₂(S)
          + 0.0000 * Δ₃(S)
          + 0.2491 * |Δ₂(S)|
          + 0.0190 * |Δ₃(S)|
          - 0.2270 * Δ₂(S)^2
          - 0.0218 * Δ₃(S)^2
          - 0.0051 * Δ₂(S)*Δ₃(S)
          + 0.3037 * c(S)
\`\`\`

Here \`C\` is common to all six candidate masks on the same donor case. It includes the intercept, recipient current readout, donor current-readout difference, common physical dose, normalized singular gap, nuisance-dose fraction and source family bit. The two raw candidate future readouts are exactly \`unpatched_readout + Δ\`; their coefficients combine with the already-present signed-shift coefficients. Candidate-invariant features do not contribute to the **relative score**. Clipping is monotone and can introduce ties, handled by the original lowest-index rule.

**Empirical source check:** Across all **8,192** exposed training donor cases, the analytically reduced formula produced exactly the same mask choices as the frozen 18-feature policy, both before and after clipping: **0 discrepancies**. Maximum preclip score difference \`1.3143e-9\`; candidate-invariant feature spread exactly \`0\`. This is a structural property of the policy's *linear functional form*, not evidence of a separately learned nine-feature policy's identical predictions.

The coefficient magnitudes place the dominant visible signal on alignment \`c(S)\`, bounded first-delayed-readout magnitude, its quadratic penalty and component cardinality. In particular, **the family bit cannot directly change within-case mask order in the already-fitted policy**. Removing family as an *input to a new ridge fit* may still alter the other learned coefficients.

## Second finding: eightfold root-held-out TRAINING ablations

The original 64 TRAINING model roots were partitioned by \`root_index % 8\`. For each fold, all models and their six candidate labels in the other 56 roots trained a new λ=10 ridge with free intercept, followed by evaluation on the eight heldout training roots. Thus each model contributed exactly **128 outcome cases** to a fold's heldout score, for **8,192** cross-fit cases per arm. These are *previously researcher-exposed training roots*, so the results are observational/exploratory rather than independent prospective heldout science.

All nine predeclared feature variants, six source-safe geometric heuristics and four pooled/label-oracle controls were retained:

| Frozen analysis arm | Correct / 8,192 | Success rate |
|---|---:|---:|
| **Full 18-feature ridge** | **3,288** | **40.14%** |
| Compact causal ridge, 10 total columns including intercept | 3,288 | 40.14% |
| Ridge without task-family bit | 3,278 | 40.01% |
| Source-safe cosine alignment rule | 3,155 | 38.51% |
| Ridge using cosine alignment and intercept | 3,155 | 38.51% |
| Ridge without delayed readout probes | 3,136 | 38.28% |
| Linear signed readout shifts and alignment | 3,129 | 38.20% |
| Ridge **without cosine alignment** | 2,295 | **28.02%** |
| Ridge without geometric/dose/alignment features | 2,183 | 26.65% |
| Fold-selected globally best fixed mask | 2,091 | 25.52% |
| Family-specific pooled mask, learned inside fold | 1,928 | 23.54% |
| Cardinality-only ridge | 1,896 | 23.14% |
| Choose largest-cardinality admissible mask | 1,822 | 22.24% |
| Choose largest min(|Δ₂|,|Δ₃|) | 1,763 | 21.52% |
| Choose largest signed shift product | 1,060 | 12.94% |
| Choose largest total absolute delayed shift | 829 | 10.12% |
| Choose largest squared delayed shift | 498 | 6.08% |
| Per-root **best fixed mask using heldout answers** | 2,921 | 35.66% |
| Per-case **six-mask oracle using heldout answers** | 4,461 | 54.46% |

The two oracle rows access heldout outcome labels and are **diagnostic ceilings only**. In this root-heldout training exercise, a candidate-specific policy can outperform even the best single fixed mask chosen retrospectively for each model because it is free to switch masks between donor cases.

### Paired root diagnostics (all exploratory)

For the nine-retrained-coefficient family, the full ridge gains over the pure cosine alignment rule by **133/8,192 = +1.62pp**. Its descriptive independent-model bootstrap 95% interval is **[+0.15,+3.21]pp**, with **31 positive, 24 negative, 9 tied roots**, and one-sided descriptive root-sign p **.2094**. The sign distribution conveys less consistent improvement than the interval, especially given the many exploratory comparisons and re-used training roots. Treat this as *incremental predictive signal to test prospectively*, not an established mechanism.

Removing cosine alignment reduces full-ridge accuracy by **993/8,192 = 12.12pp**. A descriptive paired-root 95% interval is **[+8.63,+15.67]pp** for full minus no-alignment, with 50 positive, 13 negative and one tied root. Removing delayed-readout probes costs **152/8,192 = 1.86pp**, interval **[+0.52,+3.26]pp**, with 33/22/9 positive/negative/tied roots. All effects use one independent model root as the resampling unit; the repeated donor pairs are nested observations.

A newly fitted compact causal ridge tied the full ridge at **3,288 successes**, although their selected actions **differed on 787/8,192 cases (9.61%)**, with 18/27/19 positive/negative/tied root differences. Thus equal accuracy of those *two refits* does not prove equal policies or an equivalence hypothesis. The cosine-only ridge and the simple \`argmax cosine\` heuristic selected **exactly the same mask on all 8,192 cases**. Full ridge and cosine differed on 1,615 cases (19.71%).

Other negative exploratory heuristics remain in the table; optimizing to the largest delayed response alone is a poor strategy for donor-specific *error repair*. The cosine score carries information about the particular permitted intervention's direction relative to the *known first-pass donor displacement*. This proxy uses a richer physical reference than pure component cardinality.

## Third finding: original exposed DEVELOPMENT, using action bits only

The original 64 independent AMM09R1 development roots had **512 source-defined donor pairs each** (32,768 cases). Their original full candidate feature tensors were omitted from the archived receipt. AMM10 makes **no new development action predictions or feature ablations**. The independent analyzer verifies every original selected mask against its archived success bit and compares existing strategies with retrospective fixed-mask bounds.

| Exposed DEVELOPMENT diagnostic | Correct / 32,768 | Rate |
|---|---:|---:|
| **Frozen AMM09R1 memory-only policy** | **12,502** | **38.15%** |
| Frozen memory + noisy local feedback | 11,340 | 34.61% |
| **Per-model hindsight best fixed mask** (uses test answers) | **11,509** | 35.12% |
| From-scratch equal-budget local ridge | 8,956 | 27.33% |
| Training-selected global pooled mask (0,2) | 8,038 | 24.53% |
| Training-selected family-specific pooled masks | 8,038 | 24.53% |
| Local per-mask noisy success count | 7,988 | 24.38% |
| Random admissible mask | 5,368 | 16.38% |
| Per-case six-mask oracle (uses test answers) | 18,141 | 55.36% |

**The strongest fixed mask chosen using all of DEVELOPMENT itself was also (0,2)**, recovering the same 8,038 cases as the training-selected global mask. Separate fixed-mask choices learned from the **TRAINING** wren and basalt families likewise both select (0,2) and reproduce this total. A training-family lookup cannot explain the frozen policy's benefit.

More demanding is an audit-only **best fixed permitted mask chosen after looking at all 512 outcomes within each development model**. Summed across the 64 models, this retrospectively selected model-specific fixed oracle reaches **11,509/32,768 = 35.12%**. Frozen memory reaches **12,502 = 38.15%**, a **+993/32,768 = +3.03pp** descriptive gain over that fixed-per-model ceiling. Memory exceeds that hindsight fixed-mask performance on **34 of 64 roots**, loses on 29 and ties on one; a paired-root descriptive 95% interval for this difference is **[−0.03,+6.24]pp**. Because this interval includes zero, the stronger per-root diagnostic remains **suggestive**.

Mask choice is genuinely case-dependent in the recorded development decisions: memory selects multiple admissible masks on **all 64 roots**, and selects a mask different from the global pooled mask on **21,459/32,768 cases (65.49%)**. All its selected masks belong to the **three two-component subsets**: (0,1), (0,2), (1,2). It selects no single-component intervention on these exposed development cases. The original behavioral benefit is therefore localized to routing *between pairs of canonical components*.

## Evidence and EvalMath boundary

The original AMM09R1 **prospective** development study registered an +8pp meaningful-effect threshold, matched model-root pooled repeatability shams, four Holm comparisons and an untouched confirmation population. Its corrected EvalMath preflight returned \`READY_FOR_CONFIRMATION\`, and its recorded primary effect was +13.623pp versus the globally fixed pooled policy. That result is unchanged.

AMM10 is **selection-exposed**: it fits and scores ablations inside the historical TRAINING corpus and computes new diagnostics on historical DEVELOPMENT outcome bits. It has no independent new test cohort, new matched shams, root-specific fidelity remeasurement or confirmatory release. The rooted sign tests and bootstraps reported here are descriptive. There is **no new EvalMath READY**, no new +8pp mechanism claim and no authority to access sealed roots. Original equal-dose/current-dark physical witnesses remain the relevant intervention controls.

This decomposition also prevents a premature causal interpretation of the learned 18-feature weights: the most important component of the practical selection benefit is reproducible by an **analytical directional alignment rule** in the exposed training-root test. A future mechanism claim requires a head-to-head *prospective* comparison against that stronger rule on new independently trained models.

## Prospective follow-up: AMM-11

Preregister a new namespace with fresh model training, source-frozen methods, matched physical dose, and at least the following primary competing controls: **(a)** full ridge memory, **(b)** fixed \`argmax cosine\` geometric rule, **(c)** compact fitted ridge, **(d)** training-selected pooled mask, and **(e)** same-budget local root adaptation. Test whether the full learned memory improves over **cosine alignment** by a meaningful margin, with independent-root repeatability sham and corrected EvalMath preflight. Keep the original three-pass/two-slot synthetic task for a clean bridge, followed by a separate nonlinear/TRM evaluation to test substrate transfer.

Do not tune these alternatives on the sealed AMM09R1 confirmation population. Every AMM03–09R1 confirmation population and ARG-03-N remains untouched.

## Reproduce and provenance

- Original raw TRAINING SHA256: \`015f8c78181514a0702e56ad707444699e48b377d9bf29da8ec60a42a0d48d30\` (16,143,086 bytes).
- Original raw DEVELOPMENT SHA256: \`682473974fa15db6005dab588d66d73c2a58bba1d5c8f1f4a91a93899ea88e5a\` (1,868,507 bytes).
- Frozen AMM09R1 global policy SHA256: \`b34b9a322ff5aeaa517eb5edd4bf0aced9ccf70f0bf97383d8e6cca79de2f0e9\`.
- First successful full AMM10 hosted [GitHub Actions #44305663753](https://github.com/MoralityLabAI/BPD/actions/runs/44305663753) passed all seven synthetic/source-integrity tests, raw archive SHA checks, full root-heldout training ablations and exposed development selected-action audits.
- Full hosted AMM10 JSON SHA256: \`88d7e2f6bd4c04097196c723af9667686b3e3cbcc55c37dbd99392eb7c59c648\`; artifact \`amm10-exposed-root-causal-dissection\`, ID \`14428236123\`. Its digest and all 19 arm totals are locked in \`results/RESULTS_LOCK.json\`. The CI workflow independently regenerates the full output from the original source-of-record files and enforces these locks.

Evidence scope: trained *synthetic 6D rank-three linear recurrent BPD* with source-safe first-pass donor geometry, plus post-selection exploratory cross-fitting. No direct claim about nonlinear TRMs, pretrained transformer neurons, agent-to-agent memory learning or deployed systems.
