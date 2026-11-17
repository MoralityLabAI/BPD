# SF05B Stage 2 power preflight: donor eligibility and public correction

**Post-Stage-1 design note, November 18, 2026.** This note uses the excluded-root [Stage 1 receipt](RESULTS_STAGE1.md) and **does not modify its frozen test or claim a confirmatory result**. Reserved 59+32 scientific root IDs remain untouched.

## Why 512 heldout records were inadequate

The Stage 1 exact matched-pair admission requires an authorized query, a correct recipient and donor answer, ≤0.05 prequery readout difference, and ≥0.20 future donor/recipient divergence. Only **18/512** pairs from excluded root 901 and **13/512** from excluded root 902 passed. Within each development root these are observed eligibility rates 3.52% and 2.54%.

A Wilson binomial 95% interval, **conditional on treating generated source rows as iid within a given teacher root**, is approximately:

| Root | Observed eligible rate | Wilson 95% interval |
|---:|---:|---:|
| 901 | 3.52% | [2.24%, 5.49%] |
| 902 | 2.54% | [1.49%, 4.30%] |

These intervals do **not** account for root-to-root teacher variation and cannot replace independent-root inference. They illustrate why a nominal 512-pair test on a single teacher will not resolve a 5-percentage-point intervention effect reliably.

If eligibility stayed at seed 902's observed 2.54%, 4,096 prospective generated pairs would yield roughly 104 eligible pairs on average, with only ~67% probability of reaching at least 100. The rate's 95% Wilson lower endpoint (1.49%) implies that 8,192 generated pairs would yield ~122 expected eligible, with ~98% chance of at least 100 under that **fixed** binomial approximation. These are planning calculations, not new experimental measurements.

## Prospectively improve the source-pair generator

Stage 2 should choose the authorized query role with reference to the **source labels** of the last recipient/donor write, so that the query asks for the old or new source rather than being drawn uniformly from {A,B,C,U}. This conditions only on public synthetic generator metadata, **before any model is run**. Preserve the same scalar payloads and all earlier source history. Optionally add a separately registered high-contrast stratum in which candidate scalar values differ from the prior write to the queried source by a predetermined ≥0.30, while retaining a fully unconditional diagnostic cohort.

Keep a separate *unauthorized role U* public-query cohort, with independent generated values, for leakage/preservation measurement. Avoid mixing the authorized-target eligibility numerator and the public-query harm denominator. Preserve both intention-to-test (all generated cases) and the clean-eligible analysis, since conditioning on model-correct outputs creates selection bias.

Run this new donor generator on **excluded development roots only** to estimate admission before locking fresh calibration/confirmation roots. The generator and source hashes must be frozen before scoring new roots; any posthoc adjustment is a separate preregistered experiment.

## Mechanism admission beyond oracle-feasibility

The learned predictor improved heldout mean public Δ only from 0.0414→0.0341 and 0.0688→0.0636, while the privileged Newton oracle reached ~1e-5. Stage 2 should demand a meaningful public preservation criterion on **all generated public-query cases**, for example ≤0.02 mean public Δ, alongside authorized utility; the precise threshold must be finalized after excluded-root design and before fresh test outcomes.

A primary causal comparison needs the **same correction policy** applied to appropriately matched random, signed-gradient and teacher-C physical arms. A 59-root maximum of a preregistered root-level pseudo-sham statistic can calibrate only the stated null-root distribution under iid assumptions. Confirm the matched directions' exchangeability if seeking rank-based p-values, and report nonexchangeability explicitly otherwise. Retain the complete operator's GL(32)-gauge invariance and distinguish output SVD rank 24 from arbitrary singleton BPD factors.

Minimum future deliverables: source-family-disjoint recipient/donor generator with independent authorized/public strata, a deterministic exact target oracle, an excluded-root power pilot, a **new** Stage 2 protocol with 59-root sham calibration and 32-root sealed confirmation, and root-level receipts with all negative/underpowered outcomes. ARG-03-N's 168-case heldout remains sealed.
