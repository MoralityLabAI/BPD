# AMM-09 post-development analysis correction and confirmation hold

**Recorded 2026-11-18 after the first completed DEVELOPMENT scoring, before any CONFIRMATION model has been trained or queried.** This note is an openly labeled **post-data reporting/analysis correction**. The preregistered scientific protocol `PROTOCOL.md`, model generator `models.py`, causal assay `case.py`, action-selection code and frozen experiment source `experiment.py` remain unchanged, preserving their registered SHA256 values and all root/action outcomes.

## Error discovered

The AMM09 protocol registered **memory_only minus pooled** as its primary contrast. The original `experiment.py` correctly constructs the primary EvalMath `preflight_input` from `p['memory_only']-p['pooled']` for every independently trained DEVELOPMENT model root. The source-locked corrected EvalMath receipt returned `READY_FOR_CONFIRMATION` from **+4,842/32,768 = +0.14776611328125** observed primary effect and an independently calibrated **0.064453125** repeatability sham tolerance bound.

A copied line in the original `analysis()` function instead computes **`memory_plus_local - control` for every comparison name**. For the key `memory_only_vs_pooled`, it reports **+3,266/32,768 = +0.09967041015625**, which belongs to the registered secondary `memory_plus_local_vs_pooled`. The case-level scores, policy choices, heldout target labels, physical interventions, source hashes and preregistered EvalMath pilot inputs are unaffected. This mistake impacts the development report's primary root-sign p-value and bootstrap interval and its comparison multiplicity family.

## Fixed source-preserving audit

The independent post-data file `audit_development.py` reads the archived original DEVELOPMENT raw receipt, its original training-policy lock and the historical development semantic bit lock. It validates all 64 root memberships and every selected action's scored bit against the six registered candidate arms; rejects a forbidden full-rank mask; rechecks the primary EvalMath input and pooled-repeatability sham against root case counts. It then computes **all four preregistered comparisons** from their exact named treatment and control arms. Each one-sided exact sign-test tail is calculated on independent model roots only, using their nonzero paired counts; the 2,000 fixed-root-bootstrap percentile interval uses the preregistered `AMM09/bootstrap/<comparison>` hashed RNG stream; Holm correction applies to the complete family of four. No policy was selected or changed using development data.

The corrected primary root results, calculated independently from the first-run raw receipts, are:

- `memory_only - pooled`: **+4,842 successes / 32,768 = +0.14776611328125**, root directions **55 positive / 8 negative / 1 tie**.
- One-sided exact root-sign `p = 4.880836543307357e-10`; four-test Holm adjusted `p = 1.952334617322943e-9`.
- Frozen 2,000-resample paired-root 95% percentile interval **[+0.111541748046875, +0.186614990234375]**.
- Corrected secondary `memory_plus_local - pooled`: **+3,266 / 32,768 = +0.09967041015625**, root directions **47/16/1**.
- Corrected secondary `memory_plus_local - local_ridge`: **+2,169 / 32,768 = +0.066192626953125**, root directions **47/15/2**.
- Corrected secondary `memory_plus_local - memory_only`: **−1,576 / 32,768 = −0.048095703125**, root directions **14/47/3**.

This **development** result passes the prespecified primary scientific effect floor +0.08 with a lower root-cluster bound +0.11154; its four-test adjusted p-value is below .05. The corrected EvalMath preflight was already READY using the correct primary independent-root effects. Additional preregistered fidelity **63/64** and audit-only allowed-six-mask oracle headroom **27.78 percentage points above pooled** pass their vetoes. Dose/current-readout parity also passes. The memory-only policy differs from pooled on **21,459/32,768 cases** and uses multiple masks on all **64 independent models**, ruling out the AMM08 full-mask policy collapse on this developmental instrument.

## Reproducibility and release boundary

Keep the original source-computed `results/DEVELOPMENT_LOCK.json` with its mislabeled `analysis` as an immutable historical receipt, accompanied by `results/ANALYSIS_CORRECTED.json` produced by the independent auditor. GitHub CI replays the original experimental source and **also** verifies the independent corrected analysis against the original named-arm case bits and historical EvalMath input. The raw first-run archive `BPD_AMM09_hosted_training_development_20261118.zip` is retained with SHA256 `4affd7357053e31c8c43a62a2e5ea3aa751f647af9c779e66a35ff09829a37a6`; raw `amm09_development.json` SHA256 `0423c8302c9cbb99b9413ac7e22cfcc7d0f10d921eff06c0f4a17cda33ff3925`. Raw floating point data are archived separately from the semantic case-bit digest.

**No confirmation inference has been performed.** Although this developmental study meets the predeclared effect, power, fidelity, headroom and dose admission criteria, AMM09's protocol additionally requires a **separately committed explicit human-approved release manifest** before accessing any of 128 sealed confirmation roots. That authorization has not been given. A prospective confirmation may still fail; avoid describing the current result as confirmed or generalizing it beyond trained synthetic linear BPD systems.
