# BPD-SF02: hosted registered fresh-root results

**Executed 2026-11-17. Source-locked, exploratory synthetic CPU study.** [Registered hosted workflow](https://github.com/MoralityLabAI/BPD/actions/runs/44117355061) completed successfully and verified the Git source/protocol blobs against [LOCK.json](LOCK.json) before source training. [Raw GitHub Actions artifact](https://github.com/MoralityLabAI/BPD/actions/runs/44117355061/artifacts/14342967980); [in-repository raw data](results/sf02_registered_hosted.json). Pinned PyTorch environment reported `2.5.1+cu124` on CPU.

## Registered decision

- Exploratory causal gate: **8/8 roots PASS** (minimum 6/8). All eight selected six-component interventions beat the best of 255 per-pair equal-norm random-dark sham directions by >=20 percentage points and recover the donor clean final readout on at least 60% of eligible authorized pairs.
- Engineering acceptance: **7/8 roots PASS; COMPLETE ENGINEERING GATE FAIL.** Root 1021's physical BPD reconstruction MSE is `0.0002615572011563927`, above the frozen 1e-4 tolerance. The failure is retained unchanged. All other tested engineering checks passed in the hosted receipt.
- Pooled authorized donor-readout recovery: **1633/1738 = 93.96%**. This is a ±0.10 numerical tolerance to the *donor model's own clean future output*, not a text secret-extraction rate. Complete readout-dark donor state transfer scored 100% on each root.
- 1053 eligible unauthorized-role examples across eight roots. Mean absolute future output change from a selected partial six-factor dark edit was **0.103** unweighted by root, compared with **0.093** for norm-matched random-dark edits and **0.010** under the clean donor counterfactual. Partial intervention effects are off-manifold diagnostics. None are classified as natural-language or exploitation leakage.

## Per-root audit

| Root | Eligible auth | Recovery | Best sham | Factor MSE | Public partial Δ | Public null Δ | Engineering |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1013 | 214 | 86.4% | 2.8% | 1.257e-12 | 0.014 | 0.088 | PASS |
| 1019 | 221 | 100.0% | 5.4% | 6.221e-6 | 0.104 | 0.042 | PASS |
| 1021 | 202 | 71.8% | 5.0% | 2.616e-4 | 0.088 | 0.082 | FAIL |
| 1031 | 232 | 98.3% | 7.8% | 2.813e-6 | 0.125 | 0.115 | PASS |
| 1033 | 215 | 93.0% | 8.4% | 8.874e-10 | 0.244 | 0.110 | PASS |
| 1039 | 219 | 100.0% | 8.7% | 1.700e-8 | 0.015 | 0.144 | PASS |
| 1049 | 218 | 100.0% | 10.6% | 4.052e-12 | 0.171 | 0.138 | PASS |
| 1051 | 217 | 100.0% | 5.5% | 2.498e-11 | 0.064 | 0.023 | PASS |

Every root has 384 source-disjoint test pairs, with 240 calibration pairs for greedy factor selection. Each factorization is full rank (12 components) and reconstructed independently of heldout donors. Immediate edited-readout change was zero to floating precision for all eight roots. The GL(12) shear preserved full physical weights, while singleton component assignments drifted.

## Evidence limitations and follow-up

This result demonstrates trained synthetic two-pass dynamics where current-readout-dark perturbations can carry source-specific information to a later authorized output. The partial BPD factor block predicts donor changes strongly, yet its **coordinate identity is not gauge invariant**. The eight roots share one continuous scalar-routing task family and do not meet the 59-root independent sham-calibration requirement. Matching random-dark write norm does not establish task-sensitivity exchangeability. The teacher directly receives the secrets, public context and recipient role. The study tests neither language-model source provenance nor actual secret disclosure.

Preserve root 1021's bad factor fit as an optimization failure; a separately registered recovery study can test stronger optimization or different factorization criteria without modifying the frozen source, thresholds or receipts here. A source-family-disjoint confirmation and a leak-vs-utility probe on a real open recurrent model are future work. Existing ARG-03-N sealed data remain untouched.
