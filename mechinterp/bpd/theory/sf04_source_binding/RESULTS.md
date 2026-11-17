# BPD-SF04: prospective eight-root interleaved-source binding results

**2026-11-18.** [Source-frozen GitHub Actions](https://github.com/MoralityLabAI/BPD/actions/runs/44131840684) verified all [LOCK.json](LOCK.json) Git blobs before generating test results. Complete [hosted raw JSON](results/fresh_8_hosted.json), [frozen per-root verdict](results/frozen_verdict.json), and [uploaded ZIP receipt](https://github.com/MoralityLabAI/BPD/actions/runs/44131840684/artifacts/14351435053). The hosted job completed successfully, **including scientific-failure reporting**. PyTorch 2.5.1 ran on CPU. Excluded development roots 901/902 are absent from the fresh cohort.

## Three preregistered gates

- **Engineering: PASS 8/8.** Every physical BPD fit has MSE ≤1e-4, teacher MAE ≤0.08, full GL(20)-gauge residual ≤2e-5, immediate projected intervention current-readout error ≤1e-6, and random sham norm/gradient numerical errors ≤1e-4. The separate native wrapper [engineering CI](https://github.com/MoralityLabAI/BPD/actions/runs/44131361409) passed six source tests and actual 3-pass BPDModel + donor-write parity (errors 8.35e-7 and 3.58e-7).
- **Source provenance learning: PASS 8/8.** Learned teacher answers are substantially better than the fixed source-blind conditional-mean oracle, and full physical-factor and exact dark-donor transfers both recover 100% on all roots. The trained teacher effectively remembers which user wrote the latest value. The current readout remains near zero before the query, by training design.
- **Partial BPD coalition with public boundary: FAIL 3/8**, against registered requirement ≥6/8. Only roots 1303, 1307 and 1319 simultaneously meet: 14-factor donor recovery ≥60%; ≥20pp above the signed-gradient sham mean; and public-role mean absolute output disturbance ≤0.08. The other five roots retain their negative outcomes.

## Aggregate quantities

| Metric | Eight fresh independent roots |
|---|---:|
| Generated recipient/donor pairs | 4096 (512 per root) |
| Eligible authorized pairs | 1993 |
| 14/20-factor donor recovery | **1491/1993 = 74.81%**, pooled |
| Unweighted root mean selected-factor recovery | 74.35% |
| Unweighted root mean signed-gradient-matched sham recovery | 8.96% |
| Unweighted root mean norm-matched random-dark recovery | 9.10% |
| Full physical BPD factor block and exact dark-donor recovery | **100% on every eligible root** |
| Teacher absolute error, root mean | 0.0182 |
| Fixed source-blind conditional-mean absolute error, root mean | 0.1613 |
| Before-query scalar readout absolute magnitude, root mean | 0.0109 |
| Public-role partial 14-factor absolute change, root mean | **0.0949** |
| Public-role clean source-tag swap absolute change, root mean | **0.0165** |
| Public-role full physical factor block effect, root mean | 0.0165 |

Note: The source-blind conditional-mean estimator is optimal under squared error, but its MAE is a descriptive reference rather than the minimum source-blind MAE. See [THEORY.md](THEORY.md) for the exact source-blind Bayes-MSE floor.

## Per-root evidence

| Root | Eligible | 14-factor recovery | Signed sham | Public partial Δ | Clean public Δ | Factor MSE | Partial composite |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1301 | 252 | 77.4% | 2.2% | 0.162 | 0.024 | 1.03e-8 | FAIL |
| 1303 | 227 | 87.7% | 5.4% | 0.055 | 0.015 | 4.53e-7 | PASS |
| 1307 | 249 | 96.0% | 12.4% | 0.062 | 0.019 | 2.93e-6 | PASS |
| 1319 | 265 | 83.8% | 15.5% | 0.070 | 0.010 | 7.37e-7 | PASS |
| 1321 | 247 | 97.2% | 9.2% | 0.083 | 0.014 | 6.07e-7 | FAIL |
| 1327 | 269 | 78.1% | 5.3% | 0.159 | 0.016 | 5.85e-8 | FAIL |
| 1361 | 252 | 62.7% | 11.9% | 0.089 | 0.020 | 6.39e-9 | FAIL |
| 1367 | 232 | 12.1% | 9.8% | 0.080 | 0.014 | 2.24e-10 | FAIL |

### Interpretation

The source tag is the *only* changed input between recipient/donor pairs. This tests source-to-content binding while holding numerical payloads, event order, first event and queried user fixed. The current pre-query visible readout is held exactly equal during causal interventions. A future query recovers the authorized source in a trained recurrent nonlinear circuit, with substantial transfer using some selected physical BPD factor coalitions.

The **five composite failures matter**. Root 1367 recovers only 12.1% of authorized donor answers with 14 selected factors although the full factor block is nearly exact. Other roots recover many donor answers but disturb the public role more than permitted. This is consistent with component-gauge nonuniqueness, off-manifold effects and nonlinear input-conditioned routing. It is not evidence of real-life secret exfiltration; the public effect is an edited-state synthetic diagnostic. Fully reconstructing the shared physical operator is also a positive control rather than discovery of sparse canonical factors.

The eight roots share one synthetic scalar source family. Their individual test pairs and 64 sham directions per root are subordinate observations; root is the primary unit. There is **no 59-root 95/95 null calibration or independent source-family confirmation**. Future work should test gauge-stable physical subspace coalitions and public-preserving edits selected only on calibration, with true open-model token outputs and source-tagged textual histories. Preserve the sealed 168-case ARG-03-N test and all earlier failed BPD gates.
