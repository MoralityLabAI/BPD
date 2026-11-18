# BPD-NC-03: gauge-covariant subspace engineering receipts

**Study:** BPD-NC-03, 18 November 2026 UTC.  
**Evidence class:** source-registered synthetic positive-control instrumentation, **not** natural-model mechanistic discovery or prospective confirmation.  
**Prospective protocol-only commit:** `a48871f2612f9032ddbf612747f972f701585536`.  
**Run:** [Actions 44232604891](https://github.com/MoralityLabAI/BPD/actions/runs/44232604891) (source/smoke and 8-root development both successful).  
**Raw artifact:** [nc03-eight-root-full-engineering](https://github.com/MoralityLabAI/BPD/actions/runs/44232604891/artifacts/14395498498), containing eight complete root JSON files and `NC03_SUMMARY.json`.  
**Implementation source SHA-256:** `3879482d1ea3ca894ee10884d32de0e0a4d772c2101976e2b50b7489e45fddda`.  
**Protocol SHA-256:** `25640c1c99076ae41d88492c14d715bc065c9d1b8506ed6f95cd13e789182422`.  
**Zipped artifact SHA-256:** `4e0f2713440aa89c830e3cbcc4a61e56126f30f389d2ef623323796eea605440`.

## Outcomes

Eight independently trained nonlinear shared-weight shift-register model roots, seeds **300–307**. Each received 2048 training examples, 1024 independent discovery cases and 1024 held-out evaluation examples, all new streams distinct from NC-02. Each root used 128 optimizer steps. The source CI verified nine deterministic tests and one reduced-size engineering smoke before running development.

| Metric | Native factor basis | Orthogonally rotated factor basis |
|---|---:|---:|
| Eligible independent roots | 8/8 | 8/8 |
| Min retrospective rank-2 planted-subspace overlap | 1.0000000 | 0.99999996 |
| Mean final KL(clean ∥ keep-projector) | **0.0 nats** | 4.81e-15 nats |
| Mean sham-adjusted keep-only KL advantage | **+0.935538 nats** | **+0.935538 nats** |
| Root-bootstrap 95% interval of this advantage | [0.844274, 1.031270] | [0.844274, 1.031270] |
| Mean label NLL increase when causal subspace removed | +0.676211 nats | +0.676211 nats |
| Model-blind physical-dose-eligible sham draws | 2048 / 2048 per root | Same eight *physical* shams per root |

The maximum relative physical operator difference between the two projector bases was **2.76114e-8** over all four passes and all eight roots. Maximum retained-output logit discrepancy was **3.09944e-6**. The largest factor projector transport discrepancy was **2.85274e-8**. All eight trained roots passed both context-accuracy gates and all eight zero-readout controls returned **NO_IDENTIFIABLE_SUBSPACE**.

## Primary native effect by independent root

| Seed | Difference between median matched-sham final KL and recovered-projector final KL (nats) |
|---|---:|
| 300 | +0.93764 |
| 301 | +1.15138 |
| 302 | +0.81085 |
| 303 | +1.03095 |
| 304 | +0.88030 |
| 305 | +1.09812 |
| 306 | +0.80920 |
| 307 | +0.76586 |

The registered 0.02-nat minimum causal contrast and ≤0.05-nat absolute retention-KL gates were satisfied on this engineering population. Source and raw results carry the exact family of compared interventions; both bases are evaluated against the **same eight physical projector shams** per root.

## Interpretation

NC-02's individual-coordinate co-attribution graph was sensitive to arbitrary orthogonal reparameterization of a **linear BPD factor pair**, yielding no qualifying communities in rotated coordinates. NC-03 changes the estimator: the rank-two eigenspace of the gradient second moment rotates covariantly under an orthogonal factor change. The resulting *physical* projector acts on the same recurrence and its causal outputs agree numerically. This is a validated synthetic instrument for orthogonal gauge covariance.

The source architecture is **constrained by construction**. Only four independent stage-shift lanes have nonzero weights, with a stagewise equal-gain constraint maintained after optimizer updates. The task label is explicitly defined from input context; the gradient score is **label-aligned and supervised**, unlike an unsupervised Goodfire neuron-community discovery. Retaining both genuine context-dependent lanes in every pass gives almost exact target behavior by architecture. The root family shares this mathematical scaffold, so the narrow root-level effect distribution does not measure performance on other learned models.

Orthogonal factor rotations are linear reparameterizations, not arbitrary rotations of post-nonlinearity MLP neurons. Native-neuron identity and architectural constraints can remain meaningful even when a BPD factor gauge is non-identifiable. This study therefore tests our BPD-derived subspace instrumentation and **does not rebut or replicate Goodfire's specific native-neuron result**.

## Resolution and next population

Independent unit: one fresh trained model root; **n=8**, with 8×1024 held-out evaluation cases treated as correlated within root. Primary comparison: median of eight discovery-dose-matched Haar rank-two physical shams minus retained-projector final KL. Both native and rotated arms used one predeclared estimator and a fixed two-dimensional rank. Physical dose: discovery clean-context RMS of the retained shared-linear write, within a 15% matching allowance; 2048 proposals per root all qualified. Scientific variation floor: **0.02 nats**; estimated root-bootstrap lower bound **0.844 nats** on the constrained population. The separate 95th-quantile sham floor at 95% confidence is **uncalibrated** because eight roots are below the 59 independent-root nonparametric minimum.

**Verdict:** `ENGINEERING_POSITIVE_CONTROL`. **Scientific confirmation remains sealed.** The next experiment should train an unconstrained recurrent transition with nontrivial off-route couplings and evaluate projector recovery without using external labels to select the target score. Any cross-family or >8-root confirmation needs a new prospective protocol, explicit multiple-comparison budget and EvalMath preflight; avoid reuse of seeds 300–307 as an untouched test.
