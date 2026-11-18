# BPD-NC-04: dense trained recurrent projectors, eight-root development audit

**Date:** 2026-11-18 (UTC). **Evidence class:** source-registered synthetic engineering experiment, **not scientific confirmation**.

- **Frozen prospective protocol-only commit:** `0c84c56f8f747737833f0c51201d369dcb24a790`, committed before implementing the experiment or sampling NC-04 roots.
- **CI:** [Actions run 44235766022](https://github.com/MoralityLabAI/BPD/actions/runs/44235766022), source-and-smoke and independent-root development both **success** (9/9 source tests). Initial test-harness assertion failure on the BPD wrapper was repaired before successful full execution; all runs remain visible in the PR history.
- **Complete root receipts:** [nc04-eight-trained-model-engineering-roots artifact 14397032953](https://github.com/MoralityLabAI/BPD/actions/runs/44235766022/artifacts/14397032953), containing `NC04_ENGINEERING_SUMMARY.json` and the eight full `nc04_root_{400..407}.json` files.
- **SHA-256 of exact experiment source:** `9eebadde9b32a1b9e11312e7531cd15865ab16fca928c9df7e0fb4ce129990e8`.
- **SHA-256 of frozen protocol:** `52b43e671b99f2a4c4db7002b598b9764a4d57b615ccf66878de5280521bfc85`.
- **SHA-256 of downloaded zipped raw CI artifact:** `363d3aabf242b68e7da441c57474ee477f7c3bf31ed16e59f770294e99d0df5d`.

## Registered design and task gates

Eight independent randomly initialized **dense learned recurrent models**, engineering seeds **400–407**. Each received 2,048 training examples, 384 optimizer updates, 1,024 *separate* model-only discovery examples and 1,024 held-out external-label evaluation examples. The shared trainable 20×20 physical matrix is reused at four nonlinear recurrence steps with freely learned input encoding, preweight processing and scalar readout.

All **8/8** models met the context-specific task gate (>=85% in each context), >90% dense-weight gate, native and rotated full-mask parity, rank-2 eigengap condition, eight discovery-only matched physical shams, and the zero-readout gradient null. Native and rotated arms had **8/8 eligible independent model roots** each. The *minimum* observed gap ratio across passes/roots was **1.26237** against the registered >=1.10 requirement. The weakest native root still had 289 eligible sham candidate proposals out of 2,048.

Clean-model mean held-out accuracy **increases with recurrent depth**:

| Pass | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| Accuracy, mean across eight model roots | 79.08% | 86.66% | 92.66% | **95.85%** |

Mean final clean NLL was **0.098934 nats**. Mean context-positive accuracy was **95.89%** and mean context-nonpositive accuracy **95.82%**.

## Primary causal outcome: mixed success, registered gate FAIL

The predetermined root-level primary contrast is median held-out **true-label NLL of eight dose-matched Haar keep-only shams**, minus held-out true-label NLL of the recovered two-dimensional projector. It is measured on complete four-pass native BPD closed-loop interventions.

| Independent-root statistic | Native | Orthogonally rotated |
|---|---:|---:|
| Eligible independent model roots | 8/8 | 8/8 |
| Mean external-label NLL advantage over matched shams | **+0.523956 nats** | **+0.523956 nats** |
| Root bootstrap 95% CI for sham advantage | **[+0.443194, +0.613407]** | [0.443194, 0.613407] |
| Mean keep-only minus clean-model NLL | **+0.123414 nats** | +0.123414 nats |
| Root bootstrap 95% CI for clean-to-keep NLL degradation | **[+0.077851, +0.185867]** | [0.077851, 0.185867] |
| Mean retained-model KL(clean || keep) | **0.110585 nats** | 0.110585 nats |
| Mean NLL increase from ablating recovered subspace | +0.509952 nats | +0.509952 nats |
| Mean keep-only external-label accuracy | **92.93%** | 92.93% |

**Registered scientific verdict: `NOT_RESOLVED_OR_FAILED`.** The observed mean keep-only NLL degradation **0.123414** exceeds the preregistered limit of **0.05 nats**, with its 95% root-bootstrap lower bound already greater than 0.05. The other primary contrast is positive, and the randomized controls are worse, so the recovered subspace is causally useful. The rank-two retained write is **insufficient to meet the original absolute fidelity gate** on this learned model family. Leave all original thresholds unchanged and preserve this failure.

### Root-level native results

| Seed | NLL advantage vs median sham (nats) | Keep-only NLL cost vs clean (nats) |
|---|---:|---:|
| 400 | +0.719012 | +0.132483 |
| 401 | +0.483046 | +0.141998 |
| 402 | +0.397294 | +0.307807 |
| 403 | +0.582207 | +0.032407 |
| 404 | +0.485359 | +0.108610 |
| 405 | +0.365121 | +0.054369 |
| 406 | +0.450421 | +0.102502 |
| 407 | +0.709190 | +0.107137 |

## Gauge and co-attribution diagnostics

The physical operator and complete held-out behavior remain equivariant under orthogonal changes to the exact BPD factor basis:

- Maximum relative physical projector-write map discrepancy across eight roots: **7.36103e-8**.
- Maximum keep-only held-out logit discrepancy: **2.74181e-6**.
- Maximum factor projector transport error: **3.95915e-8**.

The Goodfire-inspired native neuron co-attribution **diagnostic** discovered 0–4 cross-pass communities per root (mean **1.875**). Under orthogonally rotated factor coordinates, its mean was **1.75** (range 0–4), with varying memberships; all 16 within-root per-pass-shuffled graph controls produced zero cross-pass components in each basis. These graph diagnostics are observational, with **no registered causal graph-community test or graph-level sufficiency claim**. The projector and the graph select different mathematical objects.

The zero-readout control yielded zero intermediate write gradients, zero positive attribution hits and `UNIDENTIFIABLE` projector in all eight roots.

## Interpretation and next experiment

NC-03's nearly perfect projector retention came from a planted four-lane stage-shift architecture. NC-04 shows that after replacing the constrained transition with a fully learned dense recurrent model, the estimated two-dimensional gradient subspace still matters to external task behavior and transfers exactly between orthonormal factor bases. Its limited dimensionality produces measurable damage to clean behavior even when model-blind dose-matched sham directions do worse. These results support a **synthetic causal-instrument finding**, with a negative absolute-fidelity gate.

The *first* and *fourth* pass accuracies differ by 16.77 percentage points on this fixed input distribution, so later recurrent passes contribute to trained behavior. A selected rank-two subspace may omit important computation; assessing whether larger or variable-rank projectors recover fidelity requires a **new prospectively registered comparison**, because NC-04 fixed the rank and physical-dose budget before data.

Caveats: the model is 20D and trained on one synthetic input-defined task. Only one physical linear site is projected; the input encoder, preweight map and readout remain present. The discovery score uses model logits, but its uncentered gradient second moment is mathematically invariant to either model or external label **sign**. Shams match retained write RMS, not downstream Jacobian or task-specific effect. The eight training roots share architecture/task family and do not calibrate a nonparametric 95th-sham percentile bound at 95% confidence (>=59 independent roots are required). Protected ARG-03-N and AMM-03 confirmation cohorts stay sealed.

Any NC-05 rank/dose frontier should first freeze one selection policy, per-root model training and discovery budgets, matching allowance, complete rank-by-dose comparison family and independent-root EvalMath power audit, before running newly sampled data.
