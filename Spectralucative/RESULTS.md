# First native Spectralucative run

The registered CPU study completed on 21 November 2026 (Asia/Singapore).
[GitHub Actions run 44789012873](https://github.com/MoralityLabAI/BPD/actions/runs/44789012873)
passed all 53 tests and independently replayed all four native receipts.
The complete saved packet is in [`results/run-44789012873/`](results/run-44789012873/),
with its execution history and checksums in [`results/INDEX.json`](results/INDEX.json).

## What ran

Four independently initialized miniature TRMs used the repository's actual
`build_mini_trm` model and `BPDModel` component wrappers. Each had 80 recurrent
coordinates across the complete H/L state, including the prefix position.
One warmup preceded four measured outer transitions. A fixed rank-three basis
was built from component output-write vectors and embedded at the first board
position in the L state. The numerical definitions are in
[`PROTOCOL.json`](PROTOCOL.json) and [`README.md`](README.md).

The study retained all 16 component masks for each initialization. Each mask
changed the first physical `layers.0.down_proj` invocation of the measured
trajectory at the first board token, with the dense residual retained. The
response contains all 44 final logits across four positions and eleven head
rows. Separate measurements track the physical write and state displacement.

The four model-state hashes differ. Every receipt records random initialization,
CPU execution, zero training updates, and zero scientific or confirmation
roots. These measurements characterize the registered engineering fixture.
The runtime was Python 3.12.16, PyTorch 2.5.1+cpu, and NumPy 1.26.4. The native
study took 2.80 seconds after its process started measuring elapsed time;
dependency installation and the separate tests have their own CI steps.

## Trace expansion and its domain

All 16 full-state Jacobians failed the sufficient raw `z=1` norm condition.
Their induced infinity norms ranged from **1.8623 to 10.1061**. Each receipt
therefore leaves the raw trace-series approximation and truncation bound empty.

The explicitly damped full-state calculations use
`z = 0.5 / max(1, ||J||_infinity)`, giving `q` approximately 0.5. With twelve
terms, the absolute discrepancy between the trace sum and
`-log(det(I-zJ))` ranged from **1.11e-15 to 1.44e-11**. The analytic bound on
the omitted logarithm tail was **0.0015024**. These calculations describe the
specified scaled operators. Native stability requires its own analysis of
the undamped dynamics and their changing states.

All 16 projected matrices `A_t = Q^T J_t Q` met the raw norm condition, with
norms from **0.26582 to 0.65159**. Their twelve-term raw log discrepancies were
also within their recorded analytic tails and declared numerical slack.
The transport results below measure how much of each full-state push escapes
the projection.

The analytic tail concerns exact real arithmetic. The saved comparisons use
floating-point arithmetic, with `roundoff_certified=false` throughout.

## Transport through the native recurrence

Define `Phi_J = J_3 J_2 J_1 J_0` and `Phi_A = A_3 A_2 A_1 A_0`.
The final transport error is `||Phi_J Q - Q Phi_A||_infinity`.
The table also gives the norm envelope obtained from the recorded local
defects, and the gap `||Phi_J Q - J_0^4 Q||_infinity` caused by freezing the
first Jacobian.

| Initialization seed | Final transport error | Recorded envelope | Ordered versus frozen gap |
|---|---:|---:|---:|
| 2026112100 | 0.169751 | 38.3369 | 0.238218 |
| 2026112101 | 0.262846 | 27.6016 | 0.250263 |
| 2026112102 | 0.825595 | 87.1378 | 1.67278 |
| 2026112103 | 0.0222548 | 1.49642 | 0.0289917 |

Across the sixteen transitions, the relative leakage
`||J_t Q - Q A_t||_F / ||J_t Q||_F` ranged from **0.5123 to 0.8954**.
The chosen write-based boundary subspace therefore loses a substantial part
of each tangent push. Its projected spectral calculation needs this leakage
measurement to describe the full recurrence.

The independently reconstructed telescoping identities agreed within
**4.55e-16** in infinity norm. The envelopes were **67 to 226 times** the
observed final errors. Their conservatism comes from repeatedly applying
worst-case norm bounds; these receipts provide the actual error alongside
the envelope. Replacing the evolving Jacobians by a power of the first one
produced a measurable discrepancy on every initialization.

This basis was selected by the fixed construction in the protocol. Its
leakage is evidence about that construction on these models. A search for
better state subspaces would require a separately declared procedure.

## Component interactions across the complete mask cube

Walsh analysis uses the signed logit-delta vectors. The interaction fraction
below is the energy of degrees two through four divided by the total
nonconstant coefficient energy, so its denominator excludes the mean response.
The residual is the uniform mean squared L2 error after retaining degrees
zero and one, summed across the 44 output coordinates.

| Initialization seed | Interaction fraction of variance | Degree-at-most-one residual energy |
|---|---:|---:|
| 2026112100 | 0.425221% | 3.44729e-6 |
| 2026112101 | 0.711422% | 1.15995e-5 |
| 2026112102 | 0.749895% | 7.95758e-5 |
| 2026112103 | 0.628789% | 1.75613e-7 |

An additive response over the four component bits accounts for
**99.25% to 99.57%** of the centered response energy in these cubes. Parseval's
absolute discrepancy was at most **4.34e-19**. The residual-energy identity
had zero recorded discrepancy in all four receipts.

The Walsh degrees count interactions among mask bits. State-space transport
uses the separate rank-three inclusion map described above. Each mask also
has a recorded physical dose, which varies across the cube. The study makes
a descriptive decomposition of these responses; a mechanism comparison would
need a registered task endpoint and matched controls.

## Native parity and finite displacement

The largest all-ones logit discrepancy from the target was **1.79e-7**.
Full-state parity differed by at most **7.16e-7**, within the registered
absolute tolerance of `3e-5`. The delivered component-write formula differed
from the observed write by at most **1.31e-7**. Independent direct JVPs agreed
with dense Jacobian products to **1.27e-7** maximum absolute error.

Separate perturbations along the first Q direction were injected at the outer
state boundary. Their sizes were fixed fractions of the baseline state norm.
The following entries are the final remainder norm divided by the actual
final displacement norm after four transitions.

| Initialization seed | Relative dose 1e-4 | Relative dose 1e-3 | Relative dose 1e-2 |
|---|---:|---:|---:|
| 2026112100 | 1.1596% | 0.14794% | 0.72648% |
| 2026112101 | 1.1854% | 0.26029% | 3.32768% |
| 2026112102 | 0.18662% | 0.31627% | 3.11735% |
| 2026112103 | 9.4139% | 1.03615% | 1.02359% |

These remainders include native float32 arithmetic discrepancies alongside
finite-dose nonlinear effects. In the 9.4139% entry, the absolute remainder
is `8.64e-7` and the final displacement is `9.17e-6`. That denominator makes
the relative value large. The saved trajectories allow both quantities to
be inspected together.

## Verification and execution history

The successful workflow checked 25 frozen DT04 source files, 12 historical
locked files, and the archived DT02 receipt. The pinned historical EvalMath
preflight again returned strict exit status **2** and
**`REDESIGN_OR_PILOT_ONLY`** for its original eight-root DT05 input.

The first CI attempt stopped during native test-fixture receipt construction
because `_spectral_pair` received two values for an argument named `matrix`.
The fix renamed its operator argument. The registered four-initialization
study began in the subsequent successful run. The failed attempt remains
linked in `results/INDEX.json`.

The successful job saved all unit JSON files plus the summary and independent
verification. Recovery checked the SHA-256 of the lossless six-file bundle.
An additional local replay of those recovered files passed with the same
verification payload digest. Each unit binds 23 source hashes to its saved
matrices and observations.

From the repository root, replay the archived packet with:

```bash
python -m pip install numpy==1.26.4
python -m Spectralucative.verify --run Spectralucative/results/run-44789012873
```

The verifier checks source hashes against the checkout. Use the source revision
identified in `results/INDEX.json` when later instrument versions change those
files. The complete native experiment command is in `README.md`.
