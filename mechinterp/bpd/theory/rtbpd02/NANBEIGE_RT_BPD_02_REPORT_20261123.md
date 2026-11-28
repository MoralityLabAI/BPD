# Nanbeige RT-BPD-02 research report

November 23, 2026 · BPD draft PR #146

Reached-state weighting substantially improved compression of a fixed Nanbeige T2 write on eight exposed CAL roots. That improvement did not carry over to the signed donor direction. In the BF16 rank-2 pass-1 comparison, future-reader weighting reduced relative K-readout squared error from 0.2258 to 0.0000176, while mean matched-dose cosine fell from 0.900 to 0.805. The archive therefore supports evaluating compression and source-specific behavior separately. The independently replayed synthetic gym shows where the two objectives can agree and gives an explicit distribution-shift failure.

[Theory](THEORY.md) · [Locked registration](EXTENSION_PROTOCOL.md) · [Archive contract](ARCHIVE.md) · [Nanbeige synthesis](https://github.com/MoralityLabAI/LoopedTransformers/blob/c074e354e8bcd4d26c02104fadd289dfdad8a207/reports/nanbeige_bpd_gym/NANBEIGE_RESEARCH_SYNTHESIS_20261123.md)

## Mathematical findings

The weighted Eckart–Young lift gives a global minimizer for a fixed reached second moment and a fixed measured reader, including singular supports. A numerical reader cutoff must affect both the target and the lift. The original implementation could spend rank on a singular direction discarded by its pseudoinverse. The corrected solver reports the optimized effective loss, its spectral tail, and the full loss of that same physical lift. Exact objective values and solution sets are covariant under invertible coordinate changes. A minimum-coordinate-Frobenius-norm lift can change off support under a shear, and a tied spectral cutoff can rotate the selected subspace by 90° under an arbitrarily small perturbation.

The paired difference is the operator change acting on the midpoint plus the midpoint operator acting on the state change. Squared energies include their signed cross term. An additional reader term applies when the measured map changes. Delayed-visible rank follows by restricting the future reader to the kernel of the current reader. A fixed linear stack, or a frozen tangent stack with a controlled remainder, can serve as a rank objective. Reached moments and readers with sufficient support domination admit a transfer bound. Orthogonal source and destination supports, opposite signed gains at equal dose, cancelling paths, and nonlinear joint edits provide explicit failure cases. [Proofs and counterexamples](THEORY.md)

## Synthetic development

The six-dimensional tied recurrence stores its write in a stage slot and delivers it to delayed memory after another step. Its two delayed-visible axes are locally dark. The five candidates use rank 2, with 24 paired roots for methods that fit on training data. The physical-dose comparisons evaluated both registered pass occurrences on the same 64 independently drawn development roots, with a separate rescaling of those draws for the shift panel. Explicit state continuation verified the fixed reader to numerical precision. Each candidate intervention was sized at one-quarter of the reference write contrast.

| Method | Aligned pass-2 write MSE | Aligned pass-2 delayed MSE | Aligned matched-dose cosine | Shifted pass-2 write MSE |
| --- | ---: | ---: | ---: | ---: |
| Ordinary SVD | 334.019 | 269.263 | Undefined, zero readout | 1.317 |
| Reached-state SVD | 18.153 | 0.05381 | 0.9972 | 7,442.533 |
| Fixed-future-reader SVD | 16.971 | < 1e−26 | 1.0000 | 7,452.655 |
| BPD mask over planted atoms | 16.971 | < 1e−26 | 1.0000 | 7,452.655 |
| Seeded random write subspace | 268.306 | 191.582 | 0.7263 | 6,194.316 |

The registered fixed-reader result improved over SVD on 64 of 64 roots. The paired reduction averaged 269.263, with a range of 13.347 to 1,236.862. An independent test computes the ordinary-SVD error directly from the planted recurrence and reduces the root receipts. The BPD mask also recovers the two planted reader axes. This fixture tests a fixed bank of exact atoms, and the archived IO18 banks supply the learned-bank comparison.

The reader and BPD retained exact direction in the shifted panel, where mean signed gain after dose matching was about 250× the reference at pass 2. That gain reflects the large physical dose relative to a small delayed component. The reached fit gave negative gain on 26 of 64 shifted pass-2 roots. Its maximum pass-2 output angle between independent CAL halves was 8.29°, compared with numerical zero for the fixed reader. A fixed bank can fit a chosen finite-dimensional reader and still write poorly under a changed reach law.

EvalMath returned **REDESIGN_OR_PILOT_ONLY**. With undefined SVD directions assigned the preregistered score of zero, the direction effect was 0.500. The worst-root sham bound was 0.49958 on the bounded scale, above the registered 0.25 planning effect, and the development stream has no sealed confirmation. [Synthetic summary](SYNTHETIC_SUMMARY.json) · [Synthetic EvalMath input](SYNTHETIC_EVALMATH_INPUT.json) · [Synthetic preflight](SYNTHETIC_EVALMATH_PREFLIGHT.json) · [Hosted CPU replay and raw synthetic artifact](https://github.com/MoralityLabAI/BPD/actions/runs/45193970011)

## Nanbeige CAL archive

The replay validated 16 focused CAL cases and 96 selected vectors per backend across passes 1 and 2. It joined both worlds of each root by identity, verified the original token-list digest, and checked the archived tensor and fit hashes. All four IO18 fit cells carried the original bit-exact native all-one full-head and boundary-state qualifications. The reference T2 rank-eight sum recomposed within 1.19e−15 relative L2 in the compact physical basis. Each new fit left out both worlds of the scored root. The original exposure of the eight CAL roots, shared factor banks, and overlapping folds limit the interpretation to development.

**Native affine audit.** The CPU emulation of the separate, full BF16 down-projection used the pinned checkpoint weights, float64 contraction and BF16 round-to-nearest-even via float32. It matched 71,091 of 98,304 elements by bits and 0 of 32 complete writes. Relative L2 was 0.002130, and maximum absolute deviation was 8.0. Both recorded operands and outputs were BF16-representable. These numbers expose a host-to-native replay discrepancy. The exact all-one receipts qualify the original native runs, and the CPU emulation has no exact native closure claim.

**Write reconstruction.** The reached fit beat ordinary SVD on all eight exposed roots at each of ranks 1, 2 and 4, in both precisions and both passes. Each cell in the next table reports the rank-2 ratio of mean squared write error to mean squared T2 target write, with worlds averaged within roots. Pair-contrast weighting fit a different loss and often incurred a higher write error. The atom-mask baselines sometimes improved on ordinary SVD and sometimes did worse. Ranks 1 and 4 and all scoring fields are included in the aggregate. [Archive aggregate](NANBEIGE_CAL_ARCHIVE_AGGREGATE.json)

| Backend and pass | Ordinary SVD | Reached fit | Pair-contrast fit | SVD initializer mask | T1 mask | T2 mask | Random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 1 | 0.009897 | 0.00000483 | 0.013345 | 0.005361 | 0.005568 | 0.005309 | 0.959867 |
| BF16 2 | 0.056162 | 0.001643 | 0.728421 | 0.059100 | 0.059281 | 0.058554 | 0.742144 |
| NF4 1 | 0.009776 | 0.0000120 | 0.041920 | 0.005382 | 0.005513 | 0.005330 | 0.959770 |
| NF4 2 | 0.059300 | 0.000994 | 0.660624 | 0.062044 | 0.061572 | 0.061571 | 0.735293 |

**Fixed BF16 K reader.** The BF16 pass-1 fit uses the continuous derivative of both RMS norms and the pinned layer-0 K matrix at the mean training-fold carrier. The CPU calculation stops at K before RoPE and attention. Normalizing writes to equal physical L2 before this frozen reader exposed a difference between average squared reconstruction error and the paired donor contrast. At rank 2, the fixed-reader fit also had a 54.5° maximum output-subspace angle between leave-one-root-out fits. The reached fit's angle was 56.8°. Reached-state and fixed-reader fits used rank-14 input supports. Pair-contrast fits used rank-7 supports. A small residual on this reached cloud therefore leaves the physical directions underdetermined away from it.

| BF16 pass-1 rank-2 method | Relative K-readout MSE | Mean donor cosine at matched dose | Mean signed gain |
| --- | ---: | ---: | ---: |
| Ordinary SVD | 0.225819 | 0.900 | 0.990 |
| Reached fit | 0.00001807 | 0.824 | 0.776 |
| Fixed-reader fit | 0.00001760 | 0.805 | 0.783 |
| Pair-contrast fit | 0.025712 | 0.823 | 0.796 |
| SVD initializer mask | 0.115663 | 0.872 | 0.910 |
| T1 mask | 0.116526 | 0.871 | 0.909 |
| T2 mask | 0.116550 | 0.872 | 0.912 |
| Random | 1.846703 | 0.108 | 0.359 |

The reader fit reduced K-readout error on all eight roots. Its mean paired cosine effect after dividing by two was −0.04733. EvalMath returned **REDESIGN_OR_PILOT_ONLY** for the reused, unsealed CAL folds, with independence and the 95/95 sham bound unverified. An archive matched-dose computation constructs equal-length physical inputs to a frozen CPU reader. Native intervention dose after rounding and answer mediation remain unmeasured. [Archive preflight](NANBEIGE_CAL_EVALMATH_PREFLIGHT.json)

**Transport.** For the tied T2 target within each backend, the entire cross-pass contrast change is the state term. Pass 2 put 94.7% of BF16 and 97.8% of NF4 contrast-input energy outside the corresponding pass-1 empirical contrast support. Across backends, the mean squared state term was 264.74 at pass 1 and 2,833.74 at pass 2, compared with operator terms of 0.0151 and 7.84. The signed cross terms were −0.0571 and −10.31, respectively. These are fixed-bank physical decompositions on paired archived states. They give no finite empirical moment-domination guarantee and do not identify the causal effect of changing precision. [Transport receipts](NANBEIGE_CAL_ARCHIVE_AGGREGATE.json)

## Implications for the gym

The earlier Nanbeige synthesis used the Inkling 2 Mini work to motivate matched-depth evaluation and measuring specific later readers. The tied synthetic gym now tests that design with known physical source factors and an explicit delayed cache. Nanbeige carries residuals serially between passes, so its development replay uses an architecture-specific next-pass K tangent. The new benchmarks record present and delayed outputs, signed donor direction and gain at common dose, root-level holdouts, and output-subspace stability alongside reconstruction. The results justify keeping those endpoints distinct when comparing equal-rank SVD and BPD candidates. [Nanbeige synthesis, matched interventions](https://github.com/MoralityLabAI/LoopedTransformers/blob/c074e354e8bcd4d26c02104fadd289dfdad8a207/reports/nanbeige_bpd_gym/NANBEIGE_RESEARCH_SYNTHESIS_20261123.md#what-a-matched-intervention-changes)

Native causal validation still requires the native NF4 operator and reader arithmetic, exact parity of the considered physical patch, a verified first-pass head, rounded matched doses, and signed active/sham and clean-reader-rescue outcomes on new authorized roots. The archive provides the verified layout and precise extraction contract for that stage. This work ran zero pretrained forwards and zero GPU jobs. The 168-case ARG-03-N cohort remains sealed. [Archive and extraction contract](ARCHIVE.md)

## Reproduction and custody

The original source-only CI passed at its pinned head. The extension first committed the registration, then the proof and support audit, then the synthetic and archive runners. The hosted Ubuntu job ran 30 tests with NumPy 1.26.4 and reproduced the primary synthetic result. The local archive used NumPy 2.5.5 with one CPU thread, 65.4 seconds, and 106.6 MiB peak sampled process-private memory. It verified both pinned checkpoint shards. The receipts publish aggregate measurements and content hashes. The synthetic artifact contains the generated root rows and can be regenerated by the source command in the README. Its byte hash is host-specific because floating-point serialization can vary at the last digits. [Original CI](https://github.com/MoralityLabAI/BPD/actions/runs/45181729576) · [Extension CPU CI](https://github.com/MoralityLabAI/BPD/actions/runs/45193970011) · [Locked EvalMath planner](https://github.com/MoralityLabAI/EvalMath/tree/accf4d9040bee8daeabc6b07003bb484708996c0/skills/evalmath-mechinterp-power)
