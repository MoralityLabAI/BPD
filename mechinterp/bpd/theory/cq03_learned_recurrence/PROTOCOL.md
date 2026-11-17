# BPD-CQ-03 prospective CPU protocol: learned recurrence, masked native factors and independent delayed labels

Protocol date: 2026-11-17. Research-only BPD study on a new synthetic two-step continuous arithmetic task. The excluded engineering seed is **997**; engineering results may inform implementation checks only. The prospective locked seeds are **607, 613, 617, 619, 631, 641, 643, 647**. No ARG-03-N `test.jsonl`, real TRM checkpoints, prior SCM test gold, or private user files may be accessed.

## Causal problem

For independent `a,b ~ Uniform[-0.9,0.9]`, initialize `h0 = (0,0,a,b)` and train a tied **rank-two linear physical transition** W (four hidden coordinates) so that the scalar readout `q(h)=h[0]` satisfies `q(Wh0)=a` and `q(W²h0)=b`. W is learned from input/target examples without a prescribed channel swap. Fixed architecture: one repeated rank-2 physical operator, two calls. Report the resulting W as a learned linear state transition, not a natural-language math reasoner or nonlinear looped transformer. The second target is defined directly by `b` and independently of BPD factorization.

Train W as two 4x2/2x4 factors for exactly 520 Adam updates, LR .035, batch 96, squared error on both visible steps plus .0002 times W MSE. Seeded training samples use seed offset +390000. Initialize from seed offset +902000. Learning the physical W and subsequently fitting BPD factors are separate stages.

Freeze W. Fit **one** BPD shared rank-2 factor pair V [4,2], U [2,4] from a fresh seeded stream (+680000), exactly 360 Adam updates, LR .035, batch 96. Initialization offset +981000. The fit loss is step-one and step-two full-state trajectory MSE to the frozen physical model, plus 0.5 times full-matrix W reconstruction MSE. Both components execute at each pass, with residual delta masked off; this is a full-capacity learned BPD factorization rather than a learned sparse gate. The code must have an independent CPU backend and a repository-native `BPDModel/TiedComponentLinear` backend with parity checks. The full-mask path has an exact GL(2) basis gauge.

## Frozen selection and held-out counterfactual

After both training stages, choose component `c` by maximizing the absolute source-derived delayed derivative `|V[3,c] * dot(U[c], W_student[0,:])|` (ties prefer component zero). This uses no held-out interventions, donor target, or output labels. A calibration-only draw with RNG seed offset +510000 is permitted for supporting diagnostics; component selection is analytic and locked before held-out generation.

Create **128 held-out pairs per seed**, RNG offset +710000. Recipient and donor share `a` in [-0.8,0.8], recipient b in [-0.4,0.4], and donor b differs by a randomly signed displacement in [0.30,0.45], remaining inside the trained interval. Recipient/donor clean eligibility requires both predicted readouts within .12 of their independent a and b targets at each of the two steps. Report all generated and eligible counts; never select a replacement or extend the sample.

At the post-first-pass state h1 of the recipient, perform the selected component's native **donor-minus-recipient component write**. The forward dynamics for the second pass use the learned frozen student W. Evaluate the exact target `donor b`, with success tolerance .12. Include projected-current-dark native, unselected component (each per-pair norm-matched to selected native), random vector within current readout kernel with the same norm per pair, visible-dose control, sham and full-state donor replacement. Report immediate-readout changes and delayed outcome separately. Per-pair non-sham intervention norm errors <=1e-5; full-state is an explicitly larger-budget positive control.

## Gauge and observability

Apply one frozen invertible component shear `R=[[1,1],[0,1]]`: `V' = V R`, `U' = R^-1 U`. Verify unmasked physical `V U` remains equal within 2e-6 and report the learned singleton patch drift relative to original. Check rank of the two-row component observation `[qU^T; q W_student U^T]`, with tolerance 1e-4. It is a finite-horizon **linear** observable quotient and is not a 4D Seifert or pretrained network result.

## Primary reporting and status

The registered engineering gate requires a finite executable run of all eight seeds, no finite-norm or source-shape violations, source-verified native parity <=1e-5 when native backend is run, all source/seed hashes in the reference, and exact reference replay. The scientific positive requires >=6/8 seed-level clean eligibility >=110/128, full-state transfer >=95% of eligible, current-dark native to exceed matched random by >=20 percentage points, and no unrecognized intervention leakage. This is a separate seed-level gate; any failure must be reported unchanged. Do not replace source, seeds, thresholds or arms after producing held-out results. Report all outcomes even when the gate fails.

Keep the prospective protocol/source and native suite in a branch before opening the new held-out pairs. Use standalone unittest for engineering and GitHub Actions to run both independent and native replay. All generated labels/receipts are public synthetic; no claim of blinded clinical or general-purpose model evidence. If hosted environment changes statistics beyond float tolerance, disclose and investigate before merge. Preserve all BPD PD-01–03 and CQ-01–02 receipts and the sealed ARG-03-N evaluation.