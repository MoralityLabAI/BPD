# BPD-SF02 — trained recurrent secrecy, full-physical BPD factors and readout-invisible swaps

**Prospective exploratory CPU run, 2026-11-17.** Parent: [BPD-SF01, merged PR #28](https://github.com/MoralityLabAI/BPD/pull/28). External motivation: [Rodríguez, Spectral Faithfulness](https://github.com/idolum-ai/spectral-faithfulness).

## Evidence and tuning boundary

This protocol follows **excluded engineering pilots on seeds 901 and 902** (12D nonlinear recurrent teacher, 24-unit tanh hidden transform, 12-component full-rank factor model, two tied steps). The pilots were used to select architecture, optimization, dose, component count, cutoffs, and diagnostic arms; they **cannot** count toward a fresh prospective result. The present initial cohort is an exploratory, fresh-source **8-root** study on seeds **1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051**. Training/selection/assay streams within each root are separately seeded and nonoverlapping; the source family remains the same continuous scalar-routing family. A 95%-confidence calibration of a 95th-percentile null effect would require >=59 independent roots and is **not** established by eight.

The authorizing task is artificial and fully specified; it tests a synthetic mechanism, not a real security failure. No pretrained models, user secrets, existing CQ-03/04 heldout data, or ARG-03-N's sealed 168-case test are used. Source-locked results may be added only after protocol and implementation are committed. Never replace failures or tune registration based on the fresh eight roots.

## Teacher and task

Let independent `S_A,S_B,P ~ U[-0.8,0.8]` and role `R∈{A,B,U}` uniform. Public context includes `P,R`. The initial state `h_0∈ℝ^{12}` carries `S_A,S_B,P` at coordinates 1:4, role as 3-way one-hot at 4:7, other coordinates 0. This makes the private inputs **available to the model**. The learned **shared** physical transition is

\[
F_\theta(h)=\tanh(hW_1+b)W_2,\quad
W_1∈ℝ^{12×24},\ W_2∈ℝ^{24×12}.
\]

Both passes execute identical physical parameters. True readout `q(h)=h_0` (first coordinate), with targets `q(h_1)=P` and

\[
q(h_2)=
\begin{cases}
S_A & R=A,\\
S_B & R=B,\\
P & R=U.
\end{cases}
\]

Train `W_1,b,W_2` jointly on fresh random batches with Adam learning rate .015 for 850 updates, batch 256, loss `2 MSE(q(h_1),P) + 3 MSE(q(h_2),target) + .0001(mean W1²+mean W2²)`. The teacher is a learned nonlinear transition, with a *fixed one-coordinate* readout. It has no RLHF, tokenization, or natural-language disclosure channel.

## Independently fitted physical BPD dictionary

Freeze teacher. Fit `\hat W_2=VU` with `V∈ℝ^{24×12}, U∈ℝ^{12×12}` from random initialization, Adam .025 for 400 updates, batch 256. Minimize `1.5 MSE(VU,W_2)+MSE(tanh(h0 W1+b)VU,h1)+MSE(tanh(h1 W1+b)VU,h2)`. All 12 factors remain active. This tests physical reconstruction and interventions using a trained factorization, **not sparse gate discovery**. Full-rank GL(12) gauge remains free; singleton masks are not canonical.

## Source-matched counterfactual

Create recipient/donor pair with identical public context, role and **other** user's secret. Change only target user's secret by signed 0.3–0.9, clamped to ±0.8; for unauthorized role U vary `S_A`. At the first physical write construct each factor's donor-minus-recipient contribution:

\[
\delta_c=(\phi(h_0^d)-\phi(h_0^r))V_cU_c,\quad
\phi(h)=\tanh(hW_1+b).
\]

Project any chosen sum to `ker Dq` by zeroing its readout coordinate, then resume one clean tied transition `F_\theta(h_1^r+P_{\ker q}\sum_{c∈C'}δ_c)`. Assert immediate readout equality ≤1e-7. This projection is an intervention artifact; the exact donor is a separate control.

On a separate calibration source (`n=240`), greedily select **six** factors by smallest authorized donor-target future-readout MSE, ties by smallest index. Freeze the indices. New heldout source produces **384** recipient/donor pairs per independent root. Authorized eligibility requires clean second-pass prediction error ≤.08, first-pass recipient/donor readout difference ≤.04, and clean donor-vs-recipient future-output difference ≥.20. Public eligibility applies the first two criteria to role U. Report all denominators, excluded pairs, and every root.

## Controls, metrics and interpretation

Primary empirical descriptive endpoint on authorized eligible pairs: fraction whose post-edit output is within ±.10 of the independent donor's clean second-pass output. Compare full state dark donor patch, **all twelve** factors, unchanged recipient, and **255** per-pair equal-L2-norm random state directions in `ker Dq`. Report sham mean, maximum among 255, root-level differences and signed error; random dark directions may be nonexchangeable with selected BPD writes. Use seed-level replicated effects, never claim a rank p-value without validated exchangeability.

Public boundary: clean donor-vs-recipient change, selected six-factor change, **full-block** change, entire dark-state donor patch, and random-dark shams. A selected factor intervention can leave the reachable hidden manifold and induce unrelated output drift. Public deviation alone is **not proof of actual text leakage**.

Gauge test: apply invertible shear `R=I+0.7 e_0e_1^T`. Assert `(VR)(R^{-1}U)=VU` to float tolerance and record singleton component drift. The full block's physical effect, including nonlinear continuation, is invariant up to numeric precision. Test the native BPDModel/TiedComponentLinear full-mask two-pass execution against the independent factor implementation; use original teacher physical matrix in native target mode. Keep native parity and CPU results separately reported.

## Predeclared descriptive acceptance

Engineering: all finite values; 8 seeds; full physical gauge error ≤2e-5; teacher first/final MAE both ≤.08; BPD weight MSE ≤1e-4; numerical immediate-readout preservation ≤1e-6, full-mask native parity ≤1e-5 when executed.

Exploratory mechanism: in at least **6/8 independent roots**, ≥100 authorized eligible pairs, 6-factor donor recovery ≥0.60 and at least **0.20** greater than the best of 255 random-dark sham arms on the same eligible pairs. Full-dark and all-factor recovery should be ≥0.90. Public clean donor effect must be reported, and public partial-factor effect is *not* required to be small; a significant excess above independent-root calibrated public shams is a future question.

This 8-root run cannot establish 95/95 power or recovery of intrinsic component identities. Confirmation requires at least 59 *new* root-level sham calibrators and a separate sealed source-family-disjoint evaluation. Native 7M transformer, sparse masks, source-label binding and secret-bearing language tokens require their own prospective protocols.

## Test and provenance

Run from BPD root after dependencies:

```bash
python -m mechinterp.bpd.theory.sf02_secret_circuit.experiment --seeds 1013 1019 1021 1031 1033 1039 1049 1051 --out /tmp/sf02_results.json
```

Record Git protocol SHA, experiment SHA, independent finite-seed per-root JSON, exact PyTorch version, and native parity receipt. The protocol's original hash should remain pinned; document later corrections in separate addenda.
