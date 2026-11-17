# BPD-SF04 — Source-label binding and latent nonlinear transport across interleaved user records

**2026-11-18 prospective eight-root exploratory pilot.** Motivation: Daniel Rodríguez's [Spectral Faithfulness](https://github.com/idolum-ai/spectral-faithfulness), especially Phase II source-attribution confusion and Phase III interleaved users. Parent [BPD-SF03](https://github.com/MoralityLabAI/BPD/pull/37) found no strong incremental factor advantage after signed local-Jacobian matching (0/32 roots), with 9/32 physical factor-fit failures. This experiment changes the independent variable from a directly supplied continuous secret to a **source-to-content binding** under a repeated, trained update.

## Engineering-development exclusion

Only two source-development roots **901 and 902** were used to choose architecture, learning schedule, factor count and shortlist size. The source-tag pilot showed heldout mean absolute error ≈0.017 for root 901 at 1200 teacher updates. With 14 greedy of 20 full-rank BPD factors, source-donor recovery on excluded roots was ≈0.71 and ≈0.74; their public-side effect varied substantially (≈0.26 versus ≈0.03). These are **tuning results** and must never count as fresh inference. Fresh exploratory roots are locked **1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367**. No new-root results are to be inspected before protocol and source freeze.

## Structured interleaved-source gym

Draw independent source labels `U_1,U_2∈{A,B}`; values `V_1,V_2,P∼Uniform[-0.8,0.8]`; query role `R∈{A,B,U}`. Each trial is **three successive calls to the same physical recurrent operator** with input dimension 9. First two input events carry one-hot source name and scalar value, and the third carries the query role, public scalar and query marker. The same source may appear twice, in which case the second write overwrites its earlier value.

Correct final target is

\[
Y^* =
\begin{cases}
P& R=U\text{ (unauthorized public read)},\\
V_2 & R=U_2,\\
V_1 & R=U_1\ne U_2,\\
0 & \text{queried source never written}.
\end{cases}
\]

No messages are natural-language tokens. Distinct identities and overwritten keys make a bag-of-values solution insufficient. A source-blind oracle that sees ordered values and role but does not see source labels has conditional expectation `P` for public queries and `\frac12 V_2+\frac14 V_1` for A/B queries under this distribution. Report its heldout MAE against the true target as a provenance-negative control.

Train shared nonlinear residual transition on `h∈ℝ^{20}`:

\[
F(h,x)=h+\tanh(hA+xB+b)C,
\qquad A∈ℝ^{20×40},\ B∈ℝ^{9×40},\ C∈ℝ^{40×20}.
\]

Actual readout `q(h)=h_0`. Start with zero state. With `(h_1,h_2,h_3)` after the two writes and query, train all teacher parameters for **1200 Adam updates, lr .012, batch 256** using

\[
3\,\mathrm{MSE}(q(h_3),Y^*)+
0.4\big[\mathbb E q(h_1)^2+\mathbb E q(h_2)^2\big]
+10^{-4}\left(\mathrm{mean}(A^2)+\mathrm{mean}(C^2)\right).
\]

Thus hidden states must retain provenance-relevant information while earlier observable outputs are trained toward zero. This alone does not establish security, because the output boundary is artificial.

## BPD reconstruction and causal source swap

Freeze teacher. Train a full physical `C≈VU` with `V∈ℝ^{40×20},U∈ℝ^{20×20}` for **400 Adam updates, lr .03, batch 256** on independently generated clean teacher states, minimizing `1.5 MSE(VU,C)+MSE(φ(h,x)VU,φ(h,x)C)` across three encountered write/query passes. All 20 components are active. Compare independent rollout to native `BPDModel/TiedComponentLinear`; in the full-mask case require absolute parity ≤1e-5. Singleton factors depend on the GL(20) gauge.

For each paired example keep **identical first write, second scalar value, query role, query/public inputs and event order**, and **flip only the second event's source tag**. This changes which user owns the value. Measure source recipient and donor native teacher outcomes independently, before any BPD selection.

Use **240 development pairs per root** for greedy selection of **14/20** components by minimizing authorized donor-target MSE at the final query, with a fixed deterministic tie break. On **512 fresh test pairs per root**, eligibility is frozen: authorized query `R=A/B`, clean recipient and donor final errors ≤0.08, current before-query recipient/donor readout difference ≤0.04, and future clean donor/recipient difference >0.20. Never replace ineligible pairs.

After the second write and before the query, form the physical component donor-minus-recipient difference. Project it into the *actual current readout kernel* by setting coordinate 0 to zero, assert equality to ≤1e-7, and continue with the clean third recurrent pass. Observe:
- 14-factor greedy write (primary partial intervention).
- All 20 factors (full physical BPD factor block).
- Exact dark-projected native donor-state difference.
- 64 independent equal-L2 random-dark perturbations (weak null).
- 64 **signed one-step future Jacobian** and norm-matched random-dark perturbations (strong local-linear control).
- Public-query output effect and clean source-tag-swap baseline, separately from authorized donor recovery.

The 64 shams on one root are repeated within-root controls, not 64 independent experimental units. Keep trace of candidate factor indexes, full-block GL(20) gauge invariance, and current readout conservation. Record the public-query effect even when partial factors successfully transfer the source. Do not call edit-induced public drift a real privacy breach.

## Frozen exploratory admission, separate ledgers

**Engineering:** all 8 roots processed, finite outputs, teacher MAE ≤0.08, full physical factor weight MSE ≤1e-4, actual BPD native parity ≤1e-5 on excluded seed 901, full-block gauge equality ≤2e-5, immediate intervention readout conservation ≤1e-6.

**Provenance learning:** ≥6/8 independent roots achieve teacher MAE ≤0.08 and teacher MAE ≤0.6× source-blind conditional-mean MAE; all 20-factor donor recovery ≥0.90 and full-dark donor recovery ≥0.90 on ≥100 authorized eligible test pairs.

**Partial component routing, exploratory:** at least 6/8 independent roots must simultaneously achieve ≥0.60 greedy 14-factor donor recovery, ≥0.20 advantage over the *mean* of 64 signed-gradient shams, and public-query mean absolute output disturbance ≤0.08. Report the random-dark comparison separately. A failure of this composite gate is a genuine negative result under the frozen shortlist size and public preservation condition.

All conditions are **exploratory**, with size/thresholds informed by two excluded engineering roots. Eight roots do not calibrate a 95th-percentile 95%-confidence sham or prove uniqueness/necessity of a BPD component. Confirmation would require ≥59 independent-root calibration plus a separately source-family-disjoint sealed evaluation. **Keep all ARG-03-N 168 heldout test cases sealed; do not read or rerun other BPD/CQ datasets.**

## Reproduction

Pin source, protocol and tests Git blobs in `LOCK.json` before running the first eight roots. Execute CPU with PyTorch 2.5.1 and per-root independent RNG stream offsets. Archive every per-root eligible denominator and raw summary JSON, including failures, before interpreting results. Use source-independent tests to validate label overwrite and donor tag-only counterfactual, no-param full-operator gauge, current-readout preservation and native BPD parity.
