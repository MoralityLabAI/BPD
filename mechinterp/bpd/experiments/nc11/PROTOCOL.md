# BPD-NC-11: source-only registration for causal-Jacobian cross-model alignment

**Prospective first commit, 2026-11-18 UTC.** This protocol is committed before NC-11 code or new model/data. NC-10's original eight-pair negative result is historical development context, not NC-11's confirmation sample. NC-11 may use previously developed code as an engineering instrument, with complete provenance.

## Scientific question

NC-10 used paired, unlabeled **prewrite activation** Procrustes to carry a four-dimensional score-gradient subspace between separately trained four-pass dense recurrent models. Across eight paired training roots it failed noninferiority to B's own projector, task-specific branch separation passed, and alignment failed the permuted and matched-sham causal comparisons. NC-11 asks whether **local causal-response Jacobians**, estimated from a sharply limited unlabeled B calibration budget, provide a better cross-model chart than raw activation geometry.

This experiment has a real additional target information cost: the Jacobian aligner reads B output-score gradients on the calibration examples. It is thus **not an unsupervised, target-free transfer**, and a B-only gradient projector computed from the identical calibration examples is a mandatory equal-target-budget baseline. The B-owned full-discovery projector is a separate upper-reference arm. Model-generated scores rather than external labels define all discovery and calibration gradients.

## Task, model pairs, roots and splits

A and B are **independently initialized, separately trained** 20-dimensional `DenseLooper` models, each reusing one physical 20x20 `mix` weight through four nonlinear recurrent passes. Task A predicts sign(x0) if x4>0, else sign(x1); task B predicts sign(x0) if x4>0, else sign(x2). The positive-context task branch is shared; the negative-context branch is task-specific by construction.

- **Eight fresh paired engineering roots 1100..1107**; source-only engineering smoke uses **1099** and is excluded from the development aggregate. A's model init seed=root, B's = root+100000; both train 384 Adam updates, batch 128, LR .005, on separate 2048-input Gaussian training streams split 1/11. No shared weights or optimizers.
- Source A model-score gradient discovery uses **1024 cases split 2**, independent of calibration. An extra B **full 1024-case split-2 gradient projector** is scored solely as a reference ceiling and never influences alignment, rank admission, gain matching, or sham selection.
- **Target information budget**: exactly **128 positive-context unlabeled paired inputs**, selected by original order from a fresh 512-input split-4 calibration pool, for both Jacobian and activation Procrustes; require ≥128 positive-context cases in 512. Task labels are barred from every discovery/calibration/admission step. The B-calibration-only rank-four projector uses these **same 128 B Jacobian rows** as the Jacobian aligner, at precisely the same target-gradient budget.
- Source-only **1024 input examples split 6** are used to measure B prewrite activation physical retained-write RMS and draw eight local shams; neither model gradients nor task labels from that pool may determine the alignment. Final independent **1024 labeled evaluation examples split 3** are touched only after all ranks, alignments, shams and physical gains are fixed.
- Every A and B model must achieve ≥85% clean held-out true-label accuracy in each context; both `mix` weights must have ≥90% entries with absolute value ≥1e-4. The **independent unit is a paired trained model root**. All eight independent roots must pass technical admission for the primary claim.

## Frozen source-only estimators

1. Fit top **rank-four** per-pass physical prewrite projectors `P_A(t)` from the uncentered model-score gradient moment on 1024 A discovery cases, and `P_B_full(t)` on the full-discovery B source for descriptive upper reference. Fit `P_B_cal(t)` from 128 paired positive-context B calibration gradients. Require λ4>1e-12 and λ4/max(λ5,1e-12)≥1.001 at all four passes for all three.
2. The **causal-response Jacobian fingerprint** at pass t is `G_A[n,t,:] = ∂s_A(x_n)/∂a_A,t`, `G_B[n,t,:]=∂s_B(x_n)/∂a_B,t`, where s is the unmodified final model logit and a is the **native physical pre-mix read**. Each row is normalized to unit Euclidean length with floor 1e-12; a zero row invalidates the calibration pair. No detached self-confidence sign is applied, since A and B have a common **positive branch** score orientation.
3. For each pass, `R_J(t)=argmin_{RᵀR=I} ||G_A(t)R-G_B(t)||_F` by fixed SVD polar factor `R=UVᵀ` of `G_AᵀG_B`. Transport the A projector as `P_J(t)=R_J(t)ᵀP_A(t)R_J(t)`.
4. **Activation baseline:** on those *same 128 examples*, fit `R_act(t)` to the unnormalized physical prewrite activations `a_A(t)` and `a_B(t)`; transport `P_act=R_actᵀP_AR_act`. The frozen **permuted-pair Jacobian** control fits `R_perm` on A Jacobian rows and B Jacobian rows permuted by the root-seeded `np.random.default_rng(root+11110011)` permutation, identical at all passes.
5. Also score raw identity transport `P_A` and the B calibration-only projector `P_B_cal`. Report per-pass normalized Jacobian fit residuals and activation fit residuals, orthogonality, source-target projector overlaps, matrix symmetries, and distinct shuffled-rows permutation SHA. A high alignment residual is a failure of correspondence, not a prompt to refit.

## Equal physical dose and counterfactuals

- Use the exact target B `BPDModel/TiedComponentLinear` wrapper, full-ones target parity at all four logits max abs ≤5e-5, KL≤1e-7 nats and relative factor weight residual ≤1e-4. Four-pass closed-loop interventions use `override_read` on **all 20 native factor coordinates**, gain=α and zero residual bypass.
- Define **reference retained write dose** `D_J` as RMS of `P_J`'s clean physical write on the **independent 1024-case dose pool**. For `P_act,P_perm,P_B_cal,P_B_full,P_A`, calculate source-only `alpha = D_J / D_arm` and admit the primary paired comparisons only if each of the first three comparators has α in **[0.5, 1.5]**. Those α values are fixed before held-out outcome access; full-B and raw identity remain descriptive even if their α falls outside bounds, in which case score gain 1 with a recorded unmatched-dose flag. Record every reference, applied gain, clipped/out-of-bounds case and realized held-out displacement. The **Jacobian transfer gain is exactly 1**.
- For the primary local-sham null, generate exactly **2048 fixed-angle rank-four** candidate subspaces (θ∈{.45,.65,.85,1.05}) relative to `P_J` on target B, using the NC-06 model-blind Gaussian-complement mechanism. Fix the root-specific RNG, choose **first eight** with physical retained-write RMS within ±15% of `D_J` using the **dose-only split 6**; no held-out labels or outcome ranking. A root with fewer than eight qualifying shams is ineligible, with all counts retained.
- Source-only negative test: zero model B output head on a copy, ensuring all native intermediate write gradients are exactly zero. On each target B, transport `P_J` to a fixed root-seeded orthogonal **factor gauge Q** and verify `V'P'_JU'=VP_JU` within relative 1e-4 and retained logits within 5e-5; this verifies intervention numerics, not cross-model equivalence.
- Apply all fixed projectors and shams at **the same B physical mix site**; evaluate independent B true-label NLL, shared and task-specific contexts separately, passwise KL(clean||kept), correct count, task accuracy, and actual physical displacement. A's own P_A effect on A's task is a diagnostic for source mechanism competence. The B full-discovery reference measures target-only capacity ceiling; its high B-gradient access must be disclosed.

## Registered independent-root criteria

Four primary paired directional contrasts, one per **root** for each comparison:

- **Jacobian vs activation:** `NLL_B,shared(P_act)-NLL_B,shared(P_J)`, lower interval >**+0.01 nats**.
- **Jacobian vs permuted Jacobian:** `NLL_B,shared(P_perm)-NLL_B,shared(P_J)`, lower >**+0.01 nats**.
- **Cross-model transfer vs equal-target-gradient-budget B-calibration projector:** `NLL_B,shared(P_J)-NLL_B,shared(P_B_cal)`, upper ≤**+0.05 nats**.
- **Sham specificity:** `median_8 NLL_B,shared(P_sham)-NLL_B,shared(P_J)`, lower >**+0.02 nats**.

The 4 fixed directional comparisons use **98.75% two-sided percentile CIs** from **4096 whole-pair bootstrap resamples**, RNG seed 1110011 + comparison index×101. All eight roots need task competence, parity, eigengaps, 128-case calibration, equal dose for *activation/permuted/B_cal*, eight shams, zero-head negative control and within-B factor-gauge closure. A joint `ENGINEERING_JACOBIAN_TRANSFER` requires all four CI gates. Otherwise preserve `ENGINEERING_NULL_OR_INELIGIBLE`; no root, method, threshold, angle, rank, dose tolerance or comparison is changed after the first development run.

**Descriptive** diagnostics outside the four directional family: B-owned full-source reference, source A own task, identity transport, private-minus-shared transfer deficit, fit residuals, per-pass neural Jacobian covariance spectra, overlap with B calibration and B full projectors, and exact native/rotated code gauge parity. These cannot rescue a failed registered gate.

## Evidence, preflight and sealed populations

This is a **synthetic eight-pair engineering development study**, deliberately stronger than NC-10's activation-only input information but weaker than a cross-family confirmation. B's calibration gradients mean it is not target-blind causal transfer. The NC-06/07/08 negative gates and NC-10 first-run negative result remain unchanged. No trained language-model weights are downloaded.

The 95th-percentile matched-sham distribution with 95% confidence needs ≥59 independent sham roots for a nonparametric maximum-order-statistic tolerance bound. Eight independent paired roots do not calibrate it; a later cross-family claim requires a *new* preregistered root population, exact EvalMath mechanistic-resolution preflight and its own untouched held-out evaluation. All protected ARG-03-N, AMM-03 and unrelated confirmation cohorts remain sealed. Preserve the first raw negative/positive receipts, source locks and engineering repairs regardless of outcome.
