# BPD-v3 FLOOR-01 / BASIS-01: frozen source-only protocol
Registration: 2026-11-17. Status: PRE-DATA (neural), synthetic unit tests authorized after this commit.
Links: LoopedTransformers dark-fibers v015 PR #161, BPD v2 immutable receipts in notes/data/.

## Gate A: Dither-ensemble scoring (Propositions 6 and 7)

Independent unit: Sudoku board, never a cell, readout or repeated dither. Native cohort: the same 183 WB-Stall roots from the frozen benchmark bundle, with both SHA-pinned checkpoints gaoxin492 and yagizdevre analyzed separately. Retain all 183 roots and identify any subset already selected for v2 diagnostics. No protected ARG-03-N data may be opened. Run K=8 arithmetic-scale dither replications per board for development, with a separately declared K=32 sensitivity run if K=8 floor estimates are unstable. Fix seed schedule and log exact checkpoint, roster and tensor hashes before scoring. Native model weights must be supplied explicitly as verified local files. No automatic model download.

For readout probabilities q[b,k,t,c,y], define qbar[b,t,c,y] as the arithmetic mean over K. The plug-in target-ensemble Jensen-Shannon estimate is mean_k KL(q[b,k] || qbar[b]); aggregate equally across readout cells, then equally across boards. Record J by board and readout pass (20,41,62) and across the cohort. A bootstrap over independent boards, plus K-sensitivity, characterizes sampling; retain the K-finite plug-in downward bias caveat. Report outcome entropy, branch frequencies, KL(qbar || qbarhat), and KL(q_k || qhat_gamma_k) separately.

A noise-independent surrogate qhat[b,t,c] must satisfy the risk identity: mean_k KL(q_k || qhat) = J + KL(qbar || qhat), up to numerical error. A surrogate receiving the reference dither or a clean-trajectory signal gamma is driven; only for matched q_k and qhat_gamma_k is I(gamma;Y) >= J - mean_k KL(q_k || qhat_gamma_k) applicable. Do not infer the numeric floor from pairwise KL of near-identity controls, or infer an information leak directly from the saved v2 final KL=0.010. The CI network weights needed to replay v2 gate outputs are absent from the extracted stage-4 archive.

Prospective checks: finite identity residual <= 1e-9 nats/cell; all probabilities normalized and finite; zero JS for identical dither reads; strict JS>0 for a planted bistable two-branch toy; static barycentre risk equals JS; a perfectly driven reference has zero matched risk and saturates the theoretical leak inequality. Report failures.

## Gate B: Offline exact-factor baseline (Propositions 9 to 11)

Compare two distinct complete factorizations, without a remainder or gate training:
- Input-complete C=d_in: invertible read basis V [d_in,C], U = V^{-1} W^T. Rivals: neuron basis, full right-SVD basis (including null directions), and fixed random orthogonal basis.
- Output-complete C=d_out: invertible write basis U [C,d_out], V=W^T U^{-1}. Rivals: output coordinate basis, left-SVD write basis, and random orthogonal write basis.
Measure full-on ||(VU)^T-W||F / ||W||F, number of switched-off components under the closed-loop ascending-|a_c| ||u_c||_G prefix for beta0 in {0.01,0.03,0.10}, actual module error, certified bound, and optional output-logit KL. The gate observes only its own module input and period-modulo call role; no teacher-forced gate schedule. Compare within matched C, disclose C when contrasting input and output bases. Use an identity seminorm G in the initial CPU test; delayed Gramian use requires independently measured state Jacobians and Taylor remainder control.

Synthetic positive control: a 2D phase-pooled law with one full-support pass type and two pass types with complementary exact-zero supports. Check that the pooled exact-zero objective identifies the coordinate basis up to signed permutation, while the full-support phase has no unique basis. This validates the exact-atom premise rather than a density-based approximation. A later learned-basis objective must register its threshold, optimizer and sham before being called an identification result.

Prospective checks: relative full-on residual <= 1e-11, every partial mask meets its certified bound within 1e-10 numerical slack, all mask counts monotone in budget, and pooled exact-zero advantage positive on independent synthetic draws. Run source-only CPU tests first and attach raw receipt.

## Native neural boundary and analysis status

Native Gate A and first-readout native Gate B remain PENDING until the local pinned TRM checkpoints, complete dither logits, and module-vector inputs are present. Source-only results are labeled SYNTHETIC; they cannot support claims about Sudoku solve rates, dark component discovery, neural J_delta, or v2 leakage magnitude. B4 v2 status remains INCOMPLETE, as in notes/data/STATUS.md.

Require an EvalMath mechanistic-resolution preflight for any subsequent independent-board causal claim. Record visitation and retention by board, K perturbation sham controls, and a familywise error plan before test-time evaluation. No new native GPU test may be retroactively designated confirmatory.
