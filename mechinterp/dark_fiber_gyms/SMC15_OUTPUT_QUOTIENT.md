# BPD-OUT01: near-null observed causal directions with well-separated physical SAE atoms

**17 November 2026, source-locked interpretation addendum.** Anchors: [SMC-15 published results](https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/real_recurrence_sae_v1/SMC15_RESULTS.md), [frozen SMC-15 protocol](https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/real_recurrence_sae_v1/SMC15_PROTOCOL.md), [independent raw-receipt CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44088108070), and earlier [BPD curved basis boundaries](../bpd/theory/SAU_CURVED_BOUNDARY.md). This report adds mathematical interpretation and a source-independent linear-algebra test. It does not retrain BPD, touch the B1–B4 registered gates, or access the ARG-03-N sealed 168 cases.

## Source evidence

The SMC-15 native experiment loads exactly three frozen 44,001-parameter Sudoku recurrent teachers, with five 48→96 top-four SAE snapshots, from archived source-hash-pinned SMC-11C/13A/13C receipts. Two entirely fresh, uniquely solvable noncanonical Sudoku cohorts (64 puzzles at 30 holes, 64 at 34 holes) contribute eight preselected missing-cell goals each. SMC-15's source A had **7 and 8** naturally coactive feature pairs with *normalized nine-digit output-frame* sigma-min below **0.10**; all four B teacher SAE fits had **zero** such pairs out of **512 goals per fit per cohort**. Across source A's 15 low-output-sigma goals, **every physical decoder pair remains well conditioned** with minimum physical 48-dimensional sigma-min **at least 0.9006**. The same-source finite physical-radius-0.20 native min-direction intervention has mean output-effect ratios of **0.125** on the seven low-sigma novel30 cases and **0.152** on the eight low-sigma novel34 cases, against approximately 0.653 and 0.625 in the more common higher-sigma puzzles.

These are small descriptive subgroups under a registration that demanded 30 low-sigma goals and ten independent low-sigma puzzles per teacher/cohort. **The registered scientific verdict is UNDERPOWERED.** The fifteen instances are possible *finite-horizon task-observation quotient* witnesses, not evidence that the native hidden state is globally unobservable or that a learned BPD decomposition has discovered them.

The same experiment finds only 3–8/512 dual-active mapped SAE feature pairs in each A→B comparison. Earlier SMC-14 causal-effect correspondence must therefore be kept separate from correspondence of *naturally active* internal coordinate pairs. The archive recorded aggregate dual-activation counts but did not preserve every target top-four mask; a casewise independent audit of those counts requires source-model replay.

## The two-feature linear bound

Let H be a finite-dimensional physical hidden-state space, Y a task-specific readout space and L:H→Y the local downstream response Jacobian, for example

  L_h = P_digit D[head ∘ F^3](h),

where P_digit centers the nine Sudoku digit logits and F is the frozen native recurrent block. Let u and v be physical unit SAE decoder directions and U=[u v]. Assume sigma_min(U)=gamma>0, so the two physical coordinates are distinguishable. Also assume each of Lu and Lv is nonzero. Define

  a = ||Lu||, b = ||Lv||, nu = max(a,b),
  Q=[Lu/a, Lv/b], epsilon=sigma_min(Q).

Let c be a unit right singular vector of Q for epsilon, and set

  x=U diag(1/a,1/b) c,  w=x/||x||.

Then:

  ||Lx|| = ||Qc|| = epsilon;
  ||x|| >= gamma||diag(1/a,1/b)c|| >= gamma/nu;

therefore the physical unit vector w satisfies the **local near-null output bound**

  ||Lw|| <= epsilon * nu / gamma.

This makes the distinction precise. A high physical gamma can coexist with a small normalized output epsilon. The output projection and recurrent transition, represented by L, can identify two physical directions observationally. The finite-horizon observable quotient has a small singular direction in their span, even when the ambient dictionary does not.

For a nonlinear map G(h), the bound is a tangent statement with L=DG(h). If DG is M-Lipschitz on the physical radius-r segment, the finite edit has the additional Taylor remainder:

  ||G(h+r*w)-G(h)|| <= r * epsilon*nu/gamma + (M/2)*r^2.

SMC-15 estimated the tangent with a centered two-sided finite difference at 0.08 and separately tested finite dose 0.20. A third-derivative bound that would connect that estimate rigorously to DG has **not** been certified on the trained teachers, nor has any uniform nonlinear remainder M. Its native FP64 finite interventions remain empirical evidence.

## Source-independent computational certificate

[smc15_output_quotient.py](smc15_output_quotient.py) constructs x and w from the two-column normalized output singular system and checks the linear finite-dimensional inequality. Five tests cover an exact rank-one observed output of orthogonal physical directions, nearly parallel outputs, 64 seeded random full-rank cases, invariance under simultaneous physical-frame rotations and rejection of singular physical input frames/silent output columns.

Run from BPD root:

    python -m pytest -q mechinterp/dark_fiber_gyms/tests/test_smc15_output_quotient.py

The exact rank-one witness shows that **the physical feature count and the observable task-causal rank need not agree**, while the random and covariance tests protect the implemented bound from index/normalization errors. The theorem is written and numerically checked; it is not presented as an independently Lean-kernel-verified general inequality.

## What to measure next in BPD and LS-SAE

A recurrent BPD intervention mask should be evaluated against the *downstream observable Gramian* G_{h,k}=U^T L_{h,k}^T L_{h,k} U as well as the physical matrix factorization. The Gramian is coordinate-covariant under changes of the paired physical feature basis; the static diagonal mask commutant from the optimal-basis theorem cannot by itself establish a useful causal output quotient. If the low singular direction is estimated from outputs on a test puzzle, it needs fresh puzzle/family calibration to predict other unseen contexts.

For a new **BPD-OUT02 prospective run**, hold the teacher, target SAE and physically native feature pair fixed, choose directions on calibration-only puzzle sources, and evaluate native task-margin necessity and matched physical-norm shams on seed-disjoint 30/34-hole puzzles. Report coactivity-admission denominators, full output Jacobian spectra, exact decoder-frame singular values and the cross-depth decay/rise of the observable near-null direction. Keep source-model readouts and true target-label calibration distinct.

The 7M-parameter [LoopedTransformers LS-SAE11](https://github.com/MoralityLabAI/LoopedTransformers/blob/main/mechinterp/sae_causal_transfer/results/LS_SAE11_RESULTS.md) has an existing prospective hard-Sudoku failure. Its distinct 512D physical z_H / PCA / SAE and ACT-step semantics require an independent protocol. This 44K-teacher SMC-15 result does **not** convert that LS-SAE failure into a success or transfer the output-frame threshold directly to the larger network.
