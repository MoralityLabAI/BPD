# BPD / EFD-02: full future-Jacobian matched pair dark channels (LGA-09)

**18 November 2026 · Research addendum on prospectively registered LOOPSCOPE-12.** First native source-locked [GitHub Actions #44315883529](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44315883529), exact original [PR #200](https://github.com/MoralityLabAI/LoopedTransformers/pull/200), independently audited with Python stdlib: **`DEVELOPMENT_STRONG_FULL_J_PASS` in 3/3 historically exposed 44K TRM teachers 103,211,307.** New source seed 31063 was generated after pretest lock; the teacher roots are reused development checkpoints. No BPD factor dictionary or TiedComponentLinear native causal mediator was measured in this experiment.

## What is new

Previous LGA-08 had already restricted source-legal predecessor/goal edits to the physical 75D pair K at current-state h2, invisible to the full 10-class current head and tangent to the frozen external LGA-05 scalar E. It matched eight shams for the *single signed donor-vs-original margin derivative* and found +21.97/+27.05/+26.95 percentage points selected donor-target advantage across the three teachers. LGA-09 tested the full ten-class future Jacobian, so changing a different rival logit in first order could no longer explain the selected-versus-null difference.

For `L(h2)=head(F(F(h2)))_goal∈ℝ^10`, let `J=D_{h2[S]}L∈ℝ^{10×96}` and `K` the physical pair head/energy tangent. The restricted `B=J|K` has numerical rank **10 in all 1,536 original source-goal cases**. Consequently the **first-order future-output-dark fiber** `K∩ker J` has dimension **65**, compared with dim K=75.

For source-legal selected donor `z∈K` at natural pair L2 radius `rho`, the full-J control fixes `Bz` across all ten classes and `||z||=rho`. Decompose `z=z0+q`, where `z0` is the orthogonal projection into row(B) and `q∈K∩ker B`. The matched sham fixes `z0` and rotates `q` isotropically within the 65D future-first-order-silent fiber. Every sham has identical current-head linear response, first-order external energy, original physical L2 dose and entire native two-step ten-logit **first derivative**.

The source-selected edit preserves finite task-relevant donor information that random q orientations often destroy, pointing toward a nonlinear finite-response mechanism associated with source geometry.

## Exact original first-development findings

| Historical frozen teacher | Selected donor correct, 512 goal cases | Selected minus median eight full-J sham | 2,000 whole-base 95% CI |
|---|---:|---:|---|
| 103 | 181/512 | **+17.68 pp** | [+13.87,+21.68] pp |
| 211 | 219/512 | **+21.88 pp** | [+18.07,+25.88] pp |
| 307 | 241/512 | **+23.83 pp** | [+20.21,+27.34] pp |

All three passed their registered ≥+10pp full-J, ≥+10pp signed-margin and ≥+15pp random-K gates, source competence and reached-donor parity conditions. All 1,536 source-goal cases were admitted. The actual applied f32 linearized ten-logit response matching error was at most `4.28e-7/4.69e-7/4.22e-7` across the respective seeds.

Natural entire-state donor reconstruction recovered **100%** of independently correct donor targets among jointly competent donor/recipient cases; natural predecessor+goal pair donor reached 95.5–98.1% raw correctness. Head-only pair donor obtained 196/234/258 correct versus 181/219/241 for head+energy tangent at identical dose. The nonlinear external energy E increased on **every** selected natural-dose pair edit, by mean total +14.89/+12.08/+15.73. The extra E-row restriction still has no demonstrated added behavioral value.

All original first experimental artifacts remain unchanged: model/energy pretest lock `4d0460c9bd6fd6e342c6f0282a7f9744a7fc5082d4bf7e1f2f55af0253097ed4`, native full receipt SHA256 `480f5a5b7e133060602e1ba850e39bd25565313b68585103820f43b62305d65a`, source ten-J and physical edit gzip SHA256 `39b0cd14b4ad2a2df05b798da3d6419b263bc471494c2e3af4bfed0b34b8d75e` and `6500f8da067eedb76eea37d5a0572ed45b5a0ca44905aa3deae2849e4fe48421`. Independent original audit `9a9c746bab12088ceb797d2c41db105a4b075d44f366b5e064412cf6a99ffa21` verifies all **47,616 pair edits**, 1,536 source records, physical constraints and exact original bootstrap verdict.

## Mathematical research connection to BPD / Compactification-Theory

Let `U` be a physical write space, `A=DP` the current public readout Jacobian, `g=dE` and `B=D(P∘F^{T-t})` the future readout Jacobian. The **double-dark** local tangent subspace is

```
K0 = U ∩ ker A ∩ ker g,
K1 = K0 ∩ ker B.
```

By rank-nullity, `dim(K1) ≥ dim(U) − rank(A|U) − 1 − rank(B|K0)`. For the current two-token 96D case, 20 current readout constraints plus one independent energy row plus ten independent future logit rows leave **65** local directions in K1. Their finite write operation can still change eventual output by curvature even when `Bδ=0`, and matched control pairs can have the same nonzero Bδ while differing strongly after finite execution. This is the appropriate **second-order dark-fiber** target for mathematical formalization.

It is a physical-state rank theorem and does not depend on which BPD factor basis happens to span an equivalent write. If physical factor coordinates are rotated orthogonally, the retained projector must be transported accordingly; factor singleton selections remain gauge-relative. The original EFD-01 source-only rank/null machinery can test this bound with a newly defined double-dark Jacobian composition.

Suggested Lean proposition:

```
dim (ker A ⊓ ker g ⊓ ker B) ≥
  dim U - rank A - 1 - rank (B|_(ker A ⊓ ker g))
```

In Lean implementation, state the theorem using finite-dimensional submodules and codimension; restrict the future B to K0 before applying rank-nullity to avoid ambiguous rank units. A dense matrix theorem for a rectangular 96D native pair gives a simple engineering witness.

## Next discriminating experiment

A favorable next test is a **source-only second-order causal geometry gym**. Freeze the ten-logit full-J fiber K1 and compare true donor patch against controls that match not only ten first derivatives but also **the second-order donor-versus-original output-margin term** `(1/2) zᵀH_m z`. This isolates more of the finite nonlinear response, although the ten-class argmax remains sensitive to the other nine output Hessians. The complete ten-output Hessian-vector response is a stronger but costlier future control.

Use fixed excluded synthetic models first to establish numerical HVP parity, bracket/coverage of curvature matching under a 65D sphere, physical null/gauge tests, degenerate matching accounting and CPU costs. A prospective primary science experiment requires a separate new source roster (e.g. 31064), immutable model/code artifact lock, independent stdlib numerical auditor, and predeclared source cluster/teacher-root thresholds. If the Hessian-matched null cannot be constructed often enough, report the geometric admission failure rather than resampling winning cases. A positive second-order control is still source-conditioned in a known cyclic-copy task and cannot establish semantic feature identity or general learned reasoning.

**Experimental scope:** This finding advances the physical dark-fiber claim, while the learned energy observer's explanatory mechanism remains unverified. Transferring this technique to native trained BPDModel/TiedComponentLinear, Nanbeige layers, or recurrent Sudoku requires independent frozen protocols and new model-root confirmation.
