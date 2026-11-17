# BPD-OUT03: active SAE erasure and task-gradient geometry

**Research addendum, 17 November 2026.** This mathematical note follows the independently audited [SMC-23 target-native experiment](https://github.com/MoralityLabAI/Compactification-Theory/pull/217). It is a finite-dimensional gradient theorem with CPU witnesses. Its nonlinear bound requires a real Hessian bound before it can be certified on a trained recurrent model.

## Source-model finding

SMC-23 tested 128 new unique-solution noncanonical Sudoku puzzles, five frozen SAE components and 5,120 selected goal-level interventions. Its preregistered B501/B701 teacher gate **failed for all four SAE pairs**, despite adequate pre-outcome eligibility in all eight teacher/SAE/family cells. The hard tangent sham matched each target-native top-active SAE erasure in physical dose and was selected from 33 candidates by inactive-preactivation fidelity, without task labels.

The mean paired true-digit margin benefit among 2,096 B eligible goals was approximately -0.07373 logits. Original pretrained source A, which had substantially better Sudoku capability and somewhat larger physical edit doses, showed +2.1304 and +1.9038 logits on the two novel families. The differences involve trained model competency and other frozen physical conditions, so they are descriptive. SMC-16 and SMC-17's earlier source-mapped tests remain separately registered.

## Exact theorem

Let the active physical SAE encoder covectors form a full-row-rank matrix A of size k by d, and choose the active coordinate j. Let b=(z_j+tau)e_j for a positive code z_j and overshoot tau>0. The minimum-norm encoder-selective erasure is

\[
\delta=A^\top(AA^\top)^{-1}b.
\]

It lies in the row space of A and satisfies A delta=b. A same-dose tangent sham s satisfies As=0 and ||s||=||delta||=r. For a downstream native task-margin gradient g at the clean state, define

\[
\lambda=(AA^\top)^{-1}Ag,\qquad
g_R=A^\top\lambda,\qquad g_T=g-g_R.
\]

For linear native task margin m(h)=g dot h+c, the exact paired kill-minus-sham damage is

\[
D=m(h-s)-m(h-\delta)
  =g\cdot(\delta-s)
  =(z_j+\tau)\lambda_j-g_T\cdot s.
\]

Over the whole sphere of same-dose tangent shams, D ranges between (z_j+tau)lambda_j minus r||g_T|| and (z_j+tau)lambda_j plus r||g_T||. Consequently an SAE feature's high activation level alone does not constrain the sign of its task benefit.

For twice differentiable nonlinear m with task-margin Hessian operator norm bounded by M on both edit paths, Taylor's theorem yields

\[
|D-((z_j+\tau)\lambda_j-g_T\cdot s)|\le M r^2.
\]

The sufficient certificate for a positive effect against every same-dose tangent sham is

\[
(z_j+\tau)\lambda_j>r\|g_T\|+Mr^2.
\]

This is a *conditional* task-gradient certificate. SMC-23 measured finite numerical native responses and supplied no uniform trained-network Hessian bound for this use.

## Exact rational counterexample

In R3 take active covectors e1 and e2, choose the first coordinate with amplitude 1, and use delta=e1 and same-dose sham s=e3. Both edits satisfy the active SAE constraints, while the sham preserves both active coordinates.

- Task gradient g=(1,0,2) gives D=1-2=-1.
- Task gradient g=(1,0,-2) gives D=1+2=+3.
- With g=(3,0,0.2), the dual term 3 dominates any unit same-dose tangent contribution, whose magnitude is at most 0.2, so D is positive for all tangent shams.

Thus a fixed active-code erasure can be harmful to the task margin, helpful, or neutral according to the downstream gradient's physical projection. These are exact Fraction witnesses computed by the independently authored source.

## CPU verification

The companion [gradient split implementation](smc23_gradient_split.py) constructs the minimal physical erasure and decomposes task gradients into active-row and tangent subspaces. [Six tests](tests/test_smc23_gradient_split.py) check rational sign reversals, the sufficient positivity threshold, 64 seeded 48D full-rank cases, a bounded nonlinear quadratic remainder, orthogonal gauge covariance and malformed rank/dose rejection.

Run from the BPD root:

    python -m pytest -q mechinterp/dark_fiber_gyms/tests/test_smc23_gradient_split.py

This is a mathematical statement about constrained interventions. A frozen SAE's active mask and an external task-gradient observer can disagree because the latter also sees native directions tangent to the active encoder constraints. Matching inactive preactivations among shams reduces a particular input confound but does not guarantee that their task-gradient projections match.

## Prospective bridge

A subsequent BPD-OUT04 experiment can test whether a **calibration-frozen prediction of the target-task gradient split** explains the observed strong source-A erasure response and null independent-teacher responses. It should use new capability-matched independent teachers, fresh Sudoku seeds, strict source-only calibration, the already frozen 33-candidate hard-sham rule, independent-puzzle clustered inference, and actual native end-task margins. A gradient measured using the heldout correct digit is answer-conditioned and must be labeled as a diagnostic. Only a predictor fitted on independent calibration cases can support a new prospective transfer claim.

The old SMC-14/15/16/17/23 scientific verdicts, original BPD parameter-level source tests and sealed ARG-03-N cases remain unchanged.
