# BPD-OUT04: Class-conditioned causal observables and label uncertainty

**17 November 2026.** This source-independent finite-dimensional mathematics follows [SMC-24](https://github.com/MoralityLabAI/Compactification-Theory/pull/223), whose [complete source-locked report](https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/real_recurrence_sae_v1/SMC24_RESULTS.md) documents the prospective 0/4 failure of a calibration-trained correct-digit task-gradient predictor on 128 fresh Sudoku puzzles.

SMC-24 fitted an affine ridge map from frozen 48D hidden states to the *true-digit minus clean-rival* margin gradient on 512 calibration goals per teacher. It predicted finite native erasure effects from hidden states alone on eight new teacher/SAE/family cells. All eight had adequate independent-puzzle coverage, yet learned ridge improved over a constant calibration-mean gradient in only 2/8 cells, over the goal-slot-shuffled ridge in 4/8, and over the zero-gradient baseline in 1/8. Every original adjusted success gate failed. These results concern a particular 49-by-48 affine learner, the B501/B701 teacher checkpoints and selected physical erasures; they leave general conditional label information unresolved.

## The correct digit changes the observable itself

Consider a differentiable fixed-head model with hidden state h, clean digit logits \(\ell(h)\in\mathbb R^9\), and native readout Jacobian \(J_h=D\ell(h)\in\mathbb R^{9\times d}\). For each hypothetical correct digit y, choose the highest **clean** alternative logit as its fixed rival:

\[
c(y)=\arg\max_{k\neq y}\ell_k(h).
\]

For a physical target-feature kill \(\delta\) and a same-dose tangent sham \(s\), the local paired margin damage for that candidate digit is

\[
d_y=(J_{h,y}-J_{h,c(y)})\cdot(\delta-s).
\]

This is an explicitly **class-conditioned functional** of the same physical hidden state and Jacobian. All nine candidate effects can be computed without supplying the actual ground-truth digit. Selecting the realized effect \(d_Y\) requires knowing or estimating the latent target Y. SMC-24 predicts a 48D true-digit gradient directly from h; this theorem identifies a possible information bottleneck in that formulation.

There is no theorem saying the original Sudoku state loses the true solution. If the model's clean physical h and visible clue input determine the correct digit perfectly, a label-free estimator could identify Y and its gradient. The bounds below become informative only when a declared task population retains conditional uncertainty in Y after conditioning on the information supplied to the predictor.

## Squared-loss Bayes floor and sharp L1 witness

Fix the information accessible to a predictor, write \(p_y=\Pr(Y=y\mid H=h)\), and treat \(d_y\) as the local classwise effect at h. The conditional minimum squared prediction risk over every scalar predictor a(h) is attained by the weighted mean

\[
\mu_p=\sum_y p_y d_y.
\]

For every a, the exact identity is

\[
\mathbb E_p[(d_Y-a)^2]
=\operatorname{Var}_p(d_Y)+(\mu_p-a)^2.
\]

The irreducible squared-error floor is \(\operatorname{Var}_p(d_Y)\). It vanishes if classwise causal effects agree on the conditional support, even when the label itself is uncertain. Otherwise, predicting one scalar effect from h incurs some conditional loss unless additional label information becomes accessible.

For absolute error, any weighted median of the candidate effects minimizes the risk. In the exact two-label case \(d_1=u,d_2=v\), with conditional target weight \(p\), the optimal MAE is

\[
\min_a\mathbb E|d_Y-a|
=\min(p,1-p)|u-v|.
\]

The code contains an exact rational example with \(p=1/4\), \(u=-3\) and \(v=2\), whose irreducible absolute error is \(5/4\). This constructs a classwise identifiability obstruction while leaving the physical state and native Jacobian fixed. It is a mathematical counterexample, **not** a measured irreducible error bound for either SMC-24 trained teacher.

## Errors from using a proxy class distribution

Suppose a label-free model produces \(q_y\) in place of the true conditional \(p_y\), and predicts the class-averaged local effect \(\mu_q=\sum_y q_y d_y\). Let \(R=\max_y d_y-\min_y d_y\) and let \(\mathrm{TV}(p,q)=\tfrac12\sum_y|p_y-q_y|\). Then

\[
|\mu_q-\mu_p|\leq R\,\mathrm{TV}(p,q).
\]

The proof subtracts a common constant from all class effects, then assigns the positive and negative mass differences to the maximum and minimum effects. The inequality is sharp for two classes: \(p=(1,0)\), \(q=(0,1)\) and \(d=(-2,3)\) give difference exactly five, equal to \(R\mathrm{TV}\).

The bound does not guarantee a small *realized* error \(|d_Y-\mu_q|\) for every Y. It controls conditional mean bias given p and q. Its measurement requires a conditional target distribution and labelwise Jacobian responses. A model's softmax class probabilities can be used as q, but a calibration audit is needed before claiming that q estimates p well.

## Connection to the finite nonlinear SMC-24 experiment

A finite hidden-state edit is evaluated through a nonlinear three-block recurrent teacher, while the classwise \(d_y\) above uses the clean local Jacobian. If each fixed-rival task margin has Hessian operator norm at most M along both edit segments of physical radius r, BPD-OUT03's Taylor remainder gives

\[
|D_y-d_y|\le M r^2.
\]

The unconditional conditional-mean mismatch bound then increases by at most \(M r^2\) for a realized finite-label event. The SMC-24 source campaign provided no validated network-wide M, so its native FP64 scores remain empirical. The new mathematical program does not change SMC-24's negative predictive verdict.

The strong source-A erasure effect in SMC-23 and weak B501/B701 effects can result from different task-gradient row/tangent components. Label ambiguity offers another possible contributor to the failure of the *state-to-correct-label-gradient* predictor. This document proves neither explanation dominates on the trained teachers.

## Independent CPU verification and a next test

[smc24_label_uncertainty.py](smc24_label_uncertainty.py) calculates all nine candidate-clean-rival tangent effects from a physical logit Jacobian and target edit pair, reproduces the squared-risk decomposition, computes the class-median minimum MAE, and checks the total-variation bias bound. Eight source-independent tests cover exact finite linear readout equivalence, 9-way random risk, a sharp TV witness, exact-Fraction two-label MAE, weighted medians, zero-dispersion cases, physical orthogonal gauge invariance and invalid input rejection.

Reproduce from BPD root with the existing pinned NumPy CPU stack:

    python -m pytest -q mechinterp/dark_fiber_gyms/tests/test_smc24_label_uncertainty.py

A **new prospective model experiment**, provisionally SMC-25, should calculate the full 9×48 native digit-logit Jacobian on clean stage-one states *without using any correct test digit*, then freeze predicted effects for all nine possible targets before scoring. An outcome-independent candidate effect vector allows calibration of a probability distribution q over correct digits and can be compared with a learned state-to-gradient map at an equal model-forward budget. Require fresh independent puzzle seed families, target-blind prediction artifacts, original physical erasure and hard sham, strict source calibration, and puzzle-level uncertainty. Evaluate classwise effect prediction, native finite remainder, and predeclared label-marginalized error gains. This would distinguish task-label uncertainty from poor gradient geometry learning.

BPD expected-L0 and static mask theorems, the original source-trained SMC experimental verdicts, all previous CI receipts, and sealed ARG-03-N outcomes remain unchanged.
