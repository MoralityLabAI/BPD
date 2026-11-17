# BPD-OUT06: Bayesian abstention with a neutral causal-effect reward

**Mathematical companion, 18 November 2026.** Source experiment: [SMC-27 PR #234](https://github.com/MoralityLabAI/Compactification-Theory/pull/234), preregistered before its new data. The original SMC-27 four-stage run #44136434145 passed all source and native checks and its independent audit reconstructed 4,260 pre-outcome admitted interventions on 256 unique Sudoku puzzles. Its scientific conclusion was **FAIL (0/4 teachers passing both families)**. Selecting a scalar intervention effect threshold from the old calibration data produced small point-estimate utility gains against the original fixed threshold, with no simultaneous adjusted positive advantage over always abstaining.

This document derives a more direct decision rule for the original SMC-26/27 reward, and provides exact rational tests. It **does not claim the new Bayes policy has been evaluated on held-out trained-teacher outcomes**. SMC-27's original source/test results and earlier scientific verdicts remain fixed.

## A three-class posterior is the actual decision statistic

At a frozen physical hidden state, let the true finite effect of a selective SAE feature erasure compared with a matched physical sham have class T in {-1,0,+1}. The neutral band is the original +/-0.15 correct-digit-margin-logit interval. The action A chooses kill (-1), chooses sham (+1), or abstains (0).

The original offline reward for a non-neutral true effect assigns +1 to a matching nonzero action, -1 to an opposite nonzero action and zero to abstention. For a true neutral effect, abstention earns +1 and either action -1. Any actual nonzero action additionally pays fixed cost c=0.10.

Let the predictor's conditional class probabilities be p_+, p_-, p_0 summing to 1. The *exact expected utilities* are

\[
\mathbb E[U(+1)] = 2p_+ -1-c,\qquad
\mathbb E[U(-1)] = 2p_- -1-c,\qquad
\mathbb E[U(0)] = p_0.
\]

This follows by adding rewards by truth class. The optimal decision is the action with maximal expected utility, choosing abstention on exact ties.

When the positive class has more probability mass than the negative class, a positive intervention decision is preferable to abstention precisely when

\[
p_+ > \frac{1+c+p_0}{2}.
\]

The corresponding negative-action condition replaces p_+ with p_-. As neutral mass or intervention cost rises, abstention becomes optimal for more states. The *signed numerical magnitude of the posterior median of causal effects* is a different statistic. Its threshold cannot, in general, recover the full neutral-mass/cost decision rule.

## Exact source-independent counterexample

Consider nine candidate effects

  (+1, -1, 0, 0, 0, 0, 0, 0, 0)

and a posterior putting weight 3/5 on the +1 effect, 1/10 on -1 and 3/10 on the first zero candidate. Every other candidate has zero weight.

The probability-weighted median effect is +1, so the original threshold-0.15 sign action chooses sham. The **expected utility of this action** is

\[
2(3/5)-1-1/10=1/10.
\]

The neutral-probability-mass reward for abstention is 3/10. Hence the cost-aware Bayes policy abstains, improving expected utility by **1/5**, although the positive sign has majority probability mass. This uses exact Python Fraction arithmetic. The example is a conditional probability construction; it does not assert that either B501 or B811 has that posterior on an observed puzzle.

## Robustness to posterior error and finite-tangent mismatch

Suppose actual finite-effect sign classes t_i for each hypothetical correct digit differ from the predicted local native-tangent classes s_i. Let p be the true conditional digit distribution and q the predicted class distribution. Define

\[
\delta=\mathrm{TV}(p,q)=\tfrac12\sum_i|p_i-q_i|,\qquad
\eta=\Pr_{i\sim p}[t_i\ne s_i].
\]

For the fixed SMC-26/27 ternary reward, every action's utility ranges by at most 2 across target classes. The error in expected utility for one action is bounded by

\[
|\mathbb E_{p,t}U(a,T)-\mathbb E_{q,s}U(a,S)|
\le 2(\delta+\eta).
\]

The regret of the Bayes decision optimal for q and proxy tangent labels s, measured against the decision optimal for true p and finite classes t, therefore obeys

\[
0\le \mathrm{Regret}\le 4(\delta+\eta).
\]

The bound is conservative but applies even when the nine hypothetical tangent effects produce a wrong neutral/nonneutral classification because the physical intervention is finite. No conditional true-digit posterior p, finite class-mismatch rate eta, or trained teacher-wide nonlinear remainder was identified by SMC-27. Empirical uses of this inequality require calibration and independent source splits.

## Source-independent code and tests

The [exact rational implementation](smc27_bayes_abstention.py) computes the three class posterior masses, optimal cost-sensitive policy, original ternary utilities, and the TV/finite-mismatch regret certificate. [Eight CPU unit tests](tests/test_smc27_bayes_abstention.py) check original reward semantics, abstention ties, rising-cost thresholds, weighted-median failure, posterior mismatch, finite-class mismatch and invalid inputs. Run from the BPD repository root:

    python -m pytest -q mechinterp/dark_fiber_gyms/tests/test_smc27_bayes_abstention.py

This is independent mathematical reasoning, not a newly trained SAE, a model-level reproduction or a Lean kernel proof. It complements BPD-OUT04's labelwise Bayes squared/absolute error floors and BPD-OUT05's source-locked sign-versus-MAE distinction.

## Next prospective model experiment

SMC-28 should freeze a **class-probability-mass Bayes action rule** based on all nine candidate local native effects, with original action cost 0.10 and original neutral band +/-0.15. Use exactly the same native three-block physical intervention and 33-candidate sham controls as SMC-27, and freeze its class posterior temperature on *old training-only calibration*. Compare the expected-utility-maximizing action against always abstain, the old-calibration-selected scalar threshold from SMC-27, and uniform-class Bayes actions under identical model-forward budget. Register source-disjoint new Sudoku puzzle seeds and sample/cluster gates before any test outcome, retain all four original source-locked teachers, and audit both the conditional local class approximation and actual finite native outcomes.

The original registered SMC-27 failure against abstention cannot be converted into success through a newly defined utility method tested on the same 256 puzzles. The next experiment must use new, previously unobserved source cases and preserve the older negative verdicts.
