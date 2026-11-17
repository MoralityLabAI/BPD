"""BPD-OUT06: exact Bayes abstention for class-conditioned causal interventions.

Original SMC26/27 utility: real decision + or - yields +1 on matching
nonneutral true intervention effect, -1 on wrong/neutral, 0 for abstention
on a nonneutral effect. Neutral true effects reward abstention +1.
Any chosen nonzero action incurs fixed cost c.

For a posterior p over candidate classes, the optimal policy depends on
POSITIVE, NEGATIVE and NEUTRAL class probability masses, rather than the
weighted median of predicted effect magnitudes. All calculations use exact
Fractions, with no torch, learned teacher or heldout SMC27 data.
"""
from __future__ import annotations

from fractions import Fraction as Q


def _fraction(value):
    r=Q(value)
    if r.denominator==0:
        raise ValueError("invalid rational")
    return r


def utility(action,truth,cost=Q(1,10)):
    """Exact SMC26/27 ternary score for action/truth in {-1,0,+1}."""
    if action not in (-1,0,1) or truth not in (-1,0,1):
        raise ValueError("ternary source classes required")
    c=_fraction(cost)
    if c<0:
        raise ValueError("nonnegative action cost required")
    if truth==0:
        reward=Q(1) if action==0 else Q(-1)
    else:
        reward=Q(0) if action==0 else Q(1) if action==truth else Q(-1)
    return reward-c*int(action!=0)


def effect_truth(effect,neutral_band=Q(3,20)):
    """Original true sign class uses fixed ±0.15, including exact abstention ties."""
    x=_fraction(effect)
    b=_fraction(neutral_band)
    if b<0:raise ValueError("neutral band nonnegative")
    return 1 if x>b else -1 if x< -b else 0


def _mass(weights):
    p=tuple(_fraction(w) for w in weights)
    if len(p)<2 or any(w<0 for w in p) or sum(p)!=1:
        raise ValueError("valid normalized probability distribution with two or more labels required")
    return p


def posterior_classes(effects,weights,neutral_band=Q(3,20)):
    d=tuple(_fraction(v) for v in effects)
    p=_mass(weights)
    if len(d)!=len(p):
        raise ValueError("class-conditional candidate effect/weight dimensions must agree")
    classes=tuple(effect_truth(x,neutral_band) for x in d)
    return classes,p


def expected_actions_from_mass(p_plus,p_minus,p_neutral,cost=Q(1,10)):
    """Exact Bayes expected utility formula, deterministic abstain ties.

    E U(+1) = 2 p_plus - 1 - c
    E U(-1) = 2 p_minus - 1 - c
    E U(0) = p_neutral.

    For positive mass dominance, taking the positive action is optimal
    only if p_plus > (1+c+p_neutral)/2 and p_plus>p_minus.
    When a maximum ties the abstention score, choose abstention.
    """
    p_plus,p_minus,p_neutral=map(_fraction,(p_plus,p_minus,p_neutral))
    c=_fraction(cost)
    if min(p_plus,p_minus,p_neutral,c)<0 or p_plus+p_minus+p_neutral!=1:
        raise ValueError("normalized ternary probability masses and nonnegative cost required")
    rewards={
        1:Q(2)*p_plus-Q(1)-c,
        -1:Q(2)*p_minus-Q(1)-c,
        0:p_neutral,
    }
    optimum=max(rewards.values())
    selected=(0 if rewards[0]==optimum else
              1 if rewards[1]==optimum else -1)
    return {
        "p_plus":p_plus,"p_minus":p_minus,"p_neutral":p_neutral,
        "cost":c,"expected_utility":rewards,
        "optimal_action":selected,
        "optimal_expected_utility":optimum,
        "positive_mass_threshold":(Q(1)+c+p_neutral)/2,
        "negative_mass_threshold":(Q(1)+c+p_neutral)/2,
    }


def bayes_candidate_action(effects,weights,
                           neutral_band=Q(3,20),cost=Q(1,10)):
    classes,p=posterior_classes(effects,weights,neutral_band)
    masses={a:sum(w for t,w in zip(classes,p) if t==a)
            for a in (-1,0,1)}
    r=expected_actions_from_mass(masses[1],masses[-1],masses[0],cost)
    r["class_effects"]=tuple(_fraction(x) for x in effects)
    r["class_probability"]=p
    r["class_signs"]=classes
    r["neutral_band"]=_fraction(neutral_band)
    return r


def weighted_median_effect(effects,weights):
    d=tuple(_fraction(x) for x in effects)
    p=_mass(weights)
    if len(d)!=len(p):raise ValueError("weights and effects differ in number")
    pairs=sorted((effect,i,p[i]) for i,effect in enumerate(d))
    total=Q(0)
    for effect,i,w in pairs:
        total+=w
        if total>=Q(1,2):
            return effect
    raise ArithmeticError("posterior total probability missing")


def regret_certificate(effects,actual_classes,true_prior,
                       predicted_prior,neutral_band=Q(3,20),
                       cost=Q(1,10)):
    """Conditional regret certificate for posterior error plus tangent mismatch.

    True finite classes t_i and proxy local tangent signs s_i can differ.
    Let eta=P_p[t_i != s_i], delta=TV(p,q). For each action, expected
    rewards between true p/t and proxy q/s differ by <=2(delta+eta)
    because ternary utilities vary by at most 2 between truth classes.
    The regret of choosing argmax under q/s against oracle p/t is
    consequently bounded by 4(delta+eta).
    """
    d=tuple(_fraction(x) for x in effects)
    p=_mass(true_prior)
    q=_mass(predicted_prior)
    actual=tuple(int(t) for t in actual_classes)
    if not len(d)==len(p)==len(q)==len(actual):
        raise ValueError("actual classes, labels and two priors must align")
    if any(t not in (-1,0,1) for t in actual):
        raise ValueError("actual finite native effect class must be ternary")
    proxy=tuple(effect_truth(x,neutral_band) for x in d)
    choice=bayes_candidate_action(d,q,neutral_band,cost)["optimal_action"]
    real={a:sum(w*utility(a,t,cost) for w,t in zip(p,actual))
          for a in (-1,0,1)}
    best=max(real.values())
    chosen=real[choice]
    delta=sum(abs(x-y) for x,y in zip(p,q))/2
    eta=sum(w for w,t,s in zip(p,actual,proxy) if t!=s)
    bound=Q(4)*(delta+eta)
    regret=best-chosen
    if not Q(0)<=regret<=bound:
        raise ArithmeticError("posterior/tangent regret bound failed")
    return {
        "chosen_proxy_action":choice,
        "true_best_expected_utility":best,
        "chosen_true_expected_utility":chosen,
        "true_conditional_regret":regret,
        "posterior_total_variation":delta,
        "proxy_finite_class_mismatch":eta,
        "certified_regret_upper":bound,
    }


def exact_median_abstention_counterexample():
    """Misplaced median sign takes a lower-utility action than Bayes abstain."""
    effects=(Q(1),Q(-1),Q(0),Q(0),Q(0),Q(0),Q(0),Q(0),Q(0))
    probs=(Q(3,5),Q(1,10),Q(3,10),Q(0),Q(0),Q(0),Q(0),Q(0),Q(0))
    med=weighted_median_effect(effects,probs)
    r=bayes_candidate_action(effects,probs)
    act=effect_truth(med)
    median_utility=sum(w*utility(act,t) for w,t in
                       zip(probs,r["class_signs"]))
    assert med==1 and act==1 and median_utility==Q(1,10)
    assert r["optimal_action"]==0 and r["optimal_expected_utility"]==Q(3,10)
    return {
        "nine_candidate_median":med,"median_action":act,
        "median_expected_utility":median_utility,
        "Bayes_action":r["optimal_action"],
        "Bayes_expected_utility":r["optimal_expected_utility"],
        "improvement":r["optimal_expected_utility"]-median_utility,
    }
