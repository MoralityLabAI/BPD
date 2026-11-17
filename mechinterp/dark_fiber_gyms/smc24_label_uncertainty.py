"""BPD-OUT04: label-conditioned native task-gradient identifiability.

For a fixed clean logit vector and physical logit Jacobian, each possible
correct label defines its own clean-rival margin gradient. An observer of
the physical hidden state alone must infer the class-conditioned functional
to predict the signed outcome. Conditional label mixing has irreducible
Bayes loss unless classwise causal effects agree.

This module contains elementary finite-dimensional numerical identities.
It does not read pretrained models or assert that a Sudoku hidden state
fails to encode the actual correct answer.
"""
from __future__ import annotations

import math
from fractions import Fraction
import numpy as np


def class_margin_directions(clean_logits, logit_jacobian, kill, sham):
    """All hypothetical correct-label tangent effects under fixed clean rivals.

    For label y, competitor c_y=argmax_{k != y} clean_logit[k] and
    d_y=(J[y]-J[c_y]) dot (kill-sham). This uses all candidate labels;
    the actual correct digit is never supplied to the function.
    """
    logits=np.asarray(clean_logits,dtype=np.float64)
    jac=np.asarray(logit_jacobian,dtype=np.float64)
    a=np.asarray(kill,dtype=np.float64)
    b=np.asarray(sham,dtype=np.float64)
    if (logits.ndim!=1 or len(logits)<2 or jac.shape!=(len(logits),len(a))
        or a.ndim!=1 or b.shape!=a.shape):
        raise ValueError("required n logits, n-by-d native Jacobian and matched d-vectors")
    if not all(np.all(np.isfinite(x)) for x in (logits,jac,a,b)):
        raise ValueError("nonfinite native readout inputs")
    rivals=[]
    candidates=[]
    physical=a-b
    for y in range(len(logits)):
        alternate=logits.copy()
        alternate[y]=-float("inf")
        c=int(np.argmax(alternate))
        rivals.append(c)
        candidates.append(float((jac[y]-jac[c])@physical))
    return np.asarray(candidates),np.asarray(rivals,dtype=np.int64)


def _probability(p, n):
    p=np.asarray(p,dtype=np.float64)
    if p.shape!=(n,) or not np.all(np.isfinite(p)) or np.any(p<0):
        raise ValueError("invalid label distribution")
    if abs(float(p.sum())-1.0)>1e-10:
        raise ValueError("label distribution must sum to one")
    return p


def squared_risk_certificate(class_effects, prior, proposed):
    """Bayes risk identity: E_p[(D-a)^2] = Var_p(D)+(a-E_p D)^2."""
    d=np.asarray(class_effects,dtype=np.float64)
    if d.ndim!=1 or len(d)<2 or not np.all(np.isfinite(d)):
        raise ValueError("invalid classwise observable effects")
    p=_probability(prior,len(d))
    a=float(proposed)
    if not math.isfinite(a):
        raise ValueError("invalid proposed prediction")
    mu=float(p@d)
    variance=float(p@((d-mu)**2))
    risk=float(p@((d-a)**2))
    excess=(a-mu)**2
    if abs(risk-variance-excess)>1e-9*max(1.,risk):
        raise ArithmeticError("conditional Bayes risk decomposition failed")
    return {"bayes_mean_prediction":mu,
            "irreducible_squared_risk":variance,
            "submitted_squared_risk":risk,
            "prediction_excess_squared_risk":excess,
            "labels":len(d)}


def distribution_mismatch_certificate(class_effects, true_prior, predicted_prior):
    """TV bound: |E_p D-E_q D| <= (max D-min D) * TV(p,q)."""
    d=np.asarray(class_effects,dtype=np.float64)
    if d.ndim!=1 or len(d)<2 or not np.all(np.isfinite(d)):
        raise ValueError("invalid classwise effects")
    p=_probability(true_prior,len(d))
    q=_probability(predicted_prior,len(d))
    mean_p=float(p@d)
    mean_q=float(q@d)
    tv=float(0.5*np.abs(p-q).sum())
    spread=float(d.max()-d.min())
    upper=tv*spread
    actual=abs(mean_p-mean_q)
    if actual>upper+1e-10:
        raise ArithmeticError("probability mismatch exceeded variation bound")
    return {"true_expected_effect":mean_p,
            "predicted_expected_effect":mean_q,
            "true_vs_predicted_probability_tv":tv,
            "effect_range":spread,
            "absolute_prediction_bias":actual,
            "certified_probability_bias_upper":upper}


def optimal_absolute_prediction(class_effects, prior):
    """Bayes-optimal median for absolute loss, with deterministic left tie."""
    d=np.asarray(class_effects,dtype=np.float64)
    if d.ndim!=1 or len(d)<2 or not np.all(np.isfinite(d)):
        raise ValueError("invalid effect values")
    p=_probability(prior,len(d))
    order=np.argsort(d,kind="stable")
    cumul=np.cumsum(p[order])
    i=int(np.searchsorted(cumul,.5,side="left"))
    v=float(d[order[i]])
    mae=float(p@np.abs(d-v))
    return {"optimal_median_prediction":v,
            "irreducible_mean_absolute_error":mae,
            "label_support":int(np.count_nonzero(p))}


def exact_two_label_absolute_floor(weight_first, first, second):
    """Closed-form conditional median absolute error for two outcomes.

    If D=first with prob p and D=second with prob 1-p,
    min_a E|D-a| = min(p,1-p)*|first-second|.
    The Fraction interface makes the two-point obstruction exact.
    """
    p=Fraction(weight_first)
    x=Fraction(first)
    y=Fraction(second)
    if p<0 or p>1:
        raise ValueError("label weight must belong to [0,1]")
    return min(p,1-p)*abs(x-y)
