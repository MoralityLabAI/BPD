"""Exact finite-support information audits for BPD-SF01.

These functions evaluate supplied discrete records or fully specified finite
channels. They do not estimate leakage from LLM free text or establish a
trained network mechanism. Callers must supply independent-root data and
uncertainty intervals for empirical claims.
"""

from __future__ import annotations

from collections import defaultdict
from math import isfinite, log2
from typing import Iterable, Mapping


def _columns(fields: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(fields, str):
        return (fields,)
    return tuple(fields)


def conditional_mutual_information(
    records: Iterable[Mapping[str, object]],
    *,
    secret: str,
    signal: str | Iterable[str],
    given: str | Iterable[str] = (),
    weight: str | None = None,
) -> float:
    """Empirical I(secret; signal | given) in *bits* for finite categories.

    `records` must contain independent observational units or an explicitly
    weighted finite distribution. This point estimate alone has no CI. The
    conditioning variables must include every public fact known to the observer.
    """
    yfields = _columns(signal)
    gfields = _columns(given)
    if not yfields:
        raise ValueError("signal must contain at least one field")
    tally: dict[tuple[object, ...], dict[tuple[object, tuple[object, ...]], float]] = defaultdict(lambda: defaultdict(float))
    total = 0.0
    for row in records:
        mass = 1.0 if weight is None else float(row[weight])
        if not isfinite(mass) or mass <= 0:
            raise ValueError("record weight must be positive and finite")
        g = tuple(row[k] for k in gfields)
        s = row[secret]
        y = tuple(row[k] for k in yfields)
        tally[g][(s, y)] += mass
        total += mass
    if total == 0:
        raise ValueError("records must be nonempty")

    information = 0.0
    for entries in tally.values():
        n_g = sum(entries.values())
        by_s: dict[object, float] = defaultdict(float)
        by_y: dict[tuple[object, ...], float] = defaultdict(float)
        for (s, y), mass in entries.items():
            by_s[s] += mass
            by_y[y] += mass
        for (s, y), mass in entries.items():
            information += (mass / total) * log2(mass * n_g / (by_s[s] * by_y[y]))
    return max(0.0, information)  # only clamp floating-point cancellation


def public_channel_ledger(
    records: Iterable[Mapping[str, object]],
    *,
    secret: str,
    action: str,
    talk: str,
    given: str | Iterable[str] = (),
    weight: str | None = None,
) -> dict[str, float]:
    """Chain-rule accounting of secret disclosure through action and chat.

    Total = I(S;action,talk|context) = I(S;action|context)
          + I(S;talk|context,action). Legal poker bets can legitimately
    carry information; additional chat information is audited separately.
    """
    rows = list(records)
    g = _columns(given)
    a = conditional_mutual_information(rows, secret=secret, signal=action, given=g, weight=weight)
    t = conditional_mutual_information(rows, secret=secret, signal=talk, given=g + (action,), weight=weight)
    joint = conditional_mutual_information(rows, secret=secret, signal=(action, talk), given=g, weight=weight)
    if abs(a + t - joint) > 1e-9:
        raise ArithmeticError("finite-information chain rule failed")
    return {"action_bits": a, "additional_talk_bits": t, "total_bits": joint}


def finite_channel_information(
    channel: Mapping[object, Mapping[object, float]],
    prior: Mapping[object, float],
) -> dict[str, float]:
    """I(S;Y) and E_{S,S'} KL(P(Y|S)||P(Y|S')) in bits.

    The latter upper-bounds MI by convexity of KL in its second argument.
    Requires exact/fully specified output distributions, not sampled logits.
    A zero-probability support mismatch makes the bound infinite.
    """
    if set(channel) != set(prior) or not channel:
        raise ValueError("channel and prior keys must agree and be nonempty")
    if any(not isfinite(float(v)) or v <= 0 for v in prior.values()) or abs(sum(prior.values()) - 1) > 1e-9:
        raise ValueError("prior must be positive and normalized")
    outcomes = set().union(*(set(dist) for dist in channel.values()))
    for dist in channel.values():
        if any(not isfinite(float(v)) or v < 0 for v in dist.values()) or abs(sum(dist.values()) - 1) > 1e-9:
            raise ValueError("every channel distribution must be normalized")
    mixture = {y: sum(prior[s] * channel[s].get(y, 0.0) for s in prior) for y in outcomes}
    mi = sum(prior[s] * p * log2(p / mixture[y])
             for s in prior for y in outcomes
             if (p := channel[s].get(y, 0.0)) > 0)
    bound = 0.0
    for s in prior:
        for sp in prior:
            kl = 0.0
            for y in outcomes:
                p = channel[s].get(y, 0.0)
                if p == 0:
                    continue
                q = channel[sp].get(y, 0.0)
                if q == 0:
                    return {"mi_bits": mi, "pairwise_kl_upper_bits": float("inf")}
                kl += p * log2(p / q)
            bound += prior[s] * prior[sp] * kl
    return {"mi_bits": max(0.0, mi), "pairwise_kl_upper_bits": max(0.0, bound)}


def tied_two_pass(secret: float, authorization: float, *, leak: float = 0.0) -> tuple[float, float, float]:
    """Exact rank-two, role-conditioned tied linear map positive control.

    State h=(secret, carry, public_readout). F_A(h) =
    (0, h[0], (A + (1-A)*leak)*h[1]). Start h=(secret,0,0).
    The same matrix is executed twice. `leak` adds an intentionally unsafe
    transition for the unauthorized role. This is an *analytic* control.
    """
    h = (float(secret), 0.0, 0.0)
    for _ in range(2):
        h = (0.0, h[0], (authorization + (1.0 - authorization) * leak) * h[1])
    return h
