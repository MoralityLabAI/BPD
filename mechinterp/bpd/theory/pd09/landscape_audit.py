"""PD09 posthoc exact support weights and deterministic Gaussian quadrature.

No model training, new seed evaluations, or access to held-out checkpoints.
The calculation evaluates the *registered latent source law* on a restricted
2D orthogonal rotation submanifold. It studies the finite-bandwidth
smoothed hyperplane-mass objective, NOT its full 4D loss landscape.

This analysis was devised after the first PD09 scientific FAIL; it never
changes the original preregistered criteria or model results.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

# Weight numerators with common denominator 36, from PD06 support law.
WEIGHTED_SUPPORTS = (
    ((0,), 3), ((1,), 3), ((2,), 3), ((3,), 3),
    ((0,1), 3), ((1,2), 3), ((2,3), 3), ((0,3), 3),
    ((0,1,2), 4), ((0,1,3), 4), ((1,2,3), 4),
)


def exact_support_weights(pair: tuple[int, int]):
    i, j = pair
    if i == j or i not in range(4) or j not in range(4):
        raise ValueError("two distinct latent component indices required")
    p00 = sum(w for s,w in WEIGHTED_SUPPORTS if i not in s and j not in s)
    p10 = sum(w for s,w in WEIGHTED_SUPPORTS if i in s and j not in s)
    p01 = sum(w for s,w in WEIGHTED_SUPPORTS if j in s and i not in s)
    p11 = sum(w for s,w in WEIGHTED_SUPPORTS if i in s and j in s)
    assert p00+p10+p01+p11 == 36
    zi = p00+p01
    zj = p00+p10
    # A non-coordinate read involving i and j is exactly zero only when
    # neither active coordinate belongs to the support.
    z_diagonal = p00
    return {
        "p00":p00,"p10":p10,"p01":p01,"p11":p11,
        "z_i":zi,"z_j":zj,"z_diagonal":z_diagonal,
        "sharp_coordinate_advantage_numerator":zi+zj-2*z_diagonal,
        "denominator":36,
    }


def kernel_pair_mass(pair: tuple[int,int],theta_deg:float,sigma:float,n:int=128):
    """Expected sum of two Gaussian near-zero read kernels on pair rotation.

    The remaining two orthogonal reads are unchanged. Independent signed
    nonzero magnitudes U,V are uniformly distributed on [.75,1.25].
    """
    if sigma <= 0:
        raise ValueError("positive smoothing bandwidth required")
    m=exact_support_weights(pair)
    nodes,legendre=np.polynomial.legendre.leggauss(n)
    u=1.0+0.25*nodes
    w=0.5*legendre
    U=u[:,None]
    V=u[None,:]
    W=w[:,None]*w[None,:]
    theta=math.radians(theta_deg)
    c=math.cos(theta)
    s=math.sin(theta)
    singleton=float(np.sum(
        w*(np.exp(-((c*u)**2)/(2*sigma**2)) +
           np.exp(-((s*u)**2)/(2*sigma**2)))
    ))
    paired=0.0
    # Each read receives equally likely sign combinations.
    for a,b in ((c,s),(-s,c)):
        paired+=float(0.5*np.sum(W*(
            np.exp(-((a*U+b*V)**2)/(2*sigma**2)) +
            np.exp(-((a*U-b*V)**2)/(2*sigma**2))
        )))
    return (2*m["p00"] +
            (m["p10"]+m["p01"])*singleton +
            m["p11"]*paired)/36.0


def difference(pair,sigma):
    """Positive means 45-degree pair has *larger* near-zero kernel mass."""
    return kernel_pair_mass(pair,45,sigma)-kernel_pair_mass(pair,0,sigma)


def crossing_bandwidth(pair=(0,1),lo=0.3,hi=0.6,iterations=30):
    assert difference(pair,lo)<0 and difference(pair,hi)>0
    for _ in range(iterations):
        mid=(lo+hi)/2
        if difference(pair,mid)>0:
            hi=mid
        else:
            lo=mid
    return (lo+hi)/2


def audit():
    assert sum(w for _,w in WEIGHTED_SUPPORTS)==36
    cases=[]
    for pair in ((0,1),(1,2)):
        masses=exact_support_weights(pair)
        sigmas={}
        for sigma in (0.6,0.3,0.15):
            v0=kernel_pair_mass(pair,0,sigma)
            v45=kernel_pair_mass(pair,45,sigma)
            v44=kernel_pair_mass(pair,44,sigma)
            v46=kernel_pair_mass(pair,46,sigma)
            sigmas[str(sigma)]={
                "coordinate_pair_mass":v0,
                "diagonal_45_mass":v45,
                "diagonal_minus_coordinate":v45-v0,
                "local_peak_45_over_44":v45-v44,
                "local_peak_45_over_46":v45-v46,
            }
        cases.append({
            "pair":list(pair),
            "exact_support_weight":masses,
            "crossover_sigma":crossing_bandwidth(pair),
            "smooth":sigmas,
        })
    return {
        "study_id":"BPD-PD-09",
        "scope":"posthoc restricted 2D rotation landscape, Gaussian quadrature only",
        "source_weight_denominator":36,
        "quadrature_n":128,
        "cases":cases,
    }


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=None)
    args=ap.parse_args()
    s=json.dumps(audit(),indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(s,encoding="utf-8")
    else:
        print(s)


if __name__=="__main__":
    main()
