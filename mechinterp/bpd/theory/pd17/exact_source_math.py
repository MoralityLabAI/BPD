"""Exact finite-support source measure and hyperplane-mass certificates.

All source probabilities are Fractions from the unchanged PD12 law. The
support theorem is independent of finite-sigma neural optimization.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction
from functools import lru_cache

from mechinterp.bpd.theory.pd12.experiment import support_banks

D = 16
SINGLETON = Fraction(1, 4*D)


def bitmask(support) -> int:
    return sum(1<<int(j) for j in support)


@lru_cache(None)
def source_support_distribution():
    """Exact rational P(S) with uniform K=1..4 and clustered k>1."""
    banks=support_banks(D)
    probabilities={}
    for k in range(1,5):
        components = ((Fraction(1),banks["train"][k]),) if k==1 else (
            (Fraction(17,20),banks["cluster"][k]),
            (Fraction(3,20),banks["train"][k]),
        )
        for mass,bank in components:
            weight=mass*Fraction(1,4)/len(bank)
            for s in bank:
                m=bitmask(s)
                probabilities[m]=probabilities.get(m,Fraction(0))+weight
    assert sum(probabilities.values(),Fraction(0)) == 1
    return probabilities


def exact_hyperplane_zero_mass(read_support_bits:int):
    """For a generic continuous latent, mass at exactly zero projection."""
    if not 0<=read_support_bits<(1<<D):
        raise ValueError("16D read-support bitmask out of range")
    return sum((p for s,p in source_support_distribution().items()
                if not (s&read_support_bits)),Fraction(0))


def pair_dominance_certificate():
    """All 120 mixed-pair supports lose >=1/64 against either pure axis."""
    pair_losses=[]
    for i,j in itertools.combinations(range(D),2):
        z_ij=exact_hyperplane_zero_mass((1<<i)|(1<<j))
        di=exact_hyperplane_zero_mass(1<<i)-z_ij
        dj=exact_hyperplane_zero_mass(1<<j)-z_ij
        assert di>=SINGLETON and dj>=SINGLETON
        pair_losses.append((i,j,di,dj))
    return {
        "dimensions":D,"pairs":len(pair_losses),
        "singleton_probability":str(SINGLETON),
        "minimum_pair_loss":str(min(min(x[2],x[3]) for x in pair_losses)),
        "minimum_normalized_basis_mixed_column_gap":str(SINGLETON/D)
    }


def original_norm_tilt_profile(lam:float=1.):
    """Analytic uniform amplitude integral, no fitted latent labels."""
    if lam<0:
        raise ValueError("lambda must be nonnegative")
    if lam==0:
        phi=1.
        phi2=1.
    else:
        def phi_at(c):
            lo,hi=.75,1.25
            return (math.sqrt(math.pi)/(2*(hi-lo)*math.sqrt(c))*
                    (math.erf(hi*math.sqrt(c))-math.erf(lo*math.sqrt(c))))
        phi=phi_at(lam)
        phi2=phi_at(2*lam)
    base=[phi**k for k in range(1,5)]
    share=[v/sum(base) for v in base]
    m=sum(base)/4
    m2=sum(phi2**k for k in range(1,5))/4
    return {"lambda":lam,"phi":phi,
            "effective_cardinality_probabilities":share,
            "expected_ESS_fraction":m*m/m2}
