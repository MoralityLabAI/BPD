"""SCCG-10 conditional finite-response cache admission in exact rationals.

This reference consumes externally established error bounds; it does not
estimate or certify drift, derivative bounds, or transformer arithmetic.
"""
from fractions import Fraction as F
from math import isqrt


def q(x):
    if isinstance(x, (float, bool)):
        raise ValueError("provide exact rational numbers")
    return F(x)


def dot(a, b):
    return sum((x*y for x,y in zip(a,b)),F(0))


def mv(A, x):
    return [dot(row,x) for row in A]


def sqrt_upper(x):
    if x<0:
        raise ValueError("negative squared norm")
    scale=2**32
    z=x*scale*scale
    n=isqrt(z.numerator//z.denominator)
    return F(n if n*n==z else n+1,scale)


def require_pd(G):
    if any(G[i][j]!=G[j][i] for i in range(len(G)) for j in range(len(G))):
        raise ValueError("metric must be symmetric")
    M=[list(row) for row in G]
    while M:
        if M[0][0]<=0:
            raise ValueError("metric must be positive definite")
        M=[[M[i][j]-M[i][0]*M[0][j]/M[0][0]
            for j in range(1,len(M))] for i in range(1,len(M))]


def solve(A,b):
    m=[list(row)+[v] for row,v in zip(A,b)]
    n=len(m)
    for j in range(n):
        p=next(i for i in range(j,n) if m[i][j])
        m[j],m[p]=m[p],m[j]
        z=m[j][j]
        m[j]=[v/z for v in m[j]]
        for i in range(n):
            if i!=j:
                z=m[i][j]
                m[i]=[v-z*w for v,w in zip(m[i],m[j])]
    return [r[-1] for r in m]


def assess(G,a,C,u,*,dose,collateral,alpha,gamma,target_remainder,
           protected_remainder,min_gain,dual_multipliers=(0,1,4,16,64)):
    """Conditional lower/upper bounds on true finite control.

    G physical metric, a/C cached target/protected derivatives. alpha/gamma
    bound current minus cached derivatives in the G-induced input norm.
    Remainders bound the live finite nonlinear response uniformly over the
    dose ball. Positive certificate covers only the named span and context.
    Supplied bounds must already include input/measurement errors.
    """
    G,C=[[q(v) for v in r] for r in G],[[q(v) for v in r] for r in C]
    a,u=[q(v) for v in a],[q(v) for v in u]
    e,eta,alpha,gamma,rf,rp,gain=map(q,(dose,collateral,alpha,gamma,
                                      target_remainder,protected_remainder,min_gain))
    n=len(a)
    if not n or len(u)!=n or len(G)!=n or any(len(r)!=n for r in G+C):
        raise ValueError("incompatible coordinate dimensions")
    if e<=0 or gain<=0 or min(eta,alpha,gamma,rf,rp)<0:
        raise ValueError("invalid dose, error bounds or gain floor")
    require_pd(G)
    dose2=dot(u,mv(G,u))
    if dose2>e*e:
        raise ValueError("candidate exceeds physical dose")
    target_error=e*alpha+rf
    protected_error=e*gamma+rp
    protected2=dot(mv(C,u),mv(C,u))
    room=eta-protected_error
    safe=room>=0 and protected2<=room*room
    lower=dot(a,u)-target_error
    relaxed=eta+protected_error
    best=None
    for v in dual_multipliers:
        lam=q(v)
        if lam<0:
            raise ValueError("negative dual multiplier")
        M=[[G[i][j]+lam*sum((r[i]*r[j] for r in C),F(0))
            for j in range(n)] for i in range(n)]
        bound2=(e*e+lam*relaxed*relaxed)*dot(a,solve(M,a))
        if best is None or bound2<best[0]:
            best=(bound2,lam)
    if best is None:
        raise ValueError("at least one multiplier required")
    upper=sqrt_upper(best[0])+target_error
    if upper<gain:
        verdict="NO_HEADROOM_IN_DECLARED_SPAN"
    elif safe and lower>=gain:
        verdict="CONDITIONAL_REUSE_CANDIDATE"
    else:
        verdict="REFRESH_OR_ORDINARY"
    return dict(verdict=verdict,safe=safe,physical_dose_squared=str(dose2),
                target_lower=str(lower),target_upper=str(upper),
                collateral_upper=str(sqrt_upper(protected2)+protected_error),
                target_error=str(target_error),protected_error=str(protected_error),
                multiplier=str(best[1]),
                regret_upper=str(max(F(0),upper-lower)) if safe else None,
                scope="CONDITIONAL_FINITE_RESPONSE_IN_DECLARED_SPAN")
