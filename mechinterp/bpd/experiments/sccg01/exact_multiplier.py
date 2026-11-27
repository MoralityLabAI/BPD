"""Complete rational affine-pencil PSD decision via exact univariate sign cells."""
from itertools import combinations
import sympy as s

T = s.Symbol('lambda')


def rational(x):
    if isinstance(x, (float, bool, s.Float)):
        raise ValueError('exact rational input required')
    v = s.sympify(x)
    if not isinstance(v, s.Rational):
        raise ValueError('expected rational')
    return v


def pencil(A, B):
    A, B = (s.Matrix([[rational(x) for x in row] for row in M])
            for M in (A, B))
    if not A.rows or A.rows != A.cols or B.shape != A.shape or A != A.T or B != B.T:
        raise ValueError('symmetric nonempty aligned pencils required')
    return A + T*B


def sign_cells(expressions, lo, hi):
    """Decide all polynomials >= 0 on a closed rational interval.

    Return first witness and count of tested cells. The combined square-free
    polynomial isolates every possible sign change, including endpoints.
    """
    lo, hi = rational(lo), rational(hi)
    if lo > hi:
        raise ValueError('reversed interval')
    ps = []
    for expr in expressions:
        p = s.Poly(expr, T, domain=s.QQ)
        if p.is_zero:
            continue
        if p.degree() == 0:
            if p.LC() < 0:
                return {'feasible': False, 'witness': None, 'tested': 0, 'degree': 0}
            continue
        ps.append(p)
    if lo == hi:
        ok = all(p.eval(lo) >= 0 for p in ps)
        return {'feasible': ok, 'witness': {'kind': 'rational', 'lambda': str(lo)} if ok else None,
                'tested': 1, 'degree': 0}
    ps += [s.Poly(T-lo,T,domain=s.QQ), s.Poly(hi-T,T,domain=s.QQ)]
    P = s.Poly(1,T,domain=s.QQ)
    for p in ps:
        P = P.lcm(p.sqf_part())
    P = P.clear_denoms()[1].primitive()[1]
    # Use exact isolating intervals for all real roots, including rational
    # singleton endpoints. Refine until there are strict rational gaps.
    bits = 8
    while True:
        intervals = [pair for pair,_ in P.intervals(eps=s.Rational(1,2**bits))]
        separated = all(intervals[i][1] < intervals[i+1][0]
                        for i in range(len(intervals)-1))
        clean = all(a == b or (P.eval(a) != 0 and P.eval(b) != 0)
                    for a,b in intervals)
        if separated and clean:
            break
        bits *= 2
    tested = 0
    def result(witness):
        return {'feasible': witness is not None, 'witness': witness,
                'tested': tested, 'degree': P.degree()}
    # Any sign-feasible open cell inside [lo,hi] contains this sample.
    for left,right in zip(intervals,intervals[1:]):
        mid=(left[1]+right[0])/2
        if lo < mid < hi:
            tested += 1
            if all(p.eval(mid) >= 0 for p in ps):
                return result({'kind': 'rational', 'lambda': str(mid)})
    gs = [P.gcd(p) for p in ps]
    for a,b in intervals:
        if b < lo or a > hi:
            continue
        tested += 1
        if a == b:
            if all(p.eval(a) >= 0 for p in ps):
                return result({'kind': 'rational', 'lambda': str(a)})
        else:
            mid=(a+b)/2
            if all(g.count_roots(a,b) or p.eval(mid)>0 for g,p in zip(gs,ps)):
                return result({'kind': 'algebraic', 'polynomial': [str(c) for c in P.all_coeffs()],
                               'interval': [str(a),str(b)]})
    return result(None)


def decide(A, B, lo, hi, *, principal=False):
    """PSD feasibility; principal=True is the exponential audit comparator."""
    M = pencil(A,B)
    if principal:
        polys = [M.extract(ix,ix).det() for k in range(1,M.rows+1)
                 for ix in combinations(range(M.rows),k)]
    else:
        # Eigenvalues are real. det(zI+M) has nonnegative coefficients iff
        # it has no positive real root, iff M has no negative eigenvalue.
        polys = (-M).charpoly().all_coeffs()[1:]
    return sign_cells(polys,lo,hi)


def replay_witness(A, B, lo, hi, witness):
    """Independently check a positive witness using all principal minors."""
    M = pencil(A,B)
    polys = [s.Poly(M.extract(ix,ix).det(),T,domain=s.QQ)
             for k in range(1,M.rows+1) for ix in combinations(range(M.rows),k)]
    lo,hi = rational(lo),rational(hi)
    if witness['kind'] == 'rational':
        v = rational(witness['lambda'])
        return lo <= v <= hi and all(p.eval(v)>=0 for p in polys)
    if witness['kind'] != 'algebraic':
        return False
    P=s.Poly.from_list([rational(c) for c in witness['polynomial']],T,domain=s.QQ)
    a,b=map(rational,witness['interval'])
    if not lo <= a < b <= hi or P.eval(a)==0 or P.eval(b)==0 or P.count_roots(a,b)!=1:
        return False
    # A principal minor can have roots not in the characteristic product.
    # Refine the isolator until each nonvanishing minor has constant sign.
    gs=[P.gcd(p) for p in polys]
    zero=[p.is_zero or g.count_roots(a,b)>0 for p,g in zip(polys,gs)]
    while any(not z and p.count_roots(a,b)>0 for p,z in zip(polys,zero)):
        a,b=P.refine_root(a,b,eps=(b-a)/2,check_sqf=True)
        if a==b:
            return all(p.eval(a)>=0 for p in polys)
    return all(z or p.eval((a+b)/2)>0 for p,z in zip(polys,zero))


def certificate_pencil(cert):
    """Use the original SCCG-07 rational certificate and dose cap."""
    A=cert.block(0)
    B=[[cert.G[i][j] if i<cert.n and j<cert.n else
        -cert.e**2 if i==j==cert.n else 0
        for j in range(cert.n+1)] for i in range(cert.n+1)]
    return A,B,0,cert.cap
