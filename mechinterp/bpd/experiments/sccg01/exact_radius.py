"""Exact rational-input finite protected-reader threshold oracle.

All scalar thresholds are in units of rho0 times score gain. Exact
polynomial sign decisions use SymPy rational root isolation. No float
inputs, learned-model claims, or machine-level RMSNorm guarantees.
"""
from itertools import product
import sympy as s

t = s.Symbol("tau")


def rat(x):
    if isinstance(x, (float, bool, s.Float)):
        raise ValueError("use exact rational inputs")
    q = s.sympify(x)
    if not isinstance(q, s.Rational):
        raise ValueError("expected rational")
    return q


def col(xs):
    return s.Matrix([rat(x) for x in xs])


def matrix(rows, d=None):
    rows = [list(row) for row in rows]
    if not rows:
        if d is None:
            raise ValueError("empty matrix needs width")
        return s.zeros(0, d)
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("ragged matrix")
    if not rows[0]:
        return s.zeros(len(rows), 0)
    return s.Matrix([[rat(x) for x in row] for row in rows])


def dot(a, b):
    return (a.T*b)[0, 0]


def norm2(a):
    return dot(a, a)


def projector(U):
    if not U.cols:
        return s.zeros(U.rows)
    return U*(U.T*U).inv()*U.T


def signs_feasible(expressions):
    """Conjunction of polynomial >=0 constraints; exact sign-cell witness.

    Isolate roots of square-free product, check all open cells AND roots.
    Root witnesses identify one real root by polynomial and rational interval.
    None means every cell was checked and rejected.
    """
    ps = []
    for f in expressions:
        p = s.Poly(f, t, domain=s.QQ)
        if p.is_zero:
            continue
        if p.degree() == 0:
            if p.LC() < 0:
                return None
            continue
        if p.degree() > 4:
            raise ValueError("unexpected constraint degree")
        ps.append(p)
    if not ps:
        return dict(kind="rational", tau="0")
    P = s.Poly(1, t, domain=s.QQ)
    for p in ps:
        P = P.lcm(p.sqf_part())
    P = P.sqf_part()
    # Make isolating intervals strictly separated, including rational
    # singleton roots. This permits safe rational samples in every gap.
    bits = 8
    while True:
        intervals = [pair for pair, mult in P.intervals(eps=s.Rational(1, 2**bits))]
        if all(intervals[i][1] < intervals[i+1][0]
               for i in range(len(intervals)-1)):
            break
        bits *= 2
    if not intervals:
        samples = [s.Rational(0)]
    else:
        samples = [s.floor(intervals[0][0])-1]
        samples += [(a[1]+b[0])/2 for a,b in zip(intervals,intervals[1:])]
        samples += [s.ceiling(intervals[-1][1])+1]
    for q in samples:
        if all(p.eval(q) >= 0 for p in ps):
            return dict(kind="rational", tau=str(q))
    gs = [P.gcd(p) for p in ps]
    for lo,hi in intervals:
        if lo == hi:
            if all(p.eval(lo) >= 0 for p in ps):
                return dict(kind="rational_root", tau=str(lo))
        else:
            mid = (lo+hi)/2
            # Each non-singleton interval contains exactly one product
            # root; no endpoints are product roots. A constraint vanishes
            # there iff the gcd has a root in the interval. Otherwise its
            # sign equals the sign at any interior rational sample.
            if P.eval(lo) == 0 or P.eval(hi) == 0:
                raise ArithmeticError("unexpected root at interval endpoint")
            if all(g.count_roots(lo,hi) != 0 or p.eval(mid) > 0
                   for p,g in zip(ps,gs)):
                return dict(kind="algebraic_root",
                            polynomial=str(P.as_expr()),
                            interval=[str(lo),str(hi)])
    return None


def radical_inequality(c, L, h):
    """DNF for c sqrt(L)>=h, assuming L>=0, c rational."""
    if c == 0:
        return [[-h]]
    if c > 0:
        return [[-h], [h, c*c*L-h*h]]
    return [[-h, h*h-c*c*L]]


class ReaderProblem:
    """W: d-by-k rational physical writes; C/a: effective query rows.

    eps_mass is d*normalization_epsilon, strictly positive. dose is
    Euclidean physical edit budget. Static basis and Gram inverses
    are computed over QQ, so redundant columns cause no rank ambiguity.
    """
    def __init__(self, x, W, C, a, dose, eps_mass):
        x,a=col(x),col(a)
        d=x.rows
        W,C=matrix(W),matrix(C,d)
        e,m=rat(dose),rat(eps_mass)
        if not d or a.shape!=x.shape or W.rows!=d or C.cols!=d or e<0 or m<=0:
            raise ValueError("invalid problem shape or budget")
        ws=W.columnspace()
        Q=s.Matrix.hstack(*ws) if ws else s.zeros(d,0)
        D=C*Q
        ns=D.nullspace()
        U=Q*s.Matrix.hstack(*ns) if ns else s.zeros(d,0)
        P=projector(U)
        c=C*x
        self.fixed = D.row_join(c).rank() != D.rank()
        k=s.zeros(d,1)
        if not self.fixed and Q.cols and any(c):
            solution, parameters = D.gauss_jordan_solve(c)
            solution=solution.subs({q:0 for q in parameters})
            k=(s.eye(d)-P)*Q*solution
            if C*k!=c:
                raise ArithmeticError("exact protected solve failed")
        self.x,self.a,self.U=x,a,U
        self.p,self.b=P*x,P*a
        self.z0=(t-1)*k
        self.rho2=norm2(x)+m
        self.target0=dot(a,x)
        self.R2=s.expand(t*t*self.rho2-m-norm2(x-self.p+self.z0))
        self.E2=s.expand(e*e-norm2(self.z0))
        self.P2,self.B2,self.PB=norm2(self.p),norm2(self.b),dot(self.p,self.b)
        self.H=(self.R2+self.P2-self.E2)/2
        self.base=[t,self.R2,self.E2]
        if self.fixed:
            self.base += [t-1,1-t]

    def branches(self,gain):
        """Each returned branch is a conjunction of degree <=4 polys."""
        g=rat(gain)
        R,H=self.R2,self.H
        K=t*(self.target0+g)-dot(self.a,self.x+self.z0-self.p)
        P,B,D=self.P2,self.B2,self.PB
        n=self.U.cols
        if n==0:
            return [self.base+[-R,-H,-K]]
        if n==1:
            q=self.U[:,0]
            L=R/norm2(q)
            out=[]
            for sign in (1,-1):
                for aa,bb in product(radical_inequality(sign*dot(self.p,q),L,H),
                                     radical_inequality(sign*dot(self.b,q),L,K)):
                    out.append(self.base+aa+bb)
            return out
        # n>=2: polyhedral minimum norm <= R iff sphere intersects.
        # Every listed candidate is tested for primal feasibility.
        out=[self.base+[-H,-K]]
        if P:
            out.append(self.base+[D*H-P*K,P*R-H*H])
        if B:
            out.append(self.base+[D*K-B*H,B*R-K*K])
        det=P*B-D*D
        if det>0:
            out.append(self.base+[det*R-B*H*H+2*D*H*K-P*K*K])
        return out

    def decide(self,gain):
        """Exact existential test; lower witness is a radius sign-cell."""
        branches=self.branches(gain)
        for i,conditions in enumerate(branches):
            w=signs_feasible(conditions)
            if w is not None:
                return dict(feasible=True,branch=i,witness=w,branches=len(branches))
        return dict(feasible=False,branches=len(branches),witness=None)

    def bracket(self,bits=24):
        """Exact rational lower/upper bracket for nonnegative optimum.
        Width <= initial_upper/2**bits. Upper is infeasible, lower
        feasible; no full algebraic edit vector is reconstructed here.
        """
        if type(bits) is not int or not 1<=bits<=256:
            raise ValueError("bits must be integer 1..256")
        lo=s.Rational(0)
        hi=(self.rho2+norm2(self.a))/2+abs(self.target0)+1
        lower=self.decide(lo)
        if not lower["feasible"] or self.decide(hi)["feasible"]:
            raise ArithmeticError("initial bracket failed")
        for _ in range(bits):
            mid=(lo+hi)/2
            result=self.decide(mid)
            if result["feasible"]:
                lo,lower=mid,result
            else:
                hi=mid
        return dict(lower=str(lo),upper=str(hi),width=str(hi-lo),
                    lower_witness=lower,units="rho0_times_score_gain",
                    scope="EXACT_RATIONAL_INPUT_THRESHOLD_BRACKET")
