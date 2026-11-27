"""SCCG-09: finite collateral with nonzero baseline scores; rational inputs."""
from fractions import Fraction as F
from itertools import combinations
from math import isqrt
import sympy as sp


def q(x):
    if isinstance(x, (float, bool, sp.Float)):
        raise ValueError('exact rational inputs required')
    return F(x)


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def psd(M):
    a = [list(r) for r in M]
    while a:
        n = len(a)
        if any(a[i][i] < 0 for i in range(n)):
            return False
        zeros = [i for i in range(n) if a[i][i] == 0]
        if any(any(a[i]) for i in zeros):
            return False
        keep = [i for i in range(n) if i not in zeros]
        if not keep:
            return True
        a = [[a[i][j] for j in keep] for i in keep]
        a = [[a[i][j]-a[i][0]*a[0][j]/a[0][0]
              for j in range(1,len(a))] for i in range(1,len(a))]
    return True


def principal_psd(M):
    A = sp.Matrix(M)
    return all(A.extract(S,S).det() >= 0 for k in range(1,A.rows+1)
               for S in combinations(range(A.rows),k))


def upper_sqrt(x, scale=2**24):
    if x < 0:
        raise ValueError('negative square root')
    y = x*scale*scale
    n = isqrt(y.numerator//y.denominator)
    return F(n if n*n == y else n+1, scale)


def sign_violation(V, J, s2):
    """Exact comparison V > J sqrt(s2), s2>0."""
    if s2 <= 0:
        raise ValueError('nonpositive normalization')
    if J >= 0:
        return V > 0 and V*V > J*J*s2
    return V >= 0 or V*V < J*J*s2


class Problem:
    def __init__(self,G,H,h,c,rho2,dose,eta):
        self.G = [[q(x) for x in row] for row in G]
        self.H = [[q(x) for x in row] for row in H]
        self.h,self.c = [q(x) for x in h],[q(x) for x in c]
        self.rho2,self.e,self.eta = map(q,(rho2,dose,eta))
        n=self.n=len(self.G)
        if (not n or n!=len(self.h) or not len(self.c) or len(self.H)!=len(self.c)
                or any(len(r)!=n for r in self.G+self.H) or self.e<=0 or self.eta<0
                or any(self.G[i][j]!=self.G[j][i] for i in range(n) for j in range(n))
                or not psd(self.G)):
            raise ValueError('invalid metric, shape or budget')
        M=sp.Matrix(self.G)
        if M.det()<=0:
            raise ValueError('physical metric must be positive definite')
        self.I=[[F(v) for v in row] for row in M.inv().tolist()]
        self.offset=self.rho2-dot(self.h,[dot(row,self.h) for row in self.I])
        if self.offset<=0:
            raise ValueError('normalization offset must be positive')
        self.l=[v/self.rho2 for v in self.h]
        self.T=[[v-ci*t for v,t in zip(row,self.l)] for row,ci in zip(self.H,self.c)]
        self.b=self.e**2/(2*self.rho2)
        self.lmax=upper_sqrt(dot(self.l,[dot(row,self.l) for row in self.I]))
        self.lower=1-self.e*self.lmax

    def trace_norm(self,A):
        return upper_sqrt(sum((dot(row,[dot(r,row) for r in self.I]) for row in A),F(0)))

    def separate(self):
        cn=upper_sqrt(dot(self.c,self.c))
        return self.lower>0 and self.e*self.trace_norm(self.H)+(self.e*self.lmax+self.b)*cn<=self.eta*self.lower

    def compensated(self):
        cn=upper_sqrt(dot(self.c,self.c))
        return self.lower>0 and self.e*self.trace_norm(self.T)+self.b*cn<=self.eta*self.lower

    def endpoint(self,b):
        b=q(b); n=self.n; a=self.eta**2
        M=[[a*self.l[i]*self.l[j]-sum((row[i]*row[j] for row in self.T),F(0))
            for j in range(n)] for i in range(n)]
        cross=[a*(1+b)*self.l[i]+b*sum((row[i]*ci for row,ci in zip(self.T,self.c)),F(0))
               for i in range(n)]
        const=a*(1+b)**2-b*b*dot(self.c,self.c)
        A=[row+[v] for row,v in zip(M,cross)]+[cross+[const]]
        D=[row+[F(0)] for row in self.G]+[[F(0)]*n+[-self.e**2]]
        return A,D,const/self.e**2

    def matrix(self,b,lam):
        A,D,_=self.endpoint(b)
        return [[v+lam*u for v,u in zip(a,d)] for a,d in zip(A,D)]

    def envelope_grid(self,steps=64):
        if type(steps) is not int or not 1<=steps<=4096:
            raise ValueError('bad multiplier budget')
        if self.lower<=0:
            return None
        witnesses=[]
        for b in (F(0),self.b):
            A,D,cap=self.endpoint(b)
            if cap<0:
                return None
            for j in range(steps,-1,-1):
                lam=cap*j/steps
                M=[[v+lam*u for v,u in zip(a,d)] for a,d in zip(A,D)]
                if psd(M):
                    witnesses.append(lam)
                    break
            else:
                return None
        return tuple(witnesses)

    def violates(self,w):
        w=list(map(q,w))
        if len(w)!=self.n:
            raise ValueError('bad edit width')
        d=dot(w,[dot(row,w) for row in self.G])
        if d>self.e**2:
            raise ValueError('edit outside dose')
        s2=1+(2*dot(self.h,w)+d)/self.rho2
        y=[ci+dot(row,w) for ci,row in zip(self.c,self.H)]
        V=dot(y,y)+(dot(self.c,self.c)-self.eta**2)*s2
        J=2*dot(self.c,y)
        return sign_violation(V,J,s2)

    def exact_line(self):
        """Complete uniform safety decision for a one-dimensional span.

        Strict violation is continuous. If present even at a dose boundary,
        it is present in an open in-dose cell, hence at a rational sample.
        Isolate every root of degree<=4 sign constraints. Return a rational
        violating edit or prove absence of one; no sampling assumption.
        """
        if self.n!=1:
            raise ValueError('exact_line requires one coordinate')
        t=sp.Symbol('w')
        g=self.G[0][0]
        d=g*t*t
        s2=1+(2*self.h[0]*t+d)/self.rho2
        y=[ci+row[0]*t for ci,row in zip(self.c,self.H)]
        V=dot(y,y)+(dot(self.c,self.c)-self.eta**2)*s2
        J=2*dot(self.c,y)
        polynomials=[self.e**2-d,V,J,V*V-J*J*s2]
        P=sp.Poly(1,t,domain=sp.QQ)
        for expr in polynomials:
            p=sp.Poly(expr,t,domain=sp.QQ)
            if not p.is_zero and p.degree()>0:
                if p.degree()>4:
                    raise ArithmeticError('unexpected degree')
                P=P.lcm(p.sqf_part())
        bits=8
        while True:
            intervals=[i for i,_ in P.intervals(eps=sp.Rational(1,2**bits))]
            if all(a[1]<b[0] for a,b in zip(intervals,intervals[1:])):
                break
            bits*=2
        tested=0
        for a,b in zip(intervals,intervals[1:]):
            w=F((a[1]+b[0])/2)
            if g*w*w<self.e**2:
                tested+=1
                if self.violates([w]):
                    return dict(safe=False,witness=str(w),cells=tested,degree=P.degree())
        return dict(safe=True,witness=None,cells=tested,degree=P.degree())
