"""Exact finite-collateral certificates for a stale atlas (zero clean scores)."""
from fractions import Fraction as F
from itertools import combinations
import argparse
import json
import random


def q(x):
    if isinstance(x, (float, bool)):
        raise ValueError('exact rational inputs required')
    return F(x)


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def gram(rows, n):
    return [[sum((row[i]*row[j] for row in rows), F(0))
             for j in range(n)] for i in range(n)]


def solve(A, b):
    n = len(A)
    a = [list(row)+[b[i]] for i, row in enumerate(A)]
    for j in range(n):
        p = next(i for i in range(j, n) if a[i][j])
        a[j], a[p] = a[p], a[j]
        z = a[j][j]
        a[j] = [x/z for x in a[j]]
        for i in range(n):
            if i != j and a[i][j]:
                z = a[i][j]
                a[i] = [x-z*y for x,y in zip(a[i],a[j])]
    return [r[-1] for r in a]


def psd(A, strict=False):
    """Rational symmetric Schur complement test; zero pivots checked."""
    a = [list(row) for row in A]
    if any(len(row) != len(a) for row in a) or any(
            a[i][j] != a[j][i] for i in range(len(a)) for j in range(len(a))):
        raise ValueError('expected symmetric matrix')
    while a:
        n = len(a)
        if any(a[i][i] < 0 for i in range(n)):
            return False
        zeros = [i for i in range(n) if a[i][i] == 0]
        if zeros and (strict or any(any(a[i]) for i in zeros)):
            return False
        keep = [i for i in range(n) if i not in zeros]
        if not keep:
            return not strict
        a = [[a[i][j] for j in keep] for i in keep]
        a = [[a[i][j]-a[i][0]*a[0][j]/a[0][0]
              for j in range(1,len(a))] for i in range(1,len(a))]
    return True


def det(A):
    a = [list(row) for row in A]
    out = F(1)
    for j in range(len(a)):
        p = next((i for i in range(j,len(a)) if a[i][j]),None)
        if p is None:
            return F(0)
        if p != j:
            a[p],a[j]=a[j],a[p];out=-out
        z=a[j][j];out*=z
        for i in range(j+1,len(a)):
            f=a[i][j]/z
            a[i]=[x-f*y for x,y in zip(a[i],a[j])]
    return out


def independent_psd(A):
    # All principal minors, exponential but small verification fixtures.
    return all(det([[A[i][j] for j in S] for i in S])>=0
               for k in range(1,len(A)+1)
               for S in combinations(range(len(A)),k))


class Certificate:
    def __init__(self, G, H, h, rho2, dose, eta):
        self.G = [[q(x) for x in row] for row in G]
        self.H = [[q(x) for x in row] for row in H]
        self.h = [q(x) for x in h]
        self.rho2,self.e,self.eta = map(q,(rho2,dose,eta))
        self.n = len(self.G)
        if (not self.n or self.n!=len(h) or any(len(r)!=self.n for r in self.H)
                or self.e<=0 or self.eta<0 or not psd(self.G,strict=True)):
            raise ValueError('invalid metric or budget')
        self.offset=self.rho2-dot(self.h,solve(self.G,self.h))
        if self.offset<=0:
            raise ValueError('positive normalization mass required')
        self.H2=gram(self.H,self.n)
        self.cap=self.eta**2*self.rho2/self.e**2

    def block(self, lam):
        lam=q(lam)
        if lam<0:
            raise ValueError('negative multiplier')
        a=self.eta**2
        return [[(lam+a)*self.G[i][j]-self.rho2*self.H2[i][j]
                 for j in range(self.n)]+[a*self.h[i]] for i in range(self.n)]+[
                     [a*x for x in self.h]+[a*self.rho2-lam*self.e**2]]

    def trace_safe(self):
        tr=sum((solve(self.G,[row[j] for row in self.H2])[j]
                for j in range(self.n)),F(0))
        return self.rho2*self.e**2*tr<=self.eta**2*self.offset

    def find(self, steps=64):
        if type(steps) is not int or not 1<=steps<=4096:
            raise ValueError('invalid grid budget')
        # Endpoint is complete in the orthogonal case. Else this is a
        # sufficient rational search; failure is inconclusive.
        for i in range(steps,-1,-1):
            lam=self.cap*i/steps
            if psd(self.block(lam)):
                return lam
        return None

    def within_budget(self, w):
        w=list(map(q,w))
        if len(w)!=self.n:
            raise ValueError('wrong edit dimension')
        return dot(w,[dot(row,w) for row in self.G])<=self.e**2

    def violates(self, w):
        w=list(map(q,w))
        if not self.within_budget(w):
            raise ValueError('witness outside dose')
        denom=self.rho2+2*dot(self.h,w)+dot(w,[dot(row,w) for row in self.G])
        return self.rho2*sum((dot(r,w)**2 for r in self.H),F(0))>self.eta**2*denom


def fixture(seed, orthogonal):
    r=random.Random(seed)
    n=1+seed%4
    W=[[F(2*int(i==j)+(r.choice((-1,1)) if i==j+1 else 0))
        for j in range(n)] for i in range(n)]
    x=[F(0) if orthogonal else F(r.randint(-2,2)) for _ in range(n)]
    G=gram(W,n); h=[dot(row,x) for row in zip(*W)]
    H=[[F(r.randint(-6,6),4) for _ in range(n)] for _ in range(1+seed%3)]
    rho2=dot(x,x)+5 # side coordinate 2; positive normalization mass 1
    return Certificate(G,H,h,rho2,F(1+seed%2),F(r.randint(1,8),4))


def edge_cases():
    # Exact boundary and zero collateral; singular metric is rejected.
    p=Certificate([[1]],[[1]],[0],1,1,F(3,4))
    if p.find() is None or p.violates([1]):
        raise ArithmeticError('finite norm certificate failed')
    p=Certificate([[1]],[[0]],[0],1,1,0)
    if p.find()!=0 or p.violates([1]):
        raise ArithmeticError('zero residual certificate failed')
    p=Certificate([[1]],[[1]],[0],1,1,0)
    if p.find() is not None or not p.violates([1]):
        raise ArithmeticError('zero collateral failure missed')
    try:
        p.violates([])
    except ValueError:
        pass
    else:
        raise ArithmeticError('bad edit dimension accepted')
    try:
        Certificate([[0]],[[1]],[0],1,1,1)
    except ValueError:
        pass
    else:
        raise ArithmeticError('singular physical metric accepted')


def run(roots=96):
    edge_cases()
    from collections import Counter
    totals={}
    rows=[]
    for orthogonal in (True,False):
        c=Counter()
        for seed in range(61000,61000+roots):
            p=fixture(seed,orthogonal)
            tr=p.trace_safe();lam=p.find()
            if lam is not None and not independent_psd(p.block(lam)):
                raise ArithmeticError('PSD witness failed principal-minor replay')
            if orthogonal:
                # Independent closed-form matrix condition at the full dose.
                n=p.n
                M=[[p.eta**2*(p.rho2+p.e**2)*p.G[i][j]
                    -p.rho2*p.e**2*p.H2[i][j] for j in range(n)] for i in range(n)]
                truth=independent_psd(M)
                if truth!=(lam is not None) or (tr and not truth):
                    raise ArithmeticError('orthogonal exact decision mismatch')
                c['exact_unsafe']+=int(not truth)
            safe=tr or lam is not None
            r=random.Random(seed+300000)
            breach=False
            for _ in range(64):
                w=[F(r.randint(-8,8),8) for _ in range(p.n)]
                while not p.within_budget(w):
                    w=[v/2 for v in w]
                breach|=p.violates(w)
                c['sampled_edits']+=1
            if safe and breach:
                raise ArithmeticError('certified safe region breached')
            c['trace_safe']+=int(tr)
            c['psd_safe']+=int(lam is not None)
            c['combined_safe']+=int(safe)
            c['extra_over_trace']+=int(not tr and lam is not None)
            c['sample_breach']+=int(breach)
            c['unresolved_without_witness']+=int(not safe and not breach and not orthogonal)
            rows.append(dict(seed=seed,orthogonal=orthogonal,trace_safe=tr,
                             lambda_witness=None if lam is None else str(lam),sample_breach=breach))
        totals['orthogonal' if orthogonal else 'nonorthogonal']=dict(c)
    return dict(scope='CONSTRUCTED_RATIONAL_CERTIFICATE_ONLY',roots_per_regime=roots,
                multiplier_grid_steps=64,totals=totals,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()
    r=run()
    if a.output:
        with open(a.output,'w') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(r['totals'],sort_keys=True))
