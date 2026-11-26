"""SCCG-05 exact rare-circuit search. Constructed development only."""
from collections import OrderedDict, Counter
from fractions import Fraction as F
from itertools import combinations
from math import comb
import argparse
import json
import random
import sys
import time

N,K,R=14,4,8
ROOTS=16
PRIME=65521
DOSE, MASS, THRESHOLD=2,1,4
CAPS=(0,65536,524288)


def dot(a,b): return sum(x*y for x,y in zip(a,b))


def world(seed):
    rng=random.Random(seed)
    planted=seed%4!=0
    count=N-2 if planted else N
    ts=rng.sample(range(-9,10),count)
    columns=[[t**j for j in range(R)] for t in ts]
    groups=[]
    if planted:
        ix=rng.sample(range(count),6)
        for triple in (ix[:3],ix[3:]):
            columns.append([-sum(columns[i][j] for i in triple) for j in range(R)])
            groups.append(tuple(triple+[len(columns)-1]))
    order=list(range(N));rng.shuffle(order)
    inv={old:new for new,old in enumerate(order)}
    C=[columns[i] for i in order]
    # W=2I+signed shift; ||W||<=3 and sigma_min(W)>=1.
    W=[[0]*N for _ in range(N)]
    for j in range(N):
        W[j][j]=2
        if j+1<N: W[j+1][j]=rng.choice((-1,1))
    x=[rng.randint(-1,1) for _ in range(N)]
    if not any(x): x[0]=1
    G=[[dot(w,v) for v in W] for w in W] # W stored as physical column vectors
    h=[dot(w,x) for w in W]
    rho2=dot(x,x)+9+MASS
    queries=[]
    for q in range(3):
        a=[rng.randint(-4,4) for _ in range(count)]
        if planted:
            for i,S in enumerate(groups):
                wanted=32 if q==i else 1
                a.append(wanted-sum(a[j] for j in S[:-1]))
        queries.append([a[i] for i in order])
    circuits=[tuple(sorted(inv[i] for i in S)) for S in groups]
    return dict(seed=seed,columns=C,G=G,h=h,rho2=rho2,queries=queries,circuits=circuits)


def modular_rank(cols):
    basis={}
    for col in cols:
        v=[x%PRIME for x in col]
        for j in range(R):
            if not v[j]: continue
            if j in basis:
                f=v[j]; b=basis[j]
                v=[(x-f*y)%PRIME for x,y in zip(v,b)]
            else:
                inv=pow(v[j],-1,PRIME)
                basis[j]=[(x*inv)%PRIME for x in v]
                break
    return len(basis)


def nullspace(cols):
    n=len(cols)
    a=[[F(c[i]) for c in cols] for i in range(R)]
    piv=[]; r=0
    for j in range(n):
        i=next((i for i in range(r,R) if a[i][j]),None)
        if i is None: continue
        a[r],a[i]=a[i],a[r]
        z=a[r][j];a[r]=[v/z for v in a[r]]
        for i in range(R):
            if i!=r and a[i][j]:
                z=a[i][j];a[i]=[v-z*w for v,w in zip(a[i],a[r])]
        piv.append(j);r+=1
        if r==R: break
    basis=[]
    for j in range(n):
        if j in piv: continue
        v=[F(0)]*n;v[j]=F(1)
        for i,p in enumerate(piv): v[p]=-a[i][j]
        basis.append(tuple(v))
    return tuple(basis)


def inverse(M):
    n=len(M)
    a=[list(row)+[F(int(i==j)) for j in range(n)] for i,row in enumerate(M)]
    for j in range(n):
        i=next(i for i in range(j,n) if a[i][j])
        a[j],a[i]=a[i],a[j]
        z=a[j][j];a[j]=[v/z for v in a[j]]
        for i in range(n):
            if i!=j and a[i][j]:
                z=a[i][j];a[i]=[v-z*w for v,w in zip(a[i],a[j])]
    return tuple(tuple(row[n:]) for row in a)


def quad(v,M): return dot(v,[dot(row,v) for row in M])


def size(x):
    # Conservative per-entry recursive Python object-size accounting;
    # shared objects can be counted repeatedly. Excludes container allocator.
    n=sys.getsizeof(x)
    if isinstance(x,F): n+=sys.getsizeof(x.numerator)+sys.getsizeof(x.denominator)
    elif isinstance(x,(tuple,list)): n+=sum(size(v) for v in x)
    return n


class Search:
    def __init__(self,w,kind,cap):
        self.w=w;self.kind=kind;self.cap=cap
        self.cache=OrderedDict();self.bytes=0;self.peak=0
        self.count=Counter()

    def structure(self,U,project):
        key=(U,project)
        if key in self.cache:
            self.count['cache_hits']+=1
            value,n=self.cache.pop(key);self.cache[key]=(value,n);return value
        cols=[self.w['columns'][i] for i in U]
        self.count['modular_rank_calls']+=1
        if modular_rank(cols)==len(U): value=((),(),F(0))
        elif not project: return None # modular deficiency alone proves nothing
        else:
            self.count['rational_nullspaces']+=1
            Z=nullspace(cols)
            if not Z: value=((),(),F(0))
            else:
                G=[[self.w['G'][i][j] for j in U] for i in U]
                GZ=[[dot(row,v) for row in G] for v in Z]
                M=tuple(tuple(dot(z,v) for v in GZ) for z in Z)
                I=inverse(M)
                self.count['gram_inverses']+=1
                hx=[dot(z,[self.w['h'][i] for i in U]) for z in Z]
                value=(Z,I,quad(hx,I))
        n=size(key)+size(value)+128
        if n<=self.cap:
            while self.bytes+n>self.cap:
                _,(_,old)=self.cache.popitem(last=False);self.bytes-=old
                self.count['evictions']+=1
            self.cache[key]=(value,n);self.bytes+=n;self.peak=max(self.peak,self.bytes)
        return value

    def bound(self,U,a):
        st=self.structure(U,True)
        Z,I,px=st
        if not Z: return False
        za=[dot(z,[a[i] for i in U]) for z in Z]
        # rho2 = ||x||^2 + m, hence m+||(I-P)x||^2=rho2-px.
        self.count['geometry_bounds']+=1
        return self.w['rho2']*DOSE**2*quad(za,I)>=THRESHOLD**2*(self.w['rho2']-px)

    def finite(self,U,a):
        self.count['leaf_visits']+=1
        Z,I,_=self.structure(U,True)
        if not Z:return False
        if len(Z)!=1:raise ArithmeticError('constructed family leaf nullity exceeds one')
        self.count['finite_queries']+=1
        z=Z[0]; A=dot(z,[a[i] for i in U]); h=dot(z,[self.w['h'][i] for i in U]); G=1/I[0][0]
        if A==0:return False
        if A<0:A,h=-A,-h
        # DOSE<rho0 ensures monotonicity along the positive-gain ray.
        L=self.w['rho2']*A*A*DOSE**2/G-THRESHOLD**2*(self.w['rho2']+DOSE**2)
        B=2*THRESHOLD**2*h*DOSE
        if L>=0 and B<=0:return True
        if L<0 and B>=0:return False
        return L*L*G>=B*B if L>=0 else L*L*G<=B*B

    def solve(self,a):
        if self.kind=='lex':
            for U in combinations(range(N),K):
                if self.finite(U,a):return U
            return None
        def visit(P,start):
            self.count['nodes']+=1
            left=K-len(P); rem=N-start
            if left==0:return P if self.finite(P,a) else None
            if rem<left:return None
            U=P+tuple(range(start,N))
            if rem==left:return U if self.finite(U,a) else None
            leaves=comb(rem,left)
            # At most k+2 allowed columns: rank relaxation cheaply skips
            # independent subtrees. Larger geometric relaxations may cost
            # more than they save; fixed cutoff chosen before scoring.
            if len(U)<=R and self.structure(U,False) is not None:
                self.count['rank_pruned_leaves']+=leaves;return None
            if self.kind=='geometry' and leaves>=4 and len(U)<=R+2:
                if not self.bound(U,a):
                    self.count['geometry_pruned_leaves']+=leaves;return None
            for i in range(start,N-left+1):
                ans=visit(P+(i,),i+1)
                if ans is not None:return ans
            return None
        return visit((),0)


def atlas_arm(worlds):
    """Post-pilot comparator: enumerate rank-deficient supports once per root."""
    start=time.perf_counter(); total=Counter();rows=[];peak=0
    for w in worlds:
        search=Search(w,'lex',CAPS[2])
        atlas=[]
        for U in combinations(range(N),K):
            search.count['atlas_supports']+=1
            search.count['modular_rank_calls']+=1
            if modular_rank([w['columns'][i] for i in U])==K:continue
            Z,_,_=search.structure(U,True)
            if Z:atlas.append(U)
        search.count['atlas_dependent']+=len(atlas)
        for q,a in enumerate(w['queries']):
            before=search.count.copy();ans=None
            for U in atlas:
                if search.finite(U,a):ans=U;break
            expected=w['circuits'][q] if w['circuits'] and q<2 else None
            if ans!=expected:raise ArithmeticError('atlas decision mismatch')
            counts={k:search.count[k]-before[k] for k in search.count}
            rows.append(dict(root=w['seed'],query=q,support=ans,counts=counts))
        total.update(search.count);peak=max(peak,search.peak)
    return dict(counts=dict(total),peak_cache_bytes_proxy=peak,
                seconds=time.perf_counter()-start,rows=rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output');args=ap.parse_args()
    worlds=[world(31000+i) for i in range(ROOTS)]
    # Ground truth follows from the Vandermonde/circuit theorem, independently
    # of the numerical search; exhaustively replay all leaves for query 0.
    start=time.perf_counter(); audited=0; dependent=0
    for w in worlds:
        control=Search(w,'lex',0)
        for U in combinations(range(N),K):
            cols=[w['columns'][i] for i in U]
            if modular_rank(cols)<K:
                Z=nullspace(cols)
                if Z:
                    dependent+=1
                    if U not in w['circuits'] or len(Z)!=1:raise ArithmeticError('circuit proof mismatch')
            expected=bool(w['circuits'] and U==w['circuits'][0])
            if control.finite(U,w['queries'][0])!=expected:raise ArithmeticError('finite oracle mismatch')
            audited+=1
    audit_seconds=time.perf_counter()-start
    arms={}
    for kind,cap in [('lex',0),('rank',0),('rank',CAPS[2]),('geometry',0),('geometry',CAPS[1]),('geometry',CAPS[2])]:
        start=time.perf_counter(); total=Counter();rows=[];peak=0
        for w in worlds:
            search=Search(w,kind,cap)
            for q,a in enumerate(w['queries']):
                before=search.count.copy();ans=search.solve(a)
                expected=w['circuits'][q] if w['circuits'] and q<2 else None
                if ans!=expected:raise ArithmeticError(f'wrong decision {kind} {w["seed"]} {q}')
                counts={k:search.count[k]-before[k] for k in search.count}
                rows.append(dict(root=w['seed'],query=q,support=ans,counts=counts))
            total.update(search.count);peak=max(peak,search.peak)
        arms[f'{kind}_{cap}']=dict(counts=dict(total),peak_cache_bytes_proxy=peak,
            seconds=time.perf_counter()-start,rows=rows)
    arms['atlas_524288']=atlas_arm(worlds)
    out=dict(scope='CONSTRUCTED_DEVELOPMENT_ONLY',roots=ROOTS,queries=ROOTS*3,
             solvable=sum(2 for w in worlds if w['circuits']),supports_per_root=comb(N,K),
             audited_leaves=audited,dependent_leaves=dependent,audit_seconds=audit_seconds,
             arms=arms)
    text=json.dumps(out,indent=2,sort_keys=True)+'\n'
    if args.output:open(args.output,'w').write(text)
    print(json.dumps({k:v for k,v in out.items() if k!='arms'},sort_keys=True))
    for name,row in arms.items():print(name,row['seconds'],row['peak_cache_bytes_proxy'],row['counts'])

if __name__=='__main__':main()
