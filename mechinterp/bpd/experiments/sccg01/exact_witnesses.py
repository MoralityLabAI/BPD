"""Independent exact witnesses, using only the Python standard library."""
from fractions import Fraction as F
from itertools import combinations

def solve(delta):
    a=(F(1),F(1),delta)
    c=(F(1),F(-1),F(0))
    def squared_gain(S):
        x=[a[i] for i in S]; y=[c[i] for i in S]
        yy=sum(t*t for t in y)
        return sum(t*t for t in x)-(sum(p*q for p,q in zip(x,y))**2/yy if yy else 0)
    S=()
    for _ in range(2):
        i=max((i for i in range(3) if i not in S),
              key=lambda i:squared_gain(S+(i,)))
        S=S+(i,)
    opt=max(squared_gain(s) for s in combinations(range(3),2))
    return squared_gain(S),opt,S

def check():
    for delta in (F(1,2),F(1,10),F(1,1000),F(1,10**6)):
        g,o,S=solve(delta)
        assert S[0]==2 and g==delta**2 and o==2 and g/o==delta**2/2
    for u in (F(-1),F(-1,2),F(0),F(1,2),F(1)):
        assert min(u,-u)==-abs(u)
        assert (u-u)/2==0
    for r in (F(1,16),F(1,32),F(1,100)):
        # g(x,y)=y+x^2; C=(0,1), L=2, kappa=1.
        w=-r*r
        assert w+r*r==0 and abs(w)<=2*r*r
        assert r*r+w*w<=r*r+(2*r*r)**2
    print("PASS: exact sparse, quotient and nonlinear correction witnesses")

if __name__=="__main__":
    check()
