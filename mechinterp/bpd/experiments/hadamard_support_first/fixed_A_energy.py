"""Exact zero-shift relaxation conditioned on the complete central witness catalogue.
Retain D=B+C row sums and both separate channel energies, including singleton parity.
Relax E=B-C and nonzero correlations. Group by exact central signed-axis histogram.
"""
from pathlib import Path
from itertools import product
from functools import lru_cache
from collections import defaultdict,Counter
import csv,json,time,numpy as np
P=Path(__file__).parent
AX=((1,0),(-1,0),(0,1),(0,-1))
DS=sorted({(a[0]+b[0],a[1]+b[1]) for a,b in product(AX,repeat=2)})
LIM=49;STRIDE=128
MASK=sum(((1<<(LIM+1))-1)<<(u*STRIDE) for u in range(LIM+1))

def cost(a,d):return (3*d[0]**2+a[0]*d[0],3*d[1]**2+a[1]*d[1])
def shift_cost(bits,u,v):return (bits<<(u*STRIDE+v))&MASK

def table(counts):
    dp=np.array([[1]],dtype=object);n=0
    for a,count in zip(AX,counts):
        for _ in range(count):
            sz=4*n+1;out=np.zeros((sz+4,sz+4),dtype=object)
            for d in DS:
                u,v=cost(a,d);dst=tuple(slice(2+x,2+x+sz) for x in d)
                out[dst]|=shift_cost(dp,2*u,2*v)
            dp=out;n+=1
    return dp

@lru_cache(None)
def tab(counts):return table(counts)
@lru_cache(None)
def row(hist,b,c):
    s,*cnt=hist; a=AX[s];t=tab(tuple(cnt)); out=0
    bopts=[x for x in AX if all((i-j)%2==0 for i,j in zip(x,b))];copts=[x for x in AX if all((i-j)%2==0 for i,j in zip(x,c))]
    for B,C in product(bopts,copts):
        d=tuple(x+y for x,y in zip(B,C));z=tuple(x+y-v for x,y,v in zip(b,c,d));assert all(x%2==0 for x in z)
        pos=tuple(x//2+22 for x in z)
        if any(not 0<=x<45 for x in pos):continue
        u,v=cost(a,d);out|=shift_cost(int(t[pos]),u,v)
    return out

def coords(bits):
    while bits:
        l=bits&-bits;bits^=l;k=l.bit_length()-1;yield divmod(k,STRIDE)
def combine(a,b):
    if a.bit_count()<b.bit_count():a,b=b,a
    z=0
    for u,v in coords(b):z|=shift_cost(a,u,v)
    return z

def small_checks():
    num=0
    for cnt in [(1,0,0,0),(0,1,0,0),(1,0,1,0),(0,0,1,1),(0,1,0,2)]:
        aa=[a for a,k in zip(AX,cnt) for _ in range(k)];dp=table(cnt);n=len(aa);dct={}
        for dd in product(DS,repeat=n):
            m=tuple(sum(d[j] for d in dd) for j in range(2));cc=[cost(a,d) for a,d in zip(aa,dd)];u,v=[2*sum(z[j] for z in cc) for j in range(2)]
            if u<=LIM and v<=LIM:dct[m]=dct.get(m,0)|(1<<(u*STRIDE+v))
        for m in product(range(-2*n,2*n+1),repeat=2):assert int(dp[tuple(x+2*n for x in m)])==dct.get(m,0);num+=1
    return num

def canon_rows(u,v):
    rows=[]
    for i,(a,b) in enumerate(zip(u,v)):
        su=1 if a[0]>=0 else -1;sv=1 if b[0]>=0 else -1
        rows.append(((abs(a[0]),abs(b[0])),i,(su*a[1],sv*b[1]),(su*a[2],sv*b[2])))
    return sorted(rows)

def histkey(w):
    p,u,v=w[1:5],w[5:9],w[9:13];h=[]
    for m,a,b in zip(p,u,v):
        s=(2 if m&1 else 0)+(0 if (b if m&1 else a)&1 else 1)
        h.append((s,(a>>1).bit_count(),((4095^m^a)>>1).bit_count(),(b>>1).bit_count(),((m^b)>>1).bit_count()))
    return tuple(h)
def main():
    t=time.perf_counter();checks=small_checks();print('small checks',checks,flush=True)
    q=json.loads((P/'prior/joint_receipt.json').read_text())['quotient']['systems'];pr=json.loads((P/'prior/roster.json').read_text())['profiles'];h=defaultdict(Counter)
    for w in csv.reader(open(P/'central_witnesses.csv')):
        x=list(map(int,w));h[x[0]][histkey(x)]+=1
    out=[];c=Counter();survivors=[[] for _ in q]
    for i,hs in sorted(h.items()):
        for f,j,_ in pr[i]['roster']:
            rows=canon_rows(q[f]['U'],q[f]['all_V'][j]);assert tuple(x[0] for x in rows)==tuple(map(tuple,pr[i]['profile']))
            good=[]
            for hh,n in hs.items():
                mass=sum((s>=2)+2*(k[2]+k[3]) for s,*k in hh)
                assert (599-mass)%12==0 and (573-(92-mass))%12==0
                tu,tv=(573-(92-mass))//12,(599-mass)//12
                if min(tu,tv)<0 or max(tu,tv)>LIM:continue
                bits=1
                for H,(_,_,b,cc) in zip(hh,rows):
                    bits=combine(bits,row(H,b,cc))
                    if not bits:break
                if bits>>(tu*STRIDE+tv)&1:good.append((hh,n))
            c['hist_tests']+=len(hs);c['hist_survived']+=len(good);c['witness_pair_tests']+=sum(hs.values());c['witness_pairs_survived']+=sum(n for _,n in good);c['pair_survived' if good else 'pair_excluded']+=1
            if good:survivors[f].append(j)
            out.append(dict(profile=i,sector=f,V_id=j,retained_histograms=len(good),total_histograms=len(hs),retained_witnesses=sum(n for _,n in good),total_witnesses=sum(hs.values())))
    r=dict(scope='FIXED_CENTRAL_HISTOGRAM_D_MARGIN_AND_TWO_ENERGY_RELAXATION',counts=dict(c),remaining_by_sector=list(map(len,survivors)),retained_ids=survivors,small_checks=checks,tab_cache=tab.cache_info()._asdict(),row_cache=row.cache_info()._asdict(),seconds=time.perf_counter()-t,rows=out)
    (P/'fixed_A_receipt.json').write_text(json.dumps(r,separators=(',',':'))+'\n');print({k:v for k,v in r.items() if k not in ('rows','retained_ids')})
if __name__=='__main__':main()
