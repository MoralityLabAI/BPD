"""BPD-FSI02: prospective independent-world CPU expected-L0 toy benchmark.

Measures finite-dictionary recovery on frozen synthetic activation laws and
exact anchor-relative intervention-mask algebra. No BPD neural weights, model
checkpoint or protected ARG-03 test is accessed.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,random,math,time
from collections import Counter
from fractions import Fraction as Q
from itertools import combinations
from pathlib import Path

DIRECTIONS=((1,0),(0,1),(1,1),(1,-1),(1,2))

def det(a,b):return a[0]*b[1]-a[1]*b[0]
BASES=tuple((i,j) for i,j in combinations(range(5),2) if det(DIRECTIONS[i],DIRECTIONS[j])!=0)

def _load_fsi():
    path=os.environ.get('EVALMATH_FSI_FILE')
    if not path or not Path(path).is_file():raise RuntimeError('EVALMATH_FSI_FILE must point to pinned EvalMath source')
    spec=importlib.util.spec_from_file_location('evalmath_fsi_for_bpd_fsi02',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def seed_int(*args):
    return int.from_bytes(hashlib.sha256(('BPD-FSI02:'+':'.join(map(str,args))).encode()).digest()[:8],'little')

def law(seed,stratum):
    rng=random.Random(seed_int('law',seed))
    z=Q(1,10)+Q(rng.randrange(7),100)
    if stratum=='rotated':
        # Strictly positive masses on the two rotated support lines.
        p=Q(2,5)+Q(rng.randrange(21),100)
        return (((0,0),z),((1,1),(1-z)*p),((1,-1),(1-z)*(1-p)))
    if stratum=='cancellation':
        w=(1-z)/3
        return (((0,0),z),((1,0),w),((0,1),w),((1,1),w))
    raise ValueError('invalid stratum')

def draw(seed,split,distribution,n):
    rng=random.Random(seed_int('draw',seed,split))
    points=[x for x,p in distribution]
    cumulative=[];a=Q(0)
    for x,p in distribution:
        a+=p;cumulative.append(float(a))
    out=[]
    for _ in range(n):
        u=rng.random()
        j=next((j for j,c in enumerate(cumulative) if u<c),len(points)-1)
        out.append(points[j])
    return out

def loss_sum(basis,samples):
    a,b=(DIRECTIONS[i] for i in basis)
    return sum((a[0]*x[0]+a[1]*x[1]!=0)+(b[0]*x[0]+b[1]*x[1]!=0) for x in samples)

def true_l0(basis,distribution):
    a,b=(DIRECTIONS[i] for i in basis)
    return sum(p*((a[0]*x[0]+a[1]*x[1]!=0)+(b[0]*x[0]+b[1]*x[1]!=0)) for x,p in distribution)

def exact_minimizers(distribution):
    d={basis:true_l0(basis,distribution) for basis in BASES}
    v=min(d.values())
    return tuple(b for b in BASES if d[b]==v)

def invert(m):
    a,b=m;d=det(a,b)
    if not d:raise ValueError('singular basis')
    return ((Q(b[1],d),-Q(a[1],d)),(-Q(b[0],d),Q(a[0],d)))

def mul(a,b):return tuple(tuple(sum(a[i][k]*b[k][j] for k in (0,1)) for j in (0,1)) for i in (0,1))

def intervention_audit(best,distribution):
    if len(best)<2:return {'n_optima':len(best),'singleton_not_invariant':False,'full_block_invariant':True,'witness_bases':None}
    for first,second in combinations(best,2):
        m1=tuple(DIRECTIONS[i] for i in first)
        m2=tuple(DIRECTIONS[i] for i in second)
        r=mul(m2,invert(m1));rinv=invert(r)
        d=((Q(1),Q(0)),(Q(0),Q(0)))
        ident=((Q(1),Q(0)),(Q(0),Q(1)))
        single=mul(mul(rinv,d),r)
        block=mul(mul(rinv,ident),r)
        assert block==ident
        if single!=d:
            return {'n_optima':len(best),'singleton_not_invariant':True,'full_block_invariant':True,
                    'witness_bases':[list(first),list(second)],
                    'relative_change':[[str(single[i][j]-d[i][j]) for j in (0,1)] for i in (0,1)]}
    return {'n_optima':len(best),'singleton_not_invariant':False,'full_block_invariant':True,'witness_bases':None}

def score_world(seed,stratum,train_n,test_n):
    dist=law(seed,stratum)
    train=draw(seed,'train',dist,train_n)
    trained=min(BASES,key=lambda b:(loss_sum(b,train),b))
    random_basis=random.Random(seed_int('control',seed)).choice(BASES)
    test=draw(seed,'readout',dist,test_n)
    identity=(0,1)
    if identity not in BASES:raise AssertionError('reference basis absent')
    true_opt=exact_minimizers(dist)
    trloss=loss_sum(trained,test);ref_loss=loss_sum(identity,test);random_loss=loss_sum(random_basis,test)
    if not trloss>=0 or not ref_loss>=0:raise AssertionError('invalid L0')
    assert all(true_l0(b,dist)>=true_l0(true_opt[0],dist) for b in BASES)
    return {'seed':seed,'stratum':stratum,'true_optimal_bases':[list(b) for b in true_opt],
            'selected_basis':list(trained),'random_basis':list(random_basis),
            'selected_exact_optimal':trained in true_opt,
            'selected_train_loss':loss_sum(trained,train),
            'selected_test_loss_sum':trloss,'reference_test_loss_sum':ref_loss,
            'random_test_loss_sum':random_loss,
            'world_reduction_exact':str(Q(ref_loss-trloss,test_n)),
            'population_selected_l0':str(true_l0(trained,dist)),
            'population_optimum_l0':str(true_l0(true_opt[0],dist)),
            'mask':intervention_audit(true_opt,dist)}

def run(reg,limit=None):
    if reg.get('schema')!='BPD-FSI02/prospective-finite-basis-v1':raise ValueError('registration mismatch')
    if len(reg['seeds'])!=32 or len(set(reg['seeds']))!=32:raise ValueError('seed roster changed')
    if tuple(reg['candidate_rows'])!=tuple(f'({a},{b})' for a,b in DIRECTIONS):raise ValueError('dictionary changed')
    train_n=reg['generator']['train_draws'];test_n=reg['generator']['readout_draws']
    # 16 worlds of each type frozen before sampling.
    worlds=[score_world(seed,'rotated' if i<16 else 'cancellation',train_n,test_n)
            for i,seed in enumerate(reg['seeds'][:limit] if limit else reg['seeds'])]
    if not worlds:raise ValueError('no worlds')
    fsi=_load_fsi()
    effects=[Q(w['world_reduction_exact']) for w in worlds]
    p,mode=fsi.signflip_cluster_p(effects,seed=20261117,simulations=20000)
    cp_strata={}
    for stratum in ('rotated','cancellation'):
        sub=[w for w in worlds if w['stratum']==stratum]
        if sub:
            c=fsi.clopper_pearson_bracket(sum(w['selected_exact_optimal'] for w in sub),len(sub),Q(1,20))
            cp_strata[stratum]={'exact_optimal_count':sum(w['selected_exact_optimal'] for w in sub),
                               'n_worlds':len(sub),'iid_model_cp95_outward':[str(c['lower_outward']),str(c['upper_outward'])]}
    ci=fsi.cluster_bootstrap_ci(effects,Q(1,40),seed=20261117,resamples=4096)
    return {'schema':'BPD-FSI02/prospective-cpu-result-v1','worlds':worlds,
            'world_count':len(worlds),'readout_per_world':test_n,
            'p1':{'effect':str(sum(effects,Q(0))/len(effects)),'signflip_p':str(p),'test':mode,
                  'bonferroni_p':str(min(Q(1),2*p)),'bootstrap_97p5_interval':ci,
                  'confirm':len(worlds)>=20 and min(Q(1),2*p)<Q(1,20) and ci[0]>0},
            'p2':{'exact_optimal_worlds':sum(w['selected_exact_optimal'] for w in worlds),
                  'n_worlds':len(worlds),'by_stratum':cp_strata,
                  'caveat':'Fixed 16+16 strata: interval per stratum assumes iid generator worlds within that stratum; no pooled binomial interval.'},
            'mask':{'cancellation_worlds_with_singleton_counterexample':sum(w['stratum']=='cancellation' and w['mask']['singleton_not_invariant'] for w in worlds),
                    'full_block_invariance_algebraic':all(w['mask']['full_block_invariant'] for w in worlds)},
            'scope':'Synthetic finite-dictionary CPU experiment; no trained BPD network, ARG03 protected test untouched.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    reg=json.loads(args.registration.read_text());start=time.perf_counter()
    if args.smoke:
        reg=dict(reg,seeds=[2026111797]+reg['seeds'][1:]);res=run(reg,limit=1)
    else:res=run(reg)
    res['runtime_seconds']=round(time.perf_counter()-start,4)
    res['registration_sha256']=hashlib.sha256(args.registration.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(res,sort_keys=True,indent=2)+'\n')
    print('BPD-FSI02',len(res['worlds']),'worlds; effect',res['p1']['effect'],'p',res['p1']['signflip_p'],'mask',res['mask'])
if __name__=='__main__':main()
