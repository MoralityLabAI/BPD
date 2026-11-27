"""FUSE-07: exact oracle and fixed-cap native excerpt sweep."""
from __future__ import annotations
import argparse, copy, hashlib, json, platform, random, resource, statistics, time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import torch
try:
    from .admission import Site, select
    from .adjoint_cache import independent_residuals
    from .residual_cache import cached_residuals
    from .validate_fuse04 import load_native, close, err
except ImportError:
    from admission import Site, select
    from adjoint_cache import independent_residuals
    from residual_cache import cached_residuals
    from validate_fuse04 import load_native, close, err

MODES=('first','smallest','density','exact')
MIB=1024**2


def oracle(sites, cap):
    best=(0,0,())
    ids=[i for i,s in enumerate(sites) if s.value>0]
    for bits in range(1<<len(ids)):
        chosen=tuple(i for j,i in enumerate(ids) if bits>>j&1)
        size=sum(sites[i].size for i in chosen)
        if size>cap:continue
        value=sum(sites[i].value for i in chosen)
        best=min(best,(-value,size,chosen))
    return (-best[0],best[1],best[2])


def unit():
    rng=random.Random(3900001);cases=[]
    for j in range(256):
        n=rng.randint(1,12)
        s=[Site(rng.randint(1,20),rng.randint(0,80)) for i in range(n)]
        if j%4==0:s=[Site(x.size,3*x.size if i%3 else 0) for i,x in enumerate(s)]
        cases.append((s,rng.randint(0,sum(x.size for x in s)+3)))
    cases.extend([
        ([Site(10,60),Site(20,100),Site(30,120)],50),
        ([Site(5,10),Site(3,9),Site(3,9)],6),
        ([Site(2,2),Site(2,2),Site(4,4)],4),
        ([Site(2,0)],2),([],0),([Site(4,9)],4),
        ([Site(2**75,2**90),Site(2**75,2**90+1)],2**75),
    ])
    t=time.perf_counter(); checks=0; peak=transitions=0; misses={m:0 for m in MODES};gaps={m:0 for m in MODES}
    for sites,cap in cases:
        answer=oracle(sites,cap)
        for mode in MODES:
            p=select(sites,cap,mode)
            assert p.size<=cap and p.size==sum(sites[i].size for i in p.indices)
            assert p.value==sum(sites[i].value for i in p.indices)
            assert p.value<=answer[0]
            if mode=='exact':assert (p.value,p.size,p.indices)==answer,(sites,cap,p,answer)
            misses[mode]+=int(p.value<answer[0]);gaps[mode]+=answer[0]-p.value
            checks+=1;peak=max(peak,p.peak_frontier);transitions+=p.transitions
    try:select([Site(1,1),Site(2,2)],3,max_frontier=1)
    except RuntimeError:abort=True
    else:abort=False
    assert abort
    return dict(cases=len(cases),random_cases=256,checks=checks,exact_mismatches=0,frontier_abort=abort,peak_frontier=peak,transitions=transitions,greedy_suboptimal_cases=misses,pooled_value_shortfall=gaps,elapsed_seconds=time.perf_counter()-t)


def campaign(Base,Schedule,Config,builder,profile,passes,only=None):
    width=1024;seed=3901001+width+passes+(10000 if profile=='heterogeneous' else 0);torch.manual_seed(seed)
    a=builder(hidden_size=width,num_heads=4,l_layers=2,h_cycles=1,l_cycles=2,puzzle_emb_len=4,seq_len=16,seed=seed)
    for p in a.parameters():p.requires_grad_(False)
    ranks=(128,128,128,128) if profile=='uniform' else (32,64,96,128)
    paths=[]
    for i,block in enumerate(a.layers):
        for name in ('o_proj','down_proj'):
            rank=ranks[len(paths)];paths.append(f'layers.{i}.{name}')
            setattr(block,name,Base(getattr(block,name),rank,max_passes=passes,ablation=Config(mode='pass_mean',sphere_rescale=True),init_svd=False))
    def wrappers(model):return [model.get_submodule(p) for p in paths]
    x=a.encode_inputs(torch.randint(1,11,(2,16))).detach()
    for w in wrappers(a):w.update_running_means=True
    with torch.no_grad():
        h,l=a.init_states(2)
        for _ in range(passes//3):h,l=a.act_step(h,l,x)
    for w in wrappers(a):
        w.update_running_means=False;w.mode='component';w.schedule=Schedule(torch.rand(2,passes,20,w.C),torch.rand(2,passes,20));w.reset_pass_state()
    b=copy.deepcopy(a)
    sites=[Site.residual(w.d_in,w.d_out,w.C,passes) for w in wrappers(a)]
    caps=[4,8,16,32] if passes==6 else [16]
    plans=[];subsets={};full_work=sum(passes*w.d_in*w.C*w.d_out for w in wrappers(a))
    for m in caps:
        for mode in MODES:
            t=time.perf_counter_ns();p=select(sites,m*MIB,mode);ns=time.perf_counter_ns()-t
            assert p.value<=oracle(sites,m*MIB)[0]
            if mode=='exact':assert (p.value,p.size,p.indices)==oracle(sites,m*MIB)
            plans.append(dict(cap_mib=m,mode=mode,plan=asdict(p),planner_ns=ns,reconstruction_macs=full_work-p.value))
            subsets.setdefault(p.indices,m*MIB)
    if only is not None:assert only.issubset(subsets), (only,subsets)
    def roll(model,chosen,cap,back):
        ws=wrappers(model)
        for w in ws:w.reset_pass_state()
        model.zero_grad(set_to_none=True);inp=x.detach().requires_grad_(back)
        with torch.set_grad_enabled(back):
            chosen_ws=[ws[i] for i in chosen]
            scope=(independent_residuals(chosen_ws,variant='residual',max_bytes=cap) if back else cached_residuals(chosen_ws,mode='eval',max_bytes=cap)) if chosen else nullcontext(None)
            with scope as st:
                h,l=model.init_states(2);states=[]
                for _ in range(passes//3):h,l=model.act_step(h,l,inp);states.extend((h,l,model.digit_logits(h)))
                loss=states[-1].square().mean()
            if back:loss.backward()
        assert all(w.calls==passes for w in ws)
        if st is not None:
            assert (st.calls,st.hits,st.misses,st.reconstructions)==(len(chosen)*passes,len(chosen)*(passes-1),len(chosen),len(chosen)),asdict(st)
            assert st.peak_bytes==sum(sites[i].size for i in chosen) and st.peak_bytes<=cap and st.resident_bytes==0
            assert not st.cap_rejections and not st.bypasses
        assert all('weight_delta' not in w.__dict__ for w in ws)
        out=[s.detach() for s in states]+[loss.detach()]
        if back:out += [inp.grad.detach()]+[p.grad.detach() for p in model.parameters() if p.requires_grad]
        return out,asdict(st) if st is not None else None
    r=[]
    ref,_=roll(a,(),0,True)
    for chosen,cap in subsets.items():
        if only is not None and chosen not in only:continue
        pred,stats=roll(b,chosen,cap,True)
        failures=sum(not close(u,v) for u,v in zip(ref,pred));equal=sum(torch.equal(u,v) for u,v in zip(ref,pred));max_abs=max(err(u,v) for u,v in zip(ref,pred))
        greedy_indices=list(range(2,passes,3));greedy_failures=sum(not torch.equal(ref[i].argmax(-1),pred[i].argmax(-1)) for i in greedy_indices)
        timing={}
        for back in (False,True):
            ts=[[],[]]
            for i in range(3):
                for j in ((0,1) if i%2==0 else (1,0)):
                    t=time.perf_counter_ns();roll((a,b)[j],chosen if j else (),cap if j else 0,back);ms=(time.perf_counter_ns()-t)/1e6
                    if i>=1:ts[j].append(ms)
            timing['fwd_bwd' if back else 'forward']=dict(reference_ms=statistics.median(ts[0]),cached_ms=statistics.median(ts[1]),speedup=statistics.median(ts[0])/statistics.median(ts[1]),samples_ms=ts)
        result=dict(indices=chosen,cap_mib=cap//MIB,checks=len(ref),exact=equal,failures=failures,max_abs=max_abs,greedy_checks=len(greedy_indices),greedy_failures=greedy_failures,cache_stats=stats,physical_reconstructions=4*passes-len(chosen)*(passes-1),timing=timing)
        print(json.dumps({'subset':result}),flush=True);r.append(result)
    return dict(profile=profile,width=width,passes=passes,seed=seed,sites=[dict(path=p,rank=w.C,din=w.d_in,dout=w.d_out,**asdict(s)) for p,w,s in zip(paths,wrappers(a),sites)],baseline_reconstruction_macs=full_work,plans=plans,subsets=r,whole_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    p=argparse.ArgumentParser();p.add_argument('--excerpt-dir');p.add_argument('--output',required=True);p.add_argument('--profile',choices=('uniform','heterogeneous'));p.add_argument('--passes',type=int,choices=(6,12),default=6);p.add_argument('--only',action='append',help='repeat comma-separated subset to time');a=p.parse_args()
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');t=time.perf_counter()
    if a.profile is None:result=unit();source={'scope':'pure integer allocation oracle'}
    else:
        Base,Schedule,Config,builder,source=load_native(a.excerpt_dir);result=campaign(Base,Schedule,Config,builder,a.profile,a.passes,None if a.only is None else {tuple(map(int,x.split(','))) for x in a.only})
    out=dict(source=source,torch=torch.__version__,platform=platform.platform(),threads=torch.get_num_threads(),elapsed_seconds=time.perf_counter()-t,result=result,hashes={p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in ('admission.py','validate_fuse07.py','adjoint_cache.py','residual_cache.py','validate_fuse04.py')})
    Path(a.output).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({'output':a.output,'elapsed_seconds':out['elapsed_seconds']}),flush=True)
    if a.profile is not None:assert not any(s['failures'] or s['greedy_failures'] for s in result['subsets'])
if __name__=='__main__':main()
