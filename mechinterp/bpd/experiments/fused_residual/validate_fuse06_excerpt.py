"""FUSE-06 verified-excerpt tests; native full trainer has separate harness."""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, platform, resource, statistics, time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn
try:
    from .validate_fuse04 import load_native, close, err
    from .adjoint_cache import independent_residuals
    from .residual_cache import cached_residuals
except ImportError:
    from validate_fuse04 import load_native, close, err
    from adjoint_cache import independent_residuals
    from residual_cache import cached_residuals

CAP=16*1024**2


def retained(Base,Schedule,Config):
    checks=equal=failures=cases=failed=nonfinite=0; maximum=0.; examples=[]; rows=[]
    for scale,target in itertools.product((1.,10.,100.,10000.),('ordinary','exact','near')):
        row={'scale':scale,'target':target,'cases':0,'failures':0,'equal':0,'checks':0}
        for seed in range(3800001,3800009):
            torch.manual_seed(seed)
            w=Base(nn.Linear(64,64),32,max_passes=3,ablation=Config(mode='pass_mean',sphere_rescale=True),init_svd=False)
            with torch.no_grad():
                w.V.mul_(scale);w.U.mul_(scale)
                if target!='ordinary':w.W_target.copy_(w.reconstructed_weight()+(torch.randn_like(w.W_target)*1e-4 if target=='near' else 0.))
                x=torch.randn(2,3,1,8,64);cot=torch.randn_like(x);m=torch.randn(1,3,8,32);ds=torch.randn(2,1,3,8)
                for i in range(3):w.observe_clean_pass(x[0,i],i)
            def run(fast):
                q=copy.deepcopy(w);q.mode='component';xx=x.clone().requires_grad_();mm=m.clone().requires_grad_();dd=ds.clone().requires_grad_();out=[];losses=[];stats=[]
                for j in range(2):
                    q.reset_pass_state();q.schedule=Schedule((mm+0.25*j).sigmoid(),dd[j].sigmoid())
                    with (independent_residuals([q],variant='residual',max_bytes=CAP) if fast else nullcontext(None)) as st:
                        y=torch.stack([q(xx[j,i]) for i in range(3)])
                    assert q.calls==3 and 'weight_delta' not in q.__dict__
                    out.append(y);losses.append((y*cot[j]).sum());stats.append(st)
                partial=torch.autograd.grad(losses[1],(dd,mm),retain_graph=True)
                # Two disjoint forward graphs and a third, native parameter loss.
                loss=losses[0]+losses[1]+q.weight_delta().square().mean()*0.01
                grads=torch.autograd.grad(loss,(xx,q.V,q.U,mm,dd))
                if fast:assert all(s.hits==2 and s.misses==1 and s.resident_bytes==0 for s in stats)
                return [v.detach() for v in (*out,*partial,*grads)], [s.backward_nodes for s in stats] if fast else []
            a,_=run(False);b,nodes=run(True);bad=[]
            for name,u,v in zip(('output0','output1','partial_d','partial_m','x','V','U','m','d'),a,b):
                checks+=1;row['checks']+=1;eq=torch.equal(u,v);equal+=int(eq);row['equal']+=int(eq)
                finite=bool(torch.isfinite(u).all() and torch.isfinite(v).all());nonfinite+=int(not finite)
                z=err(u,v);maximum=max(maximum,z)
                if not close(u,v):failures+=1;bad.append(name)
            cases+=1;row['cases']+=1;failed+=int(bool(bad));row['failures']+=int(bool(bad))
            if bad and len(examples)<12:examples.append(dict(seed=seed,scale=scale,target=target,failed=bad,backward_nodes=nodes))
        rows.append(row)
    return dict(cases=cases,checks=checks,exact=equal,failed_cases=failed,failed_tensors=failures,nonfinite_tensors=nonfinite,max_abs=maximum,examples=examples,rows=rows)


def recurrent(Base,Schedule,Config,builder,width,passes):
    seed=3801001+width+passes;torch.manual_seed(seed)
    a=builder(hidden_size=width,num_heads=4,l_layers=2,h_cycles=1,l_cycles=2,puzzle_emb_len=4,seq_len=16,seed=seed)
    for p in a.parameters():p.requires_grad_(False)
    def wrappers(model):return [w for w in model.modules() if isinstance(w,Base)]
    for block in a.layers:
        for name in ('o_proj','down_proj'):
            setattr(block,name,Base(getattr(block,name),min(128,width//4),max_passes=passes,ablation=Config(mode='pass_mean',sphere_rescale=True),init_svd=False))
    x=a.encode_inputs(torch.randint(1,11,(2,16))).detach()
    for w in wrappers(a):w.update_running_means=True
    with torch.no_grad():
        h,l=a.init_states(2)
        for _ in range(passes//3):h,l=a.act_step(h,l,x)
    for w in wrappers(a):w.update_running_means=False;w.mode='component';w.schedule=Schedule(torch.rand(2,passes,20,w.C),torch.rand(2,passes,20));w.reset_pass_state()
    b=copy.deepcopy(a)
    def roll(model,fast,back):
        ws=wrappers(model)
        for w in ws:w.reset_pass_state()
        model.zero_grad(set_to_none=True);inp=x.detach().requires_grad_(back)
        with torch.set_grad_enabled(back):
            scope=(independent_residuals(ws,variant='residual',max_bytes=CAP) if back else cached_residuals(ws,mode='eval',max_bytes=CAP)) if fast else nullcontext(None)
            with scope as st:
                h,l=model.init_states(2);states=[]
                for _ in range(passes//3):h,l=model.act_step(h,l,inp);states.extend((h,l,model.digit_logits(h)))
                loss=states[-1].square().mean()
            if back:loss.backward()
        assert all(w.calls==passes for w in ws)
        if st is not None:assert st.calls==4*passes and st.reconstructions+st.hits==st.calls and st.resident_bytes==0 and st.peak_bytes<=CAP
        return [z.detach() for z in states],loss.detach(),inp.grad,st
    count=dict(tensors=0,exact=0,failures=0,greedy=0,greedy_failures=0);mx={k:0. for k in ('states','loss','input','factor','parameters')}
    def check(u,v,k):
        count['tensors']+=1;count['exact']+=int(torch.equal(u,v));count['failures']+=int(not close(u,v));mx[k]=max(mx[k],err(u,v))
    def compare():
        sa,la,ga,_=roll(a,False,True);sb,lb,gb,st=roll(b,True,True)
        for i,(u,v) in enumerate(zip(sa,sb)):
            check(u,v,'states')
            if i%3==2:count['greedy']+=1;count['greedy_failures']+=int(not torch.equal(u.argmax(-1),v.argmax(-1)))
        check(la,lb,'loss');check(ga,gb,'input')
        for u,v in zip(a.parameters(),b.parameters()):
            if u.requires_grad:check(u.grad,v.grad,'factor')
        return [float(la),float(lb)],asdict(st)
    first,st=compare();timing={}
    for back in (False,True):
        ts=[[],[]]
        for i in range(7):
            for j in ((0,1) if i%2==0 else (1,0)):
                t=time.perf_counter_ns();roll((a,b)[j],bool(j),back);ms=(time.perf_counter_ns()-t)/1e6
                if i>=2:ts[j].append(ms)
        timing['fwd_bwd' if back else 'forward']=dict(reference_ms=statistics.median(ts[0]),cached_ms=statistics.median(ts[1]),speedup=statistics.median(ts[0])/statistics.median(ts[1]),samples_ms=ts)
    opts=[torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4) for model in (a,b)];losses=[]
    for _ in range(3):
        l,_=compare();losses.append(l)
        for o in opts:o.step()
        for u,v in zip(a.parameters(),b.parameters()):
            if u.requires_grad:check(u,v,'parameters')
    # Admission order is actual native execution order; here all four sites are called per invocation.
    used=0;saved=0;total=0
    for w in wrappers(a):
        n=w.W_target.numel()*4;mac=w.d_in*w.C*w.d_out;total+=passes*mac
        if used+n<=CAP:used+=n;saved+=(passes-1)*mac
    return dict(width=width,passes=passes,seed=seed,initial_loss=first,checks=count,max_abs=mx,cache_stats=st,timing=timing,losses=losses,reference_reconstruction_macs=total,cached_reconstruction_macs=total-saved,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    p=argparse.ArgumentParser();p.add_argument('--excerpt-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--width',type=int,choices=(256,512,1024));p.add_argument('--passes',type=int,choices=(6,12));a=p.parse_args()
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    Base,Schedule,Config,builder,source=load_native(a.excerpt_dir);t=time.perf_counter()
    result=retained(Base,Schedule,Config) if a.width is None else recurrent(Base,Schedule,Config,builder,a.width,a.passes)
    out=dict(source=source,torch=torch.__version__,platform=platform.platform(),threads=torch.get_num_threads(),cap_bytes=CAP,result=result,elapsed_seconds=time.perf_counter()-t,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(a.output).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
