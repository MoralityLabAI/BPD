"""FUSE-05: registered first-order cached-primal comparison."""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, platform, statistics, time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn
try:
    from .validate_fuse04 import load_native, close, err, lifecycle as prior_lifecycle
    from .residual_cache import cached_residuals
    from .adjoint_cache import independent_residuals
except ImportError:
    from validate_fuse04 import load_native, close, err, lifecycle as prior_lifecycle
    from residual_cache import cached_residuals
    from adjoint_cache import independent_residuals

ARMS=('shared','residual','product','replay')


def scope(ws,mode,arm):
    if arm=='ref':return nullcontext(None)
    if arm=='shared' or mode=='eval':return cached_residuals(ws,mode=mode)
    return independent_residuals(ws,variant=arm)


def stress(Base,Schedule,Config,arm):
    total=dict(cases=0,output_fail=0,gradient_fail=0,nonfinite=0,forward_checks=0,grad_checks=0,bitwise_grad_checks=0,eval_fail=0,eval_checks=0,hits=0,reconstructions=0,backward_nodes=0,backward_replays=0)
    types=('x','V','U','m','d');fails={k:0 for k in types};mx={k:0. for k in ('output',*types)};examples=[];rows=[]
    for scale,target,kind,cal in itertools.product((.1,1.,3.,10.,100.,10000.),('ordinary','exact','near'),('soft','zero','one','binary'),(False,True)):
        row=dict(scale=scale,target=target,mask=kind,cal=cal,cases=0,gradient_fail=0,output_fail=0)
        for seed in range(3700001,3700009):
            torch.manual_seed(seed)
            w=Base(nn.Linear(64,64,bias=True),32,max_passes=3,ablation=Config(mode='pass_mean' if cal else 'zero',sphere_rescale=cal),init_svd=False)
            with torch.no_grad():
                w.V.mul_(scale);w.U.mul_(scale)
                if target!='ordinary':w.W_target.copy_(w.reconstructed_weight()+(torch.randn_like(w.W_target)*1e-4 if target=='near' else 0.))
                x=torch.randn(3,1,8,64);m=torch.rand(1,3,8,32)
                if kind=='zero':m.zero_()
                elif kind=='one':m.fill_(1.)
                elif kind=='binary':m=(m>.5).float()
                d=torch.rand(1,3,8) if cal else None
                if cal:
                    for i in range(3):w.observe_clean_pass(torch.randn_like(x[i]),i)
                cot=torch.randn(3,1,8,64)
            def calc(mode,which):
                train=mode=='train';q=copy.deepcopy(w);q.mode='component';q.record_inputs=True;q.reset_pass_state()
                xx=x.clone().requires_grad_(train);mm=m.clone().requires_grad_(train);dd=None if d is None else d.clone().requires_grad_(train);q.schedule=Schedule(mm,dd)
                with torch.set_grad_enabled(train):
                    with scope([q],mode,which) as st:yy=torch.stack([q(xx[i]) for i in range(3)])
                    assert q.calls==3 and len(q.recorded_inputs)==3 and 'weight_delta' not in q.__dict__
                    if st is not None:assert (st.hits,st.misses,st.reconstructions,st.resident_bytes)==(2,1,1,0)
                    g=torch.autograd.grad(yy,[xx,q.V,q.U,mm]+([] if dd is None else [dd]),cot) if train else ()
                return yy.detach(),[v.detach() for v in g],st
            if arm=='shared':
                e0,_,_=calc('eval','ref');e1,_,_=calc('eval','shared');total['eval_checks']+=3;total['eval_fail']+=int(not torch.equal(e0,e1))
            t0,g0,_=calc('train','ref');t,g,st=calc('train',arm);bad=[]
            finite=all(bool(torch.isfinite(v).all()) for v in [t0,t,*g0,*g]);outbad=not torch.equal(t0,t)
            for k,u,v in zip(types,g0,g):
                mx[k]=max(mx[k],err(u,v));total['bitwise_grad_checks']+=int(torch.equal(u,v));total['grad_checks']+=1
                if not close(u,v):fails[k]+=1;bad.append(k)
            mx['output']=max(mx['output'],err(t0,t));total['nonfinite']+=int(not finite);total['forward_checks']+=3
            for b in (total,row):b['cases']+=1;b['gradient_fail']+=int(bool(bad));b['output_fail']+=int(outbad)
            for k in ('hits','reconstructions','backward_nodes','backward_replays'):total[k]+=getattr(st,k,0)
            if bad and len(examples)<12:examples.append(dict(scale=scale,target=target,mask=kind,cal=cal,seed=seed,failed=bad,max_error=max(err(u,v) for u,v in zip(g0,g))))
        rows.append(row)
    return dict(counts=total,fail_types=fails,max_abs=mx,examples=examples,rows=rows)


def lifecycle(Base,Schedule,Config,arm):
    if arm=='shared':return prior_lifecycle(Base,Schedule,Config)
    torch.manual_seed(3700901);checks={}
    def make():return Base(nn.Linear(16,16),4,max_passes=4,ablation=Config(mode='zero',sphere_rescale=False),init_svd=False)
    def check(k,v):checks[k]=bool(v);assert v,(arm,k)
    def raises(fn):
        try:fn()
        except (RuntimeError,ValueError):return True
        return False
    def ctx(ws,**kw):return independent_residuals(ws,variant=arm,**kw)
    w=make()
    with ctx([w]) as s:
        with torch.no_grad():a=w.weight_delta()
        b=w.weight_delta();c=w.weight_delta();check('independent_nodes',b.grad_fn is not c.grad_fn and torch.equal(b,c));check('no_grad_bypass',not a.requires_grad and b.requires_grad and s.bypasses==1)
    (b+c).sum().backward();check('backward_after_exit',s.backward_nodes==2 and s.resident_bytes==0 and 'weight_delta' not in w.__dict__)
    with ctx([w]) as s:
        w.weight_delta().sum().backward();check('early_backward',s.consumed and raises(w.weight_delta))
    with ctx([w]):
        w.weight_delta()
        with torch.no_grad():w.V.add_(.1)
        check('mutation_rejected',raises(w.weight_delta) and raises(w.weight_delta))
    with ctx([w]):
        w.weight_delta();w.U=nn.Parameter(w.U.detach().clone());check('identity_rejected',raises(w.weight_delta))
    def nested():
        with ctx([w]):pass
    with ctx([w]):check('nest_rejected',raises(nested))
    try:
        with ctx([w]):raise RuntimeError('intentional')
    except RuntimeError:pass
    check('exception_restored','weight_delta' not in w.__dict__ and '_residual_cache_active' not in w.__dict__)
    w2=make();n=w.W_target.numel()*4
    with ctx([w,w2],max_bytes=n) as s:
        w.weight_delta();w2.weight_delta();w.weight_delta();check('cap',s.peak_bytes==n and s.cap_rejections==1 and s.hits==1)
    with ctx([w]) as s:
        with torch.autocast('cpu',dtype=torch.bfloat16):w.weight_delta()
        check('autocast_bypass',s.bypasses==1 and s.misses==0)
        with torch.inference_mode():w.weight_delta()
        check('inference_bypass',s.bypasses==2)
    with ctx([w]) as s:
        torch.set_float32_matmul_precision('medium')
        try:w.weight_delta();check('precision_bypass',s.bypasses==1 and s.misses==0)
        finally:torch.set_float32_matmul_precision('highest')
    w.V=nn.Parameter(w.V.detach().T.contiguous().T);assert not w.V.is_contiguous()
    with ctx([w]) as s:w.weight_delta();check('layout_bypass',s.bypasses==1 and s.misses==0)
    w=make();w.V.requires_grad_(False);z=copy.deepcopy(w);g=torch.randn_like(w.W_target)
    a=w.weight_delta();ga=torch.autograd.grad(a,w.U,g)[0]
    with ctx([z]):b=z.weight_delta()
    gb=torch.autograd.grad(b,z.U,g)[0];check('frozen_factor',torch.equal(a,b) and close(ga,gb))
    w=make();opt=torch.optim.AdamW(w.parameters(),lr=.01)
    with ctx([w]):a=w.weight_delta()
    a.square().sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    with ctx([w]) as s:b=w.weight_delta();check('fresh_after_update',s.misses==1 and not torch.equal(a,b))
    b.square().sum().backward()
    return checks


def recurrent(Base,Schedule,Config,builder,arm):
    rows=[]
    for width in (64,128,256):
        seed=3701001+width;torch.manual_seed(seed)
        a=builder(hidden_size=width,num_heads=4,l_layers=2,h_cycles=1,l_cycles=2,puzzle_emb_len=4,seq_len=16,seed=seed)
        for p in a.parameters():p.requires_grad_(False)
        def wrappers(model):return [w for w in model.modules() if isinstance(w,Base)]
        for block in a.layers:
            for name in ('o_proj','down_proj'):
                setattr(block,name,Base(getattr(block,name),min(64,width//4),max_passes=6,ablation=Config(mode='pass_mean',sphere_rescale=True),init_svd=False))
        x=a.encode_inputs(torch.randint(1,11,(2,16))).detach()
        for w in wrappers(a):w.update_running_means=True
        with torch.no_grad():
            h,l=a.init_states(2)
            for _ in range(2):h,l=a.act_step(h,l,x)
        for w in wrappers(a):w.update_running_means=False;w.mode='component';w.schedule=Schedule(torch.rand(2,6,20,w.C),torch.rand(2,6,20));w.reset_pass_state()
        b=copy.deepcopy(a)
        def roll(model,which,back):
            ws=wrappers(model)
            for w in ws:w.reset_pass_state()
            model.zero_grad(set_to_none=True);inp=x.detach().requires_grad_(back)
            with torch.set_grad_enabled(back):
                with scope(ws,'train' if back else 'eval',which) as st:
                    h,l=model.init_states(2);states=[]
                    for _ in range(2):h,l=model.act_step(h,l,inp);states.extend((h,l,model.digit_logits(h)))
                    loss=states[-1].square().mean()
                if back:loss.backward()
            assert all(w.calls==6 for w in ws)
            if st is not None:assert (st.hits,st.reconstructions,st.resident_bytes)==(20,4,0)
            return states,loss,inp.grad,st
        count=dict(tensors=0,bitwise=0,failures=0,greedy=0,greedy_failures=0);mx={k:0. for k in ('states','loss','input','factor','parameters')}
        def chk(u,v,k):count['tensors']+=1;count['bitwise']+=int(torch.equal(u,v));count['failures']+=int(not close(u,v));mx[k]=max(mx[k],err(u,v))
        def compare():
            sa,la,ga,_=roll(a,'ref',True);sb,lb,gb,st=roll(b,arm,True)
            for i,(u,v) in enumerate(zip(sa,sb)):
                chk(u,v,'states')
                if i%3==2:count['greedy']+=1;count['greedy_failures']+=int(not torch.equal(u.argmax(-1),v.argmax(-1)))
            chk(la,lb,'loss');chk(ga,gb,'input')
            for u,v in zip(a.parameters(),b.parameters()):
                if u.requires_grad:chk(u.grad,v.grad,'factor')
            return [float(la.detach()),float(lb.detach())],asdict(st)
        first,stats=compare();timing={}
        for back in (False,True):
            ts=[[],[]]
            for i in range(14):
                for j in ((0,1) if i%2==0 else (1,0)):
                    t=time.perf_counter_ns();roll((a,b)[j],('ref',arm)[j],back);ms=(time.perf_counter_ns()-t)/1e6
                    if i>=3:ts[j].append(ms)
            timing['fwd_bwd' if back else 'forward']=dict(reference_ms=statistics.median(ts[0]),cached_ms=statistics.median(ts[1]),speedup=statistics.median(ts[0])/statistics.median(ts[1]),samples_ms=ts)
        opts=[torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4) for model in (a,b)];losses=[]
        for _ in range(8):
            l,_=compare();losses.append(l)
            for o in opts:o.step()
            for u,v in zip(a.parameters(),b.parameters()):
                if u.requires_grad:chk(u,v,'parameters')
        rows.append(dict(width=width,arm=arm,initial_loss=first,checks=count,max_abs=mx,cache_stats=stats,timing=timing,losses=losses))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--excerpt-dir');p.add_argument('--output',required=True);p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    Base,Schedule,Config,builder,source=load_native(a.excerpt_dir);start=time.perf_counter();life=lifecycle(Base,Schedule,Config,a.arm);s=stress(Base,Schedule,Config,a.arm);r=recurrent(Base,Schedule,Config,builder,a.arm)
    out=dict(arm=a.arm,source=source,torch=torch.__version__,platform=platform.platform(),threads=torch.get_num_threads(),precision=torch.get_float32_matmul_precision(),lifecycle=life,stress=s,recurrent=r,elapsed_seconds=time.perf_counter()-start,script_sha256={k:hashlib.sha256(Path(__file__).with_name(k).read_bytes()).hexdigest() for k in ('validate_fuse05.py','adjoint_cache.py','validate_fuse04.py','residual_cache.py')})
    Path(a.output).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(arm=a.arm,life=len(life),stress=s['counts'],fails=s['fail_types'],max_abs=s['max_abs'],recurrent=[dict(width=q['width'],checks=q['checks'],time=q['timing']['fwd_bwd']['speedup']) for q in r],elapsed=out['elapsed_seconds']),indent=2),flush=True)
if __name__=='__main__':main()
