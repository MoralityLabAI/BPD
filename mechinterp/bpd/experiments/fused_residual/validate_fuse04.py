"""FUSE-04 residual-cache validation, synthetic CPU only."""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, platform, statistics, sys, time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn
try:
    from .residual_cache import cached_residuals
except ImportError:
    from residual_cache import cached_residuals
ATOL, RTOL = 5e-5, 5e-4

def load_native(root):
    if root is None:
        from mechinterp.bpd.components import TiedComponentLinear, PassMaskSchedule
        from mechinterp.bpd.config import AblationConfig
        from mechinterp.bpd.adapters.trm import build_mini_trm
        return TiedComponentLinear, PassMaskSchedule, AblationConfig, build_mini_trm, {'scope':'native package imports'}
    root=Path(root); a=(root/'native_excerpt.py').read_text(); b=(root/'trm_excerpt.py').read_text()
    def fnv(s):
        h=2166136261
        for x in s.rstrip().encode(): h=((h^x)*16777619)&0xffffffff
        return f'{h:08x}'
    i=a.index('@dataclass\nclass PassMaskSchedule');j=a.index('    def forward(self, x: Tensor) -> Tensor:');k=b.index('def build_mini_trm(')
    pins=dict(wrapper=fnv(a[i:j]),forward=fnv(a[j:]),config=fnv(a[a.index('@dataclass(frozen=True)'):i]),trm=fnv(b[b.index('def rms_norm('):k]),builder=fnv(b[k:]))
    assert pins==dict(wrapper='4b49f9ab',forward='08fbc12f',config='a31fa61e',trm='4a80eb7d',builder='9943d0bb'),pins
    sys.path.insert(0,str(root))
    from native_excerpt import TiedComponentLinear, PassMaskSchedule, AblationConfig
    from trm_excerpt import build_mini_trm
    return TiedComponentLinear,PassMaskSchedule,AblationConfig,build_mini_trm,dict(scope='verified original native baseline excerpts ab93e4da; target intervention method omitted and not exercised',fnv1a=pins,sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'native_excerpt.py',root/'trm_excerpt.py')})

def close(a,b): return bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.allclose(a,b,atol=ATOL,rtol=RTOL))
def err(a,b): return float((a.detach()-b.detach()).abs().max())

def stress(Base,Schedule,Config):
    rows=[]; examples=[]; totals=dict(cases=0,eval_output_fail=0,train_output_fail=0,train_gradient_fail=0,gradient_checks=0,forward_checks=0,nonfinite=0)
    mx={k:0. for k in ('output','x','V','U','m','d')}; fails={k:0 for k in ('x','V','U','m','d')}
    cache=dict(eval_hits=0,eval_misses=0,train_hits=0,train_misses=0,reconstructions_saved=0)
    for scale,target,kind,cal in itertools.product((.1,1.,3.,10.,100.,10000.),('ordinary','exact','near'),('soft','zero','one','binary'),(False,True)):
        row=dict(scale=scale,target=target,mask=kind,cal=cal,cases=0,eval_output_fail=0,train_output_fail=0,train_gradient_fail=0,max_grad_abs=0.)
        for seed in range(3600001,3600009):
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
            def calc(mode,use_cache):
                train=mode=='train';q=copy.deepcopy(w);q.mode='component';q.reset_pass_state();q.record_inputs=True
                xx=x.clone().requires_grad_(train);mm=m.clone().requires_grad_(train);dd=None if d is None else d.clone().requires_grad_(train);q.schedule=Schedule(mm,dd)
                with torch.set_grad_enabled(train):
                    with (cached_residuals([q],mode=mode) if use_cache else nullcontext(None)) as stat:
                        yy=torch.stack([q(xx[i]) for i in range(3)])
                    assert q.calls==3 and len(q.recorded_inputs)==3 and 'weight_delta' not in q.__dict__
                    if stat is not None:assert stat.hits==2 and stat.misses==1 and stat.reconstructions==1 and stat.resident_bytes==0
                    g=torch.autograd.grad(yy,[xx,q.V,q.U,mm]+([] if dd is None else [dd]),cot) if train else ()
                return yy.detach(),[v.detach() for v in g],stat
            e0,_,_=calc('eval',False);e1,_,es=calc('eval',True);t0,g0,_=calc('train',False);t1,g1,ts=calc('train',True)
            finite=all(bool(torch.isfinite(v).all()) for v in [e0,e1,t0,t1,*g0,*g1]);totals['nonfinite']+=int(not finite)
            bad=[]
            for k,a,b in zip(['x','V','U','m']+([] if d is None else ['d']),g0,g1):
                z=err(a,b);mx[k]=max(mx[k],z);row['max_grad_abs']=max(row['max_grad_abs'],z)
                if not close(a,b):fails[k]+=1;bad.append(k)
            for k,flag in [('eval_output_fail',not torch.equal(e0,e1) or not finite),('train_output_fail',not torch.equal(t0,t1) or not finite),('train_gradient_fail',bool(bad))]:totals[k]+=int(flag);row[k]+=int(flag)
            mx['output']=max(mx['output'],err(e0,e1),err(t0,t1))
            if bad and len(examples)<16:examples.append(dict(scale=scale,target=target,mask=kind,cal=cal,seed=seed,failed_gradients=bad,max_gradient_abs=max(err(a,b) for a,b in zip(g0,g1))))
            totals['cases']+=1;row['cases']+=1;totals['gradient_checks']+=len(g0);totals['forward_checks']+=6
            cache['eval_hits']+=es.hits;cache['eval_misses']+=es.misses;cache['train_hits']+=ts.hits;cache['train_misses']+=ts.misses;cache['reconstructions_saved']+=4
        rows.append(row)
    return dict(counts=totals,gradient_failures=fails,max_abs=mx,cache=cache,examples=examples,rows=rows)

def lifecycle(Base,Schedule,Config):
    torch.manual_seed(3600901)
    def make():return Base(nn.Linear(16,16),4,max_passes=8,ablation=Config(mode='zero',sphere_rescale=False),init_svd=False)
    w=make();checks={}
    def check(k,v):checks[k]=bool(v);assert v,k
    def raises(fn):
        try:fn()
        except (RuntimeError,ValueError):return True
        return False
    with torch.no_grad():
        with cached_residuals([w],mode='eval') as s:
            a=w.weight_delta();b=w.weight_delta();check('eval_hit_identity',a is b)
            w.V.add_(.1);c=w.weight_delta();check('eval_version_invalidation',s.invalidations==1 and not torch.equal(a,c))
            w.U=nn.Parameter(w.U.detach().clone()+.1);d=w.weight_delta();check('eval_identity_invalidation',s.invalidations==2 and not torch.equal(c,d))
        check('scope_restored',s.resident_bytes==0 and 'weight_delta' not in w.__dict__)
    with cached_residuals([w],mode='train') as s:
        with torch.no_grad():a=w.weight_delta()
        b=w.weight_delta();c=w.weight_delta();check('no_grad_bypass',not a.requires_grad and b.requires_grad and b is c and s.bypasses==1)
    b.sum().backward();check('backward_after_exit',w.V.grad is not None and w.U.grad is not None)
    with cached_residuals([w],mode='train') as s:
        y=w.weight_delta();y.sum().backward();check('early_backward_detected',s.consumed and raises(w.weight_delta))
    with cached_residuals([w],mode='train'):
        w.weight_delta()
        with torch.no_grad():w.W_target.add_(.01)
        check('train_mutation_raises',raises(w.weight_delta))
    with cached_residuals([w],mode='train'):
        w.weight_delta();w.V=nn.Parameter(w.V.detach().clone());check('train_identity_raises',raises(w.weight_delta))
    def nested():
        with cached_residuals([w]):pass
    with cached_residuals([w]):check('nested_rejected',raises(nested))
    try:
        with cached_residuals([w]):raise RuntimeError('intentional')
    except RuntimeError:pass
    check('exception_restored','weight_delta' not in w.__dict__ and '_residual_cache_active' not in w.__dict__)
    w2=make();n=w.W_target.numel()*w.W_target.element_size()
    with torch.no_grad(),cached_residuals([w,w2],mode='eval',max_bytes=n) as s:
        w.weight_delta();w2.weight_delta();w2.weight_delta();w.weight_delta();check('aggregate_cap',s.peak_bytes==n and s.cap_rejections==2 and s.hits==1 and s.reconstructions==3)
    with torch.no_grad(),cached_residuals([w],mode='eval') as s:
        with torch.autocast('cpu',dtype=torch.bfloat16):w.weight_delta();w.weight_delta()
        check('autocast_bypass',s.bypasses==2 and s.misses==0)
        with torch.inference_mode():w.weight_delta();w.weight_delta()
        check('inference_bypass',s.bypasses==4 and s.misses==0)
    with torch.no_grad(),cached_residuals([w],mode='eval') as s:
        torch.set_float32_matmul_precision('medium')
        try:w.weight_delta();check('precision_bypass',s.bypasses==1 and s.misses==0)
        finally:torch.set_float32_matmul_precision('highest')
    w=make();opt=torch.optim.AdamW(w.parameters(),lr=.01)
    with cached_residuals([w],mode='train'):a=w.weight_delta()
    a.square().sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    with cached_residuals([w],mode='train') as s:b=w.weight_delta();check('fresh_after_update',s.misses==1 and not torch.equal(a,b))
    b.square().sum().backward()
    x=torch.randn(1,3,16);w=make();z=copy.deepcopy(w)
    with torch.no_grad():
        a=w(x)
        with cached_residuals([z],mode='eval') as s:b=z(x);check('target_no_residual',s.calls==0 and torch.equal(a,b))
    for model in (w,z):model.reset_pass_state();model.update_running_means=True
    with torch.no_grad():
        a=[w(x+i) for i in range(3)]
        with cached_residuals([z],mode='eval') as s:b=[z(x+i) for i in range(3)];check('calibration_eval_reuse',s.hits==2 and all(torch.equal(u,v) for u,v in zip(a,b)))
    check('calibration_buffers',all(torch.equal(getattr(w,k),getattr(z,k)) for k in ('mean_comp_acts','mean_delta_out','mean_write_rms','calibrated_passes')))
    return checks

def recurrent(Base,Schedule,Config,builder):
    results=[]
    for width in (64,128,256):
        torch.manual_seed(3601001+width)
        a=builder(hidden_size=width,num_heads=4,l_layers=2,h_cycles=1,l_cycles=2,puzzle_emb_len=4,seq_len=16,seed=3601001+width)
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
        def roll(model,cache,back):
            ws=wrappers(model)
            for w in ws:w.reset_pass_state()
            model.zero_grad(set_to_none=True);inp=x.detach().requires_grad_(back)
            with torch.set_grad_enabled(back):
                with (cached_residuals(ws,mode='train' if back else 'eval') if cache else nullcontext(None)) as stat:
                    h,l=model.init_states(2);states=[]
                    for _ in range(2):h,l=model.act_step(h,l,inp);states.extend((h,l,model.digit_logits(h)))
                    loss=states[-1].square().mean()
                if back:loss.backward()
            assert all(w.calls==6 for w in ws)
            if stat is not None:assert stat.hits==20 and stat.reconstructions==4 and stat.resident_bytes==0
            return states,loss,inp.grad,stat
        count=dict(forward_equal_checks=0,output_checks=0,gradient_checks=0,parameter_checks=0,greedy_checks=0,greedy_failures=0,tolerance_failures=0)
        maxima={k:0. for k in ('states','loss','input_grad','factor_grad','parameters')}
        def compare():
            sa,la,ga,_=roll(a,False,True);sb,lb,gb,stat=roll(b,True,True)
            def chk(u,v,k,kind):maxima[k]=max(maxima[k],err(u,v));count[kind]+=1;count['tolerance_failures']+=int(not close(u,v))
            for i,(u,v) in enumerate(zip(sa,sb)):
                count['forward_equal_checks']+=int(torch.equal(u,v));chk(u,v,'states','output_checks')
                if i%3==2:count['greedy_checks']+=1;count['greedy_failures']+=int(not torch.equal(u.argmax(-1),v.argmax(-1)))
            chk(la,lb,'loss','output_checks');chk(ga,gb,'input_grad','gradient_checks')
            for u,v in zip(a.parameters(),b.parameters()):
                if u.requires_grad:chk(u.grad,v.grad,'factor_grad','gradient_checks')
            return [float(la.detach()),float(lb.detach())],asdict(stat)
        initial,stats=compare();timings={}
        for back in (False,True):
            ts=[[],[]]
            for i in range(14):
                for j in ((0,1) if i%2==0 else (1,0)):
                    t=time.perf_counter_ns();roll((a,b)[j],bool(j),back);elapsed=(time.perf_counter_ns()-t)/1e6
                    if i>=3:ts[j].append(elapsed)
            timings['fwd_bwd' if back else 'forward']=dict(reference_ms=statistics.median(ts[0]),cached_ms=statistics.median(ts[1]),speedup=statistics.median(ts[0])/statistics.median(ts[1]),samples_ms=ts)
        opts=[torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4) for m in (a,b)];losses=[]
        for _ in range(8):
            loss,_=compare();losses.append(loss)
            for o in opts:o.step()
            for u,v in zip(a.parameters(),b.parameters()):
                if u.requires_grad:maxima['parameters']=max(maxima['parameters'],err(u,v));count['parameter_checks']+=1;count['tolerance_failures']+=int(not close(u,v))
        macs=sum(w.d_in*w.C*w.d_out for w in wrappers(a));results.append(dict(width=width,timing=timings,checks=count,max_abs=maxima,initial_loss=initial,losses=losses,cache_stats=stats,reconstruction_macs_reference=6*macs,reconstruction_macs_cache=macs))
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--excerpt-dir');p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');Base,Schedule,Config,builder,source=load_native(args.excerpt_dir)
    t=time.perf_counter();life=lifecycle(Base,Schedule,Config);s=stress(Base,Schedule,Config);r=recurrent(Base,Schedule,Config,builder)
    out=dict(source=source,torch=torch.__version__,platform=platform.platform(),threads=torch.get_num_threads(),precision=torch.get_float32_matmul_precision(),lifecycle=life,stress=s,recurrent=r,elapsed_seconds=time.perf_counter()-t,script_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'residual_cache.py':hashlib.sha256(Path(__file__).with_name('residual_cache.py').read_bytes()).hexdigest()})
    Path(args.output).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('stress','recurrent')},indent=2));print('stress',s['counts'],s['gradient_failures'],s['max_abs']);print('recurrent',[(x['width'],x['timing']['forward']['speedup'],x['timing']['fwd_bwd']['speedup'],x['checks']) for x in r])
if __name__=='__main__':main()
