"""FUSE-02 CPU native validation. --excerpt-dir uses pinned local source extracts."""
import argparse, copy, itertools, json, statistics, sys, time
import torch
from torch import nn


def run(TiedComponentLinear, PassMaskSchedule, AblationConfig, build_mini_trm):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(2802)
    counts = {}; maxima = {}
    def check(a,b,key,atol=5e-5,rtol=5e-4):
        counts[key]=counts.get(key,0)+1
        maxima[key]=max(maxima.get(key,0.),float((a.detach()-b.detach()).abs().max()))
        torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
    def pair(w,x,m,d,r,offset=0,update=False):
        z=copy.deepcopy(w); z.use_fused_residual=True
        x1=x.clone().requires_grad_(); x2=x.clone().requires_grad_()
        m1=m.clone().requires_grad_(); m2=m.clone().requires_grad_()
        d1=None if d is None else d.clone().requires_grad_()
        d2=None if d is None else d.clone().requires_grad_()
        for q,mm,dd in ((w,m1,d1),(z,m2,d2)):
            q.mode='component'; q.schedule=PassMaskSchedule(mm,dd,r)
            q.reset_pass_state(pass_offset=offset); q.record_inputs=True; q.update_running_means=update
        a=w(x1); b=z(x2); check(a,b,'wrapper_output')
        g=torch.randn_like(a)
        p=[x1,w.V,w.U,m1]+([] if d1 is None else [d1]); q=[x2,z.V,z.U,m2]+([] if d2 is None else [d2])
        for u,v in zip(torch.autograd.grad(a,p,g),torch.autograd.grad(b,q,g)):check(u,v,'wrapper_gradient')
        assert w.calls==z.calls==1 and len(w.recorded_inputs)==len(z.recorded_inputs)==1
        assert list(w.state_dict())==list(z.state_dict())
        for k in ('mean_comp_acts','mean_delta_out','mean_write_rms'):check(getattr(w,k),getattr(z,k),'calibration')
        assert torch.equal(w.calibrated_passes,z.calibrated_passes)
        return z.fused_residual_calls,z.fused_residual_fallbacks
    hits=falls=0
    for mode,sphere,kind,delta,bias,route,update in itertools.product(('zero','pass_mean'),(False,True),('soft','binary','zero','one'),(False,True),(False,True),(False,True),(False,True)):
        w=TiedComponentLinear(nn.Linear(24,32,bias=bias),8,max_passes=4,ablation=AblationConfig(mode=mode,sphere_rescale=sphere),init_svd=False)
        x=torch.randn(2,5,24); m=torch.rand(2,1,5,8)
        if kind=='binary':m=(m>.5).float()
        elif kind=='zero':m.zero_()
        elif kind=='one':m.fill_(1.)
        w.observe_clean_pass(torch.randn_like(x),2)
        h,f=pair(w,x,m,torch.rand(2,1,5) if delta else None,(torch.rand(2,1,5)>.4) if route else None,2,update)
        hits+=h; falls+=f
    # Explicit fallback and severe cancellation, no assertion of equivalence on stress.
    w=TiedComponentLinear(nn.Linear(8,8,bias=False),8,max_passes=1,ablation=AblationConfig(mode='pass_mean',sphere_rescale=True),init_svd=False)
    with torch.no_grad():
        w.W_target.copy_(torch.eye(8)*.001); w.V.zero_(); w.U.zero_(); w.calibrated_passes.fill_(True); w.mean_write_rms.fill_(1.)
    w.mode='component'; w.schedule=PassMaskSchedule(torch.ones(1,1,1,8)); w.use_fused_residual=True
    w(torch.ones(1,1,8)); assert w.fused_residual_fallbacks==1
    w.reset_pass_state()
    with torch.autocast('cpu',dtype=torch.bfloat16):w(torch.ones(1,1,8))
    assert w.fused_residual_fallbacks==2
    stress=[]
    for scale in (1.,100.,10000.):
        w=TiedComponentLinear(nn.Linear(64,64,bias=False),32,ablation=AblationConfig(mode='zero',sphere_rescale=False),init_svd=False)
        with torch.no_grad():
            w.V.mul_(scale);w.U.mul_(scale);w.W_target.copy_(w.reconstructed_weight())
        w.mode='component';w.schedule=PassMaskSchedule(torch.zeros(1,1,8,32));x=torch.randn(1,8,64)
        with torch.no_grad():
            a=w(x);w.reset_pass_state();w.use_fused_residual=True;b=w(x)
        stress.append({'factor_scale':scale,'max_abs':float((a-b).abs().max()),'close':bool(torch.allclose(a,b,atol=5e-5,rtol=5e-4)),'fallbacks':w.fused_residual_fallbacks})
    results=[]
    for width in (64,128,256):
        a=build_mini_trm(hidden_size=width,num_heads=4,l_layers=2,h_cycles=1,l_cycles=2,puzzle_emb_len=4,seq_len=16,seed=2802+width)
        for p in a.parameters():p.requires_grad_(False)
        for block in a.layers:
            for name in ('o_proj','down_proj'):
                setattr(block,name,TiedComponentLinear(getattr(block,name),min(64,width//4),max_passes=6,ablation=AblationConfig(mode='pass_mean',sphere_rescale=True),init_svd=False))
        def wrappers(model):return [q for q in model.modules() if isinstance(q,TiedComponentLinear)]
        x=a.encode_inputs(torch.randint(1,11,(2,16))).detach()
        for q in wrappers(a):q.update_running_means=True
        with torch.no_grad():
            zh,zl=a.init_states(2)
            for _ in range(2):zh,zl=a.act_step(zh,zl,x)
        for q in wrappers(a):
            q.update_running_means=False;q.mode='component';q.schedule=PassMaskSchedule(torch.rand(2,6,20,q.C),torch.rand(2,6,20));q.reset_pass_state()
        b=copy.deepcopy(a)
        for q in wrappers(b):q.use_fused_residual=True
        def rollout(model,inp,back=False):
            for q in wrappers(model):q.reset_pass_state()
            model.zero_grad(set_to_none=True)
            inp=inp.detach().requires_grad_(back)
            zh,zl=model.init_states(2); states=[]
            for _ in range(2):
                zh,zl=model.act_step(zh,zl,inp);states.extend((zh,zl,model.digit_logits(zh)))
            loss=states[-1].square().mean()
            if back:loss.backward()
            assert all(q.calls==6 for q in wrappers(model))
            return states,loss,inp.grad
        def compare(back=True):
            sa,la,ga=rollout(a,x,back);sb,lb,gb=rollout(b,x,back)
            for i,(u,v) in enumerate(zip(sa,sb)):
                check(u,v,'trm_state')
                if i%3==2:assert torch.equal(u.argmax(-1),v.argmax(-1))
            check(la,lb,'loss')
            if back:
                check(ga,gb,'input_gradient')
                for u,v in zip(a.parameters(),b.parameters()):
                    if u.requires_grad:check(u.grad,v.grad,'factor_gradient')
            return float(la),float(lb)
        compare()
        timings={}
        for back in (False,True):
            ts=[[],[]]
            for i in range(14):
                for j in ((0,1) if i%2==0 else (1,0)):
                    model=(a,b)[j];t=time.perf_counter()
                    with torch.set_grad_enabled(back):rollout(model,x,back)
                    if i>=3:ts[j].append(time.perf_counter()-t)
            timings['fwd_bwd' if back else 'fwd']={'reference_ms':1000*statistics.median(ts[0]),'fused_ms':1000*statistics.median(ts[1]),'speedup':statistics.median(ts[0])/statistics.median(ts[1])}
        opts=[torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4) for model in (a,b)]
        losses=[]
        for _ in range(8):
            losses.append(compare())
            for opt in opts:opt.step()
            for u,v in zip(a.parameters(),b.parameters()):
                if u.requires_grad:check(u,v,'updated_parameter')
        results.append({'width':width,'timing':timings,'losses':losses,'fused_calls':sum(q.fused_residual_calls for q in wrappers(b)),'fallbacks':sum(q.fused_residual_fallbacks for q in wrappers(b))})
    return {'torch':torch.__version__,'threads':torch.get_num_threads(),'checks':counts,'max_abs':maxima,'wrapper_hits':hits,'wrapper_fallbacks':falls,'explicit_fallbacks':2,'stress':stress,'trm':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--excerpt-dir');args=parser.parse_args()
    if args.excerpt_dir:
        sys.path.insert(0,args.excerpt_dir)
        from native_excerpt import TiedComponentLinear, PassMaskSchedule, AblationConfig
        from trm_excerpt import build_mini_trm
    else:
        from mechinterp.bpd.components import TiedComponentLinear, PassMaskSchedule
        from mechinterp.bpd.config import AblationConfig
        from mechinterp.bpd.adapters.trm import build_mini_trm
    print(json.dumps(run(TiedComponentLinear, PassMaskSchedule, AblationConfig, build_mini_trm),indent=2))
