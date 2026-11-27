"""FUSE-03 experimental numerical guard. No change to production defaults."""
import argparse, copy, itertools, json, math, platform, sys
import torch
import torch.nn.functional as F
ATOL,RTOL=5e-5,5e-4
U,ETA=2.**-24,2.**-126

def mag(t):return float(t.detach().abs().amax())
def up(x):return math.nextafter(x*(1+1e-12)+1e-300,math.inf)
def inp(t):return up(mag(t)),0.
def add(a,b):
    A,E=a;B,G=b;return up(A+B),up(E+G+U*(A+B+E+G)+ETA)
def mul(a,b):
    A,E=a;B,G=b;return up(A*B),up(A*G+B*E+E*G+U*(A+E)*(B+G)+ETA)
def dot(a,b,k):
    A,E=a;B,G=b;g=2*k*U
    if g>=.5:return math.inf,math.inf
    return up(k*A*B),up(k*(A*G+B*E+E*G)+g/(1-g)*k*(A+E)*(B+G)+2*k*ETA/(1-g))
def envelope(x,w,m,d,cal,tau):
    """Uniform conditional roundoff envelope before sphere; masks in [0,1]."""
    X,V,Q,W=map(inp,(x,w.V,w.U,w.W_target));M=inp(m);D=(1.,0.) if d is None else inp(d);one=(1.,0.)
    A=dot(X,V,w.d_in);Y=dot(X,W,w.d_in);r=mul(M,A);z=mul((1.,add(M,D)[1]),A)
    if cal:
        q=mul((1.,add(one,M)[1]),inp(w.mean_comp_acts[tau]));r=add(r,q);z=add(z,q)
    residual=mul(D,dot(X,add(W,dot(V,Q,w.C)),w.d_in));f=add(mul(D,Y),dot(z,Q,w.C))
    if cal and d is not None:
        q=mul((1.,add(one,D)[1]),inp(w.mean_delta_out[tau]));residual=add(residual,q);f=add(f,q)
    r=add(dot(r,Q,w.C),residual)
    if w.bias is not None:
        bias=inp(w.bias);r=add(r,bias);f=add(f,bias);Y=add(Y,bias)
    return max(up(r[1]+f[1]),up(2*Y[1])) if max(r[0]+r[1],f[0]+f[1],Y[0]+Y[1])<1e30 else math.inf

def guarded_class(Base):
    class Guarded(Base):
        guard_mode='heuristic';last_bound=None
        def forward(self,x):
            self.last_bound=None
            if not self.use_fused_residual:return super().forward(x)
            def fallback():
                self.fused_residual_fallbacks+=1;self.use_fused_residual=False
                try:return super(Guarded,self).forward(x)
                finally:self.use_fused_residual=True
            if self.mode!='component' or self.schedule is None or self.update_running_means:return fallback()
            k=self.calls;tau=self.pass_offset+k;m=self.schedule.comp_mask[:,k];dm=self.schedule.delta_mask;d=None if dm is None else dm[:,k].unsqueeze(-1)
            if x.device.type!='cpu' or x.dtype!=torch.float32 or any(t.dtype!=torch.float32 for t in (self.V,self.U,self.W_target,m)) or (d is not None and d.dtype!=torch.float32) or torch.is_autocast_enabled('cpu') or torch.get_float32_matmul_precision()!='highest':return fallback()
            cal=self.ablation.mode=='pass_mean' and tau<self.mean_comp_acts.shape[0] and bool(self.calibrated_passes[tau]);sphere=cal and self.ablation.sphere_rescale
            if self.guard_mode=='heuristic':
                if not (mag(self.V)<=2 and mag(self.U)<=2 and mag(x)<=16):return fallback()
            else:
                if sphere or not bool(((m.detach()>=0)&(m.detach()<=1)).all()) or (d is not None and not bool(((d.detach()>=0)&(d.detach()<=1)).all())):return fallback()
                self.last_bound=envelope(x,self,m,d,cal,tau)
                if not math.isfinite(self.last_bound):return fallback()
            a=x@self.V;y=F.linear(x,self.W_target);dd=1. if d is None else d;z=(m-dd)*a
            if cal:z=z+(1.-m)*self.mean_comp_acts[tau]
            left=dd*y;right=z@self.U;out=left+right;mean=None
            if cal and d is not None:mean=(1.-d)*self.mean_delta_out[tau];out=out+mean
            with torch.no_grad():
                reject=not bool(torch.isfinite(out).all())
                if self.guard_mode=='heuristic':
                    amp=left.abs()+right.abs()
                    if mean is not None:amp=amp+mean.abs()
                    reject=reject or bool((amp.amax(-1)>64*(out.abs().amax(-1)+1e-3)).any())
                    if sphere:
                        ms=out.square().mean(-1,keepdim=True);eps=self.ablation.eps;reject=reject or bool(((ms-eps).abs()<=.05*eps).any() or not torch.isfinite(ms).all())
            if reject:return fallback()
            if sphere:
                eps=self.ablation.eps;frac=(1.-m.clamp(0.,1.)).mean(-1,keepdim=True);ms=out.square().mean(-1,keepdim=True);rms=(ms+eps).sqrt();desired=(1.-frac)*rms+frac*self.mean_write_rms[tau].clamp_min(eps);out=out*torch.where(ms>eps,desired/rms,torch.ones_like(rms))
            if self.bias is not None:out=out+self.bias.float();y=y+self.bias.float()
            if self.schedule.routing_mask is not None:out=torch.where(self.schedule.routing_mask[:,k].unsqueeze(-1),out,y)
            if not bool(torch.isfinite(out.detach()).all()):return fallback()
            if self.guard_mode=='envelope':
                E=self.last_bound;small=float(out.detach().abs().amin())
                if E>ATOL+RTOL*max(0.,small-E):return fallback()
            self.calls+=1
            if self.record_inputs:self.recorded_inputs.append(x.detach() if self.detach_recorded_inputs else x)
            self.fused_residual_calls+=1;return out.to(x.dtype)
    return Guarded

def stress(Base,Schedule,Config):
    Guarded=guarded_class(Base);modes=('unguarded','heuristic','envelope');rows=[];bounds={'checks':0,'violations':0,'max_ratio':0.};examples=[]
    def calc(w,x,m,d,cot,fast):
        w.mode='component';w.use_fused_residual=fast;w.reset_pass_state();xx=x.clone().requires_grad_();mm=m.clone().requires_grad_();dd=None if d is None else d.clone().requires_grad_();w.schedule=Schedule(mm,dd);y=w(xx);g=torch.autograd.grad(y,[xx,w.V,w.U,mm]+([] if dd is None else [dd]),cot);assert w.calls==1;return y.detach(),[v.detach() for v in g]
    def close(a,b):return bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL))
    def err(a,b):return float((a-b).abs().max())
    for scale,target,kind,cal in itertools.product((.1,1.,3.,10.,100.,10000.),('ordinary','exact','near'),('soft','zero','one','binary'),(False,True)):
        row=dict(scale=scale,target=target,mask=kind,calibrated_sphere=cal,cases=0)
        for mode in modes:row[mode]={k:0 for k in ('admitted','forward_fail','gradient_fail','admitted_fail','rejected_pass','fallback_nonidentical')}
        for seed in range(3500001,3500009):
            torch.manual_seed(seed);w=Base(torch.nn.Linear(64,64,bias=True),32,max_passes=1,ablation=Config(mode='pass_mean' if cal else 'zero',sphere_rescale=cal),init_svd=False)
            with torch.no_grad():
                w.V.mul_(scale);w.U.mul_(scale)
                if target!='ordinary':w.W_target.copy_(w.reconstructed_weight()+(torch.randn_like(w.W_target)*1e-4 if target=='near' else 0.))
                x=torch.randn(1,8,64);m=torch.rand(1,1,8,32)
                if kind=='zero':m.zero_()
                elif kind=='one':m.fill_(1.)
                elif kind=='binary':m=(m>.5).float()
                d=torch.rand(1,1,8) if cal else None
                if cal:w.observe_clean_pass(torch.randn_like(x),0)
                cot=torch.randn(1,8,64)
            g=copy.deepcopy(w);ref,rg=calc(w,x,m,d,cot,False);v,vg=calc(g,x,m,d,cot,True);uf,ug=not close(ref,v),not all(close(a,b) for a,b in zip(rg,vg))
            for mode in modes:
                a=row[mode]
                if mode=='unguarded':y,gr=v,vg;admit=g.fused_residual_calls>0
                else:
                    q=copy.deepcopy(w);q.__class__=Guarded;q.guard_mode=mode;y,gr=calc(q,x,m,d,cot,True);admit=q.fused_residual_calls>0
                    if mode=='envelope' and not cal and q.last_bound is not None:
                        E=q.last_bound;actual=err(ref,v);bounds['checks']+=1;bounds['violations']+=int(actual>E);bounds['max_ratio']=max(bounds['max_ratio'],actual/E if E else 0.)
                    if not admit:a['fallback_nonidentical']+=int(not torch.equal(ref,y) or not all(torch.equal(u,vv) for u,vv in zip(rg,gr)))
                ff,gg=not close(ref,y),not all(close(u,vv) for u,vv in zip(rg,gr));a['admitted']+=int(admit);a['forward_fail']+=int(ff);a['gradient_fail']+=int(gg);a['admitted_fail']+=int(admit and (ff or gg));a['rejected_pass']+=int(not admit and not uf and not ug)
                if mode!='unguarded' and admit and (ff or gg) and len(examples)<12:examples.append(dict(mode=mode,seed=seed,scale=scale,target=target,mask=kind,cal=cal,output_error=err(ref,y),gradient_error=max(err(u,vv) for u,vv in zip(rg,gr))))
            row['cases']+=1
        rows.append(row)
    totals={mode:{k:sum(r[mode][k] for r in rows) for k in rows[0][mode]} for mode in modes}
    return dict(cases=sum(r['cases'] for r in rows),totals=totals,bound_audit=bounds,examples=examples,rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--excerpt-dir',required=True);a=p.parse_args();sys.path.insert(0,a.excerpt_dir)
    from native_excerpt import TiedComponentLinear,PassMaskSchedule,AblationConfig
    from trm_excerpt import build_mini_trm
    from validate import run
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    s=stress(TiedComponentLinear,PassMaskSchedule,AblationConfig)
    print(json.dumps(dict(torch=torch.__version__,platform=platform.platform(),stress=s,guarded_recurrent=run(guarded_class(TiedComponentLinear),PassMaskSchedule,AblationConfig,build_mini_trm)),indent=2,allow_nan=False))
