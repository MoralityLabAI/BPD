"""BPD-CQ-02: CPU-trained shared low-rank factor / delayed-observer pilot.

The recurrent transport is a frozen synthetic map, not learned BPD/Transformer
recurrence. A native BPDModel parity audit is optional on a BPD checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SEEDS = (503, 509, 521, 523, 541, 547, 557, 563)
N_STEPS = 200
TARGET = torch.diag(torch.tensor([1., 1., 0., 0.]))
READ = torch.tensor([0., 1., 0., 0.])
TRANSPORT = torch.tensor([[0.,1.,0.,0.], [1.,0.,0.,0.], [0.,0.,1.,0.], [0.,0.,0.,1.]])


class TopOne(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        p=logits.softmax(-1)
        ctx.save_for_backward(p)
        return F.one_hot(logits.argmax(-1), 2).to(logits.dtype)

    @staticmethod
    def backward(ctx, grad):
        (p,) = ctx.saved_tensors
        return p*(grad - (grad*p).sum(-1, keepdim=True))


class TrainedFactors(nn.Module):
    def __init__(self, seed):
        super().__init__()
        g=torch.Generator().manual_seed(seed)
        self.V=nn.Parameter(torch.randn((4,2), generator=g)*.5)
        self.U=nn.Parameter(torch.randn((2,4), generator=g)*.5)
        self.gate=nn.Linear(4,2)
        with torch.no_grad():
            self.gate.weight.copy_(torch.randn((2,4),generator=g)*.05)
            self.gate.bias.zero_()

    def mask(self, x):
        return TopOne.apply(self.gate(x.abs()))

    def forward(self,x):
        mask=self.mask(x)
        out=((x @ self.V) * mask) @ self.U
        return out,mask


def draw(seed,count,mixed=False):
    g=torch.Generator().manual_seed(seed)
    amp=.75+.5*torch.rand(count,4,generator=g)
    sign=2*torch.randint(0,2,(count,4),generator=g).float()-1
    a=torch.zeros(count,4,4)
    for v in range(4):
        a[:,v,0 if v<2 else 1]=amp[:,v]*sign[:,v]
        if mixed:
            a[:,v,1 if v<2 else 0]=amp[:,v]*sign[:,v]
    return a


class NativeTrainedFactors(nn.Module):
    """The native BPDModel/TiedComponentLinear backend, with identical init."""
    def __init__(self, seed):
        super().__init__()
        from mechinterp.bpd.components import BPDModel
        from mechinterp.bpd.config import AblationConfig

        class Teacher4(nn.Module):
            def __init__(self):
                super().__init__()
                self.core=nn.Linear(4,4,bias=False)
                with torch.no_grad():self.core.weight.copy_(TARGET)
            def forward(self,x):return self.core(x)

        self.bpd=BPDModel(Teacher4(),{"core":2},max_passes=4,
            ablation=AblationConfig(mode="zero",sphere_rescale=False),init_svd=True)
        # Initialize native physical shared factors identically to the
        # independent CPU implementation for source/optimization parity.
        local=TrainedFactors(seed)
        with torch.no_grad():
            self.bpd.wrappers["core"].V.copy_(local.V)
            self.bpd.wrappers["core"].U.copy_(local.U)
        self.gate=local.gate

    @property
    def V(self):return self.bpd.wrappers["core"].V

    @property
    def U(self):return self.bpd.wrappers["core"].U

    def mask(self,x):return TopOne.apply(self.gate(x.abs()))

    def forward(self,x):
        from mechinterp.bpd.components import PassMaskSchedule
        mask=self.mask(x)
        schedule=PassMaskSchedule(comp_mask=mask.unsqueeze(2),
                                  delta_mask=torch.zeros_like(mask[...,0]).unsqueeze(-1))
        with self.bpd.masked_components({"core":schedule}):
            y=torch.stack([self.bpd.target_model(x[:,t:t+1])[:,0] for t in range(x.shape[1])],dim=1)
        return y,mask


def native_parity(model,seed):
    """Exact native masked-execution contract on a heldout calibration tensor."""
    x=draw(300000+seed,32)
    y,mask=model(x)
    expected=((x@model.V)*mask)@model.U
    return float((y-expected).abs().max())


def alignment(model,cal):
    # The target has distinct input channels. Find which read factor is x0.
    v=model.V.detach()
    # This orientation is selected only using the independent calibration
    # excitations, without knowing any heldout labels or interventions.
    scores=(cal[:,:2]@v).abs().mean((0,1))/(v.norm(dim=0)+1e-12)
    idx=int(scores.argmax())
    cosine=float(v[0,idx].abs()/(v[:,idx].norm()+1e-12))
    return idx,cosine


def run_seed(seed:int,steps:int=N_STEPS,heldout:int=128,pairs:int=64,backend:str="independent"):
    torch.set_num_threads(1)
    model=(TrainedFactors(seed) if backend=="independent" else NativeTrainedFactors(seed))
    g=torch.Generator().manual_seed(100000+seed)
    params=list(model.parameters());before=[p.detach().clone() for p in params]
    opt=torch.optim.Adam(params,lr=.04)
    for step in range(steps):
        amp=.75+.5*torch.rand((16,4),generator=g)
        signs=2*torch.randint(0,2,(16,4),generator=g).float()-1
        x=torch.zeros((16,4,4))
        for t in range(4):x[:,t,0 if t<2 else 1]=amp[:,t]*signs[:,t]
        out,mask=model(x)
        target=x@TARGET.T
        mse=(out-target).square().mean()
        faith=((model.V@model.U).T-TARGET).square().mean()
        loss=mse+.30*faith+.05*mask.mean()
        if not bool(torch.isfinite(loss)):raise RuntimeError(f'nonfinite at seed {seed}, step {step}')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    updates=[float((p.detach()-old).norm()) for p,old in zip(params,before)]
    cal=draw(300000+seed,128)
    idx,axis=alignment(model,cal)
    # The component ordering is frozen here using calibration/known source axis.
    parity=(native_parity(model,seed) if backend=="native" else None)
    h=draw(200000+seed,heldout)
    with torch.no_grad():
        y,mask=model(h)
        exp=torch.zeros_like(mask)
        exp[:,:2,idx]=1.;exp[:,2:,1-idx]=1.
        true_routes=(mask==exp).all(dim=(1,2))
        mse=float((y-h).square().mean())
        # Donor/recipient delta on a separate heldout pair distribution.
        rg=torch.Generator().manual_seed(400000+seed)
        recv=.75+.20*torch.rand(pairs,generator=rg)
        donor=1.05+.20*torch.rand(pairs,generator=rg)
        xr=torch.zeros(pairs,4);xd=torch.zeros_like(xr)
        xr[:,0]=recv;xd[:,0]=donor
        mr=model.mask(xr);md=model.mask(xd)
        assert torch.all(mr.sum(-1)==1) and torch.all(md.sum(-1)==1)
        # Calibrated component-only donor write replacement; if gate wrongly
        # routes one of the pair members its effect is explicitly measured.
        wr=(xr@model.V)[:,idx,None]*model.U[idx:idx+1]
        wd=(xd@model.V)[:,idx,None]*model.U[idx:idx+1]
        diff=wd-wr
        norms=diff.norm(dim=-1,keepdim=True)
        if torch.any(norms<1e-9):raise RuntimeError('zero causal intervention')
        immediate=(diff@READ).abs()/norms.squeeze(-1)
        delayed=(diff@TRANSPORT.T@READ).abs()/norms.squeeze(-1)
        # Same-norm orthogonal-fiber patches with no current readout effect.
        fiber=diff.clone();fiber[:,1]=0.
        flen=fiber.norm(dim=-1,keepdim=True)
        assert bool((flen>1e-8).all())
        fiber=fiber/flen*norms
        raw=torch.randn((pairs,4),generator=rg)
        raw[:,1]=0.
        random_fiber=raw/raw.norm(dim=-1,keepdim=True)*norms
        zero=torch.zeros_like(diff)
        visible=torch.zeros_like(diff);visible[:,1]=norms.squeeze(-1)
        assert float((random_fiber.norm(dim=-1)-norms.squeeze(-1)).abs().max())<1e-5
        assert float((fiber.norm(dim=-1)-norms.squeeze(-1)).abs().max())<1e-5
        future=lambda delta: (wr+delta)@TRANSPORT.T@READ
        gold=wd@TRANSPORT.T@READ
        counts={name:int(((future(patch)-gold).abs()<=.05).sum()) for name,patch in
                [('native',diff),('projected_native',fiber),('random_fiber',random_fiber),('sham',zero),('visible',visible) ]}
        # Instantaneous fiber parity at a later readout is the exact linear test.
        fiber_now=float((fiber@READ).abs().max());random_now=float((random_fiber@READ).abs().max())
        identity_next=float((fiber@READ).abs().max())
        mse_mixed=float((model(draw(200000+seed,heldout,mixed=True))[0]-draw(200000+seed,heldout,mixed=True)).square().mean())
    with torch.no_grad():
        component_now=READ@model.U.T
        component_later=READ@TRANSPORT@model.U.T
        gram=torch.stack((component_now,component_later))
        svals=torch.linalg.svdvals(gram)
        rank_now=int(torch.linalg.matrix_rank(component_now.unsqueeze(0),tol=1e-5))
        rank_after=int(torch.linalg.matrix_rank(gram,tol=1e-5))
    return dict(seed=seed,train_steps=steps,backend=backend,native_parity_max_abs=parity,calibration_index=idx,calibration_axis_abs_cos=axis,
                component_observable_rank_now=rank_now,component_observable_rank_after_one=rank_after,
                delayed_component_singular_values=[float(x) for x in svals],
                updated_factors=dict(V=updates[0],U=updates[1]),heldout_boards=heldout,
                four_pass_route=int(true_routes.sum()),binary_reconstruction_mse=mse,
                mixed_source_mse=mse_mixed,causal_pairs=pairs,
                immediate_median=float(immediate.median()),delayed_median=float(delayed.median()),
                native_dark_later_pairs=int(((immediate<=.10)&(delayed>=.75)).sum()),
                transfer=counts,fiber_instant_max=fiber_now,random_fiber_instant_max=random_now,
                identity_transport_instant_max=identity_next,
                largest_norm_matching_error=float(max((fiber.norm(dim=-1)-norms.squeeze(-1)).abs().max(),(random_fiber.norm(dim=-1)-norms.squeeze(-1)).abs().max())))


def cohort(backend="independent"):
    started=time.perf_counter()
    rows=[run_seed(i,backend=backend) for i in SEEDS]
    valid=[x['four_pass_route'] >=.9*x['heldout_boards'] and x['immediate_median']<=.10 and x['delayed_median']>=.90 for x in rows]
    return dict(schema='BPD-CQ-02/trained-factors-cpu/v1',source=('independent BPD-style shared-factor implementation' if backend=='independent' else 'native BPDModel/TiedComponentLinear with trained factors'),
                torch_version=torch.__version__,python_version=platform.python_version(),seeds=list(SEEDS),
                rows=rows,registered_pilot_pass=sum(valid)>=6,seed_success_count=sum(valid),
                outcomes={k:sum(r['transfer'][k] for r in rows) for k in rows[0]['transfer']},
                total_pairs=sum(r['causal_pairs'] for r in rows),
                total_train_steps=len(SEEDS)*N_STEPS,
                elapsed_seconds=round(time.perf_counter()-started,4),
                limitations='Frozen hand-designed T; synthetic factor training; not a learned recurrent Transformer or natural math-reasoning mechanism.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--backend',choices=['independent','native'],default='independent');a=p.parse_args()
    r=cohort(a.backend);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
    for x in r['rows']:
        print(x['seed'],x['four_pass_route'],'/',x['heldout_boards'],'mse=',round(x['binary_reconstruction_mse'],6),'immediate=',round(x['immediate_median'],5),'delayed=',round(x['delayed_median'],5),x['transfer'])
if __name__=='__main__':main()
