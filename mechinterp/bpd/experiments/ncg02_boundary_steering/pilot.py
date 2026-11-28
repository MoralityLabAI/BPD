"""NCG-02: development pilot on two binary triangle flow systems.

Uses a pinned, unmodified snapshot of BPD's native RP03 recurrent model.
All causal discovery, warmup and branch updates use training streams.
Ordinary development scores always have steering disabled.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, platform, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
import numpy as np
import torch
import torch.nn.functional as F
from mechinterp.bpd.experiments.training_steering_stability.model import (
    ComponentStateMemory, projectors, state_patch)
from evalmath_preflight import preflight

PIN = '402ef7764e11bc7736873d66357b8dc24b996e94'
CFG = {'width':16, 'components':16, 'input_dim':8, 'edges':6,
       'warmup':400, 'updates':80, 'batch':32, 'source_pairs':96,
       'eval_cases':512, 'train_delay':2, 'long_delay':8,
       'dose':0.75, 'gradient_ratio':0.25, 'lr':0.005}
ENGINEERING = (962600, 962601)
DEVELOPMENT = tuple(range(962610, 962618))
ARMS = ('A_task', 'B_causal', 'C_rotated')

def digest(obj):
    if isinstance(obj, bytes): return hashlib.sha256(obj).hexdigest()
    return digest(json.dumps(obj, sort_keys=True, separators=(',',':')).encode())

def hashes():
    return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in
            sorted(ROOT.rglob('*.py')) if '__pycache__' not in str(p)}

def batch(seed, stream, n, delay, slot=None):
    rng=np.random.default_rng(np.random.SeedSequence([seed, stream]))
    codes=rng.integers(0,2,(n,2,3), dtype=np.int64)
    slots=rng.integers(0,2,n) if slot is None else np.full(n,slot)
    donor=codes.copy(); donor[np.arange(n),slots,0]^=1
    enc=np.random.default_rng(seed+991); perm=enc.permutation(6)
    signs=enc.choice([-1.,1.],6)
    noise=rng.normal(0,.02,(n,2+delay,6))
    def encode(bits):
        x=np.zeros((n,2+delay,8)); x[:,:,:6]=noise
        for t in range(2):
            ids=np.arange(t*3,t*3+3)
            x[:,t,perm[ids]]+=(bits[:,t,:]*2-1)*signs[ids]
        x[:,:2,6]=1; x[:,2:,7]=1
        return torch.tensor(x,dtype=torch.float32)
    def edges(bits):
        c=bits[:,:,0]; b1=bits[:,:,1]; b2=bits[:,:,2]
        return np.stack([c,c^b1,c^b1^b2],-1).reshape(n,6)
    return {'x':encode(codes), 'donor_x':encode(donor),
            'y':torch.tensor(edges(codes),dtype=torch.float32),
            'donor_y':torch.tensor(edges(donor),dtype=torch.float32),
            'slot':torch.tensor(slots,dtype=torch.long),
            'codes':codes, 'donor_codes':donor}

def params(model): return [p for p in model.parameters() if p.requires_grad]
def norm(gs): return torch.sqrt(sum(g.square().sum() for g in gs))
def model_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def loss_gradients(model, b, ps, rotation):
    r=model.rollout(b['x']); d=model.rollout(b['donor_x'])
    task=.5*(F.binary_cross_entropy_with_logits(r['logits'],b['y'])+
             F.binary_cross_entropy_with_logits(d['logits'],b['donor_y']))
    delta,meta=state_patch(r['states'][:,1],d['states'][:,1],ps,b['slot'],CFG['dose'])
    target=model.rollout(b['x'],patch=delta,patch_step=2)['logits']
    rotated=model.rollout(b['x'],patch=delta@rotation,patch_step=2)['logits']
    aux=[F.binary_cross_entropy_with_logits(z,b['donor_y']) for z in (target,rotated)]
    ps0=params(model)
    grads=[torch.autograd.grad(loss,ps0,retain_graph=i<2) for i,loss in enumerate((task,*aux))]
    return grads, {'task_loss':float(task.detach()),
        'causal_loss':float(aux[0].detach()),'rotated_loss':float(aux[1].detach()),
        'min_dose':float(meta['delivered_norm'].min().detach()),
        'max_dose':float(meta['delivered_norm'].max().detach()),
        'zero_patches':int((~meta['valid']).sum()),
        'mean_state_radius':float(meta['recipient_state_norm'].mean().detach())}

def update(model, optimizer, grads, which):
    task=grads[0]; base=norm(task); extra=grads[which] if which else task
    scale=CFG['gradient_ratio']*base/norm(extra).clamp_min(1e-12) if which else base*0
    applied=[a+scale*b for a,b in zip(task,extra)]
    optimizer.zero_grad(set_to_none=True)
    for p,g in zip(params(model),applied): p.grad=g.detach().clone()
    before=[p.detach().clone() for p in params(model)]
    raw=float(norm(applied)); torch.nn.utils.clip_grad_norm_(params(model),1.)
    optimizer.step()
    displacement=float(norm([p.detach()-b for p,b in zip(params(model),before)]))
    return {'task_grad_norm':float(base),'aux_grad_ratio':float(scale*norm(extra)/base.clamp_min(1e-12)),
            'applied_grad_norm':raw, 'parameter_displacement':displacement}

@torch.no_grad()
def discover(model,seed):
    groups=torch.arange(CFG['components']).reshape(4,4)
    allp=projectors(model,groups); selected=[]; scores=[]
    for slot in range(2):
        b=batch(seed,30000+slot,CFG['source_pairs'],CFG['train_delay'],slot)
        r=model.rollout(b['x'])['states'][:,1]
        d=model.rollout(b['donor_x'])['states'][:,1]
        candidates=[]
        for j in range(4):
            delta,_=state_patch(r,d,allp[j:j+1],torch.zeros(len(r),dtype=torch.long),CFG['dose'])
            z=model.rollout(b['x'],patch=delta,patch_step=2)['logits']
            candidates.append(float(F.binary_cross_entropy_with_logits(z,b['donor_y'])))
        choice=int(np.argmin(candidates)); selected.append(choice); scores.append(candidates)
    chosen=groups[selected]
    return projectors(model,chosen),{'selected_groups':chosen.tolist(),
                                  'candidate_losses':scores,'selected_indices':selected}

def metrics(pred,y,codes):
    pred=np.asarray(pred,dtype=np.int64); y=np.asarray(y,dtype=np.int64)
    e=pred.reshape(-1,2,3)
    valid=((e[:,:,0]^e[:,:,1])==codes[:,:,1]) & ((e[:,:,1]^e[:,:,2])==codes[:,:,2])
    return {'exact_accuracy':float(np.all(pred==y,axis=1).mean()),
            'bit_accuracy':float((pred==y).mean()),
            'constraint_accuracy':float(np.all(valid,axis=1).mean()),
            'predictions':pred.tolist(), 'labels':y.tolist()}

@torch.no_grad()
def evaluate(model,seed,ps,rotation):
    out={}
    for name,delay in [('ordinary',CFG['train_delay']),('long_delay',CFG['long_delay'])]:
        b=batch(seed,91000+delay,CFG['eval_cases'],delay)
        z=model(b['x']); out[name]=metrics((z>=0).numpy(),b['y'].numpy(),b['codes'])
    b=batch(seed,92000,256,CFG['long_delay'])
    rr=model.rollout(b['x']); dd=model.rollout(b['donor_x'])
    delta,meta=state_patch(rr['states'][:,1],dd['states'][:,1],ps,b['slot'],CFG['dose'])
    clean=(rr['logits']>=0).numpy(); desired=b['donor_y'].numpy().astype(int)
    for name,patch in [('causal',delta),('rotated',delta@rotation)]:
        pred=(model.rollout(b['x'],patch=patch,patch_step=2)['logits']>=0).numpy()
        same=np.repeat(np.arange(2)[None,:]!=b['slot'].numpy()[:,None],3,axis=1)
        target=~same
        out[name]={'desired_exact_accuracy':float(np.all(pred==desired,axis=1).mean()),
                   'edited_bit_accuracy':float((pred[target]==desired[target]).mean()),
                   'unaffected_bit_accuracy':float((pred[same]==desired[same]).mean()),
                   'unaffected_correct_before':int(((clean==desired)&same).sum()),
                   'unaffected_corrupted':int(((clean==desired)&(pred!=desired)&same).sum()),
                   'unaffected_recovered':int(((clean!=desired)&(pred==desired)&same).sum())}
    out['dose']={'min':float(meta['delivered_norm'].min()),'max':float(meta['delivered_norm'].max())}
    return out

def run_root(seed,out):
    torch.manual_seed(seed)
    model=ComponentStateMemory(width=CFG['width'],input_dim=CFG['input_dim'],values=6,
                               components=CFG['components'])
    opt=torch.optim.Adam(model.parameters(),lr=CFG['lr'])
    for i in range(CFG['warmup']):
        b=batch(seed,1000+i,CFG['batch'],CFG['train_delay'])
        x=torch.cat([b['x'],b['donor_x']]); y=torch.cat([b['y'],b['donor_y']])
        loss=F.binary_cross_entropy_with_logits(model(x),y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params(model),1.); opt.step()
    ps,selection=discover(model,seed)
    common=model_hash(model); torch.save(model.state_dict(),out/f'root_{seed}_warmup.pt')
    rng=np.random.default_rng(seed+777); perm=rng.permutation(CFG['width'])
    sign=rng.choice([-1.,1.],CFG['width'])
    rotation=torch.tensor(np.eye(CFG['width'])[:,perm]*sign,dtype=torch.float32)
    models={arm:copy.deepcopy(model) for arm in ARMS}
    optimizers={arm:torch.optim.Adam(m.parameters(),lr=CFG['lr']) for arm,m in models.items()}
    receipt={'seed':seed,'source_selection':selection,'common_warmup_sha256':common,
             'trainable_parameters':sum(p.numel() for p in params(model)),
             'split_contract':'Independent sensor/noise samples on the same 64 semantic flow assignments',
             'branches':{arm:{'initial_sha256':model_hash(m),'trace':[]} for arm,m in models.items()}}
    for i in range(CFG['updates']):
        b=batch(seed,50000+i,CFG['batch'],CFG['train_delay'])
        datahash=digest(b['x'].numpy().tobytes()+b['y'].numpy().tobytes())
        for j,arm in enumerate(ARMS):
            g,trace=loss_gradients(models[arm],b,ps,rotation)
            trace.update(update(models[arm],optimizers[arm],g,j)); trace['data_sha256']=datahash
            receipt['branches'][arm]['trace'].append(trace)
    for arm,m in models.items():
        receipt['branches'][arm]['evaluation']=evaluate(m,seed,ps,rotation)
        receipt['branches'][arm]['final_sha256']=model_hash(m)
        torch.save({'state_dict':m.state_dict(),'projectors':ps,'rotation':rotation},out/f'root_{seed}_{arm}.pt')
    (out/f'root_{seed}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

def summarize(rows):
    summary={'scope':'DEVELOPMENT_ONLY','unit':'fresh model initialization, input encoding and training noise stream',
             'n_units':len(rows),'cases_per_root_per_arm':2*CFG['eval_cases'],
             'source_commit':PIN,'arms':{},'paired':[],
             'structural_oracle':{'exact_accuracy':1.,'constraint_accuracy':1.,'semantic_assignments':64}}
    for arm in ARMS:
        summary['arms'][arm]={k:{metric:float(np.mean([r['branches'][arm]['evaluation'][k][metric] for r in rows]))
            for metric in ('exact_accuracy','bit_accuracy','constraint_accuracy')}
            for k in ('ordinary','long_delay')}
    for r in rows:
        ev={a:r['branches'][a]['evaluation']['long_delay'] for a in ARMS}
        a,b,c=[np.all(np.array(ev[arm]['predictions'])==ev[arm]['labels'],axis=1) for arm in ARMS]
        summary['paired'].append({'seed':r['seed'],'B_minus_A':float((b.astype(float)-a).mean()),
            'B_minus_C':float((b.astype(float)-c).mean()),'C_minus_A':float((c.astype(float)-a).mean()),
            'B_recovers_A':int((b&~a).sum()),'B_corrupts_A':int((a&~b).sum())})
    return summary

def main():
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['engineering','development'],required=True)
    p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    if args.out.exists():raise SystemExit('Output directory already exists; preserve earlier runs')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True)
    roots=ENGINEERING if args.phase=='engineering' else DEVELOPMENT
    manifest={'config':CFG,'roots':roots,'source_sha256':hashes(),'source_commit':PIN,
              'torch':torch.__version__,'python':platform.python_version(),
              'selection_exposure':'one fixed recipe, development only; no confirmation population generated',
              'primary_contrast':'ordinary long-delay full-edge exact accuracy B minus A',
              'comparisons':['B-A','B-C'],'scientific_min_effect':.05,'sham_quantile':.95,'sham_confidence':.95}
    (args.out/'manifest_before_run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.perf_counter(); rows=[]
    for seed in roots:
        rows.append(run_root(seed,args.out)); print('finished',seed,flush=True)
    result=summarize(rows); (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    spec={'schema_version':1,'experiment_id':'NCG-02-'+args.phase,'independent_unit':result['unit'],
          'alpha':'0.05','scientific_min_effect':'0.05','target_effect':'0.10',
          'measurement_resolution':str(1/CFG['eval_cases']),'sham_quantile':'0.95','sham_confidence':'0.95',
          'planned_comparisons':2,'confirm_units_planned':64,'pilot_from_development_only':True,
          'heldout_sealed':False,'units_independent':True,
          'pilot':[{'unit_id':str(r['seed']),'paired_effect':str(r['B_minus_A']),
                    'sham_effects':[str(r['C_minus_A'])]} for r in result['paired']]}
    (args.out/'preflight_input.json').write_text(json.dumps(spec,indent=2)+'\n')
    (args.out/'preflight_result.json').write_text(json.dumps(preflight(spec),indent=2)+'\n')
    (args.out/'timing.json').write_text(json.dumps({'wall_seconds':time.perf_counter()-start},indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
