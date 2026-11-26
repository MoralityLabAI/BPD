"""RP04: frozen-checkpoint BPD slot-selectivity and pathway diagnostic."""
from __future__ import annotations
import argparse
import importlib.util
from dataclasses import asdict
import json
from pathlib import Path
import platform
import resource
import time
import torch
import torch.nn.functional as F
from .common import (CHECKPOINTS, CONDITIONS, DIRECTIONS, PATH_DIRECTIONS, ROUTES, ROOTS,
    SMOKE_ROOTS, PARENT_COMMIT, REPO, Recipe, checkpoint_paths, dump, input_hashes, sha,
    source_hashes, validate_lock)
from .data import make_batch
from .diagnostics import summarize_diagnostics
from .interventions import build_directions, direction_diagnostics, native_rollout, pathway_diagnostics
from .stats import summarize
from ..training_steering_stability.model import ComponentStateMemory
from ..training_steering.experiment import state_hash


def condition_record(run, ordinary, batch, step):
    logits = run['logits']
    predictions = logits.argmax(-1)
    h = ordinary['states'][:, step-1]
    return {
        'predictions':predictions.tolist(), 'correct':predictions.eq(batch.labels_h).int().tolist(),
        'cross_entropy_hybrid_mean':float(F.cross_entropy(logits,batch.labels_h)),
        'cross_entropy_receiver_mean':float(F.cross_entropy(logits,batch.labels_r)),
        'logits_abs_max':float(logits.abs().max()),
        'physical':{
            'incoming_state_delta_l2_mean':float((run['incoming_states'][:,step]-h).norm(dim=-1).mean()),
            'recurrent_write_delta_l2_mean':float((run['recurrent_writes'][:,step]-ordinary['recurrent_writes'][:,step]).norm(dim=-1).mean()),
            'next_state_delta_l2_mean':float((run['states'][:,step]-ordinary['states'][:,step]).norm(dim=-1).mean()),
            'terminal_state_delta_l2_mean':float((run['states'][:,-1]-ordinary['states'][:,-1]).norm(dim=-1).mean()),
            'all_head_change_max':float((logits-ordinary['logits']).abs().max()),
            'prefix_state_max_error':float((run['states'][:,:step]-ordinary['states'][:,:step]).abs().max()),
            'incoming_state_rms':float(run['incoming_states'][:,step].square().mean().sqrt()),
        },
    }


@torch.no_grad()
def snapshot(root, checkpoint, phase, recipe, base, batch, metadata, geometry, path):
    state = torch.load(path,weights_only=True,map_location='cpu')
    model = ComponentStateMemory(width=32,input_dim=11,values=4,components=32)
    model.load_state_dict(state,strict=True)
    model.eval()
    before=state_hash(model)
    groups=geometry['groups']; fixed=geometry['fixed_projectors']
    n=len(batch.query); step=batch.prefix_steps
    ordinary_all=native_rollout(model,torch.cat((batch.receiver,batch.donor,batch.hybrid)))
    ordinary={k:v[:n] for k,v in ordinary_all.items()}
    states=[ordinary_all['states'][i*n:(i+1)*n,step-1][::recipe.slots] for i in range(3)]
    directions,diagnostics=build_directions(*states,model.recurrent.V,model.recurrent.U,
        groups,fixed,base.edit_slot,recipe.dose)
    conditions={'clean':condition_record(ordinary,ordinary,batch,step)}
    rotated={}
    permutation,signs=geometry['permutation'],geometry['signs']
    for name in DIRECTIONS:
        delta=directions[name].repeat_interleave(recipe.slots,dim=0)
        run=native_rollout(model,batch.receiver,delta=delta,patch_step=step)
        conditions[name]=condition_record(run,ordinary,batch,step)
        rotated_delta=directions[name][:,permutation]*signs
        rotated[name+'_rotated']=direction_diagnostics(rotated_delta,model.recurrent.V,model.recurrent.U,groups,base.edit_slot)
        run=native_rollout(model,batch.receiver,delta=rotated_delta.repeat_interleave(recipe.slots,dim=0),patch_step=step)
        conditions[name+'_rotated']=condition_record(run,ordinary,batch,step)
    hybrid=states[2].repeat_interleave(recipe.slots,dim=0)
    oracle=native_rollout(model,batch.receiver,state_replacement=hybrid,patch_step=step)
    conditions['hybrid_state_oracle']=condition_record(oracle,ordinary,batch,step)
    oracle_error=float((oracle['logits']-ordinary_all['logits'][2*n:]).abs().max())
    oracle_bits=torch.equal(oracle['logits'].argmax(-1),ordinary_all['logits'][2*n:].argmax(-1))
    if oracle_error>recipe.scalar_tolerance or not oracle_bits:
        raise ValueError('Hybrid-state continuation parity failed')
    pathways={}
    for name in PATH_DIRECTIONS:
        delta=directions[name].repeat_interleave(recipe.slots,dim=0)
        pathways[name]=pathway_diagnostics(model,base.receiver,directions[name],patch_step=step)
        for route in ROUTES:
            run=native_rollout(model,batch.receiver,delta=delta,groups=groups,
                edit_slot=batch.edit_slot,route=route,patch_step=step)
            conditions[name+'__'+route]=condition_record(run,ordinary,batch,step)
    if set(conditions)!=set(CONDITIONS):
        raise ValueError('Incomplete intervention inventory')
    if state_hash(model)!=before:
        raise ValueError('Frozen checkpoint was mutated')
    return {
        'schema_version':1,'root':root,'phase':phase,'checkpoint':checkpoint,
        'checkpoint_sha256':sha(path),'batch_sha256':metadata['batch_sha256'],
        'parameter_count':sum(p.numel() for p in model.parameters()),'groups':groups.tolist(),
        'conditions':conditions,'direction_diagnostics':diagnostics,
        'rotated_diagnostics':rotated,'pathway_diagnostics':pathways,
        'checks':{'oracle_logits_max_error':oracle_error,'oracle_prediction_bits_match':oracle_bits,
                  'checkpoint_unchanged':True,'n_prediction_cases':n*len(conditions)},
        'scope':'exposed checkpoint; hybrid-derived directions use privileged counterfactual state',
    }


def run_root(root,phase,out,recipe):
    started=time.time(); cpu=time.process_time()
    directory=out/f'root_{root}'
    directory.mkdir(parents=True,exist_ok=False)
    base,batch,metadata=make_batch(root,phase,recipe)
    dump(directory/'cases.json',metadata)
    paths=checkpoint_paths(root,phase)
    geometry=torch.load(paths['geometry'],weights_only=True,map_location='cpu')
    for checkpoint in CHECKPOINTS:
        result=snapshot(root,checkpoint,phase,recipe,base,batch,metadata,geometry,paths[checkpoint])
        dump(directory/(checkpoint+'.json'),result)
        print(json.dumps({'root':root,'checkpoint':checkpoint,'complete':True}),flush=True)
    record={'root':root,'phase':phase,'checkpoint_roster':list(CHECKPOINTS),
        'artifact_sha256':{p.name:sha(p) for p in sorted(directory.glob('*.json'))},
        'elapsed_seconds':time.time()-started,'process_cpu_seconds':time.process_time()-cpu,
        'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    dump(directory/'result.json',record)
    return record


def preflight_from_summary(summary, out):
    rows=[]
    for row in summary['per_root']:
        data=row['checkpoints']['B_causal_fixed']
        n=data['hybrid_span']['counts']['targeted_n']
        effect=data['hybrid_span']['counts']['targeted_correct']-data['donor_span']['counts']['targeted_correct']
        sham=data['hybrid_span_rotated']['counts']['targeted_correct']-data['donor_span_rotated']['counts']['targeted_correct']
        rows.append({'unit_id':str(row['root']),'paired_effect':f'{effect}/{n}',
                     'sham_effects':[f'{sham}/{n}']})
    spec={'schema_version':1,'experiment_id':'BPD-RP-04-component-selectivity',
        'independent_unit':'exposed independently trained RP03 model root',
        'alpha':'.05','scientific_min_effect':'.05','target_effect':'.10',
        'measurement_resolution':f'1/{n}','sham_quantile':'.95','sham_confidence':'.95',
        'planned_comparisons':4,'confirm_units_planned':8,'pilot':rows,
        'pilot_from_development_only':True,'units_independent':True,'heldout_sealed':False,
        'heldout_scope':'fresh stochastic input draws on already exposed model roots',
        'planning_scope':'bounded diagnostic; no confirmation cohort allocated',
        'sham_scope':'paired rotated hybrid-span minus rotated donor-span; uncalibrated population tail'}
    path=Path(__file__).resolve().parent.parent/'repair_portfolio/evidence/preflight.py'
    loader=importlib.util.spec_from_file_location('rp04_pinned_preflight',path)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    verdict=module.preflight(spec)
    dump(out/'preflight_input.json',spec);dump(out/'preflight_result.json',verdict)
    return verdict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('smoke','development'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-lock',type=Path)
    args=parser.parse_args(); recipe=Recipe()
    lock=validate_lock(args.source_lock) if args.phase=='development' and args.source_lock else None
    if args.phase=='development' and lock is None:
        parser.error('Development requires --source-lock')
    torch.set_num_threads(recipe.threads)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True,exist_ok=False)
    roots=SMOKE_ROOTS if args.phase=='smoke' else ROOTS
    started=time.time()
    manifest={'experiment':'BPD-RP-04','phase':args.phase,'roots':list(roots),'recipe':asdict(recipe),
        'parent_commit':PARENT_COMMIT,'source_hashes':source_hashes(),'input_hashes':input_hashes(args.phase),
        'source_lock':lock,'started_unix':started,'python':platform.python_version(),'torch':torch.__version__,
        'evidence_class':'DEVELOPMENT_ONLY','new_training_updates':0}
    dump(args.out/'run_manifest.json',manifest)
    results=[];failures=[]
    for root in roots:
        try:
            results.append(run_root(root,args.phase,args.out,recipe))
        except Exception as exc:
            failure={'root':root,'error_type':type(exc).__name__,'message':str(exc)}
            failures.append(failure);dump(args.out/f'root_{root}'/'failure.json',failure)
    if failures:
        dump(args.out/'completion.json',{'completed':False,'assigned_roots':list(roots),
            'completed_roots':[r['root'] for r in results],'failures':failures})
        raise RuntimeError('RP04 recorded root failures; preserve every artifact')
    summary=summarize(args.out)
    dump(args.out/'summary.json',summary)
    dump(args.out/'diagnostic_summary.json',summarize_diagnostics(args.out))
    preflight_from_summary(summary,args.out)
    hashes={str(p.relative_to(args.out)):sha(p) for p in sorted(args.out.rglob('*.json'))}
    dump(args.out/'completion.json',{'completed':True,'roots':list(roots),'artifact_sha256':hashes,
        'elapsed_seconds':time.time()-started,'sum_root_cpu_seconds':sum(r['process_cpu_seconds'] for r in results),
        'process_peak_rss_kib':max(r['process_peak_rss_kib'] for r in results)})


if __name__=='__main__':
    main()
