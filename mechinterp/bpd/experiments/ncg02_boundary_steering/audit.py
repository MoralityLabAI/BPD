"""Post-run independent dense GRU replay and root-level descriptive analysis."""
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
import numpy as np
import torch
import pilot

def dense_replay(state,x):
    p={k:v.detach().cpu().numpy().astype(np.float64) for k,v in state.items()}
    x=x.numpy().astype(np.float64)
    enc=x@p['input.weight'].T+p['input.bias']
    h=np.zeros((len(x),p['output.weight'].shape[1]))
    def sigmoid(a):return 1/(1+np.exp(-a))
    for t in range(x.shape[1]):
        write=(h@p['recurrent.V'])@p['recurrent.U']
        ir,iu,ic=np.split(enc[:,t],3,axis=1)
        rr,ru,rc=np.split(write,3,axis=1)
        reset=sigmoid(ir+rr); update=sigmoid(iu+ru)
        h=(1-update)*np.tanh(ic+reset*rc)+update*h
    return h@p['output.weight'].T+p['output.bias']

def analyze(summary):
    rng=np.random.default_rng(961116)
    out={}
    for name in ('B_minus_A','B_minus_C'):
        effects=np.array([r[name] for r in summary['paired']])
        samples=rng.choice(effects,(20000,len(effects)),replace=True).mean(axis=1)
        null=np.array([np.mean(effects*np.array(sign)) for sign in
                       itertools.product([-1,1],repeat=len(effects))])
        out[name]={'mean':float(effects.mean()),'root_sample_sd':float(effects.std(ddof=1)),
                   'bootstrap_95pct_interval':np.quantile(samples,[.025,.975]).tolist(),
                   'positive_roots':int((effects>0).sum()),'zero_roots':int((effects==0).sum()),
                   'negative_roots':int((effects<0).sum()),
                   'two_sided_sign_flip_p':float((abs(null)>=abs(effects.mean())-1e-15).mean())}
    order=sorted(out,key=lambda k:out[k]['two_sided_sign_flip_p']);last=0.
    for i,key in enumerate(order):
        last=max(last,min(1.,out[key]['two_sided_sign_flip_p']*(len(order)-i)))
        out[key]['holm_adjusted_p']=last
    return out

def run():
    root=Path(__file__).resolve().parent
    report={'scope':'DEVELOPMENT_ONLY','replay_dtype':'NumPy float64 independent dense GRU equations',
            'source_hash_checks':0,'initialization_checks':0,'matched_update_data_checks':0,
            'prediction_bits_checked':0,'prediction_bit_disagreements':0,
            'labels_checked':0,'label_disagreements':0,'phases':{}}
    for phase in ('engineering','development'):
        directory=root/'results'/phase
        manifest=json.loads((directory/'manifest_before_run.json').read_text())
        for path,sha in manifest['source_sha256'].items():
            actual=hashlib.sha256((root/path).read_bytes()).hexdigest()
            assert actual==sha,(path,actual,sha)
            report['source_hash_checks']+=1
        summary=json.loads((directory/'summary.json').read_text())
        phase_result={'root_analysis':analyze(summary),'ordinary_B_vs_A':{},'training_diagnostics':{},
                      'source_group_overlap_roots':0,'causal_patch':{}}
        recovers=corrupts=0; alltrace={arm:[] for arm in pilot.ARMS}
        patchrows={arm:[] for arm in pilot.ARMS}
        for seed in manifest['roots']:
            row=json.loads((directory/f'root_{seed}.json').read_text())
            groups=row['source_selection']['selected_groups']
            phase_result['source_group_overlap_roots']+=int(bool(set(groups[0])&set(groups[1])))
            for arm in pilot.ARMS:
                branch=row['branches'][arm]
                assert branch['initial_sha256']==row['common_warmup_sha256']
                report['initialization_checks']+=1
                alltrace[arm].extend(branch['trace'])
                patchrows[arm].append(branch['evaluation']['causal'])
                checkpoint=torch.load(directory/f'root_{seed}_{arm}.pt',weights_only=True)
                for name,delay in [('ordinary',2),('long_delay',8)]:
                    data=pilot.batch(seed,91000+delay,512,delay)
                    predicted=dense_replay(checkpoint['state_dict'],data['x'])>=0
                    saved=branch['evaluation'][name]
                    disagreements=int((predicted!=saved['predictions']).sum())
                    report['prediction_bit_disagreements']+=disagreements
                    report['prediction_bits_checked']+=predicted.size
                    report['label_disagreements']+=int((data['y'].numpy()!=saved['labels']).sum())
                    report['labels_checked']+=data['y'].numel()
            for step in range(pilot.CFG['updates']):
                values={row['branches'][arm]['trace'][step]['data_sha256'] for arm in pilot.ARMS}
                assert len(values)==1;report['matched_update_data_checks']+=1
            a=row['branches']['A_task']['evaluation']['long_delay']
            b=row['branches']['B_causal']['evaluation']['long_delay']
            ca=np.array(a['predictions'])==a['labels']; cb=np.array(b['predictions'])==b['labels']
            recovers+=int((cb&~ca).sum());corrupts+=int((ca&~cb).sum())
        phase_result['ordinary_B_vs_A']={'edge_bits_recovered':recovers,'edge_bits_corrupted':corrupts,
            'complete_cases_recovered':sum(r['B_recovers_A'] for r in summary['paired']),
            'complete_cases_corrupted':sum(r['B_corrupts_A'] for r in summary['paired'])}
        for arm,rows in alltrace.items():
            phase_result['training_diagnostics'][arm]={
                'minimum_patch_norm':min(r['min_dose'] for r in rows),
                'maximum_patch_norm':max(r['max_dose'] for r in rows),
                'zero_patches':sum(r['zero_patches'] for r in rows),
                'minimum_aux_gradient_ratio':min(r['aux_grad_ratio'] for r in rows),
                'maximum_aux_gradient_ratio':max(r['aux_grad_ratio'] for r in rows),
                'mean_parameter_displacement':float(np.mean([r['parameter_displacement'] for r in rows])),
                'mean_state_radius':float(np.mean([r['mean_state_radius'] for r in rows]))}
            phase_result['causal_patch'][arm]={
                key:float(np.mean([r[key] for r in patchrows[arm]])) for key in
                ('desired_exact_accuracy','edited_bit_accuracy','unaffected_bit_accuracy')}
        report['phases'][phase]=phase_result
    report['passed']=report['prediction_bit_disagreements']==report['label_disagreements']==0
    (root/'results'/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    artifacts={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((root/'results').rglob('*')) if p.is_file()}
    (root/'ARTIFACT_HASHES.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':run()
