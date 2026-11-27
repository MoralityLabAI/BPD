"""Root-clustered, development-only analysis of the frozen RP04 contrasts."""
from __future__ import annotations
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
from statistics import mean
from .common import CHECKPOINTS, CONDITIONS, CONTRASTS, ROOTS, SMOKE_ROOTS, Recipe
from ..training_steering_stability.stats import exact_sign_flip, _holm, _bootstrap


def require(ok, message):
    if not ok:
        raise ValueError(message)


def scores(predictions, labels_h, labels_r, target, ordinary, selected=None):
    n = len(labels_h)
    require(len(predictions)==len(labels_r)==len(target)==len(ordinary)==n, 'Case lengths differ')
    require(n>0, 'Empty case record')
    for values in (predictions,labels_h,labels_r,ordinary):
        require(all(type(v) is int and 0<=v<4 for v in values), 'Invalid prediction or label value')
    require(all(type(v) is bool for v in target), 'Target mask must contain booleans')
    require(all((h!=r)==t for h,r,t in zip(labels_h,labels_r,target)),
            'Hybrid labels disagree with the targeted mask')
    chosen = list(range(n)) if selected is None else selected
    require(all(type(i) is int and 0<=i<n for i in chosen) and len(set(chosen))==len(chosen),
            'Selected case indices must be unique and in range')
    targeted = [i for i in chosen if target[i]]
    unaffected = [i for i in chosen if not target[i]]
    initial_correct = [i for i in unaffected if ordinary[i]==labels_r[i]]
    initial_wrong = [i for i in unaffected if ordinary[i]!=labels_r[i]]
    return {
        'n_cases': len(chosen), 'correct_hybrid': sum(predictions[i]==labels_h[i] for i in chosen),
        'targeted_n': len(targeted), 'targeted_correct': sum(predictions[i]==labels_h[i] for i in targeted),
        'untargeted_n': len(unaffected), 'untargeted_correct': sum(predictions[i]==labels_r[i] for i in unaffected),
        'untargeted_original_correct': len(initial_correct), 'untargeted_original_wrong': len(initial_wrong),
        'untargeted_corrupted': sum(predictions[i]!=labels_r[i] for i in initial_correct),
        'untargeted_recovered': sum(predictions[i]==labels_r[i] for i in initial_wrong),
    }


def with_rates(counts):
    output = dict(counts)
    for name, numerator, denominator in (
        ('targeted_accuracy','targeted_correct','targeted_n'),
        ('untargeted_accuracy','untargeted_correct','untargeted_n'),
        ('untargeted_corruption_rate','untargeted_corrupted','untargeted_original_correct'),
        ('untargeted_recovery_rate','untargeted_recovered','untargeted_original_wrong'),
    ):
        output[name] = counts[numerator]/counts[denominator] if counts[denominator] else None
    return output


def summarize(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'run_manifest.json').read_text())
    phase=manifest['phase']
    require(phase in ('smoke','development'), 'Unknown phase')
    expected=SMOKE_ROOTS if phase=='smoke' else ROOTS
    require(manifest['roots']==list(expected), 'Incomplete frozen root roster')
    require(manifest['recipe']==asdict(Recipe()), 'Manifest recipe differs from the frozen design')
    require(sorted(p.name for p in folder.glob('root_*') if p.is_dir())==
            sorted(f'root_{r}' for r in expected), 'Root directory inventory differs')
    recipe=Recipe(**manifest['recipe'])
    per_root=[]
    for root in expected:
        directory=folder/f'root_{root}'
        cases=json.loads((directory/'cases.json').read_text())
        expected_n=recipe.smoke_base_pairs if phase=='smoke' else recipe.base_pairs
        require(cases['root']==root and cases['n_base_pairs']==expected_n, 'Case roster differs')
        target=cases['targeted']; labels_h=cases['labels_h']; labels_r=cases['labels_r']
        query=cases['query']; edit_slot=cases['edited_slot']
        require(cases['phase']==phase and cases['n_query_cases']==4*expected_n,
                'Case phase or query count differs')
        require(all(type(t) is bool for t in target), 'Target mask must contain booleans')
        require(sum(target)==expected_n and len(target)==4*expected_n, 'All-query balance differs')
        require(all(type(q) is int for q in query) and query==list(range(4))*expected_n,
                'Expanded query order differs')
        require(len(edit_slot)==len(target) and all(type(e) is int and 0<=e<4 for e in edit_slot),
                'Edited-slot inventory differs')
        require(all(edit_slot[4*i:4*i+4]==[edit_slot[4*i]]*4 for i in range(expected_n)),
                'Edited slot changes within a base pair')
        require(all(edit_slot.count(s)==expected_n for s in range(4)), 'Edited-slot balance differs')
        require(target==[q==e for q,e in zip(query,edit_slot)], 'Target mask disagrees with query/slot metadata')
        row={'root':root,'checkpoints':{}}
        for checkpoint in CHECKPOINTS:
            snapshot=json.loads((directory/(checkpoint+'.json')).read_text())
            require(snapshot['root']==root and snapshot['checkpoint']==checkpoint, 'Checkpoint label differs')
            require(snapshot['batch_sha256']==cases['batch_sha256'], 'Paired input hash differs')
            require(set(snapshot['conditions'])==set(CONDITIONS), 'Intervention inventory differs')
            ordinary=snapshot['conditions']['clean']['predictions']
            out={}
            for condition in CONDITIONS:
                record=snapshot['conditions'][condition]
                predictions=record['predictions']
                full=scores(predictions,labels_h,labels_r,target,ordinary)
                require(all(type(c) is int and c in (0,1) for c in record['correct']) and
                        record['correct']==[int(p==h) for p,h in zip(predictions,labels_h)],
                        'Saved correctness disagrees with predictions and labels')
                by_slot={str(s):scores(predictions,labels_h,labels_r,target,ordinary,
                    [i for i,e in enumerate(cases['edited_slot']) if e==s]) for s in range(4)}
                out[condition]={'counts':full,'rates':with_rates(full),'by_edited_slot':by_slot}
            row['checkpoints'][checkpoint]=out
        per_root.append(row)
    pooled={}
    for checkpoint in CHECKPOINTS:
        pooled[checkpoint]={}
        for condition in CONDITIONS:
            records=[r['checkpoints'][checkpoint][condition]['counts'] for r in per_root]
            counts={key:sum(r[key] for r in records) for key in records[0]}
            by_slot={str(s):with_rates({key:sum(r['checkpoints'][checkpoint][condition]['by_edited_slot'][str(s)][key]
                       for r in per_root) for key in counts}) for s in range(4)}
            pooled[checkpoint][condition]={**with_rates(counts),'by_edited_slot':by_slot,
                'macro_targeted_accuracy':mean(r['targeted_correct']/r['targeted_n'] for r in records)}
    contrasts=[]; ps=[]
    for candidate,baseline,role in CONTRASTS:
        rows=[]; effects=[]
        for r in per_root:
            c=r['checkpoints']['B_causal_fixed'][candidate]['counts']
            b=r['checkpoints']['B_causal_fixed'][baseline]['counts']
            effect=Fraction(c['targeted_correct']-b['targeted_correct'],c['targeted_n'])
            effects.append(effect)
            rows.append({'root':r['root'],'candidate_correct':c['targeted_correct'],
                'baseline_correct':b['targeted_correct'],'targeted_n':c['targeted_n'],
                'effect':float(effect),'effect_fraction':str(effect),
                'untargeted_corruption_change': c['untargeted_corrupted']-b['untargeted_corrupted']})
        sf=exact_sign_flip(effects); ps.append(Fraction(sf['p_fraction']))
        contrasts.append({'candidate':candidate,'baseline':baseline,'role':role,
            'checkpoint':'B_causal_fixed','endpoint':'targeted hybrid-answer accuracy',
            'mean_effect':float(sum(effects)/len(effects)),'per_root':rows,'sign_flip':sf,
            'bootstrap_95':_bootstrap(effects,recipe.bootstrap_draws,recipe.bootstrap_seed),
            'scientific_min_effect':recipe.scientific_min_effect,
            'evidence_class':'DEVELOPMENT_ONLY','confirmation_claim':False})
    for c,p in zip(contrasts,_holm(ps),strict=True):
        c['holm_adjusted_p']=float(p); c['holm_adjusted_p_fraction']=str(p)
    changes={}
    for checkpoint in CHECKPOINTS[1:]:
        changes[checkpoint]={condition:{
            'targeted_accuracy_change':pooled[checkpoint][condition]['targeted_accuracy']-pooled['shared_warmup'][condition]['targeted_accuracy'],
            'untargeted_corruption_rate_change':(
                pooled[checkpoint][condition]['untargeted_corruption_rate']-
                pooled['shared_warmup'][condition]['untargeted_corruption_rate']
                if pooled[checkpoint][condition]['untargeted_corruption_rate'] is not None
                and pooled['shared_warmup'][condition]['untargeted_corruption_rate'] is not None
                else None)
        } for condition in CONDITIONS}
    return {'schema_version':1,'experiment':'BPD-RP-04','phase':phase,
        'evidence_class':'DEVELOPMENT_ONLY','scope':'new development inputs on previously exposed RP03 checkpoints',
        'unit':'model root; five snapshots, pairs, queries and interventions are repeated observations',
        'n_units':len(expected),'roots':list(expected),'root_roster_complete':True,
        'base_pairs_per_root':recipe.smoke_base_pairs if phase=='smoke' else recipe.base_pairs,
        'query_cases_per_root':4*(recipe.smoke_base_pairs if phase=='smoke' else recipe.base_pairs),
        'n_checkpoints_per_root':len(CHECKPOINTS),'n_conditions':len(CONDITIONS),
        'planned_comparisons':len(CONTRASTS),'actual_inferential_comparisons':len(contrasts),
        'multiplicity':'Holm across four targeted-accuracy contrasts on the B final checkpoint',
        'paired_counterfactual_control_scope':'hybrid-derived directions use privileged states',
        'corruption_reference':recipe.corruption_increase_reference,
        'pooled':pooled,'contrasts':contrasts,'changes_from_shared_warmup':changes,'per_root':per_root}
