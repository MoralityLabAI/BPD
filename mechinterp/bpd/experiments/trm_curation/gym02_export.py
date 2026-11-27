"""Stage-local V4 export. Decisions and later outcomes are separate fields."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path


SCHEMA = 'bpd-trm-gym02-stage-v4'
SHAMS = {'F_shuffled_k4', 'H_rotating_sham', 'J_balanced_sham'}


def _line(row):
    return (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def stage_rows(w, *, task, root, arm):
    scores = w.get('scores_prefix')
    if scores is None:
        return []
    norms = w['update_norms_prefix']
    stages = w['stages']
    sizes = [int(s['stage']) for s in stages]
    if sizes not in ([4], [4, 8, 12], [4, 8, 12, 16]):
        raise ValueError('invalid acquisition sequence')
    if len(scores) != sizes[-1] or len(norms) != len(scores):
        raise ValueError('invalid prefix roster')
    if not all(math.isfinite(v) for v in scores + norms):
        raise ValueError('nonfinite features')
    for key in ('parent_model_sha256', 'parent_custody_sha256', 'feedback_cohort_id'):
        if not w.get(key):
            raise ValueError('missing parent provenance')
    costs = [float(w['acquisition_start_cpu'])] + [float(s['cpu_to_stage']) for s in stages]
    if not all(math.isfinite(c) and c >= 0 for c in costs) or costs != sorted(costs):
        raise ValueError('invalid stage costs')
    prov = {'task':task,'root':root,'arm':arm,'window':w['window'],
            'parent_model_sha256':w['parent_model_sha256'],
            'parent_custody_sha256':w['parent_custody_sha256'],
            'feedback_cohort_id':w['feedback_cohort_id']}
    out = []
    for j,k in enumerate([0]+sizes):
        terminal = j == len(sizes)
        state = {'stage':k,'scores':scores[:k],'update_norms':norms[:k],
                 'feedback_loss':w['feedback_loss0'],'feedback_age':w['feedback_age'],
                 'charged_cpu_before':w['charged_cpu_before'],
                 'cpu_to_stage':costs[j]}
        action = ('ABSTAIN' if w['decision']=='ABSTAIN' else 'SELECT_BATCH') if terminal else f'ACQUIRE_{sizes[j]}'
        status = w.get('outcome_status') if terminal else None
        outcome = None
        preference = None
        if terminal:
            if action == 'SELECT_BATCH':
                state['selected_indices'] = w['selected_indices']
                if not state['selected_indices'] or any(i < 0 or i >= k for i in state['selected_indices']):
                    raise ValueError('invalid selected batch')
            outcome = {'status':status,'actual_gain':w.get('actual_gain'),
                       'cal_before':w.get('cal_before'),'cal_after':w.get('cal_after'),
                       'predicted_joint_native_parent':w.get('predicted_joint_native'),
                       'audit_probability':w.get('audit_probability'),'audit_outcome':w.get('audit_outcome'),
                       'committed':bool(w.get('committed')),'window_cpu_seconds':w['window_cpu_seconds']}
            if status in ('committed','rolled_back') and arm not in SHAMS:
                gain = outcome['actual_gain']
                if gain is None or not math.isfinite(gain):
                    raise ValueError('verified terminal missing gain')
                if outcome['cal_before'] is None or outcome['cal_after'] is None:
                    raise ValueError('missing matched feedback')
                if abs(outcome['cal_before']-outcome['cal_after']-gain)>1e-8:
                    raise ValueError('feedback arithmetic mismatch')
                if gain>0 and outcome['committed']:
                    preference = {'chosen':'SELECT_BATCH','rejected':'ABSTAIN',
                                  'objective':'positive TRAIN gain on exact verified batch'}
                elif gain<=0:
                    preference = {'chosen':'ABSTAIN','rejected':'SELECT_BATCH',
                                  'objective':'positive TRAIN gain on exact verified batch'}
        key = [task,root,arm,w['window'],k]
        out.append({'schema':SCHEMA,'record_id':hashlib.sha256(_line(key)).hexdigest(),
                    'split':'train','evidence_class':'DEVELOPMENT_ONLY','evaluation_leakage':False,
                    'provenance':prov,'state':state,'behavior_action':action,
                    'label_kind':'verified_terminal_preference' if preference else 'behavior_only',
                    'training_eligible':preference is not None,'preference':preference,'outcome':outcome})
    return out


def export_v4(out_dir: Path, tasks, roots, export_dir: Path):
    export_dir.mkdir(parents=True,exist_ok=True)
    counts=Counter()
    digest=hashlib.sha256()
    stages_path=export_dir/'stages_gym_v4.jsonl.gz'
    horizon_path=export_dir/'horizon_outcomes_gym_v4.jsonl'
    horizon_digest=hashlib.sha256()
    seen=set()
    with stages_path.open('wb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz, horizon_path.open('wb') as hf:
        for task in tasks:
            for root in roots:
                p=out_dir/task/f'root_{root}'
                res=json.loads((p/'result.json').read_text(encoding='utf-8'))
                for arm,a in res['arms'].items():
                    line=_line({'schema':SCHEMA,'split_evaluated':'eval','decision_rows_attached':False,
                                'task':task,'root':root,'arm':arm,'warm_eval':res['warm_eval'],
                                'final_eval':a['eval'],'charged_cpu_seconds':a['charged_cpu_seconds']})
                    hf.write(line);horizon_digest.update(line);counts['horizon_rows']+=1
                    with gzip.open(p/f'{arm}_windows.jsonl.gz','rt',encoding='utf-8') as f:
                        for l in f:
                            w=json.loads(l)
                            counts['windows']+=1
                            for row in stage_rows(w,task=task,root=root,arm=arm):
                                rid=row['record_id']
                                if rid in seen:
                                    raise ValueError('duplicate stage record')
                                seen.add(rid)
                                line=_line(row);gz.write(line);digest.update(line)
                                counts['stage_rows']+=1
                                counts[row['label_kind']]+=1
                                if row['outcome'] and row['outcome']['status']=='missing_unaudited':
                                    counts['missing_unaudited']+=1
    manifest={'schema':SCHEMA,'evidence_class':'DEVELOPMENT_ONLY','tasks':list(tasks),'roots':list(roots),
              'counts':dict(counts),'files':{
                  'stages_gym_v4.jsonl.gz':{'canonical_uncompressed_sha256':digest.hexdigest()},
                  'horizon_outcomes_gym_v4.jsonl':{'sha256':horizon_digest.hexdigest()}},
              'note':'Behavior rows are not positive supervision. Outcomes separate from stage-local features; no EVAL labels in decision rows.'}
    (export_dir/'manifest.json').write_bytes(_line(manifest))
    return manifest
