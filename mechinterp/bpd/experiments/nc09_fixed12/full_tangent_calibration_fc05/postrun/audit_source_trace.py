"""Replay immutable FC05 source traces without training or new searches.

This post-result auditor was published after the original calibration.
The report retains the original independent audit output.
"""
import argparse
from pathlib import Path
import base64, collections, csv, hashlib, json, math, sys, time, zipfile
import numpy as np
import torch
from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_calibration_fc05.audit import source_audit
from mechinterp.bpd.experiments.nc09_fixed12.full_tangent_fc04 import solver as frozen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root-dir", required=True, type=Path, help="Directory with the 187 extracted original files")
parser.add_argument("--archive", required=True, type=Path, help="Original run 44319021117 ZIP")
parser.add_argument("--out", required=True, type=Path, help="New output file outside the original evidence directory")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[6]
base = args.root_dir.resolve()
archive = args.archive.resolve()
out = args.out.resolve()
if out.exists():
    parser.error("Output already exists; preserve the original audit and choose a new path")
if out.is_relative_to(base):
    parser.error("Output must be outside the original evidence directory")
if not out.parent.is_dir():
    parser.error("Output directory must exist")
started = time.time()
torch.set_num_threads(1)
errors = []
def check(condition, label):
    if not condition:
        errors.append(label)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def values_close(actual, expected, label, atol=1e-10):
    check(bool(np.allclose(actual, expected, atol=atol, rtol=1e-10)), label)
def decode(receipt, label):
    b = base64.b64decode(receipt['b64_le_float64'], validate=True)
    check(hashlib.sha256(b).hexdigest() == receipt['sha256'], label + ': parameter SHA256')
    result = np.frombuffer(b, dtype='<f8').reshape(receipt['shape']).copy()
    check(np.isfinite(result).all(), label + ': finite parameters')
    return torch.from_numpy(result)

env = json.loads((base / 'ORIGINAL_ENVIRONMENT.json').read_text())
lock = json.loads((base / 'EXECUTION_LOCK.json').read_text())
manifest = json.loads((base / 'SOURCE_MANIFEST.json').read_text())
manifest_sha = sha(base / 'SOURCE_MANIFEST.json')
check(env['execution_commit_sha'] == '895433c4bc15c0985af6fc4936504bf055f76cdc', 'Execution commit')
check(env['source_manifest_sha256'] == manifest_sha == lock['source_manifest_sha256'], 'Manifest binding')
check(lock['model_roots'] == list(range(16000,16059)), 'Locked complete root roster')
for name, digest in manifest['files'].items():
    check(sha(repo / name) == digest, 'Source dependency: ' + name)
with (base / 'FC05_ORIGINAL_SHA256.csv').open() as f:
    file_rows = list(csv.DictReader(f))
for item in file_rows:
    p = base / item['file']
    check(p.stat().st_size == int(item['size_bytes']) and sha(p) == item['sha256'], 'Original file receipt: ' + p.name)
zip_sha = sha(archive)
check(zip_sha == '44b7ec94472b11444d7c71fc9668740d925c76367b27f51c0921556d93769986', 'Original ZIP SHA256')
check(archive.stat().st_size == 89927755, 'Original ZIP bytes')
with zipfile.ZipFile(archive) as z:
    check(len(z.namelist()) == 187, 'Original ZIP member count')
    check(set(z.namelist()) == {p.name for p in base.iterdir() if p.is_file()}, 'Archive and extracted member names')
    for name in z.namelist():
        check(hashlib.sha256(z.read(name)).hexdigest() == sha(base / name), 'Extracted bytes: ' + name)

census = collections.Counter()
rank_census = collections.Counter()
line_scale_attempts = collections.Counter()
line_scale_accepted = collections.Counter()
terminal = collections.Counter()
raw_delta_norms = []
condition_numbers = []
singular_minima = []
singular_maxima = []
updates_per_chart = []
source_extrema = collections.defaultdict(list)
rank_tuples = collections.Counter()
failed_charts = []
per_root = []
max_recomputed_source_target_error = 0.
max_recomputed_source_profile_error = 0.
for root in range(16000,16059):
    start_errors = len(errors)
    path = base / f'nc09_fc05_root_{root}.json'
    record = json.loads(path.read_text())
    adm = record['sham_admission']
    prefix = f'Root {root}'
    check(record['root'] == root and record['study'] == 'BPD-NC09-FC05', prefix + ': identity')
    check(record['protocol_sha256'] == env['protocol_sha256'], prefix + ': protocol hash')
    check(record['source_manifest_sha256'] == manifest_sha, prefix + ': source manifest')
    check(record['source_integrity']['verified_files'] == manifest['files'], prefix + ': dependency receipt')
    check(record['source_sha256'] == manifest['files']['mechinterp/bpd/experiments/nc09_fixed12/full_tangent_calibration_fc05/experiment.py'], prefix + ': experiment source')
    check(record['budget'] == lock['budgets'], prefix + ': full budget')
    check(record['eligible'] and record['source_audit_before_task_label_access'], prefix + ': source ordering and admission')
    source_path = base / record['source_archive']['file']
    audit = source_audit(record, source_path)
    check(audit['passed'] and audit['complete'], prefix + ': independent NumPy source audit ' + str(audit['errors']))
    ranks = record['source_rank12_optimum']['ranks']
    rank_tuples[tuple(ranks)] += 1
    for p in audit['projectors']:
        for name in ['symmetry_max','idempotence_max','trace_max_error','eigenvalue_max_error']:
            source_extrema[name].append(p['geometry'][name])
        for name in ['physical_rms_relative_error','signed_score_abs_error','score_rms_abs_error','distance_from_dp12','principal_cosine_max_error']:
            source_extrema[name].append(p[name])
        if p['minimum_pairwise_distance'] is not None:
            source_extrema['minimum_pairwise_distance'].append(p['minimum_pairwise_distance'])
        source_extrema['signed_tolerance_fraction'].append(p['signed_score_abs_error']/adm['source_signed_score_abs_tolerance'])
        source_extrema['rms_tolerance_fraction'].append(p['score_rms_abs_error']/adm['source_score_rms_abs_tolerance'])
        source_extrema['write_tolerance_fraction'].append(p['physical_rms_relative_error']/.15)
    for t, row in audit['passwise_source_candidates'].items():
        for r, c in row.items():
            source_extrema['all_candidate_eigengap_ratio'].append(c['eigengap_ratio'])
            source_extrema['all_candidate_retained_eigenvalue'].append(c['eigvalue_at_rank'])
    with np.load(source_path, allow_pickle=False) as a:
        arrays = {k:a[k].copy() for k in a.files}
    A = torch.from_numpy(arrays['a'])
    z = torch.from_numpy(arrays['z'])
    g = torch.from_numpy(arrays['g'])
    U = torch.from_numpy(arrays['U'])
    E = [torch.from_numpy(np.asfortranarray(arrays[f'E_{t}'])) for t in range(4)]
    C = [frozen.complementary_basis(e) for e in E]
    check(all(np.array_equal(c.numpy(),arrays[f'C_{t}']) for t,c in enumerate(C)), prefix + ': archived complements')
    target,tol,moments,gram = frozen.target_values(z,g,A,U,E)
    stored_target = [adm[k] for k in ['target_source_write_rms','target_source_signed_score_mean','target_source_score_rms']]
    max_recomputed_source_target_error = max(max_recomputed_source_target_error, max(abs(a-b) for a,b in zip(target,stored_target)))
    check(list(target) == stored_target, prefix + ': exact source target replay')
    dp = frozen.projectors_from_bases(E)
    accepted = []
    rng = np.random.default_rng(15_040_000 + root)
    root_updates = 0
    root_trials = 0
    root_ranks = collections.Counter()
    check(len(adm['charts']) == 32, prefix + ': 32 chart slots')
    for chart_idx, chart in enumerate(adm['charts']):
        cp = f'{prefix}, chart {chart_idx}'
        check(chart['chart'] == chart_idx, cp + ': chart ordering')
        census['chart_slots'] += 1
        census[chart['status']] += 1
        terminal[chart['terminal_reason']] += 1
        if chart['status'] == 'skipped_after_full':
            check(len(accepted) == 8 and not chart['iterations'], cp + ': skip after eight')
            continue
        check(len(accepted) < 8, cp + ': started before eighth admission')
        census['charts_started'] += 1
        initial = decode(chart['initial_parameters'], cp)
        random = torch.tensor(np.concatenate([rng.standard_normal((20-e.shape[1])*e.shape[1])/math.sqrt(20-e.shape[1]) for e in E]), dtype=torch.float64)
        check(torch.equal(initial, random), cp + ': original Gaussian RNG chart')
        next_y = initial
        actual_evals = 0
        chart_updates = 0
        previous_penalty = None
        for pos, state in enumerate(chart['iterations']):
            sp = f'{cp}, state {pos}'
            census['parameter_states'] += 1
            y = decode(state['parameters'], sp)
            check(state['iteration'] == pos and torch.equal(y,next_y), sp + ': exact parameter transition')
            leaf = y.detach().clone().requires_grad_(True)
            bases = frozen.tangent_chart(E,C,leaf)
            P = frozen.projectors_from_bases(bases)
            prof = frozen.source_tensor_profile(z,g,bases,moments,gram)
            actual_evals += 1
            f = frozen.normalized_source_residual(prof,target,tol)
            penalty = float(frozen.squared_constraint_excess(f).detach())
            rec = frozen.check_source_candidate(P,prof,target,tol,dp,accepted)
            check(frozen.projector_digest(P)['projector_sha256'] == state['projector_sha256'], sp + ': exact projector hash')
            check(rec == state['geometry_and_source'], sp + ': exact source and geometry replay')
            profile = [float(v.detach()) for v in prof]
            check(profile == state['profile'], sp + ': exact source profile replay')
            check(f.detach().numpy().tolist() == state['normalized_residual'] and penalty == state['penalty'], sp + ': residual and penalty replay')
            if rec['eligible']:
                check(chart['status'] == 'admitted' and pos == len(chart['iterations'])-1 and state['jacobian'] is None, sp + ': first eligible selection')
                selected = adm['projectors'][len(accepted)]
                check(selected['chart_index'] == chart_idx and selected['correction_updates'] == pos, sp + ': admission identity')
                check(frozen.projector_digest(P)['projector_sha256'] == selected['projector_sha256'], sp + ': selected projector hash')
                check(np.array_equal(P.detach().numpy(),arrays['sham_projectors'][len(accepted)]), sp + ': selected archive bytes')
                accepted.append(P.detach().clone())
            if state['jacobian'] is None:
                check(not state['trials'], sp + ': no trials without correction')
                continue
            check(not rec['eligible'] and penalty > 1e-14 and pos < 64, sp + ': correction conditions')
            J = torch.stack([torch.autograd.grad(f[i],leaf,retain_graph=True,create_graph=False)[0] for i in range(3)])
            sv = torch.linalg.svdvals(J)
            rank = int((sv > 1e-8).sum())
            cond = float(sv[0]/sv[-1]) if float(sv[-1]) > 1e-8 else None
            delta = -(J.T @ torch.linalg.solve(J@J.T + .001*torch.eye(3,dtype=torch.float64), f.detach()))
            raw_norm = float(delta.norm())
            raw_delta_norms.append(raw_norm)
            if raw_norm > .5:
                delta = delta * (.5/raw_norm)
                census['clipped_correction_directions'] += 1
            check(torch.equal(delta.detach(),decode(state['direction'],sp + ': direction')), sp + ': exact correction direction')
            info = state['jacobian']
            check(sv.detach().numpy().tolist() == info['jacobian_singular_values'], sp + ': exact singular values')
            check(rank == info['jacobian_rank'] and cond == info['jacobian_condition'], sp + ': rank and condition')
            check(float(delta.norm()) == info['direction_norm'] and float(delta.norm()) <= .5 + 1e-12, sp + ': clipped direction norm')
            root_ranks[rank] += 1
            rank_census[rank] += 1
            singular_minima.append(float(sv[-1]));singular_maxima.append(float(sv[0]))
            if cond is not None: condition_numbers.append(cond)
            root_updates += 1;chart_updates += 1;census['corrections'] += 1
            advanced = False
            for ix, trial in enumerate(state['trials']):
                tp = f'{sp}, trial {ix}'
                alpha = (1.,.5,.25,.125)[ix]
                check(trial['alpha'] == alpha and not advanced, tp + ': trial order and first advance')
                with torch.no_grad():
                    candidate = y + alpha*delta
                    trial_bases = frozen.tangent_chart(E,C,candidate)
                    trial_p = frozen.projectors_from_bases(trial_bases)
                    trial_prof = frozen.source_tensor_profile(z,g,trial_bases,moments,gram)
                    trial_f = frozen.normalized_source_residual(trial_prof,target,tol)
                    trial_penalty = float(frozen.squared_constraint_excess(trial_f))
                    trial_rec = frozen.check_source_candidate(trial_p,trial_prof,target,tol,dp,accepted)
                actual_evals += 1;root_trials += 1;census['line_search_trials'] += 1
                line_scale_attempts[str(alpha)] += 1
                check(frozen.projector_digest(trial_p)['projector_sha256'] == trial['projector_sha256'], tp + ': exact trial projector')
                check([float(v) for v in trial_prof] == trial['profile'], tp + ': exact trial profile')
                check(trial_f.numpy().tolist() == trial['normalized_residual'] and trial_penalty == trial['penalty'], tp + ': trial residual and penalty')
                check(trial_rec == trial['geometry_and_source'], tp + ': exact trial geometry and source')
                advance = trial_penalty < penalty - 1e-12
                check(advance == trial['advance'], tp + ': exact descent guard')
                if advance:
                    advanced = True;next_y = candidate.detach()
                    line_scale_accepted[str(alpha)] += 1
                    census['accepted_line_search_trials'] += 1
                    check(ix == len(state['trials'])-1, tp + ': immediate accepted scale')
                else:
                    census['rejected_line_search_trials'] += 1
            check(1 <= len(state['trials']) <= 4, sp + ': trial cap')
            if not advanced:
                check(len(state['trials']) == 4 and pos == len(chart['iterations'])-1 and chart['terminal_reason'] == 'no_accepted_descent_step', sp + ': no-advance termination')
        check(chart_updates == chart['correction_updates'] and chart_updates <=64, cp + ': correction count and cap')
        check(actual_evals == chart['source_metric_evaluations'], cp + ': source metric evaluation count')
        updates_per_chart.append(chart_updates)
        if chart['status'] != 'admitted':
            state = chart['iterations'][-1]
            failed_charts.append({'root':root,'chart':chart_idx,'status':chart['status'],'terminal_reason':chart['terminal_reason'],'correction_updates':chart_updates,'last_penalty':state['penalty'],'last_source_errors':chart['last_source_errors'],'final_trials':[{k:t[k] for k in ('alpha','penalty','advance','decision','profile','normalized_residual','projector_sha256')} for t in state['trials']],'later_accepted_chart_indices':[c['chart'] for c in adm['charts'][chart_idx+1:] if c['status']=='admitted']})
    check(root_updates == adm['total_jacobian_corrections'], prefix + ': correction census')
    check({str(k):root_ranks[k] for k in range(4)} == adm['jacobian_rank_census'], prefix + ': rank census')
    check(len(accepted) == 8 and adm['selected'] == 8, prefix + ': complete selected set')
    per_root.append({'root':root,'passed':len(errors)==start_errors,'original_json_sha256':sha(path),'source_archive_sha256':record['source_archive']['sha256'],'independent_numpy_source_audit_passed':audit['passed'],'selected_shams':len(accepted),'started_charts':adm['charts_actually_started'],'corrections':root_updates,'line_search_trials':root_trials,'rank_tuple':ranks,'error_count':len(errors)-start_errors})
    if (root-16000+1)%10 == 0:
        print(json.dumps({'audited_roots':root-16000+1,'elapsed_seconds':round(time.time()-started,2),'errors':len(errors)}),flush=True)

result = {
 'study':'BPD-NC09-FC05','audit_scope':'All original source archives and recorded solver states; no training and no new sampler search.',
 'original_run_id':44319021117,'execution_commit_sha':env['execution_commit_sha'],
 'original_archive':{'file':archive.name,'size_bytes':archive.stat().st_size,'sha256':zip_sha,'members':187},
 'source_manifest_sha256':manifest_sha,'source_dependency_files_verified':len(manifest['files']),
 'original_file_receipts_verified':len(file_rows),'root_roster':list(range(16000,16059)),
 'all_checks_passed':not errors,'errors':errors,'per_root':per_root,
 'trace_census':dict(census),'terminal_reasons':dict(terminal),'jacobian_rank_census':dict(rank_census),
 'line_search_scale_attempts':dict(line_scale_attempts),'line_search_scale_acceptances':dict(line_scale_accepted),
 'corrections_per_started_chart':{'min':min(updates_per_chart),'max':max(updates_per_chart),'median':float(np.median(updates_per_chart)),'mean':float(np.mean(updates_per_chart))},
 'correction_extrema':{'minimum_smallest_singular_value':min(singular_minima),'maximum_largest_singular_value':max(singular_maxima),'minimum_condition_number':min(condition_numbers),'maximum_condition_number':max(condition_numbers),'minimum_unclipped_step_norm':min(raw_delta_norms),'maximum_unclipped_step_norm':max(raw_delta_norms)},
 'accepted_source_and_geometry_extrema':{k:{'min':min(v),'max':max(v)} for k,v in source_extrema.items()},
 'rank_tuple_census':{'-'.join(map(str,k)):v for k,v in rank_tuples.items()},
 'failed_charts':failed_charts,
 'source_target_exact_replay_max_abs_error':max_recomputed_source_target_error,
 'limits':['The NumPy source audit independently recomputes source constraints and geometry from archived tensors. Trace and Jacobian replay uses the frozen helper arithmetic on the original saved source values with the original E strides restored.','This audit checks existing source evidence. It does not independently retrain models, regenerate discovery gradients from saved model states, or establish task-sham exchangeability.','No original outcome, root, source file, or threshold was modified.'],
 'elapsed_seconds':time.time()-started,
}
out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ['all_checks_passed','trace_census','terminal_reasons','jacobian_rank_census','line_search_scale_attempts','line_search_scale_acceptances','corrections_per_started_chart','correction_extrema','accepted_source_and_geometry_extrema','rank_tuple_census','elapsed_seconds']},indent=2),flush=True)
print('AUDIT_JSON',str(out),flush=True)
if errors:print('ERROR_EXAMPLES',errors[:10],flush=True)

raise SystemExit(1 if errors else 0)
