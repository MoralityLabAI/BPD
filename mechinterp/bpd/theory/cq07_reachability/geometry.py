"""BPD-CQ-07 engineering: reachable first states versus constrained BPD writes.

Reuses the exact CQ-06 learned nonlinear transition and independent source
worlds. This feasibility audit is development-only; no CQ-06 sealed cohort is
read, and no CQ-07 confirmatory claims are made.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import torch
from mechinterp.bpd.theory.cq06_nonlinear_feedback import experiment as Q

DEVELOPMENT=(997,1009,1013,1021,1033,1039,1049,1051)
ROOT=Path(__file__).resolve().parent
N_PAIRS=128
BANK_N=4096
HEAD_TOL=0.30
DOSE_BAND=0.15
MAX_ALPHA=1.5
OPT_STEPS=200
OPT_LR=.045
NONZERO=1e-6
RANK_TOL=1e-6
REACH_TOL=1e-4


def legal_preactivation_geometry(seed, teacher, factors):
    """All *first-visit* legal preactivations are p(x,y)=base+[x,y]@A.

    Every first cue has a rotated (x,y,0,-1) and the initial state is zero.
    Thus A is 2×12. The selected CQ-06 BPD write span B is also 2×12.
    If stack(A,B) has rank 4, their intersection is {0} exactly over R.
    """
    w=factors.weight().detach().double()
    q=Q.rotation(seed).double()
    a=q[:2]@w[Q.D:,:]
    b=factors.U[:2].detach().double()
    base=-q[3]@w[Q.D:,:]+teacher.bias.detach().double()
    singular=torch.linalg.svdvals(torch.cat((a,b),dim=0))
    rank=int((singular>RANK_TOL).sum())
    assert tuple(a.shape)==(2,Q.D)
    assert tuple(b.shape)==(2,Q.D)
    return {'input_basis':a,'bpd_basis':b,'base':base,
            'combined_rank':rank,'combined_singular_values':[float(s) for s in singular],
            'min_singular_value':float(singular[-1]),
            'exact_nonzero_reachability_obstructed':rank==4}


def affine_residual(pre, basis):
    """Distance of a preactivation displacement to legal first-input span."""
    b=basis.double()
    p=pre.double()
    coeff=(p@b.T)@torch.linalg.inv(b@b.T)
    return (p-coeff@b).norm(dim=1),coeff


def constrained_rank2_state(teacher,factors,recipient_cue,donor_cue,recipient_state,
                             donor_state,head_tol=HEAD_TOL,dose_band=DOSE_BAND,
                             n_steps=OPT_STEPS):
    """Only c0,c1 act. Optimize without any donor target or future readout.

    Objective fixes current head and physical dose near the *clean full donor*
    state-space dose. Always emits tanh(pre+U01*alpha), therefore in range;
    this does NOT imply reachability from a legal first cue.
    """
    n=len(recipient_cue)
    r0=torch.cat((torch.zeros(n,Q.D),recipient_cue),1)
    d0=torch.cat((torch.zeros(n,Q.D),donor_cue),1)
    with torch.no_grad():
        pre=(r0@factors.weight()+teacher.bias).detach()
        acts=((d0-r0)@factors.V[:,:2]).detach()
        write=factors.U[:2].detach()
        current=Q.logits(teacher,recipient_state).detach()
        target_dose=(donor_state-recipient_state).norm(dim=1).detach()
        head=teacher.head.weight.detach()[0]
    alpha=torch.ones((n,2),requires_grad=True)
    optimizer=torch.optim.Adam([alpha],lr=OPT_LR)
    for _ in range(n_steps):
        delta=(acts*alpha)@write
        out=torch.tanh(pre+delta)
        he=(out@head-current)/head_tol
        de=((out-recipient_state).norm(dim=1)-target_dose)/dose_band
        loss=(10*he.square()+10*de.square()+.015*(alpha-1).square().sum(1)).mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():alpha.clamp_(-MAX_ALPHA,MAX_ALPHA)
    with torch.no_grad():
        delta=(acts*alpha)@write
        state=torch.tanh(pre+delta)
        head_shift=(state@head-current).abs()
        physical_dose=(state-recipient_state).norm(dim=1)
        dose_error=(physical_dose-target_dose).abs()
        solver_feasible=(head_shift<=head_tol)&(dose_error<=dose_band)
        out_of_range=(state.abs()>1.0000001).any(1)
        raw_delta=(acts@write)
        raw_state=torch.tanh(pre+raw_delta)
    return {'state':state,'pre':pre,'delta_pre':delta,'alpha':alpha.detach(),
            'feasible':solver_feasible,'head_abs':head_shift,'dose':physical_dose,
            'dose_target':target_dose,'dose_error':dose_error,'out_of_range':out_of_range,
            'raw_state':raw_state,'raw_delta_pre':raw_delta}


def clean_bank(seed,teacher,factors,offset,n=BANK_N):
    """Actual first-pass states from independent legal source cues."""
    u0,_,labels=Q.source(seed,n,offset)
    with torch.no_grad():
        s1=Q.independent_step(teacher,factors,torch.zeros(n,Q.D),u0)
        h=Q.logits(teacher,s1)
        x=(u0@Q.rotation(seed).T)[:,0]
    return s1,h,labels[:,0],x


def match_natural_states(seed,teacher,factors,recipient_state,recipient_label,
                         patch_dose,head_tol=HEAD_TOL,dose_band=DOSE_BAND,
                         bank_n=BANK_N,offset=951000):
    """Choose among *reachable* states; no rescaling, projection, interpolation.

    Candidate selection uses only first-label equality, actual head shift and
    Euclidean physical dose. It never uses the donor's delayed label or x sign.
    Returns a zero fallback for unmatched cases; callers must count failures.
    """
    bank,head,labels,x=clean_bank(seed,teacher,factors,offset,bank_n)
    with torch.no_grad():
        current=Q.logits(teacher,recipient_state)
        dist=torch.cdist(recipient_state,bank)
        herr=(current[:,None]-head[None,:]).abs()
        derr=(dist-patch_dose[:,None]).abs()
        valid=(labels[None,:]==recipient_label[:,None]) & (herr<=head_tol) & (derr<=dose_band)
        score=herr/head_tol+derr/dose_band
        score=torch.where(valid,score,torch.inf)
        chosen=score.argmin(1)
        available=valid.any(1)
        selected=bank[chosen]
        selected_x=x[chosen]
        # Clean state provenance holds for all candidates; unavailable results
        # are descriptive only and never used as successful interventions.
        return {'state':selected,'available':available,'chosen_index':chosen,
                'selected_x_sign':torch.sign(selected_x),
                'candidate_counts':valid.sum(1),
                'head_error':herr.gather(1,chosen[:,None])[:,0],
                'dose_error':derr.gather(1,chosen[:,None])[:,0],
                'min_bank_state_dist':dist.min(1).values}


def quantiles(x):
    return [float(v) for v in torch.quantile(x.float(),torch.tensor([0.,.25,.5,.75,.9,1.]))]


def evaluate_root(seed,backend='independent',n_pairs=N_PAIRS,bank_n=BANK_N,
                  opt_steps=OPT_STEPS,gate_updates=500,record_pairs=True):
    Q.configure();torch.manual_seed(seed)
    if seed not in DEVELOPMENT:raise ValueError('CQ07 feasibility only on excluded development roots')
    t=Q.train_teacher(seed);f=Q.train_factors(seed,t)
    diag=Q.canonicalize(f)
    gate,planner,_=Q.train_policies(seed,t,f,n=4096,updates=gate_updates)
    geom=legal_preactivation_geometry(seed,t,f)
    ur,ud,u1,yr,yd=Q.donor_inputs(seed,n_pairs)
    with torch.no_grad():
        _,sr,_=Q.both(t,f,ur,u1)
        _,sd,_=Q.both(t,f,ud,u1)
    constrained=constrained_rank2_state(t,f,ur,ud,sr,sd,n_steps=opt_steps)
    with torch.no_grad():
        p=constrained['state'];feasible=constrained['feasible']
        n1=match_natural_states(seed,t,f,sr,yr[:,0],constrained['dose'],bank_n=bank_n,offset=951000)
        n2=match_natural_states(seed,t,f,sr,yr[:,0],constrained['dose'],bank_n=bank_n,offset=953000)
        joint1=feasible & n1['available']
        joint2=feasible & n2['available']
        joint12=feasible & n1['available'] & n2['available']
        raw_res,raw_xy=affine_residual(constrained['raw_delta_pre'],geom['input_basis'])
        optimized_res,optimized_xy=affine_residual(constrained['delta_pre'],geom['input_basis'])
        nonzero=constrained['delta_pre'].norm(dim=1)>NONZERO
        # Source-controlled null gets the *same first parity and physical dose
        # bin*, with a genuinely legal first cue and no post-hoc projected write.
        target=yd[:,1]
        native=Q.NativeModel(t,f) if backend=='native' else None
        def predict(states):
            mask=gate.mask(states,u1)
            out=Q.step(t,f,states,u1,mask,backend,native,tau=1)
            return Q.logits(t,out)*target>0
        yp=predict(p);yn1=predict(n1['state']);yn2=predict(n2['state'])
        yd0=predict(sd);ysh=predict(sr);yrw=predict(constrained['raw_state'])
        # Primary comparisons require joint feasibility and score any
        # unfeasible or unmatched pair as a failure in the ITT denominator.
        comp={'bpd_joint':int((joint1&yp).sum()),
              'reachable_null_1_joint':int((joint1&yn1).sum()),
              'reachable_null_2_joint':int((joint12&yn2).sum()),
              'bpd_joint_both_nulls':int((joint12&yp).sum()),
              'null1_joint_both_nulls':int((joint12&yn1).sum()),
              'full_donor_oracle':int(yd0.sum()),
              'sham':int(ysh.sum()),
              'raw_constrained_tanh_bpd':int(yrw.sum()),
              'bpd_feasible_only':int((feasible&yp).sum())}
        bx=torch.sign((ud@Q.rotation(seed).T)[:,0])
        selected_x_matches_donor=[int(((a['selected_x_sign']==bx)&a['available']&feasible).sum()) for a in (n1,n2)]
        result={
            'seed':seed,'backend':backend,'n_pairs':n_pairs,'count_unit':'one independently trained rotated recurrence seed',
            'model_geometry':{**diag,'combined_input_plus_bpd_rank':geom['combined_rank'],
                'combined_min_singular_value':geom['min_singular_value'],
                'combined_singular_values':geom['combined_singular_values'],
                'zero_intersection_certificate':geom['exact_nonzero_reachability_obstructed']},
            'counts':comp,
            'feasibility':{
                'patch_solver_feasible':int(feasible.sum()),
                'natural_null_1_matched':int(n1['available'].sum()),
                'natural_null_2_matched':int(n2['available'].sum()),
                'joint_patch_null1':int(joint1.sum()),'joint_patch_both_nulls':int(joint12.sum()),
                'nonzero_rank2_writes':int(nonzero.sum()),
                'exactly_reachable_nonzero_rank2_patches':int(((optimized_res<=REACH_TOL)&nonzero).sum()),
                'range_violations_bpd':int(constrained['out_of_range'].sum()),
                'range_violations_clean_null1':int((n1['state'].abs()>1.0000001).any(1).sum()),
                'range_violations_full_donor':int((sd.abs()>1.0000001).any(1).sum()),
                'clean_null1_matched_target_x_sign':selected_x_matches_donor[0],
                'clean_null2_matched_target_x_sign':selected_x_matches_donor[1]},
            'geometry':{'raw_preactivation_residual_quantiles':quantiles(raw_res),
                'optimized_preactivation_residual_quantiles':quantiles(optimized_res),
                'head_error_quantiles':quantiles(constrained['head_abs']),
                'dose_error_quantiles':quantiles(constrained['dose_error']),
                'physical_dose_quantiles':quantiles(constrained['dose']),
                'full_donor_dose_quantiles':quantiles(constrained['dose_target']),
                'null1_matched_candidates_quantiles':quantiles(n1['candidate_counts']),
                'null2_matched_candidates_quantiles':quantiles(n2['candidate_counts'])},
            'registration':{'head_logit_tolerance':HEAD_TOL,'physical_dose_bin_halfwidth':DOSE_BAND,
                'optimizer_steps':opt_steps,'solver_alpha_range':[-MAX_ALPHA,MAX_ALPHA],
                'independent_clean_bank_per_null':bank_n,
                'primary_intervention':'tanh constrained preactivation rank-2 BPD write',
                'primary_null':'independent legal first-cue reachable state matched on parity/head/actual dose',
                'primary_denominator':'all generated donor pairs; joint failure scored zero'},
        }
        if backend=='native':
            ind1=Q.step(t,f,p,u1,gate.mask(p,u1),'independent',None,tau=1)
            nat1=Q.step(t,f,p,u1,gate.mask(p,u1),'native',native,tau=1)
            ind2=Q.step(t,f,n1['state'],u1,gate.mask(n1['state'],u1),'independent',None,tau=1)
            nat2=Q.step(t,f,n1['state'],u1,gate.mask(n1['state'],u1),'native',native,tau=1)
            result['max_native_state_parity_error']=max(float((ind1-nat1).abs().max()),float((ind2-nat2).abs().max()))
        if record_pairs:
            result['pair_outcomes']={
                'bpd':yp.int().tolist(),'null_1':yn1.int().tolist(),'null_2':yn2.int().tolist(),
                'full_donor':yd0.int().tolist(),'sham':ysh.int().tolist(),
                'patch_feasible':feasible.int().tolist(),
                'null1_matched':n1['available'].int().tolist(),
                'null2_matched':n2['available'].int().tolist(),
                'joint_patch_null1':joint1.int().tolist(),
                'patch_dose':[float(x) for x in constrained['dose']],
                'patch_head_abs':[float(x) for x in constrained['head_abs']],
                'patch_reachability_preact_residual':[float(x) for x in optimized_res],
                'matched_null1_target_x_sign':((n1['selected_x_sign']==bx)&n1['available']).int().tolist()}
        return result


def cohort(seeds=DEVELOPMENT,backend='independent',n_pairs=N_PAIRS,bank_n=BANK_N,
           opt_steps=OPT_STEPS,gate_updates=500,record_pairs=True):
    rows=[evaluate_root(seed,backend,n_pairs,bank_n,opt_steps,gate_updates,record_pairs) for seed in seeds]
    return {'schema':'bpd.cq07.reachability.feasibility.v1',
            'evidence_class':'excluded-root engineering feasibility, no sham calibration or confirmation',
            'seeds':list(seeds),'backend':backend,'rows':rows,
            'totals':{k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']},
            'feasibility_totals':{k:sum(r['feasibility'][k] for r in rows) for k in rows[0]['feasibility']},
            'pairs_per_root':n_pairs,'torch':torch.__version__,'python':platform.python_version()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=('independent','native'),default='independent')
    p.add_argument('--seeds',nargs='*',type=int,default=list(DEVELOPMENT))
    p.add_argument('--out',type=Path)
    p.add_argument('--smoke',action='store_true',help='two excluded roots; 32 pairs and smaller reservoir')
    p.add_argument('--no-pairs',action='store_true')
    a=p.parse_args()
    if any(s not in DEVELOPMENT for s in a.seeds):raise ValueError('only development roots permitted')
    seeds=DEVELOPMENT[:2] if a.smoke else a.seeds
    res=cohort(seeds,a.backend,n_pairs=32 if a.smoke else N_PAIRS,
            bank_n=1024 if a.smoke else BANK_N,
            opt_steps=100 if a.smoke else OPT_STEPS,
            gate_updates=200 if a.smoke else 500,record_pairs=not a.no_pairs)
    print(json.dumps({'seeds':res['seeds'],'totals':res['totals'],
                     'feasibility':res['feasibility_totals']},indent=2))
    for row in res['rows']:
        print('ROOT',row['seed'],'rank',row['model_geometry']['combined_input_plus_bpd_rank'],
              'min-sv',row['model_geometry']['combined_min_singular_value'],
              'patch feasible',row['feasibility']['patch_solver_feasible'],
              'joint',row['feasibility']['joint_patch_null1'],
              'pair matched hits',row['counts']['bpd_joint'],row['counts']['reachable_null_1_joint'])
    if a.out:
        if a.out.exists():raise FileExistsError('refuse overwriting CQ-07 engineering receipt')
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(res,sort_keys=True,indent=2)+'\n')

if __name__=='__main__':main()
