"""NC13: prospective source-context and delivered-displacement diagnostic.

Only the excluded smoke and the sixteen protocol-listed development pairs are
enabled. Every run reserves its output prefix before any model initialization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.experiments.nc02.experiment import PASSES, WIDTH, sample_x, parity, binary_kl
from mechinterp.bpd.experiments.nc03.experiment import basis_transport
from mechinterp.bpd.experiments.nc04.experiment import DenseLooper, factor_basis
from mechinterp.bpd.experiments.nc05.experiment import projectors_at_ranks
from mechinterp.bpd.experiments.nc10.experiment import task_labels, procrustes, prewrite_states, transport, score_logits
from mechinterp.bpd.experiments.nc11.experiment import read_and_score_jacobian, unit_jacobians, projected_with_gain
from .dose import dose_metrics, energy_gain, calibrate_closed_displacement, select_shams

PROTOCOL_COMMIT = "069b355e8f37ec780dc6c46b82176b6f32f8e39b"
ROOTS = tuple(range(2601300, 2601316))
SMOKE = 2601299
RANK = 4
RIDGE = .05
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BUDGETS = {
    "steps":384, "train_A":2048, "train_B":2048, "source_pool":4096,
    "source_per_condition":1024, "calibration_pool":512,
    "activation_calibration":128, "target_gradient_union":8,
    "dose":1024, "evaluation":1024, "sham_proposals":2048, "shams":8,
    "search_grid":17, "search_bisection":24,
}
ARM_NAMES = ("mixed", "shared_E", "shared_D", "target_E", "target_D",
             "permuted_E", "permuted_D") + tuple(
                 f"sham_{policy}_{j}" for policy in ("E", "D") for j in range(8))
ENDPOINT_ARMS = {
    "context_gain_E":("mixed", "shared_E"),
    "context_gain_D":("mixed", "shared_D"),
    "shared_gap_reduction_D":("shared_E", "target_E", "shared_D", "target_D"),
    "target_minus_shared_D":("target_D", "shared_D"),
    "median_sham_minus_shared_D":("shared_D",) + tuple(f"sham_D_{j}" for j in range(8)),
    "shared_D_minus_clean":("shared_D", "B_clean"),
}


def _root_guard(root: int) -> None:
    if type(root) is not int or (root not in ROOTS and root != SMOKE):
        raise ValueError("NC13 permits excluded smoke 2601299 and roots 2601300..2601315 only")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array(value) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _array_hash(value) -> str:
    a = np.ascontiguousarray(_array(value))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def model_state_hash(model) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(_array_hash(value).encode())
    return h.hexdigest()


def _safe(value):
    if isinstance(value, dict):
        return {str(k):_safe(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(_safe(payload), f, sort_keys=True, separators=(",", ":"), allow_nan=False)
        f.write("\n")


def _write_npz(path: Path, arrays: dict) -> None:
    with path.open("xb") as f:
        np.savez_compressed(f, **{k:_array(v) for k,v in arrays.items()})


def _snapshot_weights(models: dict) -> dict:
    return {f"{task}__{name}":value.detach().clone().cpu().numpy()
            for task, model in models.items() for name,value in model.state_dict().items()}


def train_task(root: int, task: str, models: dict | None = None):
    """Fixed NC12 optimizer recipe on this study's fresh allowlisted roots."""
    _root_guard(root)
    if task not in ("A", "B"):
        raise ValueError("task must be A or B")
    model_seed = root if task == "A" else root + 100000
    split = 1 if task == "A" else 11
    torch.manual_seed(model_seed)
    model = DenseLooper()
    if models is not None:
        models[task] = model
    x = sample_x(root, split, 2048)
    signs = torch.where(task_labels(x, task), 1., -1.)
    rng = torch.Generator().manual_seed(model_seed + 1737701)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    for _ in range(384):
        ids = torch.randint(len(x), (128,), generator=rng)
        logits, _ = model(x[ids])
        loss = F.softplus(-2 * signs[ids] * logits[:, -1]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        accuracy = float(((logits[:,-1]>=0)==task_labels(x,task)).float().mean())
        density = float((model.mix.weight.detach().abs()>=1e-4).float().mean())
    return model, {
        "root":root, "task":task, "model_seed":model_seed, "training_split":split,
        "training_size":2048, "steps":384, "batch_size":128, "learning_rate":.005,
        "train_accuracy":accuracy, "mix_density":density,
        "last_minibatch_nll_nats":float(loss.detach()),
        "training_inputs_sha256":_array_hash(x), "model_state_sha256":model_state_hash(model),
    }


def regularized_alignment(a: Tensor, b: Tensor, prior: Tensor) -> tuple[Tensor, dict, Tensor]:
    if a.shape != (8, PASSES, WIDTH) or b.shape != a.shape:
        raise ValueError("alignment requires exactly eight paired gradient rows")
    m = torch.einsum("nti,ntj->tij", a.double(), b.double()) / 8 + RIDGE * prior.double()
    u, singular, vh = torch.linalg.svd(m, full_matrices=False)
    r = u @ vh
    residual = (torch.einsum("nti,tij->ntj",a.double(),r)-b.double()).norm(dim=(0,2))
    residual = residual / b.double().norm(dim=(0,2)).clamp_min(1e-12)
    minimum = singular.min(-1).values
    conditions = singular.max(-1).values / minimum
    diag = {
        "q":8, "ridge":RIDGE, "singular_values_by_pass":singular.tolist(),
        "condition_number_by_pass":[float(v) if math.isfinite(float(v)) else None for v in conditions],
        "minimum_singular_value_by_pass":minimum.tolist(),
        "orthogonality_max_abs":float((r.transpose(-1,-2)@r-torch.eye(WIDTH,dtype=r.dtype)).abs().max()),
        "normalized_gradient_fit_residual_by_pass":residual.tolist(),
        "admitted":bool(torch.isfinite(singular).all() and (minimum>1e-12).all()),
    }
    return r, diag, singular


def _rank_four(grads: Tensor) -> tuple[Tensor | None, Tensor, dict, Tensor]:
    projectors, diagnostics, bases = projectors_at_ranks(grads)
    gram = torch.einsum("nti,ntj->tij",grads.double(),grads.double()) / len(grads)
    spectrum = torch.linalg.eigvalsh(gram)
    diag = dict(diagnostics[4])
    diag["all_eigenvalues_by_pass"] = spectrum.tolist()
    diag["zero_gradient_rows"] = int((grads.double().norm(dim=-1)<=1e-12).sum())
    p = projectors[4]
    if diag["zero_gradient_rows"]:
        p = None
    diag["admitted"] = p is not None
    return p, bases[4], diag, spectrum


@torch.no_grad()
def _clean_capture(bpd, model, x):
    with bpd.target_capture():
        logits, states = model(x, capture=True)
        acts = torch.stack(model.acts, dim=1).double()
        writes = torch.stack(model.writes, dim=1)
    return logits, writes, acts, states


def _rms(x: Tensor) -> float:
    return float(x.double().square().mean().sqrt())


def _delivered_detail(changed_w: Tensor, clean_w: Tensor, x: Tensor) -> dict:
    delta = changed_w.double() - clean_w.double()
    out = {"rms":_rms(delta), "rms_by_pass":delta.square().mean((0,2)).sqrt().tolist(),
           "by_context":{}}
    for name, mask in (("shared", x[:,4]>0), ("specific", x[:,4]<=0)):
        out["by_context"][name] = {
            "n":int(mask.sum()), "rms":_rms(delta[mask]),
            "rms_by_pass":delta[mask].square().mean((0,2)).sqrt().tolist(),
        }
    return out


def _score(clean, changed, clean_w, changed_w, x, y):
    result = score_logits(clean, changed, clean_w, changed_w, x, y)
    clean_ok = (clean[:,-1]>=0)==y
    changed_ok = (changed[:,-1]>=0)==y
    result["correct_to_wrong"] = int((clean_ok & ~changed_ok).sum())
    result["wrong_to_correct"] = int((~clean_ok & changed_ok).sum())
    result["correct_to_correct"] = int((clean_ok & changed_ok).sum())
    result["wrong_to_wrong"] = int((~clean_ok & ~changed_ok).sum())
    result["delivered_write_displacement"] = _delivered_detail(changed_w,clean_w,x)
    kl = binary_kl(clean, changed).clamp_min(0)
    for name, mask in (("shared",x[:,4]>0), ("specific",x[:,4]<=0)):
        result["by_context"][name].update({
            "correct_to_wrong":int((clean_ok[mask] & ~changed_ok[mask]).sum()),
            "wrong_to_correct":int((~clean_ok[mask] & changed_ok[mask]).sum()),
            "correct_to_correct":int((clean_ok[mask] & changed_ok[mask]).sum()),
            "wrong_to_wrong":int((~clean_ok[mask] & ~changed_ok[mask]).sum()),
            "kl_by_pass_nats":kl[mask].mean(0).tolist(),
        })
    return result


def _source_hashes() -> dict:
    runner = Path(__file__).resolve()
    result = {str(runner.relative_to(REPO)):_sha(runner)}
    for name, module in sorted(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if (name == "mechinterp" or name.startswith("mechinterp.")) and file and Path(file).suffix == ".py":
            p = Path(file).resolve()
            if p.is_relative_to(REPO):
                result[str(p.relative_to(REPO))] = _sha(p)
    result[str((HERE/"PROTOCOL.md").relative_to(REPO))] = _sha(HERE/"PROTOCOL.md")
    return result


def _contrast_values(scores: dict, admissions: dict) -> dict:
    def n(name):
        return float(scores[name]["by_context"]["shared"]["final_nll_nats"])
    calculations = {
        "context_gain_E":lambda:n("mixed")-n("shared_E"),
        "context_gain_D":lambda:n("mixed")-n("shared_D"),
        "shared_gap_reduction_D":lambda:(n("shared_E")-n("target_E"))-(n("shared_D")-n("target_D")),
        "target_minus_shared_D":lambda:n("target_D")-n("shared_D"),
        "median_sham_minus_shared_D":lambda:float(np.median([n(f"sham_D_{j}") for j in range(8)]))-n("shared_D"),
        "shared_D_minus_clean":lambda:n("shared_D")-n("B_clean"),
    }
    out = {}
    for name, required in ENDPOINT_ARMS.items():
        reasons = [f"{a}:{r}" for a in required for r in admissions[a]["reasons"]]
        missing = [a for a in required if a not in scores]
        reasons.extend(f"unscored:{a}" for a in missing)
        value = calculations[name]() if not missing else None
        out[name] = {"value":value, "admitted":not reasons, "reasons":reasons,
                     "required_arms":list(required)}
    return out


def run(root: int, outdir: str | Path) -> dict:
    """Run the frozen recipe once, retaining attempted and failed-root artifacts."""
    _root_guard(root)
    from .verify_source import verify_source
    lock_sha = verify_source(require_runtime=root != SMOKE)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"root_{root}"
    if any(outdir.glob(prefix + "*")):
        raise FileExistsError(f"existing NC13 root artifacts block rerun: {prefix}")
    paths = {name:outdir/f"{prefix}_{name}.npz" for name in ("weights", "choices", "eval")}
    paths.update(receipt=outdir/f"{prefix}.json", frozen=outdir/f"{prefix}_frozen.json",
                 started=outdir/f"{prefix}.started.json", failure=outdir/f"{prefix}.failure.json")
    _write_json(paths["started"], {"study":"BPD-NC-13", "root":root,
                "started_utc":datetime.now(timezone.utc).isoformat(), "source_lock_sha256":lock_sha})
    models = {}
    try:
        return _execute(root, models, paths, lock_sha)
    except BaseException as exc:
        partial_path = outdir/f"{prefix}_failure_weights.npz"
        if models and not partial_path.exists():
            _write_npz(partial_path, _snapshot_weights(models))
        _write_json(paths["failure"], {
            "study":"BPD-NC-13", "root":root, "complete":False,
            "phase":"EXCLUDED_SMOKE" if root == SMOKE else "DEVELOPMENT_ENGINEERING",
            "source_lock_sha256":lock_sha, "exception_type":type(exc).__name__,
            "exception":str(exc), "traceback":traceback.format_exc(),
            "artifacts_preserved":[p.name for p in paths.values() if p.exists()],
            "partial_weights":partial_path.name if partial_path.exists() else None,
        })
        raise


def _execute(root: int, models: dict, paths: dict, lock_sha: str) -> dict:
    torch.set_num_threads(1)
    rec = {
        "study":"BPD-NC-13", "root":root, "complete":False,
        "phase":"EXCLUDED_SMOKE" if root == SMOKE else "DEVELOPMENT_ENGINEERING",
        "protocol_commit":PROTOCOL_COMMIT, "source_lock_sha256":lock_sha,
        "budgets":dict(BUDGETS), "global_failures":[], "admissions":{}, "scores":{},
        "training":{}, "construction":{}, "dose":{}, "calibrations":{},
        "target_access":{"original_gradient_inputs":0, "modified_zero_head_gradient_inputs":0,
                         "forward_calls":{}},
        "runtime":{"python":sys.version, "torch":str(torch.__version__),
                   "numpy":np.__version__, "platform":platform.platform(), "torch_threads":1,
                   "torch_build":torch.__config__.show()},
    }
    def count(category, calls=1, cases=1024):
        ledger = rec["target_access"]["forward_calls"].setdefault(category,{"calls":0,"input_rows":0})
        ledger["calls"] += calls
        ledger["input_rows"] += calls*cases
    a, ta = train_task(root, "A", models)
    b, tb = train_task(root, "B", models)
    rec["training"] = {"A":ta, "B":tb}
    count("training",384,128)
    count("training_accuracy",1,2048)
    _write_npz(paths["weights"], _snapshot_weights(models))
    ba = BPDModel(a,{"mix":WIDTH},max_passes=PASSES,
                  ablation=AblationConfig(mode="zero",sphere_rescale=False))
    bb = BPDModel(b,{"mix":WIDTH},max_passes=PASSES,
                  ablation=AblationConfig(mode="zero",sphere_rescale=False))
    factor_basis(ba,"native",root)
    factor_basis(bb,"native",root+100000)
    arrays = {}
    xsource = sample_x(root,2,4096)
    xcal = sample_x(root,4,512)
    xdose = sample_x(root,6,1024)
    mixed_ids = torch.arange(1024)
    positive_ids = torch.nonzero(xsource[:,4]>0,as_tuple=True)[0]
    shared_ids = positive_ids[:1024]
    activation_ids = torch.nonzero(xcal[:,4]>0,as_tuple=True)[0][:128]
    union_ids = torch.unique(torch.cat((mixed_ids,shared_ids)), sorted=True)
    rec["data"] = {"source_positive_count":len(positive_ids),
        "activation_positive_count":int((xcal[:,4]>0).sum()),
        "source_gradient_unique_rows":len(union_ids), "source_gradient_overlap":2048-len(union_ids),
        "source_gradient_rows_per_condition":{"mixed":len(mixed_ids),"shared":len(shared_ids)},
        "A_calibration_gradient_rows":min(8,len(activation_ids)),
        "input_sha256":{"source_pool":_array_hash(xsource),"calibration_pool":_array_hash(xcal),
                        "dose_pool":_array_hash(xdose)}}
    arrays.update(source_pool=xsource,calibration_pool=xcal,dose_inputs=xdose,
                  mixed_source_indices=mixed_ids,shared_source_indices=shared_ids,
                  source_gradient_union_indices=union_ids,activation_indices=activation_ids,
                  paired_gradient_indices=activation_ids[:8])
    if len(shared_ids)!=1024:
        rec["global_failures"].append("insufficient_shared_source_candidates")
    if len(activation_ids)!=128:
        rec["global_failures"].append("insufficient_activation_candidates")
    close_a, close_b = parity(ba,a,xdose[:64]), parity(bb,b,xdose[:64])
    count("native_parity",2,64)
    rec["closure"] = {"A":close_a,"B":close_b}
    if not close_a["passed"] or not close_b["passed"]:
        rec["global_failures"].append("native_fullmask_parity")
    dose_logits, dose_writes, dose_acts, _ = _clean_capture(bb,b,xdose)
    count("clean_dose_reference")
    u = bb.wrappers["mix"].U.detach().double().clone()
    arrays.update(dose_clean_logits=dose_logits,dose_clean_writes=dose_writes,
                  dose_clean_reads=dose_acts,B_physical_U=u)
    source_p, source_e = {}, {}
    if len(shared_ids)==1024:
        _, ga_union = read_and_score_jacobian(ba,a,xsource[union_ids])
        arrays["A_source_union_gradients"] = ga_union
        for name, ids in (("mixed",mixed_ids),("shared",shared_ids)):
            g = ga_union[torch.searchsorted(union_ids,ids)]
            p,e,d,s = _rank_four(g)
            source_p[name], source_e[name] = p,e
            rec["construction"][f"source_{name}"] = d
            arrays[f"source_{name}_eigenvalues"] = s
            arrays[f"source_{name}_basis"] = e
            if p is None:
                rec["global_failures"].append(f"source_{name}_rank_or_zero_gradient")
            else:
                arrays[f"source_{name}_projector"] = p
    projects, bases = {}, {}
    if len(activation_ids)==128:
        xact = xcal[activation_ids]
        aa, ab = prewrite_states(ba,a,xact), prewrite_states(bb,b,xact)
        count("activation_prior",1,128)
        prior, prior_diag = procrustes(aa,ab)
        prior_s = torch.linalg.svdvals(torch.einsum("nti,ntj->tij",aa,ab))
        prior_diag["singular_values_by_pass"] = prior_s.tolist()
        prior_diag["condition_number_by_pass"] = [float(v) if math.isfinite(float(v)) else None
              for v in (prior_s.max(-1).values/prior_s.min(-1).values)]
        _, ga = read_and_score_jacobian(ba,a,xact[:8])
        _, gb = read_and_score_jacobian(bb,b,xact[:8])
        rec["target_access"]["original_gradient_inputs"] = 8
        count("original_gradient_calibration",1,8)
        gau, gan = unit_jacobians(ga)
        gbu, gbn = unit_jacobians(gb)
        rec["construction"].update(activation_prior=prior_diag,A_gradient_norms=gan,B_gradient_norms=gbn)
        arrays.update(activation_A=aa,activation_B=ab,R_activation=prior,
                      activation_singular_values=prior_s,
                      activation_condition_numbers=prior_s.max(-1).values/prior_s.min(-1).values,
                      A_calibration_gradients=ga,B_calibration_gradients=gb)
        aligned, ad, singular = regularized_alignment(gau,gbu,prior)
        permuted, pd, singular_p = regularized_alignment(gau,gbu.roll(1,0),prior)
        rec["construction"].update(alignment=ad,permuted_alignment=pd)
        arrays.update(R_alignment=aligned,R_permuted=permuted,
                      alignment_singular_values=singular,permuted_singular_values=singular_p,
                      alignment_condition_numbers=singular.max(-1).values/singular.min(-1).values,
                      permuted_condition_numbers=singular_p.max(-1).values/singular_p.min(-1).values)
        pb, eb, db, sb = _rank_four(gbu)
        rec["construction"]["target_rank4"] = db
        arrays.update(target_basis=eb,target_eigenvalues=sb)
        if pb is not None:
            projects["target"] = pb
        alignment_ok = ad["admitted"] and not gan["zero_row_count"] and not gbn["zero_row_count"]
        if not alignment_ok:
            rec["global_failures"].append("alignment_singular_or_zero_gradient")
        for name in ("mixed","shared"):
            if source_p.get(name) is not None and alignment_ok:
                projects[name], bases[name] = transport(source_e[name],aligned)
        if source_p.get("shared") is not None and pd["admitted"] and not gan["zero_row_count"] and not gbn["zero_row_count"]:
            projects["permuted"], bases["permuted"] = transport(source_e["shared"],permuted)
    for name,p in projects.items():
        arrays[f"constructed_projector__{name}"] = p
    for name,basis in bases.items():
        arrays[f"constructed_basis__{name}"] = basis
    rec["admissions"] = {name:{"admitted":False,"reasons":["construction_unavailable"]} for name in ARM_NAMES}
    if "target_rank4" in rec["construction"] and not rec["construction"]["target_rank4"]["admitted"]:
        for name in ("target_E","target_D"):
            rec["admissions"][name]["reasons"] = ["target_rank_or_zero_gradient"]
    chosen = {}
    rollouts = {}
    reference_E = reference_D = None
    if "mixed" in projects:
        anchor_logits, anchor_w = projected_with_gain(bb,b,xdose,projects["mixed"],1.)
        count("anchor_dose")
        rollouts["mixed"] = (anchor_logits,anchor_w)
        reference_E = dose_metrics(dose_acts,u,projects["mixed"],1.)["retained_rms"]
        reference_D = _rms(anchor_w.double()-dose_writes.double())
        rec["reference"] = {"retained_rms":reference_E,"delivered_displacement_rms":reference_D,"gain":1.}
        anchor_ok = math.isfinite(reference_E) and reference_E>1e-12 and math.isfinite(reference_D) and reference_D>1e-12
        rec["admissions"]["mixed"] = {"admitted":anchor_ok,"reasons":[] if anchor_ok else ["invalid_anchor"]}
        chosen["mixed"] = (projects["mixed"],1.)
        if not anchor_ok:
            rec["global_failures"].append("invalid_common_anchor")
    def configure(name, p, policy):
        if reference_E is None or reference_D is None:
            return
        if policy == "E":
            result = energy_gain(reference_E,dose_acts,u,p)
        else:
            def evaluate(gain):
                _, w = projected_with_gain(bb,b,xdose,p,gain)
                count("D_search")
                return _rms(w.double()-dose_writes.double())
            result = calibrate_closed_displacement(evaluate,reference_D)
        rec["calibrations"][name] = result
        admitted = bool(result["admitted"])
        rec["admissions"][name] = {"admitted":admitted,
             "reasons":[] if admitted else [str(result["reason"])]}
        if admitted:
            chosen[name] = (p,float(result["gain"]))
    for name in ("shared","target","permuted"):
        if name in projects:
            for policy in ("E","D"):
                configure(f"{name}_{policy}",projects[name],policy)
    if "shared" in bases and reference_E is not None:
        shams, sham_diag = select_shams(dose_acts,u,bases["shared"],reference_E,root)
        rec["sham_selection"] = sham_diag
        arrays["sham_proposal_retained_rms"] = np.asarray(sham_diag["proposal_retained_rms"])
        arrays["sham_proposal_angles"] = np.asarray(sham_diag["proposal_angles"])
        arrays["sham_selected_indices"] = np.asarray(sham_diag["selected_indices"],dtype=np.int64)
        for j,p in enumerate(shams):
            name = f"sham_E_{j}"
            chosen[name] = (p,1.)
            rec["admissions"][name] = {"admitted":True,"reasons":[]}
            rec["calibrations"][name] = {"admitted":True,"gain":1.,"reason":"SELECTED_WITHIN_15_PERCENT_E"}
            configure(f"sham_D_{j}",p,"D")
    else:
        rec["sham_selection"] = {"admitted":False,"selected":0,"reason":"shared_basis_or_anchor_unavailable"}
    for name,(p,gain) in chosen.items():
        arrays[f"projector__{name}"] = p
        arrays[f"gain__{name}"] = np.asarray(gain,dtype=np.float64)
        if name not in rollouts:
            rollouts[name] = projected_with_gain(bb,b,xdose,p,gain)
            count("chosen_gain_dose_scoring")
        logit,w = rollouts[name]
        physical = dose_metrics(dose_acts,u,p,gain)
        physical["by_context"] = {k:dose_metrics(dose_acts[m],u,p,gain)
            for k,m in (("shared",xdose[:,4]>0),("specific",xdose[:,4]<=0))}
        physical["actual_closed_loop_displacement"] = _delivered_detail(w,dose_writes,xdose)
        rec["dose"][name] = physical
        arrays[f"dose_logits__{name}"] = logit
        arrays[f"dose_writes__{name}"] = w
    if "shared" in projects:
        p = projects["shared"]
        check = xdose[:64]
        native_logits,_ = projected_with_gain(bb,b,check,p,1.)
        native_map = p @ bb.wrappers["mix"].U.detach().double()
        try:
            factor_basis(bb,"rotated",root+100000)
            layer = bb.wrappers["mix"]
            rotated_p = basis_transport(p,layer.V.detach())
            rotated_map = layer.V.detach().double()@rotated_p@layer.U.detach().double()
            rotated_logits,_ = projected_with_gain(bb,b,check,rotated_p,1.)
            rotated_parity = parity(bb,b,check)
            gauge = {"rotated_fullmask":rotated_parity,
                "physical_operator_relative_error":float((native_map-rotated_map).norm(dim=(1,2)).max()/layer.W_target.double().norm().clamp_min(1e-12)),
                "closed_loop_logit_error":float((native_logits-rotated_logits).abs().max())}
            arrays.update(gauge_rotated_projector=rotated_p,gauge_V=layer.V.detach().clone(),gauge_U=layer.U.detach().clone())
        finally:
            factor_basis(bb,"native",root+100000)
        count("factor_gauge",4,64)
        gauge["admitted"] = rotated_parity["passed"] and gauge["physical_operator_relative_error"]<=1e-4 and gauge["closed_loop_logit_error"]<=5e-5
        rec["gauge"] = gauge
        if not gauge["admitted"]:
            rec["global_failures"].append("within_B_factor_gauge")
    else:
        rec["global_failures"].append("unavailable_shared_factor_gauge")
    state_before = model_state_hash(b)
    old_w, old_b = b.head.weight.detach().clone(), b.head.bias.detach().clone()
    try:
        with torch.no_grad():
            b.head.weight.zero_()
            b.head.bias.zero_()
        _, zero_g = read_and_score_jacobian(bb,b,xdose[:8])
        rec["target_access"]["modified_zero_head_gradient_inputs"] = 8
        count("modified_zero_head_gradient_diagnostic",1,8)
    finally:
        with torch.no_grad():
            b.head.weight.copy_(old_w)
            b.head.bias.copy_(old_b)
    state_after = model_state_hash(b)
    rec["zero_head"] = {"gradient_max_abs":float(zero_g.abs().max()),
        "model_state_before_sha256":state_before,"model_state_after_sha256":state_after,
        "restored_exactly":state_before==state_after}
    arrays["zero_head_gradients"] = zero_g
    if float(zero_g.abs().max())!=0. or state_before!=state_after:
        rec["global_failures"].append("zero_head_control_or_state_restore")
    search_calls = rec["target_access"]["forward_calls"].get("D_search",{}).get("calls",0)
    if search_calls>451:
        raise RuntimeError("registered D-search forward budget exceeded")
    rec["target_access"]["search_rollouts"] = search_calls
    rec["target_access"]["original_gradient_pass_rows"] = 4*rec["target_access"]["original_gradient_inputs"]
    rec["target_access"]["training_gradient_updates"] = 384
    # All choices are persisted before the first evaluation input is generated.
    rec["source_hashes_sha256"] = _source_hashes()
    rec["choice_array_hashes_sha256"] = {k:_array_hash(v) for k,v in arrays.items()}
    _write_npz(paths["choices"],arrays)
    rec["choices_frozen_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(paths["frozen"], {k:v for k,v in rec.items() if k not in ("scores", "contrasts")})
    xe = sample_x(root,3,1024)
    ya,yb = task_labels(xe,"A"),task_labels(xe,"B")
    ca,wa,_,_ = _clean_capture(ba,a,xe)
    cb,wb,_,_ = _clean_capture(bb,b,xe)
    count("clean_evaluation")
    rec["scores"]["A_clean"] = _score(ca,ca,wa,wa,xe,ya)
    rec["scores"]["B_clean"] = _score(cb,cb,wb,wb,xe,yb)
    evaluation = {"inputs":xe,"labels_A":ya,"labels_B":yb,
                  "logits__A_clean":ca,"logits__B_clean":cb,"writes__A_clean":wa,"writes__B_clean":wb}
    accs = {task:{ctx:rec["scores"][f"{task}_clean"]["by_context"][ctx]["accuracy"]
                  for ctx in ("shared","specific")} for task in ("A","B")}
    rec["competence"] = accs
    if min(v for row in accs.values() for v in row.values())<.85:
        rec["global_failures"].append("clean_context_competence")
    if min(ta["mix_density"],tb["mix_density"])<.90:
        rec["global_failures"].append("dense_mix_fraction")
    for name,(p,gain) in chosen.items():
        changed,w = projected_with_gain(bb,b,xe,p,gain)
        count("intervened_evaluation")
        rec["scores"][name] = _score(cb,changed,wb,w,xe,yb)
        evaluation[f"logits__{name}"] = changed
        evaluation[f"writes__{name}"] = w
        if not bool(torch.isfinite(changed).all() and torch.isfinite(w).all()):
            rec["admissions"][name]["reasons"].append("nonfinite_evaluation")
    for name,p in source_p.items():
        if p is not None:
            changed,w = projected_with_gain(ba,a,xe,p,1.)
            key = f"A_owned_{name}"
            rec["scores"][key] = _score(ca,changed,wa,w,xe,ya)
            evaluation[f"logits__{key}"] = changed
            evaluation[f"writes__{key}"] = w
    for name in ("A_clean","B_clean"):
        rec["admissions"][name] = {"admitted":True,"reasons":[]}
    for name,admission in rec["admissions"].items():
        admission["reasons"] = list(dict.fromkeys(admission["reasons"]+rec["global_failures"]))
        if name not in rec["scores"]:
            admission["reasons"].append("unscored")
        admission["admitted"] = not admission["reasons"]
    rec["contrasts"] = _contrast_values(rec["scores"],rec["admissions"])
    rec["descriptive"] = {}
    def n(name):
        return rec["scores"][name]["by_context"]["shared"]["final_nll_nats"]
    for label,left,right in (("mixed_deficit_vs_target_E","mixed","target_E"),
                             ("shared_deficit_vs_target_E","shared_E","target_E"),
                             ("permuted_minus_shared_E","permuted_E","shared_E"),
                             ("permuted_minus_shared_D","permuted_D","shared_D")):
        if left in rec["scores"] and right in rec["scores"]:
            rec["descriptive"][label] = n(left)-n(right)
    _write_npz(paths["eval"],evaluation)
    rec["evaluation_input_sha256"] = _array_hash(xe)
    rec["evaluation_array_hashes_sha256"] = {k:_array_hash(v) for k,v in evaluation.items()}
    rec["complete"] = True
    rec["artifacts"] = {name:{"file":paths[name].name,"sha256":_sha(paths[name])}
                        for name in ("weights","choices","frozen","eval")}
    _write_json(paths["receipt"],rec)
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=int,required=True)
    p.add_argument("--outdir",type=Path,required=True)
    args = p.parse_args()
    result = run(args.root,args.outdir)
    print(json.dumps({"root":args.root,"complete":result["complete"],
                      "global_failures":result["global_failures"],
                      "receipt":str(args.outdir/f"root_{args.root}.json")},sort_keys=True))


if __name__ == "__main__":
    main()
