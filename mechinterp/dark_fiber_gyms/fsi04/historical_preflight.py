"""FSI-04 historical CPU planning audit; old random sham was NOT chord-matched.

Read-only retrospective preflight from published FSI-03 swap receipts.
The output cannot establish the new pilot's matched-sham variation.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=ROOT/"supplements/loopscope04_arg03_receipts/loopscope04"
SOURCES={
  "roster":BASE/"rep_eval_strata.json",
  "yagizdevre":BASE/"yagizdevre_rep_swap_t4_h32.json",
  "gaoxin492":BASE/"gaoxin492_rep_swap_t4_h32.json"
}

def _load():
    src=os.environ.get("EVALMATH_PREFLIGHT_FILE")
    if not src or not Path(src).is_file():
        raise RuntimeError("EVALMATH_PREFLIGHT_FILE must point to pinned EvalMath skill preflight.py")
    spec=importlib.util.spec_from_file_location("fsi04_evalmath_power",src)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def audit(root=ROOT):
    source={k:json.loads(path.read_text()) for k,path in SOURCES.items()}
    roster=source["roster"]
    if roster["n_eval"]!=120:raise ValueError("historical source count changed")
    strata={r["case_id"]:r["stratum"] for r in roster["rows"]}
    ids=sorted(key for key,value in strata.items() if value=="S4_stalled")
    if len(ids)!=40:raise ValueError("historical S4 unit count changed")
    rows={}
    for tag in ("yagizdevre","gaoxin492"):
        record=source[tag]
        if record["mode"]!="swap" or record["settings"]["patch_step"]!=4:
            raise ValueError("wrong archived intervention")
        lookup={r["case_id"]:r for r in record["rows"]}
        if len(lookup)!=120 or set(lookup)!=set(strata):
            raise ValueError("checkpoint and roster IDs differ")
        rows[tag]=lookup
    pilot=[];summary={}
    for tag in rows:
        d=[(rows[tag][id]["arms"]["swapped_joint_zl"]["correct_cells"]-
            rows[tag][id]["arms"]["clean"]["correct_cells"])/81 for id in ids]
        s=[(rows[tag][id]["arms"]["random_joint_zl"]["correct_cells"]-
            rows[tag][id]["arms"]["clean"]["correct_cells"])/81 for id in ids]
        def mean(v):return sum(v)/len(v)
        summary[tag]={"n_root_boards":len(ids),"mean_donor_minus_clean":mean(d),
          "mean_random_minus_clean":mean(s),"mean_donor_minus_random":mean([x-y for x,y in zip(d,s)]),
          "max_abs_random_minus_clean":max(map(abs,s)),
          "donor_improves_clean":sum(x>0 for x in d),
          "donor_harms_clean":sum(x<0 for x in d)}
    for id in ids:
        y=rows["yagizdevre"][id]["arms"]
        d=y["swapped_joint_zl"]["correct_cells"]-y["clean"]["correct_cells"]
        shams=[(rows[k][id]["arms"]["random_joint_zl"]["correct_cells"]-
                rows[k][id]["arms"]["clean"]["correct_cells"]) for k in ("yagizdevre","gaoxin492")]
        pilot.append({"unit_id":id,"paired_effect":str(d)+"/81",
                      "sham_effects":[str(s)+"/81" for s in shams]})
    reg=json.loads((HERE/"REGISTRATION.json").read_text())
    an=reg["analysis"]
    spec={"schema_version":1,
       "experiment_id":"BPD-FSI04-HISTORICAL-NON-DOSE-MATCHED-S4",
       "independent_unit":"S4 root puzzle (checkpoint responses paired within root)",
       "alpha":an["alpha"],"scientific_min_effect":an["science_min_effect"],
       "target_effect":an["target_effect"],"measurement_resolution":an["measurement_resolution"],
       "sham_quantile":an["sham_quantile"],"sham_confidence":an["sham_confidence"],
       "planned_comparisons":an["planned_comparisons"],"confirm_units_planned":an["confirm_units_planned"],
       "pilot":pilot,"units_independent":True,
       "pilot_from_development_only":True,
       "heldout_sealed":False}
    pf=_load().preflight(spec)
    if pf["decision"]!="REDESIGN_OR_PILOT_ONLY":
        raise AssertionError("retrospective and uncalibrated history cannot be a green gate")
    return {
        "schema":"BPD-FSI04/archived-variation-preflight/v1",
        "scope":"40 previously published S4 roots, historical norm-matched random control; physical donor displacement differs; retrospective",
        "historical":summary,
        "pilot_preflight":pf,
        "native_gpu_pilot_executed":False,
        "minimum_independent_roots_for_95th_quantile_95pct_confidence":59,
        "fresh_s4_roots_specified":reg["sampling"]["n_stalled_pilot"],
        "effect_unit":"(correct cells at horizon32)/81",
        "source_sha256":{tag:hashlib.sha256(path.read_bytes()).hexdigest() for tag,path in SOURCES.items()},
        "registration_sha256":hashlib.sha256((HERE/"REGISTRATION.json").read_bytes()).hexdigest(),
        "heldout_seal":"ARG03 168-case test untouched",
        "decision":"DOSE_MATCHED_FRESH_PILOT_ONLY",
        "warning":"Archived sham is not angular/displacement matched. A small positive donor-minus-random contrast can result from random corruption rather than donor recovery."}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();result=audit()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("FSI04 preflight:",result["pilot_preflight"]["decision"],result["pilot_preflight"]["reasons"])
if __name__=="__main__":main()
