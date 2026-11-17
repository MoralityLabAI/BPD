"""FSI-04 experimental decision gate from TWO new dose-matched native pilot tapes.

Run only after locally scoring the exact, unscored 24-S4/12-propagation
development CSV with both pinned checkpoints. This refuses incomplete,
unpaired, reused or physically mismatched causal receipts. It never reads
the protected 168-case ARG-03 test.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os
from pathlib import Path

REG=Path(__file__).with_name("REGISTRATION.json")
ARMS=("clean","donor_joint","random_joint_dose_matched","self_control")


def _power():
    path=os.environ.get("EVALMATH_PREFLIGHT_FILE")
    if not path or not Path(path).is_file():
        raise RuntimeError("EVALMATH_PREFLIGHT_FILE missing")
    spec=importlib.util.spec_from_file_location("fsi04_evalmath_power_new",path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def _read(path):
    payload=json.loads(path.read_text())
    if payload.get("schema")!="BPD-FSI04/native-development-causal-v1":
        raise ValueError("wrong native development receipt schema")
    return payload

def _validate_record(r):
    if r["stratum"] not in ("S4_stalled","WB-Prop"):
        raise ValueError("unknown stratum")
    arms=r.get("arms",{})
    if set(arms)!=set(ARMS):raise ValueError("missing/extra causal arm")
    for k,x in arms.items():
        if not isinstance(x.get("correct_cells"),int) or not 0<=x["correct_cells"]<=81:
            raise ValueError("invalid correct cells")
        if x.get("net_cells_vs_clean")!=x.get("gained_cells_vs_clean",0)-x.get("lost_cells_vs_clean",0):
            raise ValueError("signed causal gain accounting failed")
    if arms["clean"]["correct_cells"]!=arms["self_control"]["correct_cells"]:
        raise ValueError("self-control changed task outcome")
    n_missing={v.get("missing_cells_total") for v in arms.values()}
    if len(n_missing)!=1 or not all(type(n) is int and 1<=n<=81 for n in n_missing):
        raise ValueError("missing-cell causal denominator mismatch")
    for x in arms.values():
        n=x["missing_cells_total"]
        c=x.get("missing_cells_correct")
        if type(c) is not int or not 0<=c<=n:
            raise ValueError("invalid missing-cell accuracy")
        if x.get("missing_net_vs_clean")!=x.get("missing_gained_vs_clean",0)-x.get("missing_lost_vs_clean",0):
            raise ValueError("missing-cell causal gain accounting failed")
    if arms["clean"]["missing_cells_correct"]!=arms["self_control"]["missing_cells_correct"]:
        raise ValueError("self-control changed missing-cell task outcome")
    cert=r.get("certificate",{})
    sham=cert.get("dose_matched_sham",{})
    for k in ("head_pre_readout_max_abs","head_pre_radius_max_abs",
              "zH_dose_mismatch_max_abs","zL_dose_mismatch_max_abs"):
        v=sham.get(k)
        if not isinstance(v,(int,float)) or not 0<=v<=2e-3:
            raise ValueError(f"unmatched sham physical dose/readout certificate: {k}")
    if cert.get("pre_patch_full_head_max_abs",1)>1e-3:
        raise ValueError("pre-intervention output leaked")

def evaluate(primary_path:Path,replica_path:Path,roster_manifest:Path):
    reg=json.loads(REG.read_text());manifest=json.loads(roster_manifest.read_text())
    if manifest.get("schema")!="BPD-FSI04/development-roster/v1":
        raise ValueError("unregistered roster")
    if manifest.get("registration_sha256")!=hashlib.sha256(REG.read_bytes()).hexdigest():
        raise ValueError("roster protocol hash mismatch")
    if manifest.get("ARG03_protected_test")!="untouched":
        raise ValueError("protected test is not sealed")
    src={"yagizdevre":_read(primary_path),"gaoxin492":_read(replica_path)}
    intended={x["tag"]:x["sha256"] for x in reg["models"]}
    tables={}
    n_expected=sum(manifest["counts"].values())
    if manifest["counts"]!={"S4_stalled":24,"WB-Prop":12}:
        raise ValueError("partial pilot: use full fresh 24+12 development roster")
    for tag,doc in src.items():
        if doc.get("checkpoint_tag")!=tag or doc.get("checkpoint_sha256")!=intended[tag]:
            raise ValueError("checkpoint identity mismatch")
        if doc.get("registration_sha256")!=hashlib.sha256(REG.read_bytes()).hexdigest():
            raise ValueError("registration not source-locked")
        if doc.get("eval_csv_sha256")!=manifest["csv_sha256"]:
            raise ValueError("checkpoint used a different board roster")
        rows=doc["rows"]
        if len(rows)!=n_expected:raise ValueError("native pilot incomplete")
        byid={x["case_id"]:x for x in rows}
        if len(byid)!=len(rows):raise ValueError("duplicate pilot source root")
        for row in rows:_validate_record(row)
        if sum(x["stratum"]=="S4_stalled" for x in rows)!=24:
            raise ValueError("missing independent stalled roots")
        tables[tag]=byid
    if set(tables["gaoxin492"])!=set(tables["yagizdevre"]):
        raise ValueError("unpaired checkpoint board IDs")
    ids=sorted(k for k,v in tables["yagizdevre"].items() if v["stratum"]=="S4_stalled")
    rows=[]
    for case_id in ids:
        p=tables["yagizdevre"][case_id];q=tables["gaoxin492"][case_id]
        if q["stratum"]!=p["stratum"]:raise ValueError("checkpoint stratum differs")
        effects={}
        for tag,r in (("yagizdevre",p),("gaoxin492",q)):
            v=r["arms"]
            effects[tag]={"donor_vs_clean":v["donor_joint"]["correct_cells"]-v["clean"]["correct_cells"],
                          "sham_vs_clean":v["random_joint_dose_matched"]["correct_cells"]-v["clean"]["correct_cells"],
                          "donor_vs_sham":v["donor_joint"]["correct_cells"]-v["random_joint_dose_matched"]["correct_cells"],
                          "corrected_cells":v["donor_joint"]["gained_cells_vs_clean"],
                          "corrupted_cells":v["donor_joint"]["lost_cells_vs_clean"]}
        rows.append({"unit_id":case_id,"stratum":"S4_stalled","checkpoint_effects":effects})
    a=reg["analysis"]
    pilot={"schema_version":1,"experiment_id":"BPD-FSI04/NEW-DOSE-MATCHED-DEV",
         "independent_unit":"unique S4 Sudoku puzzle root (both checkpoint outcomes clustered)",
         "alpha":a["alpha"],"scientific_min_effect":a["science_min_effect"],
         "target_effect":a["target_effect"],"measurement_resolution":a["measurement_resolution"],
         "sham_quantile":a["sham_quantile"],"sham_confidence":a["sham_confidence"],
         "planned_comparisons":a["planned_comparisons"],"confirm_units_planned":a["confirm_units_planned"],
         "pilot":[{"unit_id":r["unit_id"],
                   "paired_effect":str(r["checkpoint_effects"]["yagizdevre"]["donor_vs_clean"])+"/81",
                   "sham_effects":[str(r["checkpoint_effects"][tag]["sham_vs_clean"])+"/81"
                        for tag in ("yagizdevre","gaoxin492")]}
                  for r in rows],
         "pilot_from_development_only":True,"units_independent":True,
         "heldout_sealed":False}
    power=_power().preflight(pilot)
    stats={}
    for tag in ("yagizdevre","gaoxin492"):
        x=[r["checkpoint_effects"][tag] for r in rows]
        missing_vals=[
          (tables[tag][r["unit_id"]]["arms"]["donor_joint"]["missing_cells_correct"]-
           tables[tag][r["unit_id"]]["arms"]["clean"]["missing_cells_correct"]) /
          tables[tag][r["unit_id"]]["arms"]["clean"]["missing_cells_total"]
          for r in rows]
        stats[tag]={"roots":len(x),
          "mean_missing_cell_donor_vs_clean_fraction":sum(missing_vals)/len(missing_vals),
          "mean_donor_vs_clean_fraction":sum(z["donor_vs_clean"] for z in x)/(81*len(x)),
          "mean_sham_vs_clean_fraction":sum(z["sham_vs_clean"] for z in x)/(81*len(x)),
          "mean_donor_vs_sham_fraction":sum(z["donor_vs_sham"] for z in x)/(81*len(x)),
          "corrected_cells_sum":sum(z["corrected_cells"] for z in x),
          "corrupted_cells_sum":sum(z["corrupted_cells"] for z in x)}
    decision="REDESIGN_OR_MORE_INDEPENDENT_SHAM_CALIBRATION"
    if power["decision"]=="READY_FOR_CONFIRMATION":
        # A separate preregistration of *new* readout roots is required.
        decision="PILOT_LOOKS_RESOLVABLE_FREEZE_SEPARATE_CONFIRMATION_PROTOCOL"
    return {"schema":"BPD-FSI04/new-dose-matched-development-power/v1",
        "decision":decision,"root_count":len(ids),"score_unit":"correct_cells/81",
        "source_sha256":{"primary":hashlib.sha256(primary_path.read_bytes()).hexdigest(),
                         "replica":hashlib.sha256(replica_path.read_bytes()).hexdigest(),
                         "roster":hashlib.sha256(roster_manifest.read_bytes()).hexdigest()},
        "paired_s4":rows,"checkpoint_summaries":stats,"evalmath_power":power,
        "scope":"development-only; 24 S4 roots cannot calibrate 95%/95th sham quantile",
        "ARG03_168_test":"untouched"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yagiz",type=Path,required=True)
    p.add_argument("--gaoxin",type=Path,required=True)
    p.add_argument("--roster-manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();r=evaluate(a.yagiz,a.gaoxin,a.roster_manifest)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print(r["decision"],r["evalmath_power"]["reasons"])
if __name__=="__main__":
    main()
