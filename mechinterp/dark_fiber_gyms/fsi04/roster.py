"""FSI-04 public development-only Sudoku roster, before any native checkpoint.

Randomized MRV solved grids, uniqueness-certified clue deletion, propagated
difficulty strata, full automorphism screening against the available LOOPSCOPE02
puzzles, and SHA-256 exclusion against historical LOOPSCOPE04 representative
board IDs. The latter archive only stores hashes, so older full isomorphism
cannot be certified against those unexposed source puzzles.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from mechinterp.dark_fiber_gyms import sudoku_roster as SR
from mechinterp.loopscope02.scripts import sudoku_prop as SP

ROOT=Path(__file__).resolve().parents[3]
REG=Path(__file__).with_name("REGISTRATION.json")
OLD=ROOT/"supplements/loopscope04_arg03_receipts/loopscope04/rep_eval_strata.json"
LOOP02=ROOT/"mechinterp/loopscope02/results/benchmark_bundle.json"


def build(out_dir: Path, *, n_stalled: int = 24, n_propagation: int = 12,
          max_attempts: int = 8000) -> dict:
    reg=json.loads(REG.read_text())
    if reg["schema"]!="BPD-FSI04/native-causal-dose-matched-development-pilot/v1":
        raise ValueError("protocol changed")
    if n_stalled not in (1,24) or n_propagation not in (2,12):
        raise ValueError("only registered full or smoke roster sizes permitted")
    if not LOOP02.is_file() or not OLD.is_file():
        raise ValueError("old public exclusion archives absent; cannot certify roster")
    old=json.loads(OLD.read_text())
    known_ids={r["case_id"] for r in old["rows"]}
    if len(known_ids)!=120:raise ValueError("expected 120 historical hashed roots")
    banned,banned_wl,excluded=SR._build_exclusion_index([LOOP02])
    rng=random.Random(reg["sampling"]["seed"])
    buckets={"S4_stalled":[],"WB-Prop":[]}
    attempts=0
    for _ in range(max_attempts):
        if len(buckets["S4_stalled"])>=n_stalled and len(buckets["WB-Prop"])>=n_propagation:
            break
        attempts+=1
        target_group="S4_stalled" if len(buckets["S4_stalled"])<n_stalled else "WB-Prop"
        solution=SR.generate_diverse_solution(rng)
        low,high=(26,33) if target_group=="S4_stalled" else (34,40)
        clues=rng.randint(low,high)
        puzzle=SR.delete_clues(solution,clues,rng)
        if puzzle is None:continue
        fingerprint=hashlib.sha256(puzzle.encode()).hexdigest()
        if fingerprint in known_ids:continue
        if SR.count_solutions(puzzle)!=1:raise AssertionError("uniqueness failed")
        prop=SP.trace_propagation(puzzle)
        label="WB-Prop" if prop.solved else ("S4_stalled" if prop.num_rounds==0 else "WB-Stall_partial")
        if label!=target_group:continue
        if not SR._is_novel_and_register(puzzle,banned,banned_wl):continue
        buckets[label].append({"question":puzzle,"answer":solution,
                               "stratum":label,"origin":f"FSI04_MRV_{fingerprint[:16]}"})
    if len(buckets["S4_stalled"])!=n_stalled or len(buckets["WB-Prop"])!=n_propagation:
        raise RuntimeError(f"roster incomplete after {attempts} attempts: "+
          str({k:len(v) for k,v in buckets.items()}))
    rows=buckets["S4_stalled"]+buckets["WB-Prop"]
    if len({hashlib.sha256(x["question"].encode()).hexdigest() for x in rows})!=len(rows):
        raise AssertionError("duplicate root")
    out_dir.mkdir(parents=True,exist_ok=True)
    path=out_dir/"development.csv"
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=("question","answer","stratum","origin"))
        writer.writeheader();writer.writerows(rows)
    # Audit with the same native scorer's full isomorphism routine.
    from mechinterp.dark_fiber_gyms.loopscope04_trm import audit_boards,read_boards
    recs=read_boards(path)
    collision=audit_boards([],recs,[LOOP02],check_full_automorphisms=True)
    ids=[x["id"] for x in recs]
    if set(ids)&known_ids:raise AssertionError("source board ID reused")
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={"schema":"BPD-FSI04/development-roster/v1",
       "status":"fresh unscored development roots",
       "registration_sha256":hashlib.sha256(REG.read_bytes()).hexdigest(),
       "seed":reg["sampling"]["seed"],
       "counts":{k:len(v) for k,v in buckets.items()},
       "source":"randomized MRV solved grids + solver-checked deletion + propagation strata",
       "source_exclusion_sha256":excluded,
       "historical_rep_archive_sha256":hashlib.sha256(OLD.read_bytes()).hexdigest(),
       "historical_rep_case_ids_checked":len(known_ids),
       "historical_rep_full_automorphism_limit":"raw historical questions unavailable; excluded by exact SHA-256 root ID only",
       "full_automorphism_against_loopscope02":True,
       "collision_audit":collision,
       "csv_sha256":digest,
       "generator_attempts":attempts,
       "ARG03_protected_test":"untouched"}
    (out_dir/"development_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--max-attempts",type=int,default=100000)
    a=p.parse_args()
    manifest=build(a.out_dir,n_stalled=1 if a.smoke else 24,
                   n_propagation=2 if a.smoke else 12,max_attempts=a.max_attempts)
    print(json.dumps(manifest,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
