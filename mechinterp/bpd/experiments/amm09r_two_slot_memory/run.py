"""Fresh-process launcher for AMM09R1 engineering, training, or development.

Confirmation generation is unavailable. The original AMM09 module is never
invoked by this launcher. Each output path must be new.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parent


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("engineering","training","development"),required=True)
    parser.add_argument("--memory",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--check-lock",type=Path)
    args=parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.phase=="development" and args.memory is None:
        parser.error("development requires the complete source-locked training memory")
    if "torch" in sys.modules:
        raise RuntimeError("launch this file directly in a fresh Python process")
    manifest=json.loads((ROOT/"RUNTIME.json").read_text())
    desired=manifest["environment"]
    if any(os.environ.get(key)!=value for key,value in desired.items()):
        os.environ.update(desired)
        os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=False)
    package=types.ModuleType("_amm09r1_study")
    package.__path__=[str(ROOT)]
    sys.modules[package.__name__]=package
    m=load_module(package.__name__+".models",ROOT/"models.py")
    case=load_module(package.__name__+".case",ROOT/"case.py")
    e=load_module(package.__name__+".experiment",ROOT/"experiment.py")
    try:
        m.require_runtime()
        m.require_phase_admission(args.phase)
        if args.phase=="training":
            result=e.train()
            e.validate_memory(result)
        elif args.phase=="engineering":
            rows=[]
            e.LAST_PARTIAL={"phase":"engineering","stage":"root_fitting","rows":rows}
            for index in range(m.PHASE_N["engineering"]):
                e.LAST_PARTIAL["current_root"]=m.root_key("engineering",index)
                _,factor,metrics=m.fit("engineering",index)
                rows.append({"root":m.root_key("engineering",index),"fit":metrics,
                             "gauge":m.gauge_check(factor),
                             "case":case.score("engineering",index,factor,"engineering",m.TRAIN_PAIRS)})
            result={"experiment":m.NAME,"phase":"engineering","rows":rows,
                    "source_sha256":e.source_sha256(),"runtime_signature":m.runtime_signature(),
                    "confirmation_authorized":False}
        else:
            memory=json.loads(args.memory.read_text())
            result=e.evaluate("development",memory)
        if args.check_lock:
            if args.phase=="engineering":
                raise ValueError("engineering is not a scientific lock")
            expected=json.loads(args.check_lock.read_text())
            if e.compact(result)!=expected:
                raise AssertionError("AMM09R1 strict source/runtime/outcome replay mismatch")
    except Exception as error:
        failed={"experiment":m.NAME,"phase":args.phase,"status":"failed",
                "source_sha256":e.source_sha256(),
                "error":{"type":type(error).__name__,"message":str(error)},
                "partial":e.LAST_PARTIAL,"confirmation_authorized":False}
        if "result" in locals():
            failed["completed_result"]=result
            failed["partial"]={"phase":args.phase,"stage":"completed_result_verification"}
        failure_path=args.out.with_name(args.out.name+".failure.json")
        failure_path.parent.mkdir(parents=True,exist_ok=True)
        with failure_path.open("x") as stream:
            json.dump(failed,stream,sort_keys=True,separators=(",",":"),allow_nan=False)
            stream.write("\n")
        raise
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result,stream,sort_keys=True,separators=(",",":"),allow_nan=False)
        stream.write("\n")
    print(json.dumps({"experiment":m.NAME,"phase":args.phase,"training_policy_sha256":result.get("sha256"),
                      "n_roots":len(result["rows"]),"totals":result.get("totals"),
                      "confirmation_authorized":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
