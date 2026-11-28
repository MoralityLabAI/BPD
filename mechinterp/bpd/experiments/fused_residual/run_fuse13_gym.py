"""Serial, resumable FUSE-13 microprobe matrix; run CUDA only under VRAMorama."""
from __future__ import annotations
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from .fuse13_precision_gym import ROOTS

MODULE = "mechinterp.bpd.experiments.fused_residual.fuse13_precision_gym"

def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    p.add_argument("--allow-cuda", action="store_true")
    p.add_argument("--tier", choices=("smoke","stress"), default="smoke")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="maximum new cells in this invocation")
    a = p.parse_args()
    if a.device == "cuda" and not a.allow_cuda:
        p.error("CUDA requires explicit --allow-cuda and VRAMorama lease")
    if a.limit is not None and a.limit < 1:
        p.error("--limit must be positive")
    a.out.mkdir(parents=True, exist_ok=True)
    modes = ("fp32","bf16") if a.device == "cpu" else ("fp32","tf32","bf16","fp16")
    if a.tier == "smoke":
        params = list(itertools.product(modes, (64,), (1,6), ROOTS[:1], (1,), ("ordinary","exact")))
    else:
        params = list(itertools.product(modes, (64,128,256), (1,6,12,24), ROOTS,
                                        (1,10,100,10000), ("ordinary","exact","near")))
    # A manifest mismatch prevents accidentally mixing different implementations.
    head = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git","status","--porcelain","--untracked-files=no"], text=True).strip()
    if dirty:
        p.error("commit tracked source changes before running the matrix")
    manifest = {"head":head,"device":a.device,"tier":a.tier,"cells":len(params)}
    mp = a.out/"manifest.json"
    if mp.exists() and json.loads(mp.read_text(encoding="utf-8")) != manifest:
        p.error("manifest mismatch: choose a new output directory")
    write(mp, manifest)
    n = 0
    for mode,w,t,root,scale,target in params:
        cid = f"{a.device}_{mode}_w{w}_t{t}_r{root}_s{scale}_{target}"
        dest = a.out/(cid+".json")
        if dest.exists():
            continue
        if a.limit is not None and n >= a.limit:
            break
        cmd = [sys.executable,"-m",MODULE,"--probe","--device",a.device,
               "--mode",mode,"--width",str(w),"--passes",str(t),"--seed",str(root),
               "--scale",str(scale),"--target",target,"--output",str(dest)]
        if a.allow_cuda:
            cmd.append("--allow-cuda")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 75:
            print("Power guard pause; cell remains pending", flush=True)
            return 75
        if proc.returncode != 0:
            write(dest, {"status":"error","cell_id":cid,"returncode":proc.returncode,
                         "stdout":proc.stdout[-16000:],"stderr":proc.stderr[-16000:]})
            print(f"{cid}: error; stopping for inspection", flush=True)
            return proc.returncode
        if not dest.exists():
            raise RuntimeError(f"Missing receipt for {cid}")
        print(f"{cid}: {json.loads(dest.read_text(encoding='utf-8'))['status']}", flush=True)
        n += 1
    print(f"New receipts: {n}; planned cells: {len(params)}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
