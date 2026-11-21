"""INK-BPD-02: native PyTorch routing-control pilot, development only.

A single actual learned recurrent model is trained on source seeds, then
candidate/sham coordinate pairs are chosen from disjoint source probes using
routing-gate movement only. Fresh task roots receive native hook interventions.
There are no pretrained Inkling model weights or confirmation tests here.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch
from native_model import make_batch, train_source


def select_pairs(model, *, source_seed, dose, hook_pass=1):
    source = make_batch(source_seed, 96)
    with torch.no_grad():
        clean = model(source)
        eligible = []
        for pair in itertools.combinations(range(model.width), 2):
            changed = model(source, pair=pair, dose=dose,
                            hook_pass=hook_pass, mode="selector_only")
            if changed["admitted"].float().mean().item() < .98:
                continue
            move = (changed["beta3"][hook_pass] - clean["beta3"][hook_pass]).abs()
            eligible.append((float(move[changed["admitted"]].mean().item()), pair))
    if len(eligible) < 4:
        raise ValueError("insufficient source-only matched-dose coordinate pairs")
    target = max(eligible, key=lambda x: (x[0], tuple(-v for v in x[1])))
    disjoint = [x for x in eligible if not (set(x[1]) & set(target[1]))]
    if not disjoint:
        raise ValueError("no disjoint control admitted")
    sham = min(disjoint, key=lambda x: (x[0], x[1]))
    return {"target": list(target[1]), "sham": list(sham[1]),
            "source_gate_delta_target": target[0],
            "source_gate_delta_sham": sham[0],
            "source_pairs_admitted": len(eligible),
            "source_seed": source_seed}


def root_report(model, *, seed, n, target, sham, dose, hook_pass):
    b = make_batch(seed, n)
    with torch.no_grad():
        clean = model(b)
        ct = model(b, pair=target, dose=dose, hook_pass=hook_pass, mode="full")
        cs = model(b, pair=sham, dose=dose, hook_pass=hook_pass, mode="full")
        selector = model(b, pair=target, dose=dose, hook_pass=hook_pass, mode="selector_only")
        content = model(b, pair=target, dose=dose, hook_pass=hook_pass, mode="content_only")
    eligible = ct["admitted"] & cs["admitted"] & selector["admitted"] & content["admitted"]
    for arm in (ct, cs):
        if bool(eligible.any()):
            delivered = torch.linalg.vector_norm(arm["patched3"] - arm["clean3"], dim=-1)
            if not torch.allclose(delivered[eligible], torch.full_like(delivered[eligible], dose),
                                  atol=2e-5, rtol=1e-4):
                raise RuntimeError("unequal delivered physical patch dose")
            a = torch.linalg.vector_norm(arm["clean3"], dim=-1)
            c = torch.linalg.vector_norm(arm["patched3"], dim=-1)
            if not torch.allclose(a[eligible], c[eligible], atol=2e-5, rtol=1e-5):
                raise RuntimeError("RMS did not remain constant")
    masks = {}
    for k, v in (("clean", clean), ("target", ct), ("sham", cs),
                 ("selector", selector), ("content", content)):
        masks[k] = v["logits"].argmax(-1) == b.labels
    denom = int(eligible.sum().item())
    def avg(value):
        return float(value[eligible].float().mean().item()) if denom else None
    p = {k: avg(v) for k, v in masks.items()}
    return {"root_id": "dev-%d" % seed, "assigned": n, "joint_admitted": denom,
            "unadmitted": n-denom, "accuracy": p,
            "target_minus_sham": p["target"] - p["sham"] if denom else None,
            "target_gate_abs_change": avg((ct["beta3"][hook_pass] - clean["beta3"][hook_pass]).abs()),
            "sham_gate_abs_change": avg((cs["beta3"][hook_pass] - clean["beta3"][hook_pass]).abs()),
            "target_logit_change_l2": avg(torch.linalg.vector_norm(ct["logits"] - clean["logits"], dim=-1)),
            "sham_logit_change_l2": avg(torch.linalg.vector_norm(cs["logits"] - clean["logits"], dim=-1)),
            "target_repairs": int((masks["target"] & ~masks["clean"] & eligible).sum().item()),
            "target_corruptions": int((~masks["target"] & masks["clean"] & eligible).sum().item()),
            "target_only_correct": int((masks["target"] & ~masks["sham"] & eligible).sum().item()),
            "sham_only_correct": int((~masks["target"] & masks["sham"] & eligible).sum().item())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--check", action="store_true")
    p.add_argument("--seed", type=int, default=2701)
    p.add_argument("--train-steps", type=int, default=260)
    p.add_argument("--roots", type=int, default=12)
    p.add_argument("--cases", type=int, default=32)
    p.add_argument("--dose", type=float, default=.16)
    p.add_argument("--hook-pass", type=int, default=1)
    args = p.parse_args()
    if not (2 <= args.roots <= 128 and 8 <= args.cases <= 4096 and
            1 <= args.train_steps <= 3000 and .01 <= args.dose <= .5 and
            args.hook_pass in (0, 1, 2)):
        p.error("outside bounded synthetic pilot")
    torch.use_deterministic_algorithms(True)
    model, train = train_source(seed=args.seed, steps=args.train_steps, threads=2)
    sel = select_pairs(model, source_seed=args.seed+100003,
                       dose=args.dose, hook_pass=args.hook_pass)
    roots = [root_report(model, seed=args.seed + 200000 + i, n=args.cases,
                         target=tuple(sel["target"]), sham=tuple(sel["sham"]),
                         dose=args.dose, hook_pass=args.hook_pass)
             for i in range(args.roots)]
    params = model.state_dict()
    sha = hashlib.sha256()
    for name, value in sorted(params.items()):
        sha.update(name.encode())
        sha.update(value.detach().cpu().numpy().tobytes())
    admitted = [r for r in roots if r["joint_admitted"] > 0]
    report = {"schema": "bpd/ink-bpd02-native-dev/v1",
              "evidence_class": "source-selected, developmental native synthetic PyTorch model",
              "model": {**train, "sha256": sha.hexdigest(), "torch_version": torch.__version__},
              "selection": sel,
              "protocol": {"seed": args.seed, "train_steps": args.train_steps,
                           "roots": args.roots, "cases": args.cases,
                           "dose": args.dose, "hook_pass": args.hook_pass},
              "assigned_cases": sum(r["assigned"] for r in roots),
              "joint_admitted_cases": sum(r["joint_admitted"] for r in roots),
              "missing_controls": sum(r["unadmitted"] for r in roots),
              "mean_root_target_minus_sham": (sum(r["target_minus_sham"] for r in admitted)
                                             / len(admitted) if admitted else None),
              "per_root": roots,
              "claim_limits": ["Fixed synthetic trained checkpoint; task roots share its weights.",
                               "Matched source selection does not identify pretrained Inkling mechanisms.",
                               "Fewer than 59 independently measured causal sham roots: no confirmation."]}
    data = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.check:
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != data:
            raise SystemExit("native execution replay mismatch")
        print("NATIVE_DEVELOPMENT_REPLAY_OK")
    else:
        args.out.write_text(data, encoding="utf-8")
        print(json.dumps({"class":"development-only", "roots":len(roots),
                          "assigned":report["assigned_cases"],
                          "admitted":report["joint_admitted_cases"],
                          "mean_target_minus_sham":report["mean_root_target_minus_sham"],
                          "source_accuracy":train["source_validation_accuracy"]}))


if __name__ == "__main__":
    main()
