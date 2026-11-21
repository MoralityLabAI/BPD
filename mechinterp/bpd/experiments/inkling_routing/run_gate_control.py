"""INK-BPD-03: native gate-inversion positive control on fresh development roots.

It directly swaps the softmax selector. That treatment has no matched physical
writer dose and MUST stay separate from INK-BPD-02's matched-coordinate study.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from native_model import train_source, make_batch


@torch.no_grad()
def score(model, seed, cases, hook_pass=1):
    batch=make_batch(seed,cases)
    clean=model(batch)
    single=model(batch,hook_pass=hook_pass,gate_control="invert_at_hook")
    all_pass=model(batch,hook_pass=hook_pass,gate_control="invert_all")
    outputs={"clean":clean,"single":single,"all":all_pass}
    preds={k: x["logits"].argmax(dim=-1)==batch.labels for k,x in outputs.items()}
    if not torch.allclose(single["beta3"][0],clean["beta3"][0],atol=1e-6):
        raise RuntimeError("one-pass control changed earlier clean read")
    if not torch.allclose(single["beta3"][hook_pass],1-clean["beta3"][hook_pass],atol=1e-6):
        raise RuntimeError("one-pass gate was not inverted exactly at target pass")
    if not torch.allclose(all_pass["beta3"][0],1-clean["beta3"][0],atol=1e-6):
        raise RuntimeError("all-pass gate did not invert first native pass")
    def acc(key):
        return float(preds[key].float().mean().item())
    def disp(key):
        return float(torch.linalg.vector_norm(outputs[key]["logits"]-clean["logits"],dim=-1).mean().item())
    return {"root_id":f"dev-{seed}","assigned":cases,
            "accuracy":{k:acc(k) for k in outputs},
            "logit_l2":{"single":disp("single"),"all":disp("all")},
            "originally_correct_corrupted":{
                k:int((preds["clean"] & ~preds[k]).sum().item())
                for k in ("single","all")},
            "repaired_originally_wrong":{
                k:int((~preds["clean"] & preds[k]).sum().item())
                for k in ("single","all")}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--check",action="store_true")
    p.add_argument("--seed",type=int,default=2701)
    p.add_argument("--train-steps",type=int,default=260)
    p.add_argument("--roots",type=int,default=12)
    p.add_argument("--cases",type=int,default=32)
    args=p.parse_args()
    if not (2<=args.roots<=128 and 8<=args.cases<=4096 and 1<=args.train_steps<=3000):
        p.error("out-of-range development configuration")
    torch.use_deterministic_algorithms(True)
    model, training=train_source(seed=args.seed,steps=args.train_steps,threads=2)
    rows=[score(model,args.seed+301000+i,args.cases) for i in range(args.roots)]
    means={key:sum(r["accuracy"][key] for r in rows)/len(rows)
           for key in ("clean","single","all")}
    report={"schema":"bpd/ink-bpd03-gate-competence-dev/v1",
            "evidence_class":"post-INK-BPD-02 direct-gate developmental positive control",
            "registration":"INK_BPD03_GATE_PROTOCOL.md",
            "model":training,"assigned_cases":sum(r["assigned"] for r in rows),
            "roots":args.roots,"samples_per_root":args.cases,
            "mean_root_accuracy":means,
            "mean_single_logit_l2":sum(r["logit_l2"]["single"] for r in rows)/len(rows),
            "mean_all_logit_l2":sum(r["logit_l2"]["all"] for r in rows)/len(rows),
            "per_root":rows,
            "limits":["Gate inversion has no physically matched writer-dose sham.",
                      "All roots share one trained model and are development-only.",
                      "No pretrained Inkling checkpoint, actual Goodfire coordinator or BPD component was intervened upon."]}
    payload=json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8")!=payload:
            raise SystemExit("gate competence replay differed")
        print("INK_BPD03_REPLAY_OK")
    else:
        args.out.write_text(payload,encoding="utf-8")
        print(json.dumps({"mode":"development","assigned":report["assigned_cases"],
                          "mean_root_accuracy":means,
                          "mean_single_logit_l2":report["mean_single_logit_l2"],
                          "mean_all_logit_l2":report["mean_all_logit_l2"]}))


if __name__=="__main__":
    main()
