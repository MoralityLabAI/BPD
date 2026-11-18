"""Deterministic source-only admission receipt, no neural model benchmark."""
from __future__ import annotations
import argparse
import json
from hashlib import sha256
from .worlds import sudoku4,arc_analogue,HiddenEndingWorld,count_solutions
from .steering import DepthContract

def audit(seed:int=301)->dict:
    c=DepthContract()
    sud=sudoku4(seed,blanks=6)
    arc1=arc_analogue(seed,"train")
    arc2=arc_analogue(seed+791,"heldout")
    story=HiddenEndingWorld(seed,"train",6)
    pub=story.observe()
    public_hash=sha256(json.dumps(pub,sort_keys=True).encode()).hexdigest()
    secret_preterminal_guard=False
    try:story.ending()
    except ValueError:secret_preterminal_guard=True
    for a in ("research","research","listen","listen","share","stabilize"):
        story.step(a)
    num,_=count_solutions(sud.givens)
    return {"status":"source_only", "model_trained":False,"native_trm_run":False,
            "official_arc1_scored":False,"seed":seed,
            "fit_depths":list(c.fit),"unseen_depths":list(c.test),
            "sudoku4_unique":num==1,"sudoku4_exact_oracle":sud.grade(sud.solution),
            "arc_style_train_exact":arc1.grade(arc1.answer),
            "arc_style_heldout_exact":arc2.grade(arc2.answer),
            "arc_program_source_family_disjoint":arc1.split!=arc2.split,
            "story_preterminal_guard":secret_preterminal_guard,
            "story_public_view_hash":public_hash,
            "story_hidden_terminal_revealed":story.ending() in HiddenEndingWorld.ENDINGS}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=301)
    p.add_argument("--out");a=p.parse_args()
    result=audit(a.seed)
    print(json.dumps(result,indent=2,sort_keys=True))
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
