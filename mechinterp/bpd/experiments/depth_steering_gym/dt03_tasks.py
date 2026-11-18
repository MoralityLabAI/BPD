"""DT03 task-admission contracts, completely separate from frozen DT02 pilot.

All synthetic puzzle and story gold remains host-only. Prior experiment roots
and the sealed ARG-03-N population are never loaded.
"""
from __future__ import annotations

from itertools import permutations, product
from typing import Sequence
import torch

from .worlds import ArcCase, HiddenEndingWorld, SudokuCase, _apply_arc_rule


def sudoku4_canonical(givens: Sequence[Sequence[int]]) -> tuple[int,...]:
    """Canonicalize legal 4x4 Sudoku row/column bands, digit renaming, transpose."""
    rows = tuple(tuple(int(x) for x in row) for row in givens)
    if len(rows)!=4 or any(len(r)!=4 for r in rows):
        raise ValueError("4x4 Sudoku expected")
    if any(v not in range(5) for r in rows for v in r):
        raise ValueError("invalid Sudoku symbol")
    perms=[]
    for bands in permutations((0,1)):
        for inner0 in permutations((0,1)):
            for inner1 in permutations((0,1)):
                perms.append(tuple(b*2+j for b,inner in zip(bands,(inner0,inner1)) for j in inner))
    def canon_digits(xs):
        labels={}
        out=[]
        for v in xs:
            if v and v not in labels:
                labels[v]=len(labels)+1
            out.append(labels.get(v,0))
        return tuple(out)
    best=None
    for transpose in (False,True):
        b=rows if not transpose else tuple(tuple(rows[j][i] for j in range(4)) for i in range(4))
        for rp in perms:
            for cp in perms:
                candidate=canon_digits([b[r][c] for r in rp for c in cp])
                if best is None or candidate<best:best=candidate
    return best


def audit_puzzle_split(train:Sequence[SudokuCase], heldout:Sequence[SudokuCase])->dict:
    """Reject canonical puzzle siblings crossing independent root cohorts."""
    a={sudoku4_canonical(c.givens) for c in train}
    b=[sudoku4_canonical(c.givens) for c in heldout]
    overlaps=sum(x in a for x in b)
    return {"training_items":len(train),"heldout_items":len(heldout),
            "cross_split_isomorphic_items":overlaps,"admissible":overlaps==0}


def blank_weighted_ce(logits:torch.Tensor,gold:torch.Tensor,
                      inputs:torch.Tensor, blank_token:int=1)->torch.Tensor:
    """Supervise the unknown Sudoku cells, returning an explicit failure for none."""
    import torch.nn.functional as F
    if logits.ndim!=3 or logits.shape[:2]!=gold.shape or inputs.shape!=gold.shape:
        raise ValueError("inconsistent Sudoku input/logit/gold dimensions")
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite model logit")
    missing=(inputs==blank_token)
    if missing.sum()==0:raise ValueError("no blank cells available for reasoning supervision")
    per=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                        gold.reshape(-1),reduction="none").reshape(gold.shape)
    return (per*missing).sum()/missing.sum()


def blank_accuracy(pred:torch.Tensor,gold:torch.Tensor,
                   inputs:torch.Tensor, blank_token:int=1)->dict:
    if pred.shape!=gold.shape or inputs.shape!=gold.shape:
        raise ValueError("Sudoku label shape mismatch")
    blanks=inputs==blank_token
    if not blanks.any():raise ValueError("no blank cells")
    clues=~blanks
    return {
        "blank_cells":int(blanks.sum()),
        "blank_accuracy":float((pred[blanks]==gold[blanks]).float().mean()),
        "clue_preservation":float((pred[clues]==gold[clues]).float().mean())
            if clues.any() else None,
        "whole_board_exact":float((pred==gold).all(dim=-1).float().mean()),
    }


ARC_ROLE_IDS={"demo_in":7,"demo_out":8,"test_in":9}
ARC_SEPARATOR=10
ARC_END=11


def arc_demo_tokens(case:ArcCase, *,n_demo:int=3)->tuple[int,...]:
    """Serialized public demonstrations and query, with no host test gold."""
    if n_demo < 1 or len(case.train)<n_demo:
        raise ValueError("insufficient public ARC demonstrations")
    toks=[]
    def add(role,grid):
        if len(grid)!=4 or any(len(row)!=4 for row in grid):
            raise ValueError("DT03 bounded ARC encoder requires 4x4 grids")
        toks.append(ARC_ROLE_IDS[role])
        for row in grid:
            toks.extend(int(v)+1 for v in row)
        toks.append(ARC_SEPARATOR)
    for inp,out in case.train[:n_demo]:
        add("demo_in",inp)
        add("demo_out",out)
    add("test_in",case.test_input)
    toks.append(ARC_END)
    if any(t<1 or t>11 for t in toks):
        raise ValueError("public ARC token outside native 11 vocabulary")
    return tuple(toks)


def native_arc_query_positions(n_demo:int=3)->tuple[int,...]:
    """Input is 3*(demo_input+demo_output) plus query; each grid is 16 cells."""
    if n_demo<1:raise ValueError("one demo required")
    offset=n_demo*2*(1+16+1)
    return tuple(offset+1+i for i in range(16))


def public_reachable_policy(observation:dict)->str:
    """One host-independent public policy that achieves every hidden gate."""
    if observation.get("terminal"):raise ValueError("already terminal")
    remaining=max(0,observation.get("horizon",12)-observation["day"])
    trust_needed=max(0,2-observation["trust"])
    evidence_needed=max(0,2-observation["evidence"])
    token_needed=int(not observation["token"])
    stability_after_research=max(0, 2-(observation["stability"]-evidence_needed))
    required=trust_needed+evidence_needed+token_needed+stability_after_research
    if required>remaining:
        raise ValueError("unreachable all-ending-safe public goal under remaining horizon")
    if observation["trust"]<2:return "listen"
    if observation["evidence"]<2:return "research"
    if not observation["token"]:return "share"
    return "stabilize"


def rollout_reachable(root:int,horizon:int=12)->dict:
    world=HiddenEndingWorld(root,"dt03",horizon=horizon)
    actions=[]
    for _ in range(horizon):
        obs=world.observe()
        action=public_reachable_policy({**obs,"horizon":horizon})
        actions.append(action)
        world.step(action)
    ending=world.ending()
    return {"actions":actions,"ending":ending,
            "terminal_success":ending!="unfinished",
            "hidden_gate_accessed_by_policy":False}
