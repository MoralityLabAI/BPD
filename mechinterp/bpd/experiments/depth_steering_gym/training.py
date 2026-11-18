"""Depth-frozen supervised TRM training contract and optional 7M Sudoku bridge.

A trained checkpoint is supplied by the caller. Training/evaluation depth sets
are disjoint. Loss supervision never reads hidden story-ending or official ARC
answers from an evaluator-side source file. This is a harness, not an assertion
that any pretrained TRM solves the tasks.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol
import torch
from torch.nn import functional as F
from .steering import DepthContract

@dataclass(frozen=True)
class TrainingBatch:
    inputs: torch.Tensor
    gold: torch.Tensor
    mask: torch.Tensor
    family: str
    group_root: str

class RecurrentTaskAdapter(Protocol):
    def output_at_depth(self, inputs:torch.Tensor, depth:int)->torch.Tensor: ...
    def protected_readout_at_depth(self, inputs:torch.Tensor,depth:int)->torch.Tensor: ...


def supervised_depth_loss(adapter:RecurrentTaskAdapter, batch:TrainingBatch,
                          contract:DepthContract,*,weights:tuple[float,...]|None=None):
    if batch.family not in ("sudoku4","sudoku9","arc_style","arc1_user_supplied","storyworld"):
        raise ValueError("unknown task family")
    if weights is None:weights=tuple(1. for _ in contract.fit)
    if len(weights)!=len(contract.fit) or sum(weights)<=0 or any(w<0 for w in weights):
        raise ValueError("invalid depth weights")
    losses=[]
    for depth,w in zip(contract.fit,weights):
        # A checker wraps output_at_depth to trap queries to heldout depths.
        logits=adapter.output_at_depth(batch.inputs,depth)
        if logits.shape[:-1]!=batch.gold.shape or logits.shape[:-1]!=batch.mask.shape:
            raise ValueError("logits, label and mask dimensions disagree")
        if logits.shape[-1]<2:raise ValueError("categorical logits required")
        per=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.gold.reshape(-1),reduction="none")
        denom=batch.mask.float().sum().clamp_min(1)
        losses.append(float(w)*(per*batch.mask.reshape(-1)).sum()/denom)
    return sum(losses)/sum(weights)


def audit_depth_test(adapter:RecurrentTaskAdapter, batch:TrainingBatch,contract:DepthContract):
    with torch.no_grad():
        out={}
        for depth in contract.test:
            logits=adapter.output_at_depth(batch.inputs,depth)
            predicted=logits.argmax(-1)
            if predicted.shape!=batch.gold.shape:raise ValueError("prediction shape mismatch")
            correct=((predicted==batch.gold)*batch.mask).sum()
            out[str(depth)]={"valid_tokens":int(batch.mask.sum()),
                             "token_accuracy":float(correct/batch.mask.sum().clamp_min(1)),
                             "task_root":batch.group_root,"family":batch.family}
        return out


class NativeSudokuTRMBridge:
    """Explicit integration with existing BPD PureSudokuTRM 7M adapter.

    The real checkpoint must be supplied by caller; this bridge does not
    download weights or claim official solve rates. TRM uses nested H/L loops
    and records a boundary readout at H-cycle commits, not every L pass.
    """
    def __init__(self,model):
        from mechinterp.bpd.adapters.trm import TRMBPDAdapter
        self.model=model
        self.adapter=TRMBPDAdapter(model)
    def audit(self, inputs, gold, *, steps=(4,8,16,32,64)):
        from mechinterp.bpd.adapters.trm import evaluate_sudoku_horizon
        result=evaluate_sudoku_horizon(self.model,inputs,gold,eval_outer_steps=steps)
        return {"type":"native_7m_sudoku_trm", "exact_by_step":result.board_exact_by_step,
                "cell_acc_by_step":result.cell_acc_by_step,
                "period2_oscillation":result.period2_oscillation_rate,
                "step_flip_tail":result.step_flip_rate_tail,
                "provenance":"caller supplied checkpoint, no download"}
