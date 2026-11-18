"""Exact small-task generators for BPD recurrent training/steering experiments.

All targets are independently checked. ARC-AGI-1 JSON is loaded from an
explicit user-supplied directory; synthetic ARC-style rules never receive an
official ARC-1 accuracy label. Hidden storyworld endings are revealed only
at terminal evaluation, and world/seed siblings are kept in one split.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import random
from typing import Iterable

Grid = tuple[tuple[int, ...], ...]


def grid(value: Iterable[Iterable[int]], *, max_size: int = 30) -> Grid:
    rows = tuple(tuple(int(v) for v in r) for r in value)
    if not rows or not rows[0] or len(rows) > max_size or len(rows[0]) > max_size:
        raise ValueError("grid outside 1..30 dimensions")
    if len(set(map(len, rows))) != 1 or any(v not in range(10) for row in rows for v in row):
        raise ValueError("ARC grids must be rectangular with colors 0..9")
    return rows


def sudoku_legal(board: Grid, *, require_complete: bool = False) -> bool:
    n = len(board)
    side = int(n ** .5)
    if n not in (4, 9) or side*side != n or any(len(row)!=n for row in board):
        return False
    if any(x not in range(n+1) for row in board for x in row):
        return False
    for group in (
        list(row) for row in board
    ):
        vals = [v for v in group if v]
        if len(vals) != len(set(vals)) or (require_complete and len(vals) != n):
            return False
    groups = []
    for i in range(n):
        groups.append([board[r][i] for r in range(n)])
    for br in range(0,n,side):
        for bc in range(0,n,side):
            groups.append([board[r][c] for r in range(br,br+side) for c in range(bc,bc+side)])
    for group in groups:
        vals=[v for v in group if v]
        if len(vals)!=len(set(vals)) or (require_complete and len(vals)!=n):
            return False
    return True


def count_solutions(board: Grid, limit: int = 2) -> tuple[int, Grid | None]:
    if not sudoku_legal(board) or limit < 1:
        return 0, None
    n=len(board)
    cells=[list(x) for x in board]
    found=0
    first=None
    def walk():
        nonlocal found,first
        candidates=[]
        for r,c in product(range(n), repeat=2):
            if cells[r][c]:
                continue
            used=set(cells[r]) | {cells[x][c] for x in range(n)}
            s=int(n**.5)
            used|={cells[x][y] for x in range(r//s*s,r//s*s+s)
                                  for y in range(c//s*s,c//s*s+s)}
            choices=[v for v in range(1,n+1) if v not in used]
            if not choices:
                return
            candidates.append((len(choices),r,c,choices))
        if not candidates:
            found+=1
            if first is None:
                first=tuple(tuple(row) for row in cells)
            return
        _,r,c,opts=min(candidates)
        for value in opts:
            cells[r][c]=value
            walk()
            if found >= limit:
                break
        cells[r][c]=0
    walk()
    return found,first


@dataclass(frozen=True)
class SudokuCase:
    id: str
    givens: Grid
    solution: Grid
    split: str
    def grade(self, proposal: Grid) -> bool:
        if not sudoku_legal(proposal,require_complete=True): return False
        return all(not clue or proposal[r][c]==clue
                   for r,row in enumerate(self.givens) for c,clue in enumerate(row)) and proposal==self.solution


def sudoku4(seed: int, split: str="train", blanks: int=6) -> SudokuCase:
    if not 1<=blanks<=12:
        raise ValueError("4x4 blank count must be 1..12")
    rng=random.Random(seed)
    base=[[((r*2+r//2+c)%4)+1 for c in range(4)] for r in range(4)]
    bands=[0,1];rng.shuffle(bands)
    rows=[band*2+j for band in bands for j in rng.sample([0,1],2)]
    stacks=[0,1];rng.shuffle(stacks)
    cols=[band*2+j for band in stacks for j in rng.sample([0,1],2)]
    labels=list(range(1,5));rng.shuffle(labels)
    full=tuple(tuple(labels[base[r][c]-1] for c in cols) for r in rows)
    puzzle=[list(row) for row in full]
    positions=list(product(range(4), repeat=2));rng.shuffle(positions)
    removed=0
    for r,c in positions:
        keep=puzzle[r][c];puzzle[r][c]=0
        count,_=count_solutions(tuple(tuple(row) for row in puzzle))
        if count!=1:
            puzzle[r][c]=keep
        else:
            removed+=1
        if removed==blanks:break
    if removed<blanks:
        raise ValueError("this seed does not admit requested uniquely solvable blank count")
    return SudokuCase(f"sudoku4-{split}-{seed}",tuple(tuple(r) for r in puzzle),full,split)


@dataclass(frozen=True)
class ArcCase:
    id: str
    train: tuple[tuple[Grid,Grid], ...]
    test_input: Grid
    answer: Grid | None
    split: str
    source: str
    def model_view(self) -> dict:
        return {"id":self.id,"train":[{"input":a,"output":b} for a,b in self.train],"test":[{"input":self.test_input}]}
    def grade(self, proposed: Grid) -> bool:
        if self.answer is None:
            raise ValueError("official ARC answer is withheld; supply host-owned gold")
        return proposed==self.answer


def load_arc1_task(path: str | Path, *, split: str="evaluation") -> ArcCase:
    """Standard public ARC-AGI-1 JSON format. Answer may remain withheld."""
    path=Path(path)
    data=json.loads(path.read_text(encoding="utf-8"))
    demos=tuple((grid(pair["input"]),grid(pair["output"])) for pair in data["train"])
    if len(demos)<1 or not data["test"]:
        raise ValueError("ARC task needs train examples and a test input")
    if len(data["test"])!=1:
        raise ValueError("one ARC test item per episode; split multi-test task separately")
    test=data["test"][0]
    return ArcCase(path.stem,demos,grid(test["input"]),
                   grid(test["output"]) if "output" in test else None,
                   split,"ARC-AGI-1 local JSON")


def _apply_arc_rule(name: str, item: Grid) -> Grid:
    if name=="flip_h": return tuple(tuple(reversed(row)) for row in item)
    if name=="recolor_1_2": return tuple(tuple(2 if v==1 else v for v in row) for row in item)
    if name=="rotate_180": return tuple(tuple(reversed(row)) for row in reversed(item))
    if name=="fill_zero": return tuple(tuple(3 if v==0 else v for v in row) for row in item)
    raise ValueError("unknown synthetic ARC-style program")


def arc_analogue(seed:int,split:str="train") -> ArcCase:
    rng=random.Random(seed)
    program=rng.choice(("flip_h","recolor_1_2") if split=="train" else ("rotate_180","fill_zero"))
    examples=[]
    for _ in range(4):
        g=tuple(tuple(rng.randrange(3) for _ in range(4)) for _ in range(4))
        examples.append((g,_apply_arc_rule(program,g)))
    return ArcCase(f"arc-style-{split}-{seed}",tuple(examples[:3]),examples[3][0],examples[3][1],split,
                   "synthetic ARC-style program, not official ARC-1")


@dataclass(frozen=True)
class StoryState:
    day: int=0
    trust: int=0
    evidence: int=0
    stability: int=2
    token: bool=False
    secret_cause: int=0
    closed: bool=False


class HiddenEndingWorld:
    """Exact compact storyworld with delayed ending and non-disclosed oracle.

    The host holds `secret_cause` and the terminal label. Model observations
    omit those fields. Same `world_root` must not cross train/test splits.
    """
    ACTIONS=("listen","research","stabilize","share","advance")
    ENDINGS=("quiet_alliance","restored_archive","open_council","unfinished")

    def __init__(self,root:int,split:str="train",horizon:int=6):
        if horizon<3:raise ValueError("horizon >= 3")
        self.root=root;self.split=split;self.horizon=horizon
        self._hidden_gate=random.Random(root).randrange(3)
        self.state=StoryState(secret_cause=self._hidden_gate)

    def observe(self)->dict:
        s=self.state
        return {"day":s.day,"trust":s.trust,"evidence":s.evidence,
                "stability":s.stability,"token":s.token,"terminal":s.closed,
                "actions":self.ACTIONS,"world_id":f"world-{self.root}"}

    def step(self,action:str)->dict:
        if action not in self.ACTIONS or self.state.closed:
            raise ValueError("invalid action or terminal world")
        s=self.state
        trust,evidence,stability,token=s.trust,s.evidence,s.stability,s.token
        if action=="listen":trust+=1
        if action=="research":evidence+=1;stability-=1
        if action=="stabilize":stability+=1
        if action=="share":token=evidence>=2 and trust>=1
        if action=="advance":stability-=1
        s=StoryState(s.day+1,max(-2,min(4,trust)),min(evidence,4),
                     max(0,min(4,stability)),token,s.secret_cause,s.day+1>=self.horizon)
        self.state=s
        return self.observe()

    def ending(self)->str:
        if not self.state.closed:
            raise ValueError("ending is sealed until terminal")
        s=self.state
        if s.secret_cause==0 and s.trust>=2 and s.token and s.stability>=1:
            return "quiet_alliance"
        if s.secret_cause==1 and s.evidence>=2 and s.stability>=2:
            return "restored_archive"
        if s.secret_cause==2 and s.token and s.stability>=2:
            return "open_council"
        return "unfinished"

    def commitment(self)->str:
        """Stable host-side provenance commitment, not a cryptographic hiding guarantee."""
        raw=f"world-v1|{self.split}|{self.root}|{self.horizon}|{self._hidden_gate}"
        return sha256(raw.encode()).hexdigest()
