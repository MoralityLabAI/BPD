"""Shared recipe, seed domains, roots and native model access for SCCG-10 MMS-TRM.

All new draws use the ``SCCG10-MMS-TRM`` seed domain, so no nuisance stream
coincides with CURATE-02 (``BPD-TRM-CURATE-02``) even on a reused root.
The TRAIN edge partition (fit/feedback/tune) is CURATE-02's ``edge_views``
with split seed 20261127; this keeps edge-level custody identical while
every example draw inside a view is fresh.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from ..trm_curation.data import edge_batch
from ..trm_curation.model import ComponentTRM
from ..trm_curation.scheduler import edge_views
from ..training_steering.data import make_pairs
from ..training_steering_stability.data import binding_batch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

# Historical roots: replay and descriptive drift only, never promotion.
HISTORICAL_ENGINEERING_ROOTS = (28000, 28001)
HISTORICAL_DEVELOPMENT_ROOTS = tuple(range(28101, 28109))
# Fresh roots. Registry check (git grep over *.py/*.json/*.md, 2026-11-27):
# used ranges 26101-26108, 27101-27108, 28000-28001, 28101-28108, 29000-29001
# (eval_suite_36). 29200-29201 and 29301-29308 do not occur as roots.
ENGINEERING_ROOTS = (29200, 29201)
DEVELOPMENT_ROOTS = tuple(range(29301, 29309))
RESERVED_FORBIDDEN = ('ARG-03-N',)

SEED_DOMAIN = 'SCCG10-MMS-TRM'


def seed10(root: int, domain: str, index: int = 0) -> int:
    raw = f'{SEED_DOMAIN}:{root}:{domain}:{index}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little') % (2**63 - 1)


@dataclass(frozen=True)
class Recipe:
    # Architecture (identical to CURATE-02 TRM_w32_r29).
    width: int = 32
    components: int = 29
    low_cycles: int = 1
    slots: int = 4
    values: int = 4
    noise: float = 0.03
    threads: int = 2
    split_seed: int = 20261127
    # Warmup (identical to CURATE-02 Phase A).
    warmup_updates: int = 6000
    warmup_batch_edges: int = 12
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    train_delays: tuple[int, ...] = (1, 2, 3)
    # Curation phase.
    curation_learning_rate: float = 0.001
    proposals_per_window: int = 16
    selected_per_window: int = 4
    exploration: float = 0.20
    pair_exposure_cap: int = 4
    feedback_batch_edges: int = 12
    feedback_delay: int = 15
    # Mechanism.
    panel_edges: int = 24
    panel_delays: tuple[int, ...] = (2, 7, 31)
    dose: float = 0.75
    collateral_budget: float = 0.25
    support_k: int = 8
    support_beam: int = 4
    w_ordinary: float = 1.0
    w_target: float = 0.5
    w_preserve: float = 0.5
    # Evaluation.
    eval_delays: tuple[int, ...] = (2, 7, 31, 63, 127)
    primary_delay: int = 31
    eval_n: int = 512
    binding_n: int = 256
    tune_eval_edges: int = 144

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def recipe_from(d: dict[str, Any]) -> Recipe:
    fields = Recipe.__dataclass_fields__
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in fields}
    return replace(Recipe(), **kw)


def make_trm(recipe: Recipe) -> ComponentTRM:
    return ComponentTRM(width=recipe.width, components=recipe.components,
                        low_cycles=recipe.low_cycles, input_dim=3 + recipe.slots + recipe.values,
                        values=recipe.values)


def views(recipe: Recipe) -> dict[str, tuple]:
    return edge_views(recipe.split_seed)


def state_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode('ascii'))
        h.update(str(tuple(value.shape)).encode('ascii'))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def warmup_batch(fit_edges: tuple, root: int, step: int, recipe: Recipe):
    delay = recipe.train_delays[step % len(recipe.train_delays)]
    g = torch.Generator(device='cpu').manual_seed(seed10(root, 'warmup_edges', step))
    idx = torch.randint(0, len(fit_edges), (recipe.warmup_batch_edges,), generator=g).tolist()
    batch = edge_batch(tuple(fit_edges[i] for i in idx), root=root,
                       seed=seed10(root, 'warmup_nuisance', step), delay=delay,
                       split='train', noise=recipe.noise)
    return (torch.cat((batch.receiver, batch.hybrid), 0),
            torch.cat((batch.labels_r, batch.labels_h), 0), delay)


@torch.no_grad()
def eval_tune(model: ComponentTRM, root: int, recipe: Recipe, *, delays=(2, 7, 31),
              tag: str = 'tune') -> dict[str, Any]:
    """Ordinary steering-off accuracy on the TRAIN tune edge view (engineering metric)."""
    model.eval()
    tune = views(recipe)['tune']
    out = {}
    for d in delays:
        b = edge_batch(tune, root=root, seed=seed10(root, f'{tag}_d{d}', 0), delay=d,
                       split='train', noise=recipe.noise)
        x = torch.cat((b.receiver, b.hybrid), 0)
        y = torch.cat((b.labels_r, b.labels_h), 0)
        logits = model(x)
        out[str(d)] = {'n': int(len(y)), 'correct': int((logits.argmax(-1) == y).sum()),
                       'accuracy': float((logits.argmax(-1) == y).float().mean()),
                       'cross_entropy': float(F.cross_entropy(logits, y))}
    return out


@torch.no_grad()
def eval_final_development(model: ComponentTRM, root: int, recipe: Recipe) -> dict[str, Any]:
    """Final development draw: historical EVAL tuple partition, fresh SCCG10 seeds.

    Only called by the frozen development runner. Never used for engineering
    selection of caps, thresholds, sites, doses or policy.
    """
    model.eval()
    out: dict[str, Any] = {'split': 'eval', 'steering_enabled': False, 'delays': {}, 'binding': {}}
    for d in recipe.eval_delays:
        b = make_pairs(seed10(root, f'final_dev_d{d}', 0), recipe.eval_n, root_seed=root, delay=d,
                       split='eval', slots=recipe.slots, values=recipe.values, noise=recipe.noise)
        logits = model(b.receiver)
        c = (logits.argmax(-1) == b.labels_r).int()
        out['delays'][str(d)] = {'n': recipe.eval_n, 'correct_count': int(c.sum()),
                                 'accuracy': float(c.float().mean()),
                                 'cross_entropy': float(F.cross_entropy(logits, b.labels_r)),
                                 'batch_sha256': b.fingerprint(), 'correct': c.tolist()}

    class _R:  # binding_batch reads slots/values/noise only
        slots, values, noise = recipe.slots, recipe.values, recipe.noise
    for d in (2, 7, recipe.primary_delay):
        x, y, meta = binding_batch(root, f'sccg10_final_bind_d{d}', recipe.binding_n, d, 'eval', _R)
        logits = model(x)
        c = (logits.argmax(-1) == y).int()
        out['binding'][str(d)] = {'n': recipe.binding_n, 'correct_count': int(c.sum()),
                                  'accuracy': float(c.float().mean()),
                                  'cross_entropy': float(F.cross_entropy(logits, y)),
                                  'batch_sha256': meta['batch_sha256']}
    return out


def train_warmup(root: int, recipe: Recipe, *, log_every: int = 500) -> tuple[ComponentTRM, dict]:
    torch.manual_seed(seed10(root, 'init', 0))
    model = make_trm(recipe)
    init_sha = state_sha256(model)
    fit = views(recipe)['fit']
    opt = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                            weight_decay=recipe.weight_decay)
    t_cpu0, t_wall0 = time.process_time(), time.perf_counter()
    curve = []
    for step in range(recipe.warmup_updates):
        model.train()
        x, y, _ = warmup_batch(fit, root, step, recipe)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
        opt.step()
        if step == 0 or (step + 1) % log_every == 0:
            cpu = time.process_time() - t_cpu0
            ev = eval_tune(model, root, recipe, tag='warmup_curve')
            curve.append({'step': step + 1, 'train_ce': float(loss.detach()), 'cpu_seconds': cpu,
                          'wall_seconds': time.perf_counter() - t_wall0,
                          **{f'tune_d{d}': ev[d]['accuracy'] for d in ev}})
            print(json.dumps({'root': root, **curve[-1]}), flush=True)
    receipt = {'root': root, 'init_state_sha256': init_sha, 'state_sha256': state_sha256(model),
               'optimizer': 'AdamW', 'updates': recipe.warmup_updates,
               'cpu_seconds': time.process_time() - t_cpu0,
               'wall_seconds': time.perf_counter() - t_wall0, 'curve': curve,
               'parameter_count': model.parameter_count()}
    return model, receipt
