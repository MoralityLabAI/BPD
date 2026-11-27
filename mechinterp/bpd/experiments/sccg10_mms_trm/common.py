"""Shared roots, seeds, source hashing and native loaders for SCCG-10 / MMS-TRM."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

from mechinterp.bpd.experiments.trm_curation.campaign import CampaignRecipe, _make_trm
from mechinterp.bpd.experiments.trm_curation.signal_bank import model_state_sha256

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CURATE = HERE.parent / 'trm_curation'
SCCG10_REF = HERE.parent / 'sccg10_native_mms'

CURATE02_PIN = 'af43e48b1ed472eef88bedd0cfa8f4f8aefb49b5'
# Historical CURATE-02 roots: replay and diagnostic development only, never promotion.
HISTORICAL_ENGINEERING_ROOTS = (28000, 28001)
HISTORICAL_DEVELOPMENT_ROOTS = tuple(range(28101, 28109))
# Fresh SCCG-10 roots. Searched every BPD remote branch on 2026-11-27: the only
# textual hits for 292xx/293xx were float substrings in unrelated receipts.
ENGINEERING_ROOTS = (29200, 29201)
DEVELOPMENT_ROOTS = tuple(range(29301, 29309))
CHECKPOINT_UPDATES = (0, 1, 2, 4, 8, 16, 32, 64)
PANEL_DELAYS = (2, 7, 31)
DOMAIN = 'BPD-SCCG10-MMS'


def sccg_seed(root: int, domain: str, index: int = 0) -> int:
    """Domain-separated nuisance seed; disjoint from CURATE-02 seeds by prefix."""
    raw = f'{DOMAIN}:{root}:{domain}:{index}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little') % (2**63 - 1)


def recipe() -> CampaignRecipe:
    return CampaignRecipe()


def make_model(state: dict[str, torch.Tensor] | None = None):
    m = _make_trm(recipe())
    if state is not None:
        m.load_state_dict(state)
    return m


def load_checkpoint(path: Path):
    return make_model(torch.load(path, map_location='cpu', weights_only=True))


def historical_warmup_path(root: int) -> Path:
    return CURATE / 'results' / 'engineering' / 'phase_a' / f'root_{root}' / 'TRM_w32_r29.pt'


def load_admission():
    """Import the unchanged SCCG-10 exact rational admission reference by path."""
    spec = importlib.util.spec_from_file_location('sccg10_admission', SCCG10_REF / 'admission.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TEXT_SUFFIXES = ('.py', '.md', '.json', '.yml', '.jsonl', '.txt', '.toml')


def file_sha256(path: Path) -> str:
    """LF-normalized SHA-256 for text files, raw bytes for binaries."""
    data = path.read_bytes()
    if path.suffix in TEXT_SUFFIXES:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def source_hashes() -> dict[str, str]:
    files = sorted(HERE.glob('*.py'))
    files += [CURATE / n for n in ('model.py', 'data.py', 'observer.py', 'cache.py', 'transaction.py',
                                   'scheduler.py', 'signal_bank.py', 'campaign.py', 'reference.py',
                                   'memory_cap.py')]
    files += [SCCG10_REF / n for n in ('admission.py', 'THEORY.md', 'PROTOCOL.md')]
    files += [REPO / 'mechinterp/bpd' / n for n in ('components.py', 'config.py')]
    return {str(p.relative_to(REPO)).replace('\\', '/'): file_sha256(p) for p in files if p.is_file()}


def accounting_dict(cap) -> dict[str, Any]:
    return asdict(cap.accounting())


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')


__all__ = ['model_state_sha256']
