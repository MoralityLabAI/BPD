"""BPD-PD-17 prospective paired original/radial-camouflage source.

Source labels and support masks are returned for generator audit / postfit
physical evaluation only. The planned fit functions accept only dense x.
All 32 scientific root seeds were committed to CONFIG before this file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from mechinterp.bpd.theory.pd12.experiment import sample_batch, rotation_for_seed
from mechinterp.bpd.theory.pd15.experiment import CELL as PD15_CELL
from mechinterp.bpd.theory.pd15.experiment import source_for_seed as pd15_source

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
D = 16
CELL = PD15_CELL
FAMILIES = tuple(CONFIG["families"])


def _camouflage(mask: torch.Tensor, q: torch.Tensor, *,
                amp_seed: int, radius_seed: int):
    """Draw continuous signed amplitudes with cardinality-independent energy."""
    if mask.ndim != 4 or mask.shape[-3:] != (4, 1, D):
        raise ValueError("Expected original PD12 [boards,4,1,16] masks")
    if q.shape != (D,D):
        raise ValueError("Wrong Q")
    shape = mask.shape
    gam = torch.Generator().manual_seed(amp_seed)
    grad = torch.Generator().manual_seed(radius_seed)
    b = .9 + .2*torch.rand(shape, generator=gam, dtype=torch.float64)
    signs = torch.where(torch.randint(0,2,shape,generator=gam).bool(),1.,-1.)
    radius = 1. + .1*torch.rand(shape[:-1]+(1,),generator=grad,dtype=torch.float64)
    m = mask.double()
    scale=(b.square()*m).sum(dim=-1,keepdim=True).sqrt()
    if bool((scale<=0).any()):
        raise AssertionError("PD12 cardinalities must be >=1")
    latent=(radius*signs*b*m/scale).float()
    x=(latent.squeeze(2)@q.T).unsqueeze(2)
    return x, latent, radius


def training_source(seed:int, law:str, *, boards:int|None=None):
    """Paired source banks share Q and exact original train support masks."""
    if law not in CONFIG["source_laws"]:
        raise ValueError(law)
    q, original, _, mask, latent = pd15_source(seed,boards=boards)
    if law=="original":
        x, z, radius=original,latent,latent.square().sum(-1,keepdim=True).sqrt()
    else:
        streams=CONFIG["source_laws"]["radial_camouflage"]["deterministic_streams"]
        x,z,radius=_camouflage(
            mask,q,
            amp_seed=streams["train_amplitude_seed_offset"]+seed,
            radius_seed=streams["train_radius_seed_offset"]+seed
        )
    return {
        "seed":seed,"law":law,"q":q,"x":x,"mask":mask,"latent":z,
        "radius":radius,
        "q_sha256":hashlib.sha256(q.numpy().tobytes()).hexdigest(),
        "pool_sha256":hashlib.sha256(x.numpy().tobytes()).hexdigest(),
        "support_sha256":hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
    }


def heldout_source(seed:int, law:str, family:str, *,
                   boards:int|None=None):
    """Same PD12 heldout support bank for both laws; only amplitudes differ."""
    if law not in CONFIG["source_laws"] or family not in FAMILIES:
        raise ValueError((law,family))
    idx=FAMILIES.index(family)
    n=CONFIG["source_rng"]["eval_boards_per_family"] if boards is None else boards
    generator=torch.Generator().manual_seed(
        CONFIG["source_rng"]["original_heldout"]+seed+
        CONFIG["source_rng"]["heldout_family_stride"]*idx
    )
    q=rotation_for_seed(seed,D)
    x,mask,latent=sample_batch(generator,n,q,cell=CELL,family=family)
    if law=="radial_camouflage":
        streams=CONFIG["source_laws"]["radial_camouflage"]["deterministic_streams"]
        x,latent,radius=_camouflage(
            mask,q,
            amp_seed=streams["heldout_amplitude_seed_offset"]+seed+
                     CONFIG["source_rng"]["heldout_family_stride"]*idx,
            radius_seed=streams["heldout_radius_seed_offset"]+seed+
                        CONFIG["source_rng"]["heldout_family_stride"]*idx
        )
    else:
        radius=latent.square().sum(-1,keepdim=True).sqrt()
    return {
        "seed":seed,"law":law,"family":family,"q":q,
        "x":x,"latent":latent,"mask":mask,"radius":radius,
        "support_sha256":hashlib.sha256(mask.numpy().tobytes()).hexdigest()
    }
