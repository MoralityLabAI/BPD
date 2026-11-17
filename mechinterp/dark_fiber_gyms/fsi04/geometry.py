"""FSI-04: per-cell dose-matched head-fibre and z_L random sham.

A sphere-preserving donor transplant and its random sham have the same initial
Euclidean displacement from the clean recipient at every board cell. The
random head direction is tangent to the recipient's head-null sphere and
independent of the donor orientation conditional on the donor's physical dose.
"""
from __future__ import annotations

import math
import torch


def matched_random_joint(rec_h, donor_h, rec_l, donor_l, head, board_start, seed):
    """Return random (z_H,z_L) control, with donor displacement matched per cell.

    rec_h / donor_h: [1,97,D] float32, where donor_h already preserves
    recipient's full-head logits and per-cell radius. Only board cells change.
    rec_l / donor_l: [1,97,D_L] float32; z_L direction is sampled independently,
    with the exact Euclidean donor-replacement displacement as its dose.
    """
    if any(x.dtype != torch.float32 for x in (rec_h, donor_h, rec_l, donor_l)):
        raise ValueError("FSI04 requires FP32 states")
    if rec_h.shape != donor_h.shape or rec_l.shape != donor_l.shape:
        raise ValueError("recipient/donor shape mismatch")
    if rec_h.ndim != 3 or rec_h.shape[0] != 1 or rec_l.ndim != 3 or rec_l.shape[0] != 1:
        raise ValueError("pass one 3D model state at a time")
    if not 0 <= board_start < rec_h.shape[1] or board_start >= rec_l.shape[1]:
        raise ValueError("invalid board start")
    from mechinterp.dark_fiber_gyms.core import row_basis, project_visible

    basis=row_basis(head).to(device=rec_h.device)
    old=rec_h[:,board_start:,:]
    donor=donor_h[:,board_start:,:]
    vis=project_visible(old,basis)
    orth_old=old-vis
    orth_donor=donor-vis
    rad=orth_old.norm(dim=-1,keepdim=True)
    if bool((rad < 1e-6).any()):
        raise ValueError("degenerate head-null radius")
    visible_error=(project_visible(donor,basis)-vis).abs().max().item()
    radius_error=(orth_donor.norm(dim=-1,keepdim=True)-rad).abs().max().item()
    if visible_error > 1e-3 or radius_error > 1e-3:
        raise ValueError("donor must be an audited sphere-preserving head-null transplant")

    # Match the chord length exactly. Orthogonal unit vectors at equal radius
    # differ by distance r*sqrt(2*(1-cos(theta))).
    chord=(donor-old).norm(dim=-1,keepdim=True)
    cos=(1.0-(chord*chord)/(2*rad*rad)).clamp(-1.0,1.0)
    sin=(1.0-cos*cos).clamp_min(0.0).sqrt()
    gen=torch.Generator(device="cpu").manual_seed(int(seed))
    noise=torch.randn(tuple(old.shape),generator=gen,dtype=torch.float32).to(old.device)
    tangent=noise-project_visible(noise,basis)
    u=orth_old/rad
    tangent=tangent-(tangent*u).sum(dim=-1,keepdim=True)*u
    norm=tangent.norm(dim=-1,keepdim=True)
    if bool((norm < 1e-6).any()):
        raise ValueError("random tangent degeneracy")
    tangent=tangent/norm
    random_h=rec_h.clone()
    random_h[:,board_start:,:]=vis+rad*(cos*u+sin*tangent)

    old_l=rec_l[:,board_start:,:]
    donor_diff=donor_l[:,board_start:,:]-old_l
    dose_l=donor_diff.norm(dim=-1,keepdim=True)
    znoise=torch.randn(tuple(old_l.shape),generator=gen,dtype=torch.float32).to(old_l.device)
    znoise=znoise/znoise.norm(dim=-1,keepdim=True).clamp_min(1e-10)
    random_l=rec_l.clone()
    random_l[:,board_start:,:]=old_l+dose_l*znoise

    h_chord_diff=((random_h[:,board_start:,:]-old).norm(dim=-1,keepdim=True)-chord).abs().max().item()
    l_chord_diff=((random_l[:,board_start:,:]-old_l).norm(dim=-1,keepdim=True)-dose_l).abs().max().item()
    head_w=head.float().to(old.device)
    head_error=((random_h[:,board_start:,:]-old)@head_w.T).abs().max().item()
    h_radius_error=(random_h[:,board_start:,:].norm(dim=-1)-old.norm(dim=-1)).abs().max().item()
    certificate={
        "head_pre_readout_max_abs":float(head_error),
        "head_pre_radius_max_abs":float(h_radius_error),
        "zH_dose_mismatch_max_abs":float(h_chord_diff),
        "zL_dose_mismatch_max_abs":float(l_chord_diff),
        "donor_zH_dose_mean":float(chord.mean().item()),
        "donor_zL_dose_mean":float(dose_l.mean().item()),
        "random_seed":int(seed),
        "geometry":"sphere-preserving per-cell same-chord zH; independent same-L2-displacement zL",
    }
    if any(not math.isfinite(float(v)) for k,v in certificate.items() if k.endswith("_abs") or k.endswith("_mean")):
        raise ValueError("nonfinite dose certificate")
    if max(head_error,h_radius_error,h_chord_diff,l_chord_diff) > 2e-3:
        raise AssertionError(f"physical dose or full-readout parity failure: {certificate}")
    return random_h,random_l,certificate
