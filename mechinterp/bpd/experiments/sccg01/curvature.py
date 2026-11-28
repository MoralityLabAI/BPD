"""SCCG-02: analytic-bound reference, floating evaluation only."""
from __future__ import annotations
import math
import numpy as np
from .frontier import array, scalar, rank_svd


def rms_bounds(center, radius, eps, gamma=None):
    x = array(center, 1, "center")
    radius, eps = scalar(radius, "radius"), scalar(eps, "eps")
    d = len(x)
    if d == 0:
        raise ValueError("empty state")
    g = np.ones(d) if gamma is None else array(gamma, 1, "gamma")
    if g.shape != x.shape:
        raise ValueError("scale width mismatch")
    gmax = float(np.max(np.abs(g)))
    xmin = max(0., float(np.linalg.norm(x))-radius)
    rho = math.hypot(xmin, math.sqrt(d*eps))
    if not math.isfinite(rho) or rho == 0:
        raise ValueError("normalization tube is singular or nonfinite")
    gain = math.sqrt(d)*gmax/rho
    curvature = (2/math.sqrt(3))*gain/rho
    if eps > 0:
        curvature = min(curvature, 48*gmax/(25*math.sqrt(5*d)*eps))
    if not np.isfinite([gain, curvature]).all():
        raise ValueError("nonfinite bound")
    return dict(gain=gain, curvature=curvature, rho_min=rho,
                radius=radius, scale_max=gmax)


def normalized(x, eps, gamma):
    d = len(x)
    rho = math.hypot(float(np.linalg.norm(x)), math.sqrt(d*eps))
    if rho == 0 or not math.isfinite(rho):
        raise ValueError("singular normalization")
    return array(math.sqrt(d)*gamma*(x/rho), 1, "normalized")


def reader_bounds(block2, block3, query, radius, eps, gamma=None):
    a,b,q = (array(v,1,name) for v,name in zip(
        (block2,block3,query),("block2","block3","query")))
    eps = scalar(eps,"eps")
    d = len(b)
    g = np.ones(d) if gamma is None else array(gamma,1,"gamma")
    if a.shape != b.shape or q.shape != b.shape or g.shape != b.shape:
        raise ValueError("reader widths")
    rb = rms_bounds(b,radius,eps,g)
    ln,kn = rb["gain"],rb["curvature"]
    factor=float(np.linalg.norm(q))/math.sqrt(d)
    ls,ks=factor*ln,factor*kn
    lb,kb=ls/4,ks/4+ls*ls/(6*math.sqrt(3))
    D=min(2*math.sqrt(d)*rb["scale_max"],
          float(np.linalg.norm(normalized(b,eps,g)-normalized(a,eps,g)))
          +ln*rb["radius"])
    result=dict(rms=rb, gap_gain=ls,gap_curvature=ks,
                beta_gain=lb,beta_curvature=kb,content_difference_bound=D,
                message_gain=ln+lb*D,
                message_curvature=kn+2*lb*ln+kb*D,
                scope="FIXED_QUERY_AND_SECOND_BLOCK")
    if not np.isfinite([v for v in result.values() if isinstance(v,float)]).all():
        raise ValueError("nonfinite reader bound")
    return result


def recurrence_tube(clean_jacobian_bounds, gain_bounds, curvature_bounds,
                    injection_bounds, dose_bounds, tube_radii):
    names=("j","G","K","B","dose","radius")
    xs=[array(x,1,name) for x,name in zip(
        (clean_jacobian_bounds,gain_bounds,curvature_bounds,
         injection_bounds,dose_bounds,tube_radii),names)]
    n=len(xs[0])
    if not n or any(len(x)!=n or np.any(x<0) for x in xs):
        raise ValueError("nonempty aligned nonnegative bounds required")
    r=E=0.
    rows=[dict(step=0,radius_bound=r,remainder_bound=E)]
    for t,(j,G,K,B,e,R) in enumerate(zip(*xs)):
        if j>G:
            raise ValueError("clean Jacobian bound exceeds uniform gain bound")
        if r>R:
            raise ValueError(f"tube radius not admitted at step {t}")
        next_r=min(G*r,j*r+.5*K*r*r)+B*e
        next_E=j*E+.5*K*r*r
        r,E=float(next_r),float(next_E)
        if not np.isfinite([r,E]).all():
            raise ValueError("nonfinite recurrence envelope")
        rows.append(dict(step=t+1,radius_bound=r,remainder_bound=E))
    return dict(rows=rows,scope="CONDITIONAL_ADDITIVE_INJECTION_REAL_BOUNDS")


def fixed_radius_control(x, W, C, a, dose, eps, *, rcond=1e-12):
    """Solve local fixed-radius scalar RMS gap problem on span(W)∩ker C.
    W: d-by-k actual physical writes; C and a: effective raw query rows.
    Inverse normalization and score use rho = sqrt(||x||²+d eps).
    Float rank threshold and all delivered residuals are exposed.
    """
    x,W,C,a=(array(v,n,name) for v,n,name in
             [(x,1,"x"),(W,2,"W"),(C,2,"C"),(a,1,"a")])
    dose,eps,rcond=scalar(dose,"dose"),scalar(eps,"eps"),scalar(rcond,"rcond")
    d=len(x)
    if not d or W.shape[0]!=d or C.shape[1]!=d or a.shape!=x.shape or rcond>=1:
        raise ValueError("shape or rank threshold")
    rho=math.hypot(float(np.linalg.norm(x)),math.sqrt(d*eps))
    if rho==0 or not math.isfinite(rho):
        raise ValueError("singular normalization")
    U,s,_=np.linalg.svd(W,full_matrices=False)
    r=int(np.sum(s>rcond*s[0])) if s.size else 0
    Q=U[:,:r]
    if r:
        _,vh,cr=rank_svd(C@Q,rcond)
        T=Q@vh[cr:].T
    else:
        cr=0
        T=np.empty((d,0))
    p,b=T.T@x,T.T@a
    R,B=float(np.linalg.norm(p)),float(np.linalg.norm(b))
    delta=np.zeros(d)
    case="ZERO"
    if dose>0 and R>0 and B>0 and len(p):
        if len(p)==1:
            if 2*R<=dose and float(b@p)<0:
                delta=T@(-2*p)
                case="LINE_SIGN_FLIP"
        else:
            c=p/R
            best=b/B
            if float(np.linalg.norm(R*best-p))<=dose:
                delta=T@(R*best-p)
                case="UNCONSTRAINED_CAP"
            else:
                # Avoid squaring an arbitrarily large dose/R.
                ratio=min(2.,dose/R)
                alpha=1-.5*ratio*ratio
                bp=float(b@c)
                perpendicular=b-bp*c
                pn=float(np.linalg.norm(perpendicular))
                if pn<=np.finfo(float).eps*B*8:
                    e=np.eye(len(c))[int(np.argmin(np.abs(c)))]
                    perpendicular=e-c*float(c@e)
                    pn=float(np.linalg.norm(perpendicular))
                w=alpha*c+math.sqrt(max(0.,(1-alpha)*(1+alpha)))*perpendicular/pn
                delta=T@(R*w-p)
                case="ACTIVE_CAP"
    y=x+delta
    rho2=math.hypot(float(np.linalg.norm(y)),math.sqrt(d*eps))
    if not rho2:
        raise ValueError("singular edited state")
    actual=a@y/rho2-a@x/rho
    result=dict(delta=delta, target_gain=float(a@delta/rho),
                actual_gap_change=float(actual),dose=float(np.linalg.norm(delta)),
                radius_residual=abs(float(np.linalg.norm(y)-np.linalg.norm(x))),
                protected_gap_residual=float(np.linalg.norm(C@y/rho2-C@x/rho)),
                span_residual=float(np.linalg.norm(delta-Q@(Q.T@delta))),
                rank=r,protection_rank=cr,tangent_dimension=T.shape[1],
                projected_radius=R,rcond=rcond,case=case,
                scope="NUMERICAL_FIXED_RADIUS_BOUNDARY_GAP")
    if not np.isfinite(delta).all() or not np.isfinite(
            [v for v in result.values() if isinstance(v,float)]).all():
        raise ValueError("nonfinite result")
    return result
