"""Source-only MoE BPD routing and content diagnostics; no native hooks."""
import math


def sigmoid(x):
    if not math.isfinite(x):
        raise ValueError("nonfinite logit")
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    a = math.exp(x)
    return a / (1 + a)


def vector(x, n=None):
    x = tuple(x)
    if not x or (n is not None and len(x) != n) or not all(map(math.isfinite, x)):
        raise ValueError("invalid finite vector")
    return x


def support_ids(n, allowed, k):
    ids = tuple(range(n)) if allowed is None else tuple(allowed)
    if (type(k) is not int or not 1 <= k <= len(ids) or len(set(ids)) != len(ids)
            or any(type(e) is not int or not 0 <= e < n for e in ids)):
        raise ValueError("invalid support or top-k")
    return ids


def route(logits, bias, k=8, allowed=None):
    z = vector(logits)
    b = vector(bias, len(z))
    ids = support_ids(len(z), allowed, k)
    s = tuple(map(sigmoid, z))
    ranks = vector([x+y for x,y in zip(s,b)], len(z))
    selected = tuple(sorted(ids, key=lambda e: (-ranks[e], e))[:k])
    mass = sum(s[e] for e in selected)
    if mass <= 0:
        raise ValueError("zero mixture mass")
    p = tuple(s[e]/mass if e in selected else 0. for e in range(len(z)))
    return selected, p


def certify_box(logits, bias, writes, ranges, k=8, allowed=None, slack=0.):
    """Certify original top-k within support for simultaneous gain perturbations.

    Each write is a full logit vector; ranges bound additive gain changes.
    Real-arithmetic interval argument; supplied slack is not a machine proof.
    """
    z = vector(logits)
    b = vector(bias, len(z))
    selected, _ = route(z, b, k)
    ids = support_ids(len(z), allowed, k)
    if not math.isfinite(slack) or slack < 0 or len(writes) != len(ranges):
        raise ValueError("invalid box")
    lo, hi = list(z), list(z)
    for w, bounds in zip(writes, ranges):
        w = vector(w, len(z))
        a, c = vector(bounds, 2)
        if a > c:
            raise ValueError("reversed interval")
        for e, v in enumerate(w):
            lo[e] += min(a*v, c*v)
            hi[e] += max(a*v, c*v)
    lower = vector([sigmoid(x)+y for x,y in zip(lo,b)])
    upper = vector([sigmoid(x)+y for x,y in zip(hi,b)])
    missing = tuple(e for e in selected if e not in ids)
    rivals = set(ids)-set(selected)
    gap = min(lower[e] for e in selected)-max(upper[e] for e in rivals) if rivals else None
    return {"certified": not missing and (gap is None or gap > 2*slack),
            "lower_gap": gap, "missing": missing, "selected": selected}


def fixed_derivative(logits, bias, values, adjoint, write, k=8, allowed=None, scale=8., content_dot=None):
    """Directional utility derivative at fixed membership; report ties separately."""
    z = vector(logits)
    b = vector(bias,len(z))
    w = vector(write,len(z))
    a = vector(adjoint)
    v = tuple(vector(row,len(a)) for row in values)
    if len(v) != len(z) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid value vectors or scale")
    selected,p = route(z,b,k,allowed)
    s = tuple(map(sigmoid,z))
    rivals = set(support_ids(len(z),allowed,k))-set(selected)
    gap = min(s[e]+b[e] for e in selected)-max(s[e]+b[e] for e in rivals) if rivals else None
    if gap is not None and gap <= 0:
        raise ValueError("fixed-membership derivative undefined at ranking tie")
    q = [sum(x*y for x,y in zip(a,row)) for row in v]
    mean = sum(p[e]*q[e] for e in selected)
    mass = sum(s[e] for e in selected)
    routing = scale/mass*sum(s[e]*(1-s[e])*w[e]*(q[e]-mean) for e in selected)
    content = 0.
    if content_dot is not None:
        dv = tuple(vector(row,len(a)) for row in content_dot)
        if len(dv) != len(z):
            raise ValueError("invalid content derivative")
        content = scale*sum(p[e]*sum(x*y for x,y in zip(a,dv[e])) for e in selected)
    return {"routing": routing, "content": content, "total": routing+content, "gap": gap}


def path_terms(p0, p1, v0, v1, scale=8.):
    """Four counterfactual mixtures with independently supplied routing/content."""
    p0 = vector(p0)
    p1 = vector(p1,len(p0))
    if (min(*p0,*p1) < 0 or not math.isclose(sum(p0),1.,abs_tol=1e-10)
            or not math.isclose(sum(p1),1.,abs_tol=1e-10) or not math.isfinite(scale) or scale <= 0):
        raise ValueError("invalid normalized mixtures")
    v0 = tuple(vector(row) for row in v0)
    if len(v0) != len(p0) or any(len(row)!=len(v0[0]) for row in v0):
        raise ValueError("invalid baseline content")
    v1 = tuple(vector(row,len(v0[0])) for row in v1)
    if len(v1) != len(p0):
        raise ValueError("invalid changed content")
    def mix(p,v):
        return tuple(scale*sum(w*row[d] for w,row in zip(p,v)) for d in range(len(v0[0])))
    def sub(x,y):
        return tuple(a-b for a,b in zip(x,y))
    y00,y10,y01,y11 = mix(p0,v0),mix(p1,v0),mix(p0,v1),mix(p1,v1)
    return {"baseline":y00,"route_only":sub(y10,y00),"content_only":sub(y01,y00),
            "interaction":sub(sub(y11,y10),sub(y01,y00)),"total":sub(y11,y00),
            "tv":sum(abs(a-b) for a,b in zip(p0,p1))/2}
