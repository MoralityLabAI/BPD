"""Exact single-slot donors; one nuisance draw shared by all four queries."""
import math
import torch

from ..training_steering.data import PairBatch, root_encoding
from .reference import legal


def edge_batch(edges, *, root, seed, delay, split='train', noise=.03):
    edges = tuple(edges)
    if not edges or type(delay) is not int or delay < 0 or not math.isfinite(noise) or noise < 0:
        raise ValueError('invalid edge batch recipe')
    for r, h, s in edges:
        if (type(s) is not int or not 0 <= s < 4 or not legal(r, split)
                or not legal(h, split) or [i for i in range(4) if r[i] != h[i]] != [s]):
            raise ValueError('edge violates partition or single-slot semantics')
    n = len(edges)
    rng = torch.Generator(device='cpu').manual_seed(seed % (2**63-1))
    addr, val = root_encoding(root, 4, 4)
    receiver = torch.tensor([r for r, _, _ in edges])
    hybrid = torch.tensor([h for _, h, _ in edges])
    slot = torch.tensor([s for _, _, s in edges])
    order = torch.rand((n,4), generator=rng).argsort(dim=1, stable=True)
    base = torch.zeros((n,delay+5,11), dtype=torch.float32)
    base[:,:4,0] = 1
    base[:,4:4+delay,1] = 1
    base[:,-1,2] = 1
    row = torch.arange(n)[:,None]
    base[row,torch.arange(4)[None,:],3+addr[order]] = 1
    if delay:
        t = torch.arange(4,4+delay)[None,:]
        base[row,t,3+addr[torch.randint(4,(n,delay),generator=rng)]] = 1
        base[row,t,7+val[torch.randint(4,(n,delay),generator=rng)]] = 1
    if noise:
        base[:,:,3:] += noise*torch.randn((n,delay+5,8),generator=rng)
    query = torch.arange(4).repeat(n)

    def features(m):
        x = base.clone()
        x[row,torch.arange(4)[None,:],7+val[m.gather(1,order)]] += 1
        x = x.repeat_interleave(4,dim=0)
        x[torch.arange(n*4),-1,3+addr[query]] += 1
        return x

    xr,xh = features(receiver),features(hybrid)
    vr,vh = receiver.repeat_interleave(4,dim=0),hybrid.repeat_interleave(4,dim=0)
    yr,yh = vr.gather(1,query[:,None]).squeeze(1),vh.gather(1,query[:,None]).squeeze(1)
    return PairBatch(receiver=xr,donor=xh.clone(),hybrid=xh,
                     labels_r=yr,labels_d=yh.clone(),labels_h=yh,
                     edit_slot=slot.repeat_interleave(4),query=query,
                     values_r=vr,values_d=vh.clone(),values_h=vh,prefix_steps=4)
