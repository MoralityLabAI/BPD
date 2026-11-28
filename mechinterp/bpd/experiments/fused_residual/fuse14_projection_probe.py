"""FUSE-14A/B: experimental frozen projection cast reuse, standalone parity probe."""
from __future__ import annotations

import argparse
import copy
import json
from contextlib import contextmanager
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def frozen_projection_scope(modules):
    """Only dense, bias-free, frozen FP32 Linear weights in eager autocast.

    One scope per joint forward/backward step; caller forbids concurrent use,
    weight writes (including .data), offload, checkpoint recompute, compile,
    and autocast stream migration. Unsupported calls use original forward.
    Ordinary versioned mutation after first use is rejected within the scope.
    """
    entries, originals = {}, []
    stats = {"hits": 0, "misses": 0, "fallbacks": 0}

    def build(m, original):
        def forward(self, x):
            w = self.weight
            old = entries.get(id(self))
            if old is not None and (w is not old[0] or w._version != old[1] or w.requires_grad):
                raise RuntimeError("frozen projection changed during scope")
            dev = w.device.type
            eligible = (type(self) is nn.Linear and self.bias is None
                        and not w.requires_grad and w.dtype == torch.float32
                        and w.is_contiguous() and dev in ("cpu", "cuda")
                        and x.device == w.device and x.dtype in (torch.float32, torch.get_autocast_dtype(dev))
                        and torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
                        and torch.is_autocast_enabled(dev)
                        and not torch.jit.is_tracing())
            if not eligible:
                stats["fallbacks"] += 1
                return original(x)
            dtype = torch.get_autocast_dtype(dev)
            if dtype != torch.bfloat16:
                stats["fallbacks"] += 1
                return original(x)
            stream = torch.cuda.current_stream(w.device).cuda_stream if dev == "cuda" else None
            key = (w.device, dtype, stream, tuple(w.shape), tuple(w.stride()), w.data_ptr())
            if old is not None:
                if old[2] != key:
                    raise RuntimeError("projection cast scope changed")
                cast = old[3]
                stats["hits"] += 1
            else:
                cast = w.detach().to(dtype)
                entries[id(self)] = (w, w._version, key, cast)
                stats["misses"] += 1
            return F.linear(x, cast)
        return MethodType(forward, m)

    try:
        for m in modules:
            if type(m) is not nn.Linear or any(m is prev[0] for prev in originals):
                raise ValueError("expected distinct exact nn.Linear modules")
            if "forward" in m.__dict__:
                raise ValueError("refuse prepatched projection")
            originals.append((m, m.forward))
            m.forward = build(m, originals[-1][1])
        yield stats
        for m, _ in originals:
            old = entries.get(id(m))
            if old is not None and (m.weight is not old[0] or m.weight._version != old[1] or m.weight.requires_grad):
                raise RuntimeError("frozen projection changed during scope")
    finally:
        for m, _ in originals:
            del m.forward
        entries.clear()


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.qkv_proj = nn.Linear(width, 3 * width, bias=False).requires_grad_(False)
        self.gate_up_proj = nn.Linear(width, 6 * width, bias=False).requires_grad_(False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(3 * width, width, bias=False)

    def forward(self, h):
        q, k, v = self.qkv_proj(h).chunk(3, -1)
        h = h + self.o_proj(torch.tanh(q * k) * v)
        g, u = self.gate_up_proj(h).chunk(2, -1)
        return h + self.down_proj(F.silu(g) * u)


def run_cell(device, width, visits, seed):
    torch.manual_seed(seed)
    base = Block(width).to(device)
    inputs = [torch.randn(2, 5, width, device=device) for _ in range(2)]
    shapes = {(width, 3 * width), (width, 6 * width)}  # saved transposed frozen weights
    results = []
    from contextlib import nullcontext
    for candidate in (False, True):
        block = copy.deepcopy(base)
        opt = torch.optim.AdamW((p for p in block.parameters() if p.requires_grad), lr=1e-3, foreach=False)
        xs = [x.detach().clone().requires_grad_() for x in inputs]
        saved = set()
        def pack(t):
            if t.dtype == torch.bfloat16 and tuple(t.shape) in shapes:
                saved.add(t.untyped_storage().data_ptr())
            return t
        with (frozen_projection_scope([block.qkv_proj, block.gate_up_proj]) if candidate else nullcontext({})) as stats:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                with torch.autocast(device, dtype=torch.bfloat16):
                    ys = []
                    for x in xs:
                        h = x
                        for _ in range(visits):
                            h = block(h)
                        ys.append(h)
                    loss = sum(y.float().square().mean() for y in ys)
                loss.backward()
            grads = [p.grad.detach().clone() for p in block.parameters() if p.requires_grad]
        opt.step()
        tensors = [loss, *ys, *(x.grad for x in xs), *grads, *block.parameters()]
        tensors += [v for p in opt.param_groups[0]["params"] for v in opt.state[p].values() if isinstance(v, torch.Tensor)]
        results.append(([t.detach().cpu() for t in tensors], len(saved), dict(stats)))
    a, b = results
    finite = all(torch.isfinite(t).all().item() for t in a[0] + b[0])
    bitwise = all(x.dtype == y.dtype and x.shape == y.shape and torch.equal(x.contiguous().reshape(-1).view(torch.uint8), y.contiguous().reshape(-1).view(torch.uint8)) for x, y in zip(a[0], b[0]))
    return {"width": width, "visits": visits, "seed": seed, "finite": finite, "bitwise": bitwise, "native_saved_shape_storages": a[1], "candidate_saved_shape_storages": b[1], **b[2]}


def guards(device):
    m = nn.Linear(8, 24, bias=False).to(device).requires_grad_(False)
    x = torch.randn(2, 8, device=device, requires_grad=True)
    with frozen_projection_scope([m]) as stats:
        m(x)
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            m(x)
        with torch.autocast(device, dtype=torch.bfloat16):
            m(x)
            m(x)
        assert stats == {"hits": 1, "misses": 1, "fallbacks": 2}
    assert "forward" not in m.__dict__
    try:
        with frozen_projection_scope([m]), torch.autocast(device, dtype=torch.bfloat16):
            m(x)
            with torch.no_grad():
                m.weight.add_(1)
            m(x)
    except RuntimeError as exc:
        assert "changed" in str(exc)
    else:
        raise AssertionError("mutation accepted")
    assert "forward" not in m.__dict__
    with frozen_projection_scope([m]) as stats, torch.autocast(device, dtype=torch.bfloat16):
        m(x)
        assert stats["misses"] == 1
    m.requires_grad_(True)
    with frozen_projection_scope([m]) as stats, torch.autocast(device, dtype=torch.bfloat16):
        m(x).float().sum().backward()
        assert stats["fallbacks"] == 1 and m.weight.grad is not None
    assert "forward" not in m.__dict__
    return {"disabled_or_no_grad_fallback": True, "mutation_rejected": True,
            "fresh_scope": True, "trainable_fallback": True, "restored": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [run_cell(args.device, w, t, s) for w in (16, 32, 64) for t in (2, 6, 12) for s in (14001, 14002)]
    print(json.dumps({"device": args.device, "torch": str(torch.__version__), "rows": rows, "guards": guards(args.device)}, indent=2))
    assert all(r["finite"] and r["bitwise"] and r["misses"] == 2 for r in rows)


if __name__ == "__main__":
    main()
