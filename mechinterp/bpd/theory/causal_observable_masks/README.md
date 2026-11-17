# BPD-CQ-01: causal quotient of optimal sparse-basis masks

An isolated exact-Fraction instrument connecting [BPD's expected-L0 optimum-set theorem](../OPTIMAL_BASIS_AND_MASKS.md) to [delayed observable quotients](../../../../papers/shared_weights_dark_fibers/CAUSAL_QUOTIENT_ARG03.md) and to the [SCM-20–23 causal-fiber experiments](https://github.com/MoralityLabAI/Compactification-Theory/tree/main/research/dimensional_atlas/gym).

It asks which BPD component masks are **observationally invariant for k additional loops**, distinguishing immediate invisibility, delayed visibility, and global basis-independent interventions. It includes an exact counterexample showing finite-horizon invariant binary masks need not form a block-mask Boolean algebra on a restricted reachable subspace.

From BPD repo root:

```bash
PYTHONPATH=mechinterp/bpd/theory python -m unittest discover -s mechinterp/bpd/theory/causal_observable_masks/tests -v
PYTHONPATH=mechinterp/bpd/theory python -m causal_observable_masks.reference --out mechinterp/bpd/theory/causal_observable_masks/results/cq01_cpu_reference.json --check
```

The first command uses no PyTorch, GPU, remote services, or model weights. `reference --check` replays all 384 deterministic rational worlds from the registered script, verifies source hashes and the fixed result. The numerical output is a mathematical instrument audit rather than empirical BPD parameter recovery.

The companion retrospective [SCM21_RECEIPT_CROSSWALK.md](SCM21_RECEIPT_CROSSWALK.md) keeps the native-write norm mismatch visible. A proper trained BPD test requires independent candidate-subspace calibration and paired held-out outcome measurements.

The standalone module path keeps this exact Fraction audit independent of BPD's optional Torch runtime and its package-level imports. This is a CI dependency correction only; registered mathematical source, protocol and receipt hashes remain unchanged.
