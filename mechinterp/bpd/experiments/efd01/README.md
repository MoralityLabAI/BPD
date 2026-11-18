# BPD-EFD-01: energy-fiber dimension and delayed observability

This is the **source-only mathematical instrument** for the registered [EFD-01 protocol](PROTOCOL.md). The protocol was committed first at `495ec447bc817ed2fccf899544db56c1d5a5bdca`.

The physical energy-dark fiber at a state is the nullspace of the stacked present-readout Jacobian and scalar energy gradient:

```
K = ker([J_current; dE])
delayed_capacity = rank(J_future @ basis(K))
```

Source-only test:

```bash
python -m pip install numpy>=1.26
python -m unittest discover -s mechinterp/bpd/experiments/efd01 -p 'test_energy_fibers.py' -v
```

The `energy_fibers.py` instrument performs full SVD, numerical rank accounting, gauge transport-ready physical projection, native field decomposition, an algebraic antisymmetric representation and matched-K random-sham generation. Tests include finite-step curvature reversal of local descent. No teacher checkpoint, training, future holdout, model-root inference, or native BPD causal experiment is part of this source-only change.

LOOPSCOPE-10 integrates the same geometry with original frozen TRM teachers on an **independently registered development-only** source. BPD native factor operations and certified physical write equivalence require a separate follow-up.
