# BPD-FSI02: prospective 32-world finite-basis and causal-mask experiment

**Protocol frozen before sampling:** `mechinterp/bpd/experiments/fsi02_registration_v1.json`, registration commit `b97c1f0211eaa403f1192d8f4cf1f561a98b80ab` (17 November 2026). Runner code is in `fsi02_run.py`. This is an exact-rational, finite-dictionary structural CPU experiment. There are no neural BPD decomposition fits or ARG-03 private test results.

## Frozen design

Each of 32 independent synthetic worlds supplies 64 training activation samples and 256 held-out readout samples. The five candidate row directions are `(1,0)`, `(0,1)`, `(1,1)`, `(1,-1)`, `(1,2)`; exhaustive evaluation considers every invertible row pair. The first 16 worlds have support on the two rotated lines `(1,1)` and `(1,-1)`, with a unique optimum read basis `(1,1),(1,-1)`. The remaining 16 have support at `(1,0)`, `(0,1)`, `(1,1)` with equal positive masses, creating exactly three optimal read bases and a cancellation/shear ambiguity. Each world has separate seeded training/readout streams, and the model chooses the lowest empirical expected-L0 basis using a fixed tie-break without access to readout samples.

## CPU results

| Measurement | Observed |
|---|---:|
| World-optimal basis selected under exact source law | 32/32 |
| Mean held-out active-row reduction vs. fixed coordinate basis | `1773/4096 = 0.4328613` |
| Rotated worlds, mean reduction | `0.8552246` |
| Cancellation-tied worlds, mean reduction | `0.0104980` |
| Cancellation worlds with a noncommuting singleton mask across optimal bases | 16/16 |
| Whole-block mask invariant under exact conjugation | 32/32 |

The 32-world prospective signed-difference test, using 20,000 seeded Monte Carlo sign flips, returned `1/20001` two-sided p with Bonferroni-adjusted value `2/20001`. The descriptive 97.5% bootstrap interval for the mean difference is `[0.2708,0.5957]` active rows. The effect is **established within the constructed two-stratum toy generator**, subject to independent-world and sign-symmetry assumptions. This numerical test is supplementary to the exact finite-basis law that explains the rotated cohort's large improvement.

Among cancellation worlds, the three optimal bases have the same exact expected L0. Relative reparameterizations between them can carry shear. The singleton mask `diag(1,0)` fails to commute for an optimal basis pair, while `I` commutes identically. This is a **finite algebraic mask counterexample**, not an observation of a learned transformer's latent causal channels.

## Inference boundary

The 16 rotated worlds were constructed to favor a non-coordinate read basis. Their large benefit over `(e1,e2)` is expected from the exact source distribution. The cancellation cohort's small deviations arise from sampling noise on equal-optimum bases. The pooled 32-world fraction of exact-optimum selections is **descriptive**, because the cohort mixture is fixed 16/16; conditional Clopper–Pearson brackets are reported separately for each independently sampled stratum. Results generalize only to the specified finite dictionary and toy activation laws. The existing BPD B1–B4 neural gates remain unchanged and ARG-03's 168-case held-out test remains sealed.

## Replay

Use the pinned EvalMath FSI-01 module at merge `5004cab9b366a412290a83945cda76987bb3cd6c`:

```bash
EVALMATH_FSI_FILE=../EvalMath/src/spiral_ln/evaluation_finite_sample.py \
  PYTHONPATH=. python -m unittest discover -s mechinterp/bpd/experiments -p 'test_fsi02_cpu.py' -v
EVALMATH_FSI_FILE=../EvalMath/src/spiral_ln/evaluation_finite_sample.py \
  python mechinterp/bpd/experiments/fsi02_run.py \
  --registration mechinterp/bpd/experiments/fsi02_registration_v1.json \
  --out /tmp/bpd_fsi02_replay.json
```

The committed `fsi02_cpu_world_receipt.json` preserves every selected basis, exact source optimal set, train/test L0 count, random-basis control and relative-mask witness, with the frozen Git blob identifiers. The CI compares every regenerated world and inference result, allowing only the run-duration metadata to change.
