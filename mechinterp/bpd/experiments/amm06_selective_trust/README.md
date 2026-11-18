# AMM-06: selective trust of noisy causal feedback

[PROTOCOL.md](PROTOCOL.md) was registered first on 2026-11-18 at commit `4ede3b9290903dbf22a81c44811c9a00246ac2ee`, before implementation or experimental model training. AMM-05's directional result found that cross-model memory alone outscored an always-updated local policy; its 128-model confirmation cohort remains sealed.

AMM-06 tests a **preregistered, held-out local-feedback trust gate**. The model receives 8 source-disjoint calibration donor pairs with 16 noisy binary component labels. A root-level hidden judge has either .10 or .45 flip probability, balanced independently of the source family. The decision model is blind to the regime and receives the same 16 noisy labels and eleven-dimensional SVD-canonical causal-response features as the always-updated and local-ridge controls.

The global memory is ridge regression on 64 freshly trained synthetic 4D rank-two recurrent model roots, each with 128 independently generated donor pairs. For each new model, the trust gate fits a residual from the *first four* calibration pairs, evaluates its noisy predictive likelihood against frozen memory on the *last four*, and accepts the full 16-label update only when the **mixture log Bayes factor exceeds ln 2**. Both accepted and rejected paths choose one causal BPD component per test case using the same three unlabeled continuation readouts. The true donor label and donor-future continuation are audit-only; all active candidate patches have the same L2 dose.

## Source-locked CPU workflow

Requires Python 3.12 and PyTorch CPU:

```bash
python -m unittest mechinterp.bpd.experiments.amm06_selective_trust.test_experiment -v

python -m mechinterp.bpd.experiments.amm06_selective_trust.experiment \
  --phase training --out /tmp/amm06_train.json

# Only after results/TRAINING_LOCK.json has been committed:
python -m mechinterp.bpd.experiments.amm06_selective_trust.experiment \
  --phase development --memory /tmp/amm06_train.json --out /tmp/amm06_dev.json

python -c 'import json; d=json.load(open("/tmp/amm06_dev.json")); json.dump(d["preflight_input"],open("/tmp/amm06_pilot.json","w"))'
python /path/to/pinned/EvalMath/skills/evalmath-mechinterp-power/scripts/preflight.py \
  --pilot /tmp/amm06_pilot.json --strict --out /tmp/amm06_pf.json
```

Pinned corrected EvalMath planner: `MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a`. The independent unit is one trained teacher/factor pair; 256 heldout cases per root and 256 disjoint repeatability cases are nested observations. The preregistered minimum effect is +8 percentage points, the target +15, with 64 independent development roots and a 95/95 sham tolerance envelope. Additional predeclared quality-stratum/acceptance/headroom *vetoes* must pass before any release is considered.

**All 128 prospective AMM-06 confirmation model roots remain sealed until a separate human-approved source-lock release manifest**. The CLI fails closed on a missing/invalid EvalMath receipt or release document before fitting any confirmation model. No AMM-03/04/05 protected confirmation root or ARG-03-N test data are used.

## Interpretation

This is a synthetically trained **linear** recurrent BPD causal-routing experiment with a modeled noisy judge. It does not establish a real agent's understanding, general neural mechinterp transfer, nonlinear transformer reasoning, or live Buzz/Nostr memory.
