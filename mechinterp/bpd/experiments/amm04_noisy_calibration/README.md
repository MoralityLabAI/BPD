# BPD-AMM-04: model-conditioned memory under noisy calibration

The pre-data registration is [PROTOCOL.md](PROTOCOL.md), committed at `31e46035` before experimental source, training or evaluation. The previous [AMM-03 developmental null](../amm03_geometry_memory/RESULTS.md) showed eight clean local pairs saturating the available causal decisions. AMM-04 uses four local calibration pairs with each component feedback bit independently flipped at fixed probability 0.35, while reserving separate clean heldout donor pairs.

The repeated model training architecture, external task labels, SVD-canonical BPD factor geometry and equal-dose current-dark interventions remain disclosed. The cross-model ridge prior is trained only on 64 new model roots, each with 96 donor pairs, and frozen before evaluating 64 disjoint development roots. Both `memory_calibrated` and `local_only` see exactly **8 noisy local component observations** per model. `local_only` uses signed-gain geometry to break feedback ties. Memory has its explicitly disclosed amortized training cost. The new `repeatability` split measures variation of the *same fixed local-only policy* on fresh pairs.

## Development decision (2026-11-18)

The study has completed its development cohort. Read [RESULTS.md](RESULTS.md)
and the locked `results/DEVELOPMENT_LOCK.json` and `results/PREFLIGHT.json`.
**The corrected EvalMath preflight returned REDESIGN_OR_PILOT_ONLY. All
128 confirmation model roots remain sealed and must not be accessed.**
The registered primary gain was +3.61pp with only three positive independent
model roots, below the preregistered +8pp scientific minimum. The source
remains unchanged for exact replay, and the earlier historical model cohorts
are unaffected. Do not fabricate a READY preflight receipt or run
`--phase confirmation`.

## CPU experiment

```bash
python -m unittest mechinterp.bpd.experiments.amm04_noisy_calibration.test_experiment -v
python -m mechinterp.bpd.experiments.amm04_noisy_calibration.experiment --phase training --out /tmp/amm04_training.json
python -m mechinterp.bpd.experiments.amm04_noisy_calibration.experiment --phase development --memory /tmp/amm04_training.json --out /tmp/amm04_development.json
python -c 'import json; x=json.load(open("/tmp/amm04_development.json")); json.dump(x["preflight_input"],open("/tmp/amm04_pilot.json","w"))'
python /path/to/EvalMath/skills/evalmath-mechinterp-power/scripts/preflight.py --pilot /tmp/amm04_pilot.json --strict --out /tmp/amm04_preflight.json
```

Pin EvalMath to its corrected commit `6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a` (PR #51), and use a local CPU PyTorch 2.10 environment. A strict preflight exit code 2 means development-only evidence. The 128 independent confirmation model roots are sealed until a new preflight and registered effect/headroom gates support evaluation. The developer may also apply the scientific hold even when preflight returns READY. Protected ARG-03-N and AMM-03 confirmation roots remain unaccessed.

Source-only unit tests train `engineering` roots and include native BPDModel/TiedComponentLinear `PassMaskSchedule` parity when run inside the full BPD checkout. Raw JSON outputs contain audit-only clean calibration bits; policy functions consume the separately generated noisy-feedback object. Full heldout case bits, training source hashes, root labels and actual physical dose diagnostics permit exact replay.
