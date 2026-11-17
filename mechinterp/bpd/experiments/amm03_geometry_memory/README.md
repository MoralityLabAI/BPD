# BPD-AMM-03: gauge-stable, cross-model strategy memory

[PROTOCOL.md](PROTOCOL.md) was committed on 2026-11-18 at `cae5a676` before experiment source, training models, or development data. The goal is to test whether memory learned from the causal geometry of **independently trained rank-two recurrent models** helps select a component beyond eight-pair per-model calibration. AMM-02's family-only policy collapsed to the pooled strategy.

The external donor target comes from the input task, defined before physical or BPD fitting. Every scored case receives one matched-dose current-dark component write and one learned tied recurrent continuation. Each evaluation root has 8 disjoint calibration donor pairs with both candidate interventions measured, followed by 128 independent held-out donor pairs. `memory_calibrated` and `local_only` share those 16 local labeled intervention measurements. Memory additionally uses its explicitly counted amortized training corpus.

## Run on CPU

From the BPD repository root with Python 3.11+ and PyTorch CPU installed:

```bash
python -m unittest mechinterp.bpd.experiments.amm03_geometry_memory.test_experiment -v

python -m mechinterp.bpd.experiments.amm03_geometry_memory.experiment --phase training --out /tmp/amm03_training.json
python -m mechinterp.bpd.experiments.amm03_geometry_memory.experiment --phase development --memory /tmp/amm03_training.json --out /tmp/amm03_development.json
python -c 'import json; x=json.load(open("/tmp/amm03_development.json")); json.dump(x["preflight_input"],open("/tmp/amm03_preflight_input.json","w"))'
python /path/to/pinned/EvalMath/skills/evalmath-mechinterp-power/scripts/preflight.py --pilot /tmp/amm03_preflight_input.json --out /tmp/amm03_preflight.json
```

The strict EvalMath variant is `--strict`; exit code 2 and `REDESIGN_OR_PILOT_ONLY` mean the registered confirmation cohort **stays sealed**. Do not score the 128 confirmation roots unless the prospective gate passes:

```bash
python -m mechinterp.bpd.experiments.amm03_geometry_memory.experiment --phase confirmation --memory /tmp/amm03_training.json --preflight /tmp/amm03_preflight.json --out /tmp/amm03_confirmation.json
```

The code checks source hashes and confirms eligibility *before* touching any confirmation root. Each `--out` uses exclusive file creation. Engineering tests only train excluded roots. CI independently replays case bits and the pinned EvalMath preflight. Raw receipts contain complete binary case outcomes, deterministic training and root diagnostics, and exact source hashes.

## Evidence limits

The target is a **trained synthetic tied linear recurrence**. Family cues, task labels and physical weights are part of the declared instrument. A positive result would show model-conditioned intervention policy transfer on this task; it would not establish general model reasoning, nonlinear transformer mechanism discovery, live agent-to-agent memory transfer or deployed Buzz functionality. Protected ARG-03-N evaluation data remain sealed.
