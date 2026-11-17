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

## Scientific confirmation hold after the completed development run

**The 128 AMM-03 confirmation roots are sealed. Do not run `--phase confirmation`, including with the historical `READY_FOR_CONFIRMATION` receipt.** The source-pinned planner reported READY because all 64 development root effects were exactly zero and its naive normal-power calculation treated that zero variation as certainty. The registered target of +20 percentage points is unsupported by the development result; a test-answer oracle had just 0.12 percentage points of additional room beyond local calibration. [RESULTS.md](RESULTS.md) contains the audit and the retained historical preflight receipt.

The frozen experiment source was left unchanged for source-hash replay. **Its original CLI checks the older planner output and is not an override of this scientific hold.** A new experiment with fresh roots and a separately registered higher-resolution causal contrast is needed before any confirmation work. EvalMath's follow-up correction is tracked in [EvalMath PR #51](https://github.com/MoralityLabAI/EvalMath/pull/51).

The strict EvalMath variant is `--strict`; its archived AMM-03 output is retained to document the planning anomaly. Future planning should use the fixed planner, and any non-READY verdict blocks confirmation before root access.

The code checks source hashes and its original preflight eligibility fields *before* touching any confirmation root. Each `--out` uses exclusive file creation. Engineering tests only train excluded roots. CI independently replays case bits and the pinned EvalMath preflight. Raw receipts contain complete binary case outcomes, deterministic training and root diagnostics, and exact source hashes.

## Evidence limits

The target is a **trained synthetic tied linear recurrence**. Family cues, task labels and physical weights are part of the declared instrument. A positive result would show model-conditioned intervention policy transfer on this task; it would not establish general model reasoning, nonlinear transformer mechanism discovery, live agent-to-agent memory transfer or deployed Buzz functionality. Protected ARG-03-N evaluation data remain sealed.
