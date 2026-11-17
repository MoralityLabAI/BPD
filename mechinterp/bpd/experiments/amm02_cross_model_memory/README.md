# BPD-AMM-02: memory transfer across independently trained recurrent models

Source of record: [PROTOCOL.md](PROTOCOL.md), registered at `ce390c8`, and pre-development [AMENDMENT_SHAM.md](AMENDMENT_SHAM.md), registered at `b811ab1`.

This expands the earlier AMM-01R analytic positive instrument to 4D **separately Adam-trained rank-two recurrent physical models** with **separately trained and SVD-canonicalized BPD factor pairs**. A fixed policy observes the opaque task family and chooses one of the two canonical component indices for a matched-dose, current-dark donor write. Task labels are defined directly from the source inputs before model training and before BPD fitting.

The policies compare frozen family-keyed training memory, the pooled best component, swapped family memory, a per-pair random component, and equal-dose isotropic dark controls. Train/engineering/development/confirmation root seeds are separate. Within-root cases remain repeated outcomes.

## CPU commands

From BPD repository root:

```bash
python -m unittest mechinterp.bpd.experiments.amm02_cross_model_memory.test_experiment -v
python -m mechinterp.bpd.experiments.amm02_cross_model_memory.experiment --phase engineering --out /tmp/amm02_engineering.json

python -m mechinterp.bpd.experiments.amm02_cross_model_memory.experiment --phase training --out /tmp/amm02_training.json
python -m mechinterp.bpd.experiments.amm02_cross_model_memory.experiment --phase development --memory /tmp/amm02_training.json --out /tmp/amm02_development.json
python -c 'import json; d=json.load(open("/tmp/amm02_development.json")); json.dump(d["preflight_input"],open("/tmp/amm02_preflight_input.json","w"))'
python path/to/pinned/EvalMath/skills/evalmath-mechinterp-power/scripts/preflight.py --pilot /tmp/amm02_preflight_input.json --strict --out /tmp/amm02_preflight.json
```

If strict preflight fails, leave the `confirmation` phase sealed and publish the complete development receipt with a `REDESIGN_OR_PILOT_ONLY` verdict. A successful strict preflight permits the separately registered 128-root confirmation command:

```bash
python -m mechinterp.bpd.experiments.amm02_cross_model_memory.experiment --phase confirmation --memory /tmp/amm02_training.json --preflight /tmp/amm02_preflight.json --out /tmp/amm02_confirmation.json
```

The CLI opens any `--out` path with exclusive creation. It requires a source-hashed training memory, a matching preflight `READY_FOR_CONFIRMATION` receipt and an allowed phase before generating any confirmation model.

## Evidence boundaries

The label comes from the donor input, independently of both neural training and decomposition. This is a synthetic **linear trained-model** test of fixed component-index strategy transfer, with an externally defined behavioral objective. The BPD factor basis is selected by weight-derived SVD; the experiment does not establish unique learned neuronal features. It uses no real LLM agent, no signed Buzz exchange, no pretrained model download, and no ARG-03-N protected test data. AMM-01R receipts remain unchanged.
