# AMM-08: seven-action, three-pass recurrent BPD strategy memory

[PROTOCOL.md](PROTOCOL.md) was registered first at commit `88855522` on November 18, 2026, before AMM-08 source or model training. It includes the disclosed eight-seed engineering-only architecture search. The chosen 6D rank-three recurrence is independently trained from scratch at every model root; a separate BPD V/U factorization is fitted and SVD-canonicalized for seven possible physical component subsets.

Each donor changes the inputs used for two delayed behavioral readouts plus one nuisance input. A fixed physical dose and immediate-readout-dark projection apply to **each of seven candidate masks**; the agent chooses exactly one. Success requires repairing **both** task-defined donor outputs after two further learned tied passes. The labels are fixed independently of model fitting. The candidate-gated donor continuation is *audit-only*.

The global memory trains on 64 independent model roots with 128 donor cases/root and all seven action labels. Every new root receives 16 local calibration pairs, seven candidate labels per pair, each independently flipped with known p=.2, for **112 noisy feedback observations**. The proposed memory-plus-local residual policy and strong local-ridge comparator receive identical calibration and physical response-probe budgets. They each see seven patched candidate continuations and an unpatched reference at two readout horizons, or 16 unlabeled measurements per heldout case. A fixed memory-only policy, local count policy, pooled mask, random mask, equal-dose isotropic random-dark shams, zero patch and larger-dose full donor are registered controls.

The independent unit is **one trained physical model and BPD factorization**, with disjoint root-specific training, calibration, evaluation and repeatability streams. Repeated donor cases and seven interventions inside a root remain repeated observations. There are 64 training and 64 development roots, 512 heldout and 512 repeatability cases per development model, and **128 sealed confirmation roots**. The registered primary floor is +8 percentage points, with an independent-world 95/95 sham tolerance bound from 64 roots. Confirmatory source release additionally requires a nontrivial per-case oracle headroom and 56/64 source-validation fidelity roots.

## CPU workflow

Use Python 3.12 and CPU PyTorch 2.10 from the BPD repository root:

```bash
python -m unittest mechinterp.bpd.experiments.amm08_rank3_causal_memory.test_experiment -v
python -m mechinterp.bpd.experiments.amm08_rank3_causal_memory.experiment --phase training --out /tmp/amm08_train.json

# Commit results/TRAINING_LOCK.json from the reproducible first run before development:
python -m mechinterp.bpd.experiments.amm08_rank3_causal_memory.experiment --phase development --memory /tmp/amm08_train.json --out /tmp/amm08_dev.json
python -c 'import json; x=json.load(open("/tmp/amm08_dev.json"));json.dump(x["preflight_input"],open("/tmp/amm08_pilot.json","w"))'
python /path/to/pinned/EvalMath/skills/evalmath-mechinterp-power/scripts/preflight.py --pilot /tmp/amm08_pilot.json --strict --out /tmp/amm08_preflight.json
```

Pinned corrected planner: `MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a`. The hosted workflow first runs engineering-only source tests and trains new root memory. It blocks development until the memory digest and case-bit hashes are committed, and never runs confirmation. The CLI opens output files exclusively and checks a separate preflight plus explicit human-authorized release manifest *before* reaching a single confirmation model root.

Historical ARG-03-N and AMM03–07 confirmation populations remain sealed. The physical engine is a **synthetically trained linear recurrent model**, without claims of nonlinear TRM generality, unique neurons in pretrained language models, real agent memory exchange, or deployed Buzz relay functionality.
