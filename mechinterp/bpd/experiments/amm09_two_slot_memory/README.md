# BPD-AMM-09: two-slot BPD causal-routing memory

[PROTOCOL.md](PROTOCOL.md) was registered on **2026-11-18** at commit `f6a96069`, before AMM09 experimental source, training or development data. The AMM08 [development finding](../amm08_rank3_causal_memory/RESULTS.md) showed its case-conditioned memory defaulting to the unrestricted full three-component action on 99.50% of cases. AMM09 forbids that action, restricting every agent to the same **six admissible masks**, each using one or two of the three rank-three SVD-canonical BPD components.

## Registered physical test

Each new independent model root trains a genuine 6D rank-three, three-pass weight-tied recurrent physical operator from three externally defined successive output targets, then fits a separately initialized rank-three BPD factorization. Donor edits change one nuisance and two target source coordinates. The selected component-subset patch is projected current-dark and normalized to the common per-case physical L2 donor-write dose. A successful intervention repairs both independent donor targets on learned future passes two and three. Full donor-state replacement is a distinct-dose contextual control; the three-component mask never enters the scored action set.

Every policy sees the same 18-dimensional per-mask causal-response feature vector and **14 unlabeled delayed-readout measurements per case** (2 common plus 2 for each of six masks). New model roots receive 16 disjoint calibration pairs with six noisy candidate success reports per pair, independently flipped with known 20% noise, giving **96 equal-budget local labels** for the memory-plus-local and local-ridge arms. The global memory additionally trains using 64 fresh independent roots, with 128 donor cases per root, and freezes an 18-coefficient ridge policy at four-decimal precision.

The **primary question** is whether case-conditioned **memory_only** exceeds the **strongest fixed pooled permitted mask** selected on the *same global training corpus*. The registered secondaries compare memory-plus-local against pooled, local ridge and memory-only. All four root-level positive comparisons receive Holm adjustment. The independent sample is **64 newly trained development model roots**, each with 512 heldout and 512 independent repeatability donor cases. A 95/95 nonparametric sham calibration and the fixed +8-point scientific minimum govern confirmation admission.

## CPU reproducibility

Run from BPD repository root with Python 3.12 and CPU PyTorch 2.10:

```bash
python -m unittest mechinterp.bpd.experiments.amm09_two_slot_memory.test_experiment -v

python -m mechinterp.bpd.experiments.amm09_two_slot_memory.experiment --phase training --out /tmp/amm09_training.json

# Only after publishing the matching results/TRAINING_LOCK.json:
python -m mechinterp.bpd.experiments.amm09_two_slot_memory.experiment --phase development --memory /tmp/amm09_training.json --out /tmp/amm09_development.json

python -c 'import json;d=json.load(open("/tmp/amm09_development.json"));json.dump(d["preflight_input"],open("/tmp/amm09_pilot.json","w"))'
python /path/to/pinned/EvalMath/skills/evalmath-mechinterp-power/scripts/preflight.py --pilot /tmp/amm09_pilot.json --strict --out /tmp/amm09_preflight.json
```

Pinned corrected EvalMath planner: `MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a`. The CI workflow first runs engineering-only tests and trains the 64 global-memory roots. It refuses development until the training policy, root membership and candidate case bits are committed. Complete raw per-case training and development receipts and strict preflight are archived. It never enters the confirmation phase.

**All 128 AMM09 confirmation roots are sealed** until both the prospective EvalMath and fidelity/headroom/dose gates pass and an explicit separate human-approved source-lock release is committed. AMM03–08 and ARG-03-N protected populations are also untouched.

**Evidence class:** synthetic 6D rank-three *linear* trained recurrent BPD causal-routing under a two-slot intervention budget. It establishes no general nonlinear transformer/LLM mechanism, actual agent learning or deployed Buzz/Nostr memory transfer.
