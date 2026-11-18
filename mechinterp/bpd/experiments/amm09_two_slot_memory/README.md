# BPD-AMM-09: two-slot BPD causal-routing memory

[PROTOCOL.md](PROTOCOL.md) was registered on **2026-11-18** at commit `f6a96069`, before AMM09 experimental source, training or development data. The AMM08 [development finding](../amm08_rank3_causal_memory/RESULTS.md) showed its case-conditioned memory defaulting to the unrestricted full three-component action on 99.50% of cases. AMM09 forbids that action, restricting every agent to the same **six admissible masks**, each using one or two of the three rank-three SVD-canonical BPD components.

## Registered physical test

Each new independent model root trains a genuine 6D rank-three, three-pass weight-tied recurrent physical operator from three externally defined successive output targets, then fits a separately initialized rank-three BPD factorization. Donor edits change one nuisance and two target source coordinates. The selected component-subset patch is projected current-dark and normalized to the common per-case physical L2 donor-write dose. A successful intervention repairs both independent donor targets on learned future passes two and three. Full donor-state replacement is a distinct-dose contextual control; the three-component mask never enters the scored action set.

Every policy sees the same 18-dimensional per-mask causal-response feature vector and **14 unlabeled delayed-readout measurements per case** (2 common plus 2 for each of six masks). New model roots receive 16 disjoint calibration pairs with six noisy candidate success reports per pair, independently flipped with known 20% noise, giving **96 equal-budget local labels** for the memory-plus-local and local-ridge arms. The global memory additionally trains using 64 fresh independent roots, with 128 donor cases per root, and freezes an 18-coefficient ridge policy at four-decimal precision.

The **primary question** is whether case-conditioned **memory_only** exceeds the **strongest fixed pooled permitted mask** selected on the *same global training corpus*. The registered secondaries compare memory-plus-local against pooled, local ridge and memory-only. All four root-level positive comparisons receive Holm adjustment. The independent sample is **64 newly trained development model roots**, each with 512 heldout and 512 independent repeatability donor cases. A 95/95 nonparametric sham calibration and the fixed +8-point scientific minimum govern confirmation admission.

## Prospective development completed; confirmation sealed

The original first-run 64-model developmental cohort is archived in the
[original hosted artifact](https://github.com/MoralityLabAI/BPD/actions/runs/44281548742).
[RESULTS.md](RESULTS.md) contains every arm, the four registered
independent-root comparisons, fidelity and physical-dose audits.
The preregistered primary **memory_only minus strongest fixed pooled
mask (0,2)** was **+14.78 percentage points** across 32,768 heldout
cases, **55 positive / 8 negative / 1 tied model roots**. Its corrected
four-test Holm p was **1.95e-9**, and root-bootstrap 95% interval
**[+11.15,+18.66]pp**. It passes the registered +8pp minimum,
while memory-plus-local adaptation was weaker than memory alone.

Two post-development audit qualifications apply. First, the
frozen `experiment.py.analysis()` mislabels the primary comparator's
treatment numerator, although the preregistered `preflight_input`
uses the right memory-only-versus-pooled contrast. The separately
committed [analysis amendment](AMENDMENT_POST_DEV_ANALYSIS.md),
`audit_development.py` and `results/ANALYSIS_CORRECTED.json`
restore all four root tests from the original policy and candidate
bits without modifying frozen model or action code.

Second, [NUMERIC_REPLAY_LIMITATION.md](NUMERIC_REPLAY_LIMITATION.md)
documents 27/49,152 changed training outcome bits across CPU
workers with the same model source. **Exact cross-host retraining
is unresolved.** The dedicated original-receipt audit downloads
and hashes the original hosted evidence and verifies its true
root-level analysis. The main CI reports nonportable retraining
as an explicit warning and never re-evaluates an already scored
development population.

Corrected EvalMath returned `READY_FOR_CONFIRMATION` using
the correctly registered effect and a **6.45pp** 95/95 sham bound;
63/64 model roots meet the fidelity veto and the allowed six-mask
oracle has 27.78pp headroom over pooled. **ALL 128 AMM09
CONFIRMATION ROOTS REMAIN SEALED.** The protocol additionally requires
a separately committed *human-approved* source-policy-preflight
release manifest, which is absent. The replay limitation calls
for further prospective deterministic-backend validation before
interpreting a confirmation run. Do not invoke `--phase confirmation`
or imply real-agent/LLM transfer.

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
