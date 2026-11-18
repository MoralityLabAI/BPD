# BPD-AMM-05: case-conditioned causal-response strategy memory

Prospective registration [PROTOCOL.md](PROTOCOL.md) committed first at
`375a9510` with a pre-source readout-budget clarification in
[AMENDMENT_PRE_SOURCE.md](AMENDMENT_PRE_SOURCE.md). AMM-04's noisy
four-pair local calibration gain was only +3.61 percentage points and
supported by three independent roots. AMM-05 replaces its *constant
per-model component selection* with an independently learned **per-case
component routing** decision.

Each root trains its own 4D rank-two weight-tied physical recurrence and
a separate two-component BPD factorization, with the target supplied
externally from its input task before decomposition. Both relevant and
nuisance input coordinates change in every donor pair. Each candidate
BPD write is projected to the current readout's kernel and normalized
to a common physical dose. The selected write is followed by a real
learned shared pass.

The action learner receives eleven fixed, SVD-canonical causal-response
features per candidate and only *three unlabeled future-readout
measurements* per donor pair (one common unpatched baseline and two
candidate patches). The donor's full future continuation, independent
goal and scored success bits are reserved for audit. Both compared
policies receive the same four local calibration pairs with eight
independently corrupted p=.35 binary feedback labels. The proposed
memory policy additionally uses 64 independently trained model roots,
128 clean labeled donor cases per root and fixed ridge penalties.
The local-only comparator fits its own per-root ridge on precisely the
same eight noisy labels, with equal probe and inference budgets.

The independent unit is one newly trained physical recurrence plus
BPD decomposition. All 128 repeated heldout donor pairs and 128
repeatability pairs per development root are retained, including
failures. The registered primary science minimum is +8 percentage
points, with the corrected EvalMath 95th-quantile/95%-confidence
independent-world sham tolerance gate and three Holm-adjusted contrasts.

## Completed development decision: 2026-11-18

The 64 independent model-root development cohort is complete. [RESULTS.md](RESULTS.md)
records all candidate policy outcomes, cluster-level uncertainty and source
provenance. Memory+noisy-local improved on equal-budget local ridge by
**+5.87 percentage points**, with 38 positive, 12 negative and 14 tied
independent model roots. The repeatability-calibrated 95/95 variation
floor is **+14.06 points**, so the corrected EvalMath preflight gave
`REDESIGN_OR_PILOT_ONLY`. **All 128 AMM-05 confirmation model roots
remain sealed.** Do not create a confirmation release manifest or call
`--phase confirmation` based on this cohort. The immutable code and
source hashes are preserved so the hosted workflow can reproduce every
training/development root and per-case action.

## Run on CPU

Python 3.12 and CPU PyTorch 2.10.0 are pinned in hosted CI.

```bash
python -m unittest mechinterp.bpd.experiments.amm05_case_routing.test_experiment -v

# After frozen source is committed:
python -m mechinterp.bpd.experiments.amm05_case_routing.experiment \
    --phase training --out /tmp/amm05_training.json

# Only after training-memory manifest is committed:
python -m mechinterp.bpd.experiments.amm05_case_routing.experiment \
    --phase development --memory /tmp/amm05_training.json \
    --out /tmp/amm05_development.json

python -c 'import json; d=json.load(open("/tmp/amm05_development.json")); json.dump(d["preflight_input"],open("/tmp/amm05_pilot.json","w"))'
python path/to/pinned/EvalMath/skills/evalmath-mechinterp-power/scripts/preflight.py --pilot /tmp/amm05_pilot.json --strict --out /tmp/amm05_preflight.json
```

Pinned corrected EvalMath is `MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a`.
`REDESIGN_OR_PILOT_ONLY` stops confirmation and keeps all 128
prospective confirmation models sealed. Even a READY preflight cannot
run the confirmation CLI without a separately committed, source-locked
explicit release manifest. Source tests only execute excluded
`engineering` roots, including native BPDModel/PassMaskSchedule parity
on hosted CI. No AMM-03/04 confirmation roots or ARG-03-N test split
may be touched.

**Claim boundary:** Synthetic trained 4D *linear* recurrence and
manually registered two-component intervention instrument. No live
Buzz/Nostr relay, trained LLM agent, nonlinear TRM mechanism,
cross-architecture transfer or general model alignment result is
claimed.
