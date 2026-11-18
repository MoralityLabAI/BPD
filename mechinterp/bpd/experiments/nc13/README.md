# NC13: source context and delivered displacement

**Original result: `ENGINEERING_NO_RECOVERY_OR_INELIGIBLE`.** All sixteen registered pairs completed and passed global integrity checks. The complete context-E comparison failed its improvement gate; every D endpoint was ineligible. Read the [original results](RESULTS_20261118.md), [aggregate](results/NC13_ORIGINAL_SUMMARY.json), [per-root table](PER_ROOT_20261118.csv) and [post-result next-step review](NEXT_STEP_REVIEW.md).

The [protocol](PROTOCOL.md) was committed first at `069b355e8f37ec780dc6c46b82176b6f32f8e39b`, before implementation and new model data. It reserves sixteen paired roots at eight target gradient inputs, comparing mixed/shared source discovery under retained-energy and actual closed-loop-displacement calibration.

The mixed-source transfer at gain1 supplies one common reference per pair. Source conditioning preserves the 1024-row discovery budget and raw-gradient convention. Displacement matching can use up to451 additional B dose-pool rollouts per pair; all target forward access is recorded separately from gradient access. Its bounded search leaves failed matches visible.

`DESIGN_PREFLIGHT.json` uses exposed NC12 effects only as prior design context. Its `REDESIGN_OR_PILOT_ONLY` output permits this engineering diagnostic and supplies no confirmation readiness for NC13. The [cross-thread bridge](CROSS_THREAD_BRIDGE.md) links the study to DT03 and the existing LOOPSCOPE12 response controls.

## Source checks

```sh
python -m mechinterp.bpd.experiments.nc13.verify_source
python -m pytest mechinterp/bpd/experiments/nc13 -q
python -m mechinterp.bpd.experiments.nc13.experiment --root 2601299 --outdir /tmp/nc13-excluded-smoke
```

Development roots require the complete execution lock and accepted runtime. Each output root is reserved before training, and existing reservations prevent replacement. The workflow enables the original development job only for the initial pull-request opening. Later documentation updates run source checks while retaining the first development archive.

## Audit the original archive

Download [original artifact 14434614677](https://github.com/MoralityLabAI/BPD/actions/runs/44322140651/artifacts/14434614677) and verify its ZIP SHA-256 against [ORIGINAL_ARTIFACT_MANIFEST.json](ORIGINAL_ARTIFACT_MANIFEST.json). After extraction:

```sh
python -m mechinterp.bpd.experiments.nc13.audit --receipts /path/to/original --summary /path/to/original/NC13_ORIGINAL_SUMMARY.json --out /path/to/new-audit.json
python -m mechinterp.bpd.experiments.nc13.audit_physical_receipts --receipts /path/to/original --out /path/to/new-physical-audit.json
```

The first auditor was committed before development and ran in the original CI. The second was added after the lock to inspect archived physical arrays; it performs no model evaluations. Both preserve scientific admission failures.
