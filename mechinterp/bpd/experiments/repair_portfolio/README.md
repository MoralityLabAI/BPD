# BPD-RP-01: quantized component repair portfolios

**All four assigned roots had zero errors in the unchanged quantized deployment.** Each repair allocation preserved 1,024/1,024 terminal answers, so recoveries and corruptions are both zero. The complete independent replay passes. This gym provides an audited engineering fixture; its current task/damage recipe has no correct-answer repair headroom.

Start with [RESULTS.md](RESULTS.md) for measured outcomes. [APPLICATIONS.md](APPLICATIONS.md) develops uses for Behavioral Parameter Decomposition, including a proposed Delayed Pointer Repair gym. The [frozen protocol](PROTOCOL.md) records the training, quantization and search recipe.

## Scope

One learned V/U pair for the shared mix matrix and its learned CI router are trained through the unchanged BPD engine. Deployment quantizes V/U to symmetric 3-bit codes, disables the target residual, and recomputes CI at each reached message. Four global gate offsets are selected with labeled calibration cross-entropy.

Greedy, independent multistart and dynamic beam each receive 172 logical calibration rollout calls per root. Common bootstrap work is shared, for 496 physical calls per root. The final 256 examples are generated after all policy choices. Four separately trained roots supply the independent units; examples share the fixture's eight semantic bit/query patterns.

The evidence class is PILOT_ONLY_ENGINEERING. The official planner's scoped inputs and verdicts are in [evidence/](evidence/). Protected confirmation populations are outside this task.

## Inspect and replay the saved run

The complete raw records are compressed into one deterministic ZIP_LZMA archive per root under receipts/. Each includes its learned checkpoint and every candidate's reached-input/gate trace. Small JSON records also remain directly readable in the smoke and pilot folders. The [receipt manifest](receipt_manifest.json) records every member's SHA256 and verifies byte-for-byte extraction.

From this experiment directory, use Python's zipfile support to expand the archives:

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
for path in sorted(Path('receipts').glob('*.zip')):
    with ZipFile(path) as archive:
        archive.extractall('.')
PY
sha256sum -c lock.sha256
python audit_independent.py pilot_results --output replay_audit.json
```

The read-only auditor independently rebuilds proposals and retention, then replays the physical recurrence with saved weights and gates. It recalculates recovery/corruption from raw outputs. The CI transformer remains outside the independent reimplementation; archived CI inputs and applied gates receive explicit trace checks.

## Reproduce the engineering run

The repository's BPD core and native fixture must be present. The combined downloadable package includes the ten pinned upstream files needed by this fixture. Runtime versions appear in each run_manifest.json; requirements.txt records the execution environment.

From the BPD repository root:

```bash
python -m unittest mechinterp.bpd.experiments.repair_portfolio.test_experiment
python -m mechinterp.bpd.experiments.repair_portfolio.experiment --phase pilot --source-lock mechinterp/bpd/experiments/repair_portfolio/source_lock.json --output rp01_reproduction
```

The runner requires a new output directory and checks its implementation hashes against source_lock.json. Reproduction repeats the same assigned roots. A changed task, damage recipe or search budget requires a separately versioned experiment.

## Provenance

The upstream BPD commit is 6d3ca23a817095f38789751004594777f54cf7d8. [upstream_sources.json](evidence/upstream_sources.json) pins the exact core and fixture bytes.

Source freeze commit 1f2a87fd16b70ba78098c5060988a3124232e613 preceded the four-root run. The 26-file source/development lock has SHA256 846523b58693c40ac06fe97ed77b817c392953daa83a9a780482c8975c7b40da. The result report and packaging utility were added after the experiment; frozen experiment files stayed unchanged.

[pilot_execution_receipt.json](pilot_execution_receipt.json) records the remote source identity and run chronology. [pilot_independent_audit.json](pilot_independent_audit.json) contains all four replay reports. The earlier smoke root is preserved separately as development evidence.
