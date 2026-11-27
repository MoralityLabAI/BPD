# BPD-RP-04: component-local memory steering

The amended eight-root replacement development run completed with
**1,064,960 / 1,064,960 prediction bits independently replayed**
and **5,713,396 checks passed**. On final B, the true single-slot hybrid
fixed-span edit exceeded the ordinary donor fixed-span edit by
**8.691 percentage points** at the matched 0.75 state L2 dose.
Evidence remains **DEVELOPMENT_ONLY** and the EvalMath decision remains
**REDESIGN_OR_PILOT_ONLY**.

See [RESULTS.md](RESULTS.md) for full root-level effects, geometries,
audit and raw GitHub Actions artifact custody. The previous arithmetic
failure and new independent seed domain are documented in
[Amendment 01](AMENDMENT_01_AUDIT_ARITHMETIC.md).

From the BPD repository root:

```bash
python -m pip install -r mechinterp/bpd/experiments/component_selectivity/requirements.txt
python -m unittest discover -s mechinterp/bpd/experiments/component_selectivity -t . -p 'test_*.py' -v
python -m mechinterp.bpd.experiments.component_selectivity.experiment --phase smoke --out /tmp/rp04_smoke
python -m mechinterp.bpd.experiments.component_selectivity.experiment --phase development --source-lock mechinterp/bpd/experiments/component_selectivity/SOURCE_FREEZE.json --out /tmp/rp04_development
python -m mechinterp.bpd.experiments.component_selectivity.audit mechinterp/bpd/experiments/component_selectivity/results/development --out /tmp/rp04_audit.json
```

Each output directory must be fresh. The development command requires the published source lock. Existing RP03 models and their original experiment receipts retain their hashes.

## Audit amendment and replacement development

[Amendment 01](AMENDMENT_01_AUDIT_ARITHMETIC.md) records the failed original
eight-root audit, the float32 operation-order repair, and a new development
input-seed domain. The first-attempt original local raw receipts are unavailable
in the current workspace and have not been uploaded to this branch.

The PR workflow runs a fresh engineering check of the amended auditor.
Following publication of `SOURCE_FREEZE.json`, it executes the full replacement
eight-root development diagnostic, independently replays it, and uploads all
receipts as a named GitHub Actions artifact. The development study remains
`DEVELOPMENT_ONLY`; source, checkpoint, and result hashes are retained.
