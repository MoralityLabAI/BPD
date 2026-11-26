# BPD-RP-04: component-local memory steering

Development results are pending. RP04 deepens the RP03 memory-edit diagnostic through the native BPD read/write factors and the GRU carry path. It uses frozen checkpoints from eight exposed model roots and fresh development input draws. All results remain DEVELOPMENT_ONLY.

The [protocol](PROTOCOL.md) fixes six state directions at L2 dose 0.75, matched rotations, and carry/write route controls. [THEORY.md](THEORY.md) explains why projecting into selected V columns can affect other component reads, how a dual edit isolates those reads, and why isolation can cost state norm. The selected U rows supply a separate limit on writable gate directions.

From the BPD repository root:

```bash
python -m pip install -r mechinterp/bpd/experiments/component_selectivity/requirements.txt
python -m unittest discover -s mechinterp/bpd/experiments/component_selectivity -t . -p 'test_*.py' -v
python -m mechinterp.bpd.experiments.component_selectivity.experiment --phase smoke --out /tmp/rp04_smoke
python -m mechinterp.bpd.experiments.component_selectivity.experiment --phase development --source-lock mechinterp/bpd/experiments/component_selectivity/SOURCE_FREEZE.json --out /tmp/rp04_development
python -m mechinterp.bpd.experiments.component_selectivity.audit mechinterp/bpd/experiments/component_selectivity/results/development --out /tmp/rp04_audit.json
```

Each output directory must be fresh. The development command requires the published source lock. Existing RP03 models and their original experiment receipts retain their hashes.
