# Auditing the original FC05 evidence

These checks read the first original 59-root archive from [run 44319021117](https://github.com/MoralityLabAI/BPD/actions/runs/44319021117). The ZIP SHA256 is `44b7ec94472b11444d7c71fc9668740d925c76367b27f51c0921556d93769986`. Extract its 187 files into a dedicated directory and keep new audit outputs elsewhere.

The original environment used Python 3.12.16, NumPy 2.3.5, SciPy 1.17.0 and Torch 2.5.1+cpu. Exact trace comparison relies on the frozen arithmetic and restored source-basis layout. The auditor checks all 55 dependency hashes before reporting its result. Other numerical environments can change floating-point bytes, which an exact replay audit must disclose.

From the repository root, the registered aggregate recomputes source and task audits, the original bootstrap summaries and the EvalMath decision:

```bash
python -m mechinterp.bpd.experiments.nc09_fixed12.full_tangent_calibration_fc05.aggregate \
  --root-dir /absolute/path/to/extracted-original \
  --out-dir /absolute/path/to/new-aggregate-output \
  --archive-source https://github.com/MoralityLabAI/BPD/actions/runs/44319021117
```

The post-result trace auditor checks every saved Y state and line-search trial. It recomputes all archived Jacobians and their correction vectors:

```bash
python -m mechinterp.bpd.experiments.nc09_fixed12.full_tangent_calibration_fc05.postrun.audit_source_trace \
  --root-dir /absolute/path/to/extracted-original \
  --archive /absolute/path/to/bpd-nc09-fc05-original-59-root-44319021117.zip \
  --out /absolute/path/to/new-source-audit.json
```

The output directory must already exist. The auditor rejects an existing output file and an output path inside the original evidence directory. It returns exit status 1 when its checks fail. Its elapsed-time field changes with each execution.

The published trace auditor adds command-line paths and output safeguards to the script used for the independent post-run review. Its audit arithmetic remains unchanged. [PUBLISHED_EVIDENCE.json](../results/PUBLISHED_EVIDENCE.json) records both script digests, while [FC05_SOURCE_AUDIT.json](../results/FC05_SOURCE_AUDIT.json) preserves the first independent audit output. These audit commands perform zero model training and zero new intervention rollouts.
