# Original smoke comparison repair

The first full smoke used root 15999 once, under source commit 9e8a51c4c3633327ecd14c75a2fc48c963dc03b3 and execution 9d6df3fb0ccfd7a582df77540537d6ccdba56768. [Run 44317319540](https://github.com/MoralityLabAI/BPD/actions/runs/44317319540) retains its failed workflow conclusion. Its original [artifact 14433326335](https://github.com/MoralityLabAI/BPD/actions/runs/44317319540/artifacts/14433326335) has ZIP SHA256 `23e95ea757082dfef9bb417bb9dc04381aa7ef026c8526ece99b160cfacf5bb7`.

The model admitted eight shams and passed the independent source and task audits. The runner's comparison failed because the saved logical E values were replayed as row-major arrays. The live E values are column-major views sliced from torch.linalg.eigh, with strides (1,20). The arithmetic produced differences of at most 2.6506574712925612e-15 in the selected projectors after the layout change. On identical archived C-layout inputs, FC04 and FC05 still selected byte-identical projectors.

Restoring E to its original column-major layout reproduces all eight original projector hashes with both samplers. All 33 original parameter-state hashes and the 25 correction updates also agree exactly. U retains its original row-major layout. This repair changes replay construction only. The live experiment, sampler, thresholds, controls and original outcomes retain their source bytes.

A second runner check compared live tuple ranks with the aggregate's JSON-list schema. Re-reading the saved original JSON passes every technical admission check. The runner now audits that saved representation.

The separate replay workflow reads the first original artifact, verifies its root and NPZ hashes, repeats both independent audits, and checks the complete original source trajectory. It performs zero model training and zero new task evaluation. The first smoke failure remains in the record. Calibration release requires the repaired replay audit to pass before its execution lock is committed.
