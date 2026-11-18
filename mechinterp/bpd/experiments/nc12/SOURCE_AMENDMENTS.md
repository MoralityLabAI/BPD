# NC-12 predevelopment source amendment, 2026-11-18

The prospective frozen study design remains at protocol-only commit `6a4b5ebc16fae730f17b0e5dac5564eaf419c071`. No scientific threshold, model-root population, gradient budget, target-access limit, root bootstrap family, sham admission, or comparison changed.

The first [source-only GitHub Actions run #44283519514](https://github.com/MoralityLabAI/BPD/actions/runs/44283519514) verified the original source Git blobs and passed **15/16** source tests. Its final reduced smoke test failed at the `q=32` within-B orthogonal factor-gauge audit because the `basis_transport` function was used without an import. The gated full eight-root development job was **skipped**, so none of roots **1200–1207** was trained or evaluated in this run. The excluded root **1199** was exercised in the failing test only.

Prior to any development execution, the implementation added the one missing import from NC-03, commit `734f44647144ff375aa83a597eb765cc4bc9b69c`. The source blob SHA changed to `6341ea32e1af3fa7b4725ced536c5a07ef0814ee`, recorded in `EXECUTE_LOCK.json`. The CI source gate now requires an explicit amendment entry with the skipped development job and source-only scope. The first failed CI receipt is retained for auditing.

Any NC-12 full development results must be reported under this amended **predevelopment** source, with the original source-only protocol unchanged. Scientific confirmation remains sealed.
