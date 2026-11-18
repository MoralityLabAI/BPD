# BPD-NC-12: low-target-gradient causal transfer frontier

**Prospective protocol-only first commit:** `6a4b5ebc16fae730f17b0e5dac5564eaf419c071` in [PROTOCOL.md](PROTOCOL.md).

NC-11 failed to transfer a causal projector between separately trained nonlinear recurrent models with **128 B Jacobian calibration examples**: a B-only rank-four projector using the same 128 target gradients won. NC-12 tests the registered nested **4, 8, 16, 32** B-gradient budgets.

Each newly trained model pair consists of two independently initialized dense 20D four-pass tied-`mix` models, with an externally defined task shared on `x4>0` and private input signal on `x4<=0`. The source A projector uses a model-only 1024-case discovery sample. Both A-assisted transport and target-only B projection use an identical *prefix* of B gradients from the first 32 positive-context cases in a separate 512-case calibration pool. Every comparator has access to the same 128 paired unlabeled A/B activation cases.

For q below hidden width 20, pure Jacobian Procrustes has an arbitrary null-space rotation. NC-12 registers **a fixed activation-anchored polar alignment**:

```
R_q = polar((G_A(q)^T G_B(q))/q + 0.05 * R_activation)
P_transfer(q) = R_q^T P_A R_q
```

The 0.05 ridge is frozen before any NC-12 model training. A permuted-case Jacobian alignment breaks the pairing through a fixed one-row cyclic shift. Direct target B rank-four, activation-only transport and local physical dose-matched shams are scored on identical untouched held-out examples.

The four preregistered directional tests are repeated at q={4,8,16,32}, creating one **16-comparison** family with **99.6875% Bonferroni** root-bootstrap intervals. One independent training pair, rather than pass, prompt, or sham, is the inference unit. An engineering low-query success requires q=4 or q=8 to satisfy all four gates over 8/8 roots. A failed result stays `ENGINEERING_NULL_OR_INELIGIBLE` with all raw receipts.

New development roots **1200–1207**, excluded root **1199**. Per root, two separately trained models with 384 Adam steps and separate 2048-case training streams; 1024 A gradient discovery, 512 paired calibration inputs, 1024 dose-only inputs, 1024 independent final evaluation inputs and 2048 sham proposals for each q. B 1024-gradient high-budget reference is explicitly descriptive and generated only after primary arm choices are frozen.

Run tests from BPD repository root:

```bash
python -m pytest -q mechinterp/bpd/experiments/nc12/test_experiment.py
python -m mechinterp.bpd.experiments.nc12.experiment --root 1199 --steps 384 --discovery-size 256 --dose-size 256 --evaluation-size 256 --proposals 256 --out /tmp/nc12_smoke.json
```

For full engineering development, execute roots 1200–1207 with the original registered budgets and aggregate the eight independent root receipts:

```bash
python -m mechinterp.bpd.experiments.nc12.aggregate --directory /tmp/nc12_results --out /tmp/nc12_results/NC12_ORIGINAL_SUMMARY.json
```

A source-only checksum lock and excluded smoke gate precede the eight-pair job. All NC-10/11 first-run negative results remain unchanged. Protected ARG-03-N, AMM-03 and future sham-calibration/confirmation samples remain sealed.
