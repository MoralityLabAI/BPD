# FSI-04: Dose-matched native causal correction pilot

**Status:** registered, development-only, source and CPU tests. Registration was committed **before** the fresh Sudoku roster is generated at `mechinterp/dark_fiber_gyms/fsi04/REGISTRATION.json`. The ARG-03-N 168-case protected test remains sealed. Native model execution requires the two **local** pinned 7M TRM checkpoints. CI never downloads or evaluates those checkpoints.

## Scientific question and design

We need a causal improvement in *correct Sudoku cells*, with a sham that changes the actual physical state by the same amount. Historical LOOPSCOPE-04 swaps often altered logits or donor alignment while yielding little task improvement. In particular, archived `yagizdevre` S4 stalled boards at step 4/horizon 32 averaged donor-joint minus clean **−2.28 percentage points** correct cells and random-joint minus clean **−5.77 points**, so donor versus random looked **+3.49 points** even though donor harmed task correctness. On `gaoxin492` S4 the corresponding numbers were **+0.37, +2.13 and −1.76 points**. These are descriptive **retrospective** data on 40 source boards per checkpoint.

The old random control resampled head-null and `z_L` orientations on their spheres, which matched state norms but could change the *distance moved from the clean recipient* relative to a correlated donor. The [FSI-04 dose matcher](dark_fiber_gyms/fsi04/geometry.py) corrects this.

The fixed `t=4`, `sigma=0.35`, horizon-32 pilot uses the same recipient, same-board perturbed donor, `z_H` head-null sphere transplant and full board `z_L` donor transplant as the historical experiment. The new random sham:

- keeps the exact recipient head-visible component and per-cell radius, with a random tangent direction rotated by the **same geodesic angle** as the donor transplant, so the per-cell `z_H` displacement chord matches;
- perturbs `z_L` along an independent random direction with the exact **per-cell Euclidean displacement magnitude** of the donor `z_L` transplant;
- retains the recipient prefix and checks the complete pre-continuation readout, head-cell radius, and each layer's per-cell physical dose discrepancy.

Each root runs `clean`, `donor_joint`, `random_joint_dose_matched`, and `self_control` in one fixed-budget continuation. The donor reference trajectory supplies the first-successor signed directional cosine. The raw receipt records all arms, **wrong-to-correct** cell gains, **correct-to-wrong** losses, net correct cells, exact board solves, and geometry certificates.

The preregistered **primary** exploratory endpoint is
\[
d_i=\frac{\mathrm{correctCells}_{\rm donor}(i)-\mathrm{correctCells}_{\rm clean}(i)}{81}
\]
on independent new `S4_stalled` Sudoku roots. The practical minimum is `+0.10` (about eight additional correct cells per board). A stronger `+0.15` target effect is tested against the calibrated variation floor. The primary checkpoint is `yagizdevre`, with `gaoxin492` as a heterogeneity replication. Donor minus matched random and donor-aligned next-logit displacement are secondary causal diagnostics. Improving only upon a destructive random control does not pass the primary task gate.

## Existing CPU preflight

Run `historical_preflight.py` using the merged EvalMath power-skill preflight. It reads the previously published, **old non-dose-matched** 40 S4 roots from each checkpoint and explicitly marks them retrospective.

The positive-effect threshold is compared with independent-root sham variation. For a nonparametric upper bound on a 95th-percentile sham with 95% confidence, the maximum observed sham requires **at least 59 independent calibration roots**. The fresh **24-S4-root pilot** is intentionally too small for that tolerance guarantee and is strictly developmental; it cannot become a confirmation test merely by adding iterations or measuring many cells. A second, fresh calibration roster of 59+ S4 roots is required for this particular tail guarantee before an independently sealed confirmation cohort. The current registration plans 256 independent confirmation roots **only after** resolving sham variation and pilot efficacy.

Historical continuous donor alignment, exact Sudoku solve changes and the selection of `t=4` are already publicly inspected. Therefore this protocol registers a *new exploratory instrument* and future test design; it does not reclassify the old 120-board data as confirmation.

## Fresh development roster and limits

The [roster generator](dark_fiber_gyms/fsi04/roster.py) uses randomized MRV full Sudoku solutions, an exact unique-solution clue-deletion solver, and synchronous naked/hidden-single propagation to select **24 S4-stalled** roots and **12 WB-Prop** negative-control roots. It rejects all exact/digit-relabel/full-grid automorphism equivalents of the 256 prior LOOPSCOPE-02 source puzzles and compares SHA-256 puzzle identities with the 120 historical LOOPSCOPE-04 root IDs. The historical archive contains hashes rather than source puzzles, so full-automorphism exclusion against those 120 cannot be independently certified. Do not call the new roster universally contamination-free.

CI produces a small six-board smoke roster, with the raw development.csv and hash-bearing manifest as artifacts. A full roster is created locally with the same script. **Never include ARG-03 test.jsonl in the exclusions or readout command.**

## Source and native commands

From the BPD repository root:

```bash
# First generate solver-checked unscored development puzzles.
python -m mechinterp.dark_fiber_gyms.fsi04.roster \
  --out-dir runs/fsi04/roster --max-attempts 8000

# In a source-only environment, run old-data preflight, a guaranteed red gate.
export EVALMATH_PREFLIGHT_FILE=/path/to/EvalMath/skills/evalmath-mechinterp-power/scripts/preflight.py
python -m mechinterp.dark_fiber_gyms.fsi04.historical_preflight \
  --out runs/fsi04/historical_preflight.json

# Only with *locally available SHA-pinned* 7M weights, after roster inspection:
python -m mechinterp.dark_fiber_gyms.fsi04.native_pilot \
  --repo-root . --checkpoint-file /local/step_21700 --tag gaoxin492 \
  --eval-csv runs/fsi04/roster/development.csv \
  --roster-manifest runs/fsi04/roster/development_manifest.json \
  --exclude-csv mechinterp/loopscope02/results/benchmark_bundle.json \
  --out runs/fsi04/gaoxin_development.json --device cuda:0

python -m mechinterp.dark_fiber_gyms.fsi04.native_pilot \
  --repo-root . --checkpoint-file /local/model.pt --tag yagizdevre \
  --eval-csv runs/fsi04/roster/development.csv \
  --roster-manifest runs/fsi04/roster/development_manifest.json \
  --exclude-csv mechinterp/loopscope02/results/benchmark_bundle.json \
  --out runs/fsi04/yagiz_development.json --device cuda:0
```

The native runner rejects missing weights, an altered roster manifest, SHA mismatch, an incorrectly labelled difficulty stratum, and geometric dose/readout parity violations. The runner itself imports no ARG-03 data. Retain checkpoint and board source hashes in each output and report all arms, including failed/silent controls. Generate further independent calibration and confirmation roots only after the developmental effect clears the *mechanistic* threshold and a valid matched-sham envelope is available.

## Evidence classification

The CPU work is a mathematical/instrumentation verification and fresh-board generation. Historical FSI-03 outcomes remain retrospective real native model evidence. Any new run on the generated pilot roots is **exploratory neural evidence**, and needs a model-file SHA and raw per-root receipt before it can be described as executed. Neither one-percent logit-margin changes nor stable neural embeddings substitute for the prespecified ten-percentage-point *correct-cell* gain.
