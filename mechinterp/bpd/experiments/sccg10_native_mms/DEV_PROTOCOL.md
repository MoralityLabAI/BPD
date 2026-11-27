# SCCG-10 / MMS-TRM development protocol (prospectively frozen)

Machine-readable lock: `DEV_FREEZE.json` (source SHA-256 of every executed
file, recipe, arm configurations, roots, statistics). `devrun.py` refuses to
start if any locked file differs. Engineering receipts that set the frozen
values are under `results/engineering/` and are summarized in RESULTS.md §2.
This plan was committed before any development root was trained.

## Units, custody, roots
* Units: independently initialized ComponentTRM roots 29301–29308 (fresh;
  registry check in common.py). Each root's warmup weights, optimizer moments,
  RNG and data cursor are shared by its arms. Probes, windows, edges and query
  slots are never units.
* Data: TRAIN fit view for training and candidate proposals; TRAIN feedback
  view (12 edges, delay 15) for calibration, acceptance and controller value;
  final development = historical EVAL tuple partition with fresh SCCG10 seed
  domains (`final_dev_d*`, `sccg10_final_bind_d*`). The TRAIN tune view is
  not read in development. ARG-03-N and reserved cohorts are never read.
  Historical roots 28000/28001/28101–28108 are replay and diagnostics only.
* Architecture and warmup: CURATE-02 TRM_w32_r29 (5,380 parameters, width 32,
  rank 29, one low visit per token, 64 carried coordinates), 6,000 AdamW
  updates at lr 3e-3, train delays 1/2/3, 2 CPU threads, 6 GiB Win32 job cap.

## Arms (identical proposal streams, interleaved least-CPU-first per root)
A, B, C, D, E, F as defined in trainer.py. The role of "A" (strongest
ordinary training with full reinvestment) is fixed in `DEV_FREEZE.json`
`arm_roles` from engineering TRAIN-tune results. The uniform arm is
always reported. Causal auxiliary training is closed: the frozen behavioral
gate failed on root 29200.

## Comparisons
* Fixed total CPU: every arm gets the frozen cap. An arm stops before a
  window whose predicted cost would exceed the cap, and overruns are logged.
  All audits, refreshes, previews, rollbacks, cache and atlas operations,
  and controller measurements are charged.
* Fixed update: every arm runs the frozen number of windows, with a safety
  CPU stop that is logged.

## Endpoints and statistics
* Primary: ordinary steering-disabled final-development delay-31 accuracy
  (512 cases/root), D − A at fixed CPU, paired by root.
* Family (Holm, α = 0.05, two-sided exact root sign-flip): D−A (primary),
  D−C, D−B, D−E. Seeded root bootstrap intervals are descriptive.
* Practical floor: +5 percentage points (planning value inherited from
  CURATE-02, not established). Nothing below the floor is called a benefit.
* Secondary, descriptive, no reselection: d2/7/63/127, all-distinct binding
  d2/7/31, final CE, windows/accepted/rejected, refresh rate, atlas coverage
  (verified reuse fraction), physical dose of used directions, verified
  target gain and protected change, CPU by category, peak memory.
* EvalMath strict mechinterp-power preflight (pinned
  `repair_portfolio/evidence/preflight.py`): pilot = development roots,
  paired effect D−A, sham = E−A (matched shuffled mechanism), declared
  `heldout_sealed: false`. The expected verdict is REDESIGN_OR_PILOT_ONLY.
  No confirmatory claim is possible from this cohort.

## Stop and reporting rules
Every root and arm is reported, including failures and overruns. The primary
endpoint is not reselected. A negative or null primary result is the result.
If the strongest ordinary arm wins, the report names the limiting cost from
the measured ledger.
