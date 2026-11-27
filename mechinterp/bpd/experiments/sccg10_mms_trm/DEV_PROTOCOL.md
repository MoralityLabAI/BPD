# SCCG-10 / MMS-TRM development protocol (frozen by DEV_LOCK.json)

Evidence class: `DEVELOPMENT_ONLY`. This document and `DEV_LOCK.json` are
committed before any development root is warmed or scored. The trainer
refuses development roots when the live source hashes, budget or arm roster
differ from the lock (`economics.py --source-lock`).

## Units and custody
* Roots 29301..29308 (fresh, checked against every BPD branch). The independent
  unit is the model root. Panels, probes, checkpoints, queries and slots are
  not units.
* Common warmup per root: CURATE-02 TRM_w32_r29 recipe, 6,000 updates, delays
  1/2/3, AdamW lr 3e-3, wd 1e-4, clip 1, 2 CPU threads, 6 GiB Job Object cap.
  The competence gate (TRAIN d2, d7 ≥ 0.80; binding d2, d7 ≥ 0.75) is recorded. A root
  that fails is kept, reported and analysed as assigned.
* Curriculum proposals come from the TRAIN fit view, calibration from the TRAIN
  feedback view (domain-separated `BPD-SCCG10-MMS` seeds), and the mechanism
  panel from the TRAIN feedback view. The held EVAL partition is scored once per
  arm at the end. It is the historical CURATE-02 EVAL partition, so it is
  exposed, not sealed. No ARG-03-N or other protected cohort is read.

## Arms and budgets (frozen from engineering)
Candidate previews use the batched analytic AdamW preview (`--preview-mode batched`),
chosen on engineering because it reproduced transactional selections exactly at
2.2–3.0× lower CPU.
A_ordinary, B_ord_grad, C_mech_fresh, C_probe (fixed-update only, exact replay
of C), D_atlas, E_shuffled, F_obs_only. Fixed-update: 80 windows. Fixed-CPU:
100 process CPU-seconds per arm after warmup, charging the mechanistic setup
(slot projectors) to every mechanistic arm. A window starts only if its
estimated cost fits; otherwise the arm takes the affordable plain window or
stops, and any overrun is reported. Atlas 16 MiB; policy `Policy()` defaults
(max age 8 committed updates, trace residual ≤ 1.0, factor-frame residual
≤ 0.25); no-audit action for 4 windows when the first-order audit value
`mean(top-4 predicted) − mean(all predicted)` does not exceed
`(audit_cpu / plain_cpu − 1) · max(mean(all predicted), 0)`.

## Endpoints and family
Primary: fixed-CPU ordinary steering-disabled delay-31 accuracy, D_atlas minus
A_ordinary, per root. Secondary: D − C_mech_fresh, D − B_ord_grad, D − E_shuffled
(all fixed-CPU, delay 31). Exact two-sided root sign-flip, Holm over the four,
seeded root bootstrap. Descriptive only: fixed-update contrasts, other delays,
binding, feedback losses, panel hybrid flips and corruption, atlas reuse,
refresh and trace residuals, CPU and memory ledger.

The practical floor of +5 pp is an inherited planning value, not an established
threshold. The sham is E − A per root. The EvalMath strict preflight runs with
`heldout_sealed = false`, so its decision is `REDESIGN_OR_PILOT_ONLY` whatever
the outcome. No confirmatory claim can come from this cohort.

## Closed gates
Causal auxiliary training stays closed. The CURATE-02 behavioral
selective-control admission failed, and no fresh frozen behavioral gate has
passed. Negative and null outcomes are kept and reported with the same detail
as positive ones.
