# SCCG-10 / MMS-TRM: native selective-control cache economics gym

Status: design and engineering reference only; not a frozen empirical
registration. Parent BPD main ab93e4da986beef87897f8d1244c7fe7564b5bde.
No training run or protected cohort access authorized by this document.

## Source anchors and integration
Main includes SCCG-01..09 (#168, #171, #173) and INK-BPD-06 (#161).
The native curation implementation lives on another line of development:
BPD #170, pinned af43e48b1ed472eef88bedd0cfa8f4f8aefb49b5,
mechinterp/bpd/experiments/trm_curation/{model,data,observer,cache,campaign,
transaction,reference}.py, RESULTS.md, PROTOCOL.md, V2_RUNBOOK.md.
#170 is draft; do not assume it or RP03/RP04 are present on main.
Integrate using an isolated worktree/branch from that exact source, then
merge/cherry-pick only reviewed SCCG dependencies with parity checks.
Keep PRs #155/#156/#157/#170 and existing locks/receipts intact. Do not
merge unrelated drafts as part of this assignment.

CURATE-02 has a 5380-param width-32 rank-29 streaming vector TRM with
64 carried coordinates, two tied block visits per token. It reports
ordinary d31 uniform 46.90%, ordinary-gradient 50.24%, mechanistic
49.15%, shuffled 47.27%, equal-CPU reinvestment 49.44%, and a 2.27x
same-checkpoint cache speedup in an exposed RAM benchmark. Its Phase B
behavioral selective-control admission failed. These are source-reported
development results; this gym must preserve that negative gate.
Its observer uses post-write joint-state margins, protected all other
addresses over multiple declared query delays, suffix-only continuation,
and finite full-hybrid transport. Cache keys currently invalidate at
checkpoint/frame change. Do not relax strict keys for ordinary cached
gradients, states, optimizer previews or activations.

## Scientific question
Can a separate cross-checkpoint mechanism atlas reduce the cost of
candidate selection enough to increase ordinary steering-disabled
learning utility under the same real CPU/RAM budget? Distinguish:
(1) safe finite response prediction, (2) cache economy, (3) autonomous
unsteered training benefit. A local RMS reader certificate cannot be
substituted for full native terminal-logit protection. No causal training
cross if behavioral transport admission remains failed.

## Phase 0: source and custody
Reproduce native focused tests, zero-hook parity, exact suffix and pass
offset parity, write reconstruction, complete transactional rollback
including explicit RNG and factor call counters; hash all inputs.
Use exposed engineering 28000/28001 and exposed development
28101..28108 only for historical replay, debugging, and descriptive drift
measurement. No promotion. Inspect existing root registries before
allocating fresh engineering and development IDs (proposed 29200..29201,
29301..29308; do not use if any overlap). Generate fresh domain-hashed
TRAIN fit, feedback, tune and final development draws with disjoint
nuisance seeds and root-specific encodings. Group edge orientations.
Never read ARG-03-N or any reserved test. Never select sites, rank, dose,
margin budget or cache thresholds using final evaluation.

## Phase 1: checkpoint drift and transport eligibility
Reuse the exact ComponentTRM architecture and generator. For fresh
engineering roots use 6000 ordinary warmup updates, delays 1/2/3,
AdamW lr .003 wd .0001 clip 1, 2 CPU threads; follow inherited
competence screen and report full learning curves. Train-only feedback
panel: proposed 24 semantic edges balanced over four edit slots and
transition types, one common nuisance draw expanded into four queries;
delays (2,7,31), fixed panel solely for drift/calibration. At warmup
and after 1,2,4,8,16,32,64 committed updates measure real receiver states,
physical write frames, protected signed relative-logit rows at all
addresses/delays, longest-delay target, clean margins, singular values,
condition number, rank threshold, rowspace drift, and current finite
continuation. Distinguish carried-state edits from actual per-visit
native factor writes. Compare exact versus approximate fixed-update
subspace; log residuals, and do not claim exact membership by tolerance.
Only current full trajectory derivatives can justify true terminal
linear map drift; a sampled difference or sketch is not a uniform bound.
Local RMS fixed-query reader experiments have their own scoped endpoint.

Candidate supports k=4,8 and full joint state; same admissible span
for reference/cached arms, rank-matched random/SVD and joint circuit
controls, equal physical dose (planning .75; tune .25/.75/1.5/3 only on
TRAIN), positive collateral planning grid .25/.5/1. Use SCCG-01 on
current derivative, SCCG-02 only with certified domain, SCCG-09 for
local normalized reader, SCCG-10 for conditional suffix error bounds.
Record target gain, delivered write amplitude, actual protected margin
changes, targeted flips among eligible wrong answers, collateral
corruption among initially correct unaffected answers, zero-rank
supports, all denominators. Full hybrid is privileged; match .75 dose
and retain natural-dose comparator separately. If insufficient
recoverable headroom or behavioral selectivity, stop causal auxiliary
training. Keep diagnostic cache experiment separate. Require a written
pre-data amendment with quantitative behavior thresholds after engineering,
rather than inventing new thresholds on the scored cohort.

## Phase 2: economic cache calibration
Two separate namespaces: exact existing cache with immutable full keys;
atlas proposals bound to origin and destination checkpoint, module,
visit/step kind, input/nuisance hash, response definition, support,
factor-frame and metric identity, dtype and dose. Store detached data
only. Transport or invalidate on change of physical frame/metric;
factor gauge changes may change support algebra. Never reuse stale
gradients or optimizer state as if current. Rerun finite native
continuation before using a cached proposal on a new checkpoint.
If there is no rigorous uniform error bound, mark heuristic proposal
and require actual per-candidate finite verification; never certify
unqueried supports safe. An audit failure or no certificate sends the
candidate to refresh or ordinary fallback, not to an infeasibility label.
Exact structural atlas reuse requires actual verified fixed quotient;
near-null numerical columns are not exact rational dependencies.
SCCG-10 upper bound is only on the declared span and depends on supplied
error/remainder bounds. Do not treat it as a training benefit bound.

Compare fixed-update and fixed CPU separately. Cache tiers 0,4,16,64 MiB
on same engineering stream; 256 optional engineering ceiling if useful.
Enforce 6 GiB process-tree cap via cgroup v2 (or Windows Job Object commit
limit + RSS monitor, report semantics). Fail closed if unavailable.
Two CPU threads; no paid GPU. Log payload/keys/metadata, allocator and
process-tree peak, cache hits and ages, invalidation, validation, read/
write/eviction cost, actual CPU/wall, data/observer forward/backward
visits, VJPs, support/solver calls, exact certificates, probe clone
and rollback time, committed and discarded updates, unique train draws.
No estimate substituted for measured resource accounting. Price atlas
construction and full live error measurement, including periodic
refreshes. Resource gain must survive comparison to exact same-checkpoint
memoization and ordinary reinvestment. Paired same-update deterministic
parity applies only to exact computations; approximate treatments
require finite response checks and their own independent endpoints.

## Phase 3: proposed paired development arms
Freeze final roster, time cap, architecture, data and policy before
new development scoring, after engineering only. Use common warmup
weights, optimizer moments, RNG/cursor and identical proposal streams
where eligibility permits; equalize actual capacity, update schedule
and all diagnostic work in the fixed-update comparison. For fixed CPU
all arms receive the same cap (planning 100 CPU-s/root after warmup,
chosen on engineering before freeze; include audits and fallback work).
Precommit an upper bound on completing an action or stop safely at
the last completed affordable action, report overruns. No free audit.
A: ordinary training/reinvest all compute (strongest ordinary curation
   competitor selected on TRAIN tune; report uniform also).
B: fresh ordinary gradient curation (match proposal/search budget).
C: fresh mechanistic observer and joint-support search; exact
   same-checkpoint cache allowed.
D: adaptive atlas-assisted mechanistic proposal policy, with certificate
   or explicit native verification, exact fallback and reinvestment.
E: age/support/dose/compute-matched shuffled mechanism information,
   same policy and verification; never knowingly commit a violating edit.
F: observation-only sham, same audit work, ordinary proposal decisions.
Optional exact paired replay arm isolates caching from policy changes.
No crossed causal auxiliary loss unless fresh behavioral gate passes.
Use bounded candidate panels and transactional optimizer previews only
where their estimated decision value exceeds audit cost; validate
ordering on TRAIN tune, never pay repeatedly for near-tied actions.

Primary scientific endpoint: ordinary unsteered d31 accuracy per root,
D minus A under fixed total CPU. Secondary required contrast D-C
(does caching help the mechanistic policy?), D-E (mechanism specificity),
D-B (strong ordinary curation); predeclare Holm family and effect
direction. Report d2/7/63/127, binding, loss, per-root paired gains,
recovery/corruption and cost without reselecting primary. Engineering
primary is time/correctness parity for exact reuse; approximate branch
needs observed held-out TRAIN audit of violations and coverage.
Planning effect floor +5 percentage points (inherited, not established);
freeze practical collateral guardrail and sham tail before any inference.
Use EvalMath strict mechinterp-power preflight; if exposed, unsealed,
sham uncalibrated, insufficient roots or below-floor headroom,
label REDESIGN_OR_PILOT_ONLY. Independent model roots, not probes,
support draws, repeated checkpoints or query slots, are units.
No confirmatory runs without a separate prospective plan.

## Phase 4: deliver
Additive code, tests, immutable source lock before generated data,
per-root receipts and checkpoint hashes, independent saved-checkpoint
replay, stop/failure ledger, separate actual time/memory table, no
promotion on negative results. DataGoblin exports TRAIN-only accepted
and rejected examples with candidate lineage, exact/heuristic scope,
cost, and verifier outcomes; exclude tune/dev/test from training intake.
Prove math and verify small exact fixtures first. If gate fails, return
negative result and remaining bottleneck. No claim of an advantage until
matched-budget training and fresh root evaluation support it.
