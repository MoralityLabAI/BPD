# BPD-TRM-CURATE-01: selective counterfactual curriculum

Status: development scaffold; protocol proposal, no training outcomes or promotion.
Parent: RP04 aa4b55d8c968084c1414846441cb539356ba7995 (PR #156).
The RP03/RP04 source locks and result bytes remain unchanged. RP03 (#155) is
the historical GRU reference. Repeat task-only GRU on fresh paired roots if
making any architecture claim. RP03 failed its sparse transport admission;
RP04 found donor contamination, large span residual and a carry contribution.
See RP04 RESULTS.md for the matched-dose 31.05% vs 22.36% target result and
16.68% vs 22.45% collateral corruption.

## Implemented instruments and limits

- reference.py: exhaustive directed Hamming-one catalogue of the fixed
  modulo-five partitions, zero-collateral linear frontier in a physical
  support, joint single/pair beam expansion and first-order update gain.
  The reference is float64, not interval certified. Five local reference
  tests passed before publication; native model and data tests are supplied
  but have not yet been run against a full BPD checkout.
- data.py: explicitly materializes admitted semantic edges under RP03's
  operation semantics and root-specific input encodings, with one noise,
  distractor and write-order draw for all four queries. Donor equals true
  hybrid; avoid counting duplicate ordinary examples as extra information.
- model.py: one native TiedComponentLinear and shared residual block
  across low visits and high commits. Token streaming with joint
  [z_H,z_L] cut state; no rereading past input tokens, no oracle features.
  Width 32, rank 29, one low visit per token gives 5,380 trainable parameters
  by construction, matching RP03's parameter count, but has 64 carried
  coordinates and twice as many block visits as GRU steps. This is a
  vector TRM adaptation, not the LOOPSCOPE attention TRM. The latter
  is a separate prospective replication using LoopedTransformers #402.
- No optimizer/curriculum runner, cache, independent native replay or hard
  cap launcher is included yet. Run the native tests before any pilot.

## Semantic curation

Train has 153 tuples, 912 directed edges, 228 per slot, 48 ordered
slot/value-transition types, one connected component. Evaluation has 103
tuples, 312 directed edges, 78 per slot, 24 transition types, components
of sizes 5, 14, 84. This creates a transition-support shift; report it.
Use only TRAIN graph edges for discovery, calibration, feedback or caching.
Choose augmentation seeds by domain hash and separate TRAIN fit, TRAIN
calibration, TRAIN scheduler-tune purposes. Group both orientations of an
edge when splitting fitting from calibration. Shared vertices are allowed
and explicitly recorded; all historical evaluation and sealed receipts
stay untouched. Final fresh evaluation draws only after freeze.

A catalogue entry is root, edge, augmentation, delay, checkpoint digest,
physical module, recipient visit, observer definition and factor-frame digest.
Record all four query results, target and protected centered-logit margins,
ordinary error, edge-specific corruption/recovery, full-hybrid natural
displacement norm, projected residual, signed readout response and nonlinear
prediction error. A candidate is an intervention on a genuine earlier
state, not a future-label feature.

## Phase A: competence and matched architecture

Preselect engineering roots distinct from RP02-04 and previous MMS cohorts.
Use ordinary CE, same training streams and root encodings, delays 1/2/3.
Record full warmup curve (RP03 used 6000 updates); predeclare fixed selection
rule with delay 2 and 7 competence and all-distinct binding. Run GRU and
TRM with uniform data; report fixed-update and separately fixed total CPU
comparisons, parameters, carried coordinates, block visits and measured
runtime. Also compare a width-16 state-size-matched TRM descriptively.
RP03 delay 31 and dose 0.75 are common reference points; measure headroom
and dose grid 0.25/0.75/1.5/3 in new engineering roots. A failed competence
or causal admission gate stops progression; retain every failure.

## Phase B: discover selective control

At the post-write cut form a joint state s=(z_H,z_L). For fixed shared
suffixes obtain target-margin row a and protected-margin Jacobian C by
differentiating the actual continuation, including recurrent carry.
Use all three unaffected addresses (all centered output differences) and
declared intermediate delay readouts with identical query policy; avoid
silent substitution of a terminal-only observer. Source/recipient timing
must follow actual visits. Use distinct high, low and joint physical sites.

For a physical candidate basis B, orthonormal Q spans B. Let Z span
ker(C Q). The exact-arithmetic zero-collateral scalar optimum is
eps * ||Z^T Q^T a||, achieved in direction Q Z Z^T Q^T a normalized to
dose when nonzero. Numerical rank and residual must be reported. For
positive collateral budget use BPD SCCG-01 #168's support frontier;
evaluate actual finite-dose outcomes. An empirical Hessian estimate is
a diagnostic, never a uniform Taylor certificate. Rank-zero/zero-gain
cases abstain and stay in denominators.

Discovery candidate bases: RP03 contiguous V groups, canonicalized native
V or physical U write spans at their actual sites, TRAIN-only hybrid
difference SVD, and joint pair expansions with bounded beam, rank 2/4/8.
Actual state and block-write edits use different physical spaces and must
not be conflated; compare each to its own same-site controls. The high/low
joint state basis is derived in its joint coordinates, not by labeling a
single-block V column as a 64-dimensional state actuator. Apply genuine
native write interventions separately with full factor reconstruction and
equal physical displacement. Reference joint search is a bounded proposal
heuristic, not a global sparse optimum. SCCG-01 shows greedy can miss
cancelling pairs even with perfect local attribution.

For each candidate compare matched same-site random direction, signed
permutation, rank-matched SVD, equal-dose full hybrid and unmatched natural
full-state replacement. Also match the signed full observer response and
physical dose when feasible (MOLT-06 #403); log infeasibility and retain
the full denominator. The stronger sham tests effects beyond that linear
observer. Primary selective admission: target improvement over random
with collateral corruption <= 10% of initially correct protected queries,
plus improvement over RP03-style donor span at equal dose; tune numerical
thresholds only on engineering roots. Do not promote from passive probes.

## Phase C: mechanistic data hill-climb (runner to build)

Maintain an exact edge table. Sample uniform coverage by slot, value
transition, delay and nuisance class. Reserve 20% uniform exploration;
cap each edge at 4 presentations per scheduling window and keep original
sampling distribution as a separate diagnostic. Use only past feedback.
Candidate utility is improvement in TRAIN-calibration objective per
charged CPU time. Compute predicted gain -g_cal dot delta_theta for
candidate updates, using actual optimizer deltas from cloned complete
optimizer/model state when measuring. SGD dot-product estimates with a
shared preconditioner are cheap screening proxies. Report predicted and
realized improvements and rank correlations. A conditional lower bound
subtracts L||delta||^2/2 only if a uniform bound L is justified; otherwise
use actual calibration replay and call it empirical. Retain random and
loss-ranked candidates with identical proposal quotas. Use train-only
separate scheduler tune views, never final evaluation to pick data.

Candidate acceptance: fit on selected edge mini-batches with equal
ordinary labels in every comparable arm; verify measured TRAIN calibration
loss improvement and protected margin/corruption checks. Reject negative
or stale candidates, with rollback including optimizer/RNG. Compare
the complete policy against ordinary no-curation, not just accepted steps.
A loss for causal training uses all four query labels and target hybrid CE
plus protected teacher-relative margin/CE terms. Include donor/hybrid CE
for every arm with deduplicated or matched weights. Hold auxiliary
gradient norm ratio at RP03's 0.25 for first comparison; learn a
receiver-conditioned displacement only on TRAIN. Gate causal auxiliary
on held-in calibration admission and track ordinary steering-off accuracy.

Paired arms: uniform CE, loss-mined CE, mechanistic-curated CE, uniform
causal loss, mechanistic-curated causal loss, shuffled-score control.
Add equal-budget reinvested ordinary training and compute-matched
instrumented observe-only arms. Freeze proposal cap, refresh schedule,
sham controls and all contrast families before new development roots.
Keep fixed-update and fixed CPU budgets separate to avoid MMS-06's
overfitting confound. At least 8 new independent model roots for a
development screen, never treat query rows as independent model roots.
Primary endpoint ordinary delay-31 accuracy with steering disabled.
Report delay-2/7/63/127, all-distinct binding, query-balanced recall,
recovery/corruption and resource endpoints. Require >=5 pp ordinary
gain over uniform CE, positive paired comparison vs loss-mining and
matched controls, preflight power and separate confirmation protocol
before promotion. Existing eight exposed roots cannot calibrate a 95/95
population sham bound.

## Memory policy

Whole campaign process-tree ceiling 6 GiB with 12 GB available; execute
one arm/root worker at a time. On Linux use an actual cgroup v2 limit with
memory.max=6442450944 and swap disabled, charge child CPU from cgroup stats,
record memory.peak/events and wall time. On Windows use an appropriate
Job Object commit-memory cap plus process-tree RSS/host pressure monitoring,
and label different memory semantics. A watchdog alone is not a hard cap.
Fail closed if enforcement is unavailable. Start cache disabled then
4/16/64/256 MiB payload tiers; expand only after observed positive saved
CPU at equal outputs and complete cost accounting. No requirement to fill
6 GiB. Exact graph and feature templates are cheap; cache detached
current-checkpoint observer/suffix maps or measured reusable gradients
only after complete binding and stale rejection. For cross-epoch
historical cotangents use separate approximate policy and parity controls.
MOLT-07/08 use forecast saved CPU and conditional error bounds; charge
metadata, keys, serialization and resident peak, not payload alone.
Use DataGoblin MMS-06 canonical factor maps and FastGradientBank when
integrating #157, with frozen refs and measured actual amortization.
No extra GPU/cloud spend.

## Required next runner and tests

Native data/model tests, independent recurrence and intervention replay,
exact gradient parity, all-query donor checks, signed observer matching,
finite-dose residuals, semantic/seed custody, cache invalidation, full
optimizer rollback, matched work and process-tree accounting. Source lock
before scoring new model roots. Record all aborted runs and failures.

Reference verification:
    python -m unittest mechinterp.bpd.experiments.trm_curation.test_reference -v
    python -m unittest mechinterp.bpd.experiments.trm_curation.test_model -v

Cross-repo references:
- BPD RP03 #155, RP04 #156, native MMS #157, SCCG-01 #168, INK-BPD-06 #161.
- LoopedTransformers #402 (attention TRM), #403 (stacked observer),
  #406 (error fallback), #407 and #408 (native observer/distillation).
- DataGoblin #85 (MMS-06 measured costs), #91/#92 (observer byte allocation).
