# BPD-TRM-CURATE-02: train-only curriculum instruments and Neon handoff

Parent: merged PR #169 (c559a482a7026fd43ddb22fbaae935ba44d661e7).
This change adds scheduler, candidate optimizer probe and signed observer.
Local isolated tests: four passed (PyTorch 2.10 CPU, NumPy; linear fixture for
observer/optimizer, not full native TRM). Historical RP03/RP04 frozen files
unchanged. No new native training, hard cap, measured speedup or confirmation.
A dedicated hosted workflow now includes the earlier native data/model tests.

## Split and scheduler
624 fit, 144 feedback, 144 tune directed training edges, each spanning all
48 ordered slot transitions. Both orientations stay together, vertices may
overlap. Freeze split seed and hashes before any run. The entire original
153-tuple training partition remains legal; this tighter edge-view split is
additional and applies to all compared arms. 312 historical evaluation edges
are not read by fit/feedback/tune. A separate freshly seeded final draw is
required. Do not infer unseen-tuple generalization between the training views.

Candidate scores are signed predicted feedback-objective gain / charged
CPU seconds. The scheduler checks edge membership, unique rows, epoch
identity, finite scores and positive cost, caps unordered pair exposure and
uses a deterministic 20% exploration quota. A nonpositive gain falls back
to ordinary random draws from the remaining eligible proposals. Shuffled
scores preserve the gain/cost score multiset. Uniform, loss and mechanistic
arms must receive the same proposal pool and eligibility constraints;
candidate generation should explicitly balance slot/transition/delay.
Return actual chosen count and failures; a cap can prevent a full batch.

## Observer and optimizer probe
observe_edge evaluates four query schedules at declared delays. Shorter
delays are separate counterfactual continuations, not hidden native probes.
All are differentiated w.r.t. the common post-write joint state. Each
address contributes three relative-logit rows, target at longest delay,
three protected addresses at all delays. Keep full signed stack; a
terminal-only sham is weaker. The hybrid finite-dose readout is privileged,
not an autonomous model feature. Measure actual task transport and
protected corruption for every intervention. Rank-zero protected-null
supports stay in denominators; use SCCG positive-budget solver separately.

probe_update measures the actual clipped optimizer step, predicted gain
from the calibration gradient dot parameter delta, and realized loss
change on the same feedback cohort. Always rollback model tensor state,
optimizer state, existing grads, training flags and global CPU RNG
(Torch/NumPy/Python). External RNG generators, BPD mutable call counters,
Python attributes and cache/sampler side effects are outside this helper;
isolate callbacks and test full native replay before using for acceptance.
It measures probe process CPU including snapshot and rollback. Cache
immutable feedback gradients only when model, objective, batch and
optimizer identity are exact. No autograd graphs held in a persistent bank.

## Next steps on local machine
1. Run focused tests and independent recurrence and optimizer rollback
   checks. Verify exact parameter count, native write reconstruction and
   suffix replacement. Implement exact suffix-only continuation with
   pass offset parity; compare gradients/values to the full-rollout oracle.
2. Add sequential CPU runner and enforce whole-tree 6 GiB cap (cgroup
   v2 memory.max=6442450944, swap 0; Windows Job Object alternatives label
   commit limit semantics). No cloud/GPU. Fail closed if cap unavailable.
3. On fresh engineering model roots, compare uniform GRU and TRM on
   shared warmup data, complete competence and delay/dose screens.
   Use fit-only semantic edges across curation arms; same nuisance seed
   domains and all-query weighting. Baseline historical RP03 reproduced
   separately. Stop failed gates and keep receipts.
4. For a competent TRM, first measure gradient-prediction and selective
   control utility on a bounded train-only cohort. Compare ordinary CE
   gradient alignment vs alignment to a feedback objective incorporating
   fixed-dose targeted transport and protected preservation. Freeze the
   objective and calibration split before seeing new outcomes. Distinguish
   privileged hybrid diagnostics from learned intervention directions.
   Match all candidate proposal work. Stratify by delay and transition.
5. Initial scheduler pilot: uniform, loss-ranked, ordinary feedback
   gradient alignment, mechanistic feedback alignment, shuffled scores.
   Equal updates, then separate equal total CPU comparison with ordinary
   compute reinvestment. Start tiny (16 proposals/window, 4 selected,
   20% exploration, 4 presentations/unordered edge/window); profile and
   freeze before fresh development. Do not claim a five-point gain from
   engineering tuning. Only add the causal auxiliary loss in a second
   crossed campaign after actual control admission.
6. Use detached current-checkpoint observer maps at 0/4/16/64/256 MiB
   payload tiers, exact cache equivalence, time/byte receipts, separate
   approximate reuse policy only after validation. Stop growing after
   break-even. Preserve CPU budget parity including candidate probes,
   cache refresh, serialization, rejected proposals and verification.
7. At least eight independent fresh model roots for development with
   source-lock and separate RNG domains. Delay-31 ordinary steering-off
   accuracy primary; report other delays, all-distinct binding, target
   recovery, collateral errors, candidate prediction correlations,
   chosen-edge distribution, root-paired comparisons and resource curves.
   No confirmation or promotion without EvalMath preflight and a separate
   registered confirmation cohort.

Full runner, native observer parity test, controlled finite-dose selection,
real cache and enforceable memory launcher remain Neon implementation
tasks; the source here is an instrument, not completed campaign.
