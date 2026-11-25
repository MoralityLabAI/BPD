# BPD-NBR-02: transported-risk gates in the CQ-06 nonlinear recurrence

Engineering/development registration, 2026-11-25. Committed before
implementation or new model-root generation. This extends BPD-NBR-01
(PR #147) and the merged CQ-06 (PR #45); their original gates, roots
and verdicts remain unchanged. No sealed ARG-03-N or CQ confirmation
data are read.

## Source and budget

Use unchanged CQ-06 experiment.py Git blob
6f9e03268ea5c3d8a706ee547b8e76da296e86e7 on main, including the 12D
tanh teacher, rank-four tied BPD factors, SVD canonicalization,
teacher 850 steps, factor 600 steps, feedback gate 500 steps.
Fresh engineering roots 202611250201 through 202611250208 inclusive;
disjoint from CQ-06 development, sham and confirmation seeds.
Root 202611250200 is excluded from all results for derivative/parity
tests. Python 3.12, Torch 2.5.1 CPU, NumPy 1.26.4, one Torch thread.
256 paired-z task cases/root (CQ-06 source offset 810000) and 128
donor pairs/root (CQ-06 donor_inputs). Independent unit = retrained root.
Record all roots, including poor source competence, without replacement.
Mark teacher/full BPD source competence at >=90% two-readout correctness;
no scientific success threshold in this development pilot.

All selected masks keep exactly two of four second-pass components.
First-pass tests keep two of four first-pass components and run the
second pass fully. Compare at equal cardinality, not equal FLOPs or
physical edit norm. Every masked outcome uses actual
BPDModel/TiedComponentLinear through CQ-06 NativeModel. Independently
compute the same states and check parity. Analytic scores require
counterfactual full preactivations and readout derivatives; these are
diagnostic model-access policies, not demonstrated efficient inference.

## A. First-pass delayed-observable mask

Let x be the reached-state/cue row vector, a=x V, w_i=a_i U_i, p=sum_i
w_i+b, s=tanh(p). Current logit = s c^T, with c=teacher.head.weight.
Write-coordinate derivative g0=c*(1-s^2). For a hypothetical second
cue q, s2=tanh([s,q] V U+b), g2=c*(1-s2^2), and the first-preactivation
gradient for the delayed readout is g1(q)=(g2 W_state^T)*(1-s^2),
where W_state=(V U)[:12,:]. The local observable signature of component
i is [w_i dot g0, w_i dot g1(q_j)/sqrt(8)] for j=0..7.

Fix eight future-cue probes using CQ-06 source(seed,8,922000,paired_z=True);
keep only its second cues. This uses the known source distribution
and separate calibration draws, never the actual held-out second cue
or answer. Do not tune probes or choose by held-out outcomes.
Current-only signature is w_i dot g0. First-pass selection must give
identical masks to the two examples sharing a first cue; assert this
for every arm (random mask is sampled once per pair).

Arms: physical diagonal, current-only diagonal, delayed observable
diagonal, delayed observable greedy, delayed joint Taylor oracle
(exhaustive six k=2 masks), uniform random. Rank by squared vector
norm; greedy adds the component minimizing current omitted summed
signature norm; joint enumerates all six fixed-cardinality masks.
Lowest coordinate index breaks ranking ties, fixed PATTERNS order
breaks exact mask ties. No target labels enter policy scores.
Score actual first and second logits after native masked first pass
and full second pass on the actual cue. Compare to full BPD logits
and independent task truth. Record squared current/actual future
logit error, mean two-readout error, both-correct and per-readout
accuracy; report per-root paired contrasts. Hypothetical-probe
Taylor risk is a proxy, not the observed future error.

## B. Terminal feedback gate

First pass full. Given actual reached first state and actual second
cue, compute second preactivation component writes and terminal
readout derivative as above. Compare physical diagonal, terminal
observable diagonal, observable greedy, joint Taylor oracle, trained
CQ-06 feedback scores top two, uniform random; also the exact finite
logit-fidelity oracle over the six second-pass masks (diagnostic
ceiling, not a cheap baseline). Every mask is scored on the same
full-first state. Score exact native output error relative to
full BPD and task truth. The full reference is not available free;
count information access (full preactivation, derivative, probes,
candidate forwards); no efficiency claim.

## C. Feedback intervention

Use CQ-06 donor_inputs to generate independently defined recipient
and donor labels, which agree on first parity and oppose on delayed
parity. Construct the exact existing CQ-06 partial first-pass
component-(0,1) donor preactivation edit, tanh it and subtract the
recipient state. Project onto the current readout kernel and normalize
to the original raw edit's physical L2 dose. Compare against a
seeded isotropic projected norm-matched direction and zero sham.
Include the genuine reached full-donor state as a separate
reachability control (unmatched dose). Do not call projected states
on-manifold; record coordinates outside [-1,1] and current logit drift.
Zero projected/raw norms: zero patch, flag and retain the pair.
No editing of teacher or BPD weights.

For each policy from B, compare recomputed versus frozen second-pass
mask (the mask selected at clean recipient), both with exactly k=2.
Apply each pair of masks to the identical patched reached state and cue.
Record donor-target hit, actual output error relative to the full
BPD continuation on that same patched state, mask switches,
same-dose and dark errors, and native versus independent parity.
Do not select donor patch direction based on the policy or labels.
Random mask is fixed across feedback/frozen and patch arms. Sham
must give identical recomputed and frozen masks and outputs.
These patch outcomes are conditional on the specified interventions,
not evidence of discovering naturally reachable causal variables.

## Evidence, audit and continuation

Engineering gate: derivative checks on excluded root, native versus
independent max absolute state/logit error <= 1e-5 for all scored
masks, exact cardinality, first-pair mask invariance, finite outputs,
zero sham, projected equal-dose/dark tolerance <=1e-5 (nondegenerate
rows); full finite oracle no worse than every terminal mask in
per-example squared full-logit fidelity within 1e-5 absolute tolerance.
No post-hoc exclusions or replacement. Save raw per-case masks,
logits, proxy risks and targets, root parameter hashes and source
provenance. Independent arithmetic auditor recomputes all task
and donor outcomes, matched cardinality, oracle ordering, and root
summaries from saved arrays; does not retrain or reexecute models.
Keep all source competence failures and engineering errors.

This is a new development pilot, not a confirmatory EvalMath study.
Primary descriptive contrasts: first-pass delayed joint/greedy vs
physical diagonal on actual delayed squared logit error; terminal
joint/greedy vs physical on native logit error; donor recomputed
vs frozen joint and greedy on target hits and fidelity to reached
full continuation. Report every comparison and root. No scientific
p-values. A prospective continuation requires a separately locked
59+ independent sham/calibration population and fresh confirmation
roots, derivative-access matched policy and numerical tolerance,
preregistered minimum effect and power plan, and EvalMath gate.
