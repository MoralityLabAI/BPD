# BPD-NBR-01: observable-risk sparse masks

Source-only bridge from NBR-01/01B and NBR-02A in LoopedTransformers.
Prospective synthetic arithmetic test: this protocol is committed before
the new generated worlds are scored. The external NBR results are already
observed and used only for motivation. No BPD checkpoint, sealed dataset,
learned controller, or native causal result is used here.

## External evidence

LoopedTransformers PR #355, registered run 45519229947, found temporal
native minus shuffled-ID cosine +0.747095 / +0.814789 on 48 eval / 48 rep
boards, while residualized native minus fixed-mixture cosine was
-0.006790 / -0.010723 (registered FAIL for basis specificity).
PR #356, exploratory run 45522596410, artifact 14980502397, used eight
orthogonal coordinate rotations preserving the complete write and
its local transported derivative. At ACT3 L0, 64 physical-ranked
coordinates, native vs rotated bounded physical loss was
0.113398 / 0.356810 (eval) and 0.109747 / 0.358471 (rep).
Transported loss was 0.589493 / 0.405501 (eval) and
0.637443 / 0.422559 (rep). Max full parity error 4.48e-15.
These are first-order reconstruction metrics, not finite intervention
or learned gate results. Calibration/Eval were 48/48 exact at ACT16,
rep 46/48. ACT3 median-site wrong counts 0/2/3 in cal/eval/rep;
a conditional recovery detector is too class-limited there.

## BPD extension

See THEORY.md for the finite-horizon error Gramian, exact counterexamples,
cost-aware mask risk and the local nonlinear approximation boundary.
This extends BPD-CQ-01 without altering its exact equivalence or common
centralizer theorem. BPD-OUT06's posterior utility is a distinct
decision layer; a small observable squared error does not certify
task-level utility or calibrate class probabilities.

## Frozen arithmetic gym

Generate 128 independent deterministic synthetic roots, each an 8D
physical state, native signed-permutation write dictionary U, signed
integer activation a, rank-at-most-two integer readout C, signed
permutation recurrence T. Every root has two large native coefficients
(4..7 magnitude) and six small coefficients (0..2 magnitude), with
random signs. T and U each have independent seeded permutation and signs.
C has two rows, with entries drawn from -3..3 inclusive.
Use visits 0,1,2 (unit weights). Use a source-independent fixed orthogonal
Walsh rotation on each consecutive four-coordinate block; Q entries
are +/-1/2 within blocks. Transform (a,U) jointly so the full physical
write and all three observed outputs are unchanged. Each basis gets
the same exact cardinality k=2, no dimension or compute savings claim.
All calculations of writes, scores, mask risks and ordering use exact
rational arithmetic, lexicographic tie breaking. Use Python random.Random
seed 20261125061, 128 roots in order; 0..127. No seeds replaced.

Within each basis compare:
- physical diagonal: keep two largest |a_i|^2 ||u_i||^2;
- observable diagonal: keep two largest |a_i|^2 sum_h ||C T^h u_i||^2;
- observable greedy: add one coordinate at a time minimizing the exact
  current omitted observable risk; final k=2;
- exact physical oracle and exact observable oracle (all 28 masks);
- uniform random mask expectation (all 28 masks, not Monte Carlo).

Report exact full-write / horizon-output parity, exact direct-risk /
Gramian-risk parity for every enumerated mask; mean bounded physical
and observable losses E/(||full||^2+E), defined as 0 when both vanish;
raw mean risk; exact per-root wins/ties/losses and mask disagreement.
Summaries by basis and by horizon (present-only is a secondary
diagnostic for the same selected masks). All outcomes retained.
This is a constructed mechanism stress test: the oracle is a lower
bound by construction and is not a trainable baseline. No inferential
p-values or neural-generalization claim. The preregistered engineering
gate is exact parity and oracle dominance for every root and mask.
Scientific claims about native BPD require a separate experiment.

## Follow-up gate (proposal, not registered)

Reuse BPD-CQ-05/06/08 source-competent teachers and their true physical
component writes. On fresh independent roots, fit a horizon-aware
selection policy on calibration-only trajectories using a frozen
budget and matched local derivative-access cost. Compare against
physical magnitude, diagonal observable and equal-cardinality random,
a static present-readout observer, exact small-rank oracle ceiling,
and learned BPD gating. Separate observed prediction versus actual
finite outcome; log future head changes, task success, current-dark
dose, feedback re-computation vs frozen masks, and utility after action
cost. Calibrate sham and prospective root-level thresholds using the
existing EvalMath requirements; do not touch existing sealed data.
Full nonlinear closed-loop effects, out-of-distribution basis recovery
and wall-clock speed remain separate endpoints.
