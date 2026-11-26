# BPD-RP-03: fixed and live steering subspaces during training

The selected recipe uses **6,000 warmup updates**, **delay 31** for its long
readout and **hidden-state L2 dose 0.75**. Engineering passed the ordinary
competence and long-delay headroom checks. Sparse transport failed its admission
criterion and retains **`TRANSPORT_GATE_FAILED`**. This experiment tests how
training changes that initially weak intervention policy. Its evidence class
remains **DEVELOPMENT_ONLY**.

## Question and independent unit

Does a causal loss through a fixed hidden-state subspace improve ordinary
delayed-memory answers after training? B minus A measures its task benefit.
B minus C tests the direction of that training signal. B minus D compares the
fixed subspace with one recomputed from the same component indices as the model
trains. These are one three-contrast family on ordinary long-delay accuracy.

A root contains a fresh initialization, an input encoding and a training stream.
Its four descendants share the common warmup and form paired treatments.
Examples, diagnostic checkpoints and treatment descendants remain repeated
measurements within that root. The development roster contains eight roots,
27101 through 27108. All eight belong in the analysis.

## Engineering exposure and admission

RP02 ended near 50% ordinary accuracy with weak targeted transport. The
[exposed checkpoint diagnostic](diagnostics/rp02_transport_receipt.json)
motivated moving the intervention to the hidden state before a recurrent step.
This site changes both the carry and the recurrent write. The full-hybrid-state
control reproduced native hybrid continuation in that diagnostic. Its natural
displacement and privileged hybrid input make it a structural ceiling.

The [initial engineering plan](engineering/PLAN.md) used roots 27000 and 27001
for task-only learning curves. Their [2,000-update results](engineering/RESULTS_2000.md)
showed incomplete address binding. The [extension plan](engineering/EXTENSION_PLAN.md)
then prescribed continued training to 6,000 updates, with the original optimizer
state and fixed checkpoints at 3,000 and 6,000. The earliest candidate had to
reach 80% ordinary accuracy and 75% all-distinct address-binding accuracy on
both exposed roots at both training-partition delays 2 and 7.

The [selection receipt](engineering/extension_results/selection.json) chooses
**6,000 updates**, with `ENGINEERING_COMPETENCE_GATE_PASSED`. The 3,000-update
candidate failed on root 27000. At 6,000, both roots reached 99.61% ordinary
accuracy at delay 7, which exposed a ceiling for the intended five-point gain.
The [admission plan](engineering/ADMISSION_PLAN.md) therefore fixes a bounded
search for a longer delay and a usable state-edit dose using these same exposed training roots.
Every candidate and failure must remain in its receipt.

| Final recipe item | Selected value |
|---|---|
| Common warmup | 6,000 task-only updates |
| Long delay | 31 distractor steps |
| Hidden-state L2 dose | 0.75, selected by the prespecified fallback |
| Evidence from engineering | Exposed training data on two roots |

The delay rule chooses the shortest candidate among 31, 63 and 127 for which
both roots have ordinary accuracy between 55% and 90%. The dose candidates are
0.75, 1.5 and 3. At the chosen delay, select the smallest dose with pooled
targeted hybrid accuracy of at least 25% and corruption of at most 20% of
originally correct unaffected answers. If every delay fails, use 127 with
`HEADROOM_GATE_FAILED`. If every dose fails, maximize targeted accuracy minus
that corruption rate, break ties toward the smaller dose and retain
`TRANSPORT_GATE_FAILED`. A pool with zero originally correct unaffected answers
receives corruption rate 1 in this rule. Binding, rotated-control and oracle
results remain descriptive. Each dose gets its own discovery on the same
512 calibration pairs; the causal cohort is also shared across doses. Any
fallback and failed admission gate must be stated with the final receipt.
The [admission receipt](engineering/admission_results/selection.json) selects
delay 31, with ordinary TRAIN accuracies 75.78% and 82.81%. At dose 0.75, pooled
targeted transport is 46/256 (17.97%) and 49/202 originally correct unaffected
answers are corrupted (24.26%). Doses 1.5 and 3 increase targeted transport to
25.78% and 41.02%, while corruption rises to 40.10% and 63.37%. All doses fail
the joint gate. Dose 0.75 maximizes the prescribed fallback score, and
`TRANSPORT_GATE_FAILED` remains part of the experiment's scope. Full hybrid-state
replacement reaches 205/256 targeted answers (80.08%) in this exposed readout.
These engineering estimates select a development recipe and provide no held-out
result. The sparse groups have yet to demonstrate reliable semantic transport.

RP02 and RP03 change warmup, intervention site and other recipe choices
together. The paired comparisons within RP03 support the experiment's causal
interpretation. Historical protected populations remain untouched.

## Task, labels and evaluation barrier

Four addressed slots each hold one of four values. Every sequence writes each
slot once, passes distractors carrying an explicit NOOP type and then queries
one slot. Training delays cycle through 1, 2 and 3. The input has exact
operation-type bits and Gaussian noise of standard deviation 0.03 on the
address and value channels. Each root fixes independent permutations of those
input encodings. The prediction head uses semantic value labels.

Base-four encodings of memory tuples define a fixed semantic split. Residues
2, 3 and 4 modulo 5 form the 153 training tuples; residues 0 and 1 form the 103
evaluation tuples. Receiver, donor and hybrid must all belong to their assigned
partition. A hybrid changes the selected receiver slot to the donor's different
value. Half the cases query that edited slot. The rest query an unaffected slot.
The donor can differ in other slots, so a full donor-state replacement can
corrupt otherwise correct answers. Rejection sampling under these constraints
defines the finite task distribution.

Every arm receives receiver, donor and hybrid as ordinary supervised inputs.
Each counterfactual label is therefore available through an ordinary example
in every treatment. The neural forward interface accepts features only.
Discovery and fixed training anchors use the training partition. Final scored
batches are generated after all paired updates finish, using the evaluation
partition and separate seed purposes.

## Architecture and intervention

`ComponentStateMemory` has width 32, a GRU-style recurrence and 5,380 trainable
parameters. Its hidden-to-gate operator executes the repository's native
`TiedComponentLinear`, with one 32-component V/U pair shared across sequence
steps. Component gates remain open and residual delta remains zero. The input
map, output head and V/U factors all train. The stored `W_target` buffer does
not contribute to execution. This recipe directly trains the component model;
the post-hoc `BPDTrainer` and learned importance network are outside its scope.

After warmup, 512 training-only pairs at delay 2 identify a group for each
semantic slot. The candidate family contains eight contiguous groups of four
component indices. Each candidate is tested by a physical hidden-state edit.
Mean hybrid cross-entropy over all cases with a given edited slot chooses that
slot's group. Stable candidate order resolves exact ties. Groups can overlap
across slots. The complete loss matrix, targeted accuracies, selected groups
and invalid-edit counts are retained.

Let V_G contain the selected four V columns. Reduced QR gives an orthonormal
basis Q and projector P = QQ^T. The candidate groups must have full numerical
rank. Rank failure stops the run and remains a recorded failure. Projectors
depend on the span, so column signs and basis changes within that span leave
the intervention unchanged.

For receiver and donor hidden states immediately after all writes, the raw edit
is `(stop_gradient(h_d) - h_r) @ stop_gradient(P)`. A valid edit is normalized
to physical L2 dose 0.75 and added before the first distractor. Both
the recurrent operator and the GRU carry receive the edited state. Raw norms at
or below 1e-9 yield a logged zero edit. The continuation's hybrid-label
cross-entropy supplies the auxiliary loss. The donor state and the projector
are detached; gradients still reach the receiver trajectory and continuation.

## Four paired training arms

| Arm | Applied training gradient |
|---|---|
| A_task | Ordinary receiver, donor and hybrid task gradient |
| B_causal_fixed | Task gradient plus the causal auxiliary gradient through the warmup projector |
| C_rotated_fixed | Task gradient plus an auxiliary gradient under a fixed signed permutation of B's physical edit |
| D_causal_live | Task gradient plus the causal auxiliary gradient through the current V-column projector |

B retains its warmup projector throughout paired training. D recomputes its
projector from the same selected component indices before each update, then
detaches it. All parameters remain trainable in every arm. C uses one
root-seeded coordinate permutation and sign pattern throughout training. The
rotation preserves the physical edit norm for each example.

Each arm starts from the identical warmup model and a fresh AdamW optimizer.
Paired training lasts 240 updates with the same 48 pairs per update. Learning
rate is 0.003, weight decay 0.0001 and global gradient clipping is 1.0. Every
arm executes three ordinary trajectories and all three intervention
trajectories per pair. It computes the task gradient and all three auxiliary
gradients on each update, for four gradient evaluations per arm.

The applied auxiliary gradient has 0.25 times that arm's task-gradient L2 norm
before global clipping and AdamW. A applies the task gradient alone while
computing the same auxiliary evaluations. Degenerate gradients contribute a
logged zero. Realized optimizer updates can differ as models diverge; the
receipts retain their norms, the gradient accounting and physical edit doses.
There is no checkpoint selection or early stopping within the paired stage.

## Fixed anchors and the meaning of stability

Each root has one fixed set of 128 training-partition anchor pairs at delay 31.
Readouts occur at paired steps 0, 80, 160 and 240 for every arm. The
identical step-zero model permits a shared initial readout. These observations
never alter the training recipe or group choices.

The geometry readout records each slot's normalized projector distance,
`||P_current - P_warmup||_F / sqrt(2 * 4)`, together with cosine and distance
between realized fixed and live edits. Transport readouts retain targeted
hybrid accuracy and unaffected-answer recovery and corruption. The signed
permutation, live projector and full-hybrid-state controls receive descriptive
readouts on the same anchors.

Holding a physical projector fixed does not hold its semantic role fixed.
B−D estimates the effect of these two training policies, including their
different learning trajectories. An association between geometric drift and
transport cannot establish that drift mediates a task-accuracy effect. Role
stability requires the joint geometric and causal diagnostics, with that
limitation retained in reporting.

## Evaluation and statistical family

Each arm receives 512 ordinary examples at delay 2 and 512 at delay 31.
Steering is disabled on all ordinary examples. Ordinary long-delay
accuracy is the primary endpoint. Common inputs, label order and batch hashes
must match across all four treatments.

An additional 256 causal pairs use evaluation tuples at the long delay.
The common fixed projector supplies the main descriptive causal readout in
every arm, which makes the intervention policy comparable across treatments.
Current-projector and rotated edits are also scored. Targeted queries measure
transport toward the edited memory value; unaffected queries measure answer
preservation. Recoveries and corruptions reference the same arm's ordinary
answer on those cases. Full-hybrid-state replacement is a privileged positive
control with an unmatched natural dose. Its logits should match native hybrid
continuation because the suffixes coincide.

The address-binding readout uses 64 all-distinct evaluation memories, each
queried at all four addresses, for 256 ordinary long-delay cases per arm.
A predictor that ignores the query and chooses a single value for each memory
has exactly 25% accuracy on this balanced diagnostic. The four queries remain
repeated measurements of their base sequence and model root.

For ordinary long-delay accuracy, compute equally weighted root means and
three paired contrasts: B−A, B−C and B−D. Report the per-root effects and paired
recovery/corruption counts for each contrast. The exact two-sided sign-flip
calculation enumerates all root signs under signed exchangeability; Holm
adjustment covers the three-contrast family. The descriptive root bootstrap
uses 10,000 draws and seed 20261126. The scientific reference floor is an
absolute five-percentage-point gain. A ten-point target belongs only to the
planning calculation.

The development experiment has eight independent root units and 4,096 repeated
ordinary examples per endpoint per arm. IID accuracy, causal transport,
address binding and drift diagnostics receive descriptive summaries. The
small root count limits uncertainty estimates and sham-tail calibration.
All outcomes retain `DEVELOPMENT_ONLY`, including positive nominal tests or
effects above the reference floor. A positive B−C alone can result from harm
under C, so task benefit requires the B−A comparison.

## Execution, provenance and independent replay

Smoke roots 27010 and 27011 run the selected recipe. Their root-level B−A
effects and C−A sham variation enter the source-pinned EvalMath preflight.
Both engineering roots and both smoke roots remain exposed development work.
No empirical 95/95 population sham bound is inferred from them.

After smoke, publish the complete source and recipe before running roots
27101 through 27108. `SOURCE_FREEZE.json` binds the remote commit, source
hashes, recipe and roster. Development execution verifies that lock and
requires a fresh output directory. All assigned roots and failures stay in the
record. Any recipe change after freeze requires a visible amendment and a new
prospective roster.

Retain the warmup state, selected geometry, four final model states, per-step
training receipts, final predictions and labels, and fixed-anchor outputs.
The independent audit reimplements the recurrence and physical interventions
to replay saved-checkpoint outputs and aggregate scores, and checks source
hashes and training-accounting receipts. It does not establish a replay of the
complete optimizer trajectory. Audit outcomes and any numerical differences
must be reported with the results.
