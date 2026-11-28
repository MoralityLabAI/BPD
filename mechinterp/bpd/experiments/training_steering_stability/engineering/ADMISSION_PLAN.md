# RP03 TRAIN-only admission of delay and hidden-state dose

This plan precedes the admission measurements. It uses the two exposed
engineering roots, 27000 and 27001, at their saved 6,000-update checkpoints.
The completed 2,000-update study and its separately planned continuation stay
unchanged. The planning record under `../planning/` remains
`REDESIGN_OR_PILOT_ONLY`; this exercise selects a measurement recipe before
fresh development work. It provides no scientific confirmation.

## Fixed delay grid

Each saved model receives fresh TRAIN-partition ordinary cohorts of 512 pairs
at delays **31, 63 and 127**. Ordinary accuracy uses receiver sequences with
steering disabled. Each model also receives a fresh address-binding cohort of
256 cases at every delay: 64 admitted all-distinct memories expanded to all
four late query addresses. The query change preserves each base sequence's
write order, distractors and noise. All memories retain the TRAIN residues;
the query-blind modal-value baseline is exactly 25% in the binding cohorts.

Select the shortest delay for which **both** roots have ordinary receiver
accuracy in **[0.55, 0.90]**, inclusive. If every candidate fails, select 127
and record `HEADROOM_GATE_FAILED`. Binding results describe address memory and
do not alter this selection rule. Retain every delay endpoint.

## Fixed hidden-state dose grid

At the selected delay, evaluate doses **0.75, 1.5 and 3.0**. At each dose and
root, discover four slot-specific groups from the same eight contiguous
groups of four V columns. Use 512 fresh TRAIN pairs at delay 2. For each
candidate group, form the detached orthogonal projector onto its V-column
span, project donor-minus-recipient prefix state, and normalize the resulting
hidden-state delta to the proposed L2 dose. Apply it before token 4, so it
enters the recurrent operator and the GRU carry. Select minimum mean hybrid
cross-entropy separately by edited slot, using the RP03 discovery criterion.
All cases for the edited slot enter its loss, including unaffected queries.
Candidate order breaks ties and groups may overlap. Save all eight-by-four
loss and targeted-accuracy matrices.

One fresh 256-pair TRAIN causal cohort per root is shared across doses.
Calibration pairs also remain shared across doses. The comparison retains:

- the selected fixed hidden-state projector edit;
- the same delta under one root-specific signed coordinate permutation,
  preserving its physical L2 norm;
- full replacement by the true hybrid prefix state, with its native hybrid
  continuation checked for equality.

The full-state control uses privileged hybrid input and its natural radius.
Report its actual dose separately. It is a structural continuation ceiling.

## Dose selection and measurements

Pool targeted correctness counts across the two roots for the fixed projector
arm. Its untargeted original-answer corruption rate divides corruptions by
the count of originally correct unaffected cases. Also retain corruption
counts over all unaffected cases, recoveries, complete predictions, and both
original and hybrid labels.

Choose the smallest dose with pooled targeted accuracy **at least 0.25** and
pooled untargeted original-answer corruption rate **at most 0.20**. If no dose
passes, choose the largest value of targeted accuracy minus that corruption
rate, breaking ties toward the lower dose, and record
`TRANSPORT_GATE_FAILED`. If a cohort has zero originally correct unaffected
cases, its pooled corruption rate is conservatively 1 for this rule. Sham
and oracle outcomes do not choose the dose. Retain the full dose grid.

Record per-case recipient, donor and edited-state radii; raw and delivered
delta norms; validity masks; actual incoming-state displacement; prior-state
equality; whole-head changes; and terminal-state distances. Each matched sham
must preserve the corresponding intervention norm. The original checkpoint
state and the new class's unpatched ordinary computation must agree.

## Execution and exposure

No optimizer updates occur. The model parameters, native BPD factor path,
input/output dimensions and root encodings come from the saved RP02 recipe.
All generated receiver, donor and hybrid memories use the TRAIN partition.
Seeds use SHA-256 of
`BPD-RP-03:admission:{root}:{domain}:{index}:{delay}` with distinct purposes for
ordinary, binding-candidate, calibration, causal and rotation cohorts.
No eval-partition examples or additional candidate delays or doses enter this
run. Every choice here uses the same two previously exposed roots.

Before execution, save `admission.py` and a manifest containing this plan,
executed source hashes, checkpoint hashes and environment versions. Refuse to
overwrite an existing result directory. Save complete delay/dose outcomes,
geometry, source checks, a machine-readable `selection.json`, and a compact
report. Verify predictions and selection from the saved checkpoints after the
run. The bounded grid and stated fallback rules finish this admission task.
