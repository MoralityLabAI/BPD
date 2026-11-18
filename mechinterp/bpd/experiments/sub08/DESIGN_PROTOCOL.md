# BPD-SUB-08: prospective full-state source competence

Registered 2026-11-18 before any SUB-08 model training. Research issue: #125.
Base: SUB-07 commit c18d40bd3a8c574080d1d124b2fde6e7ac990c2f.

## Scope and decision

This phase asks whether a fixed six-step tied GRU can reliably learn the entire
final state of the finite transducer. The independent unit is one fresh model
initialization and its minibatch stream. The common dataset is fixed; results
describe training-seed reliability conditional on this task and partition.

The only prospective cohort is roots 1900 through 1907. At least seven of all
eight attempted roots must pass every source gate below for
`SOURCE_COMPETENCE_PASS`. Otherwise the result is `SOURCE_COMPETENCE_HOLD`.
Any missing, runtime-failed, or nonfinite root counts as a failure. Every root
stays in the denominator. There is one registered engineering decision and no
mechanistic hypothesis test or scientific-confirmation population in this phase.

The bounded design pilots are excluded from those eight roots. Their complete
results and the candidate-selection decision will be committed before a
`COMPETENCE_LOCK.json` releases the prospective cohort. No post-release
hyperparameter change, checkpoint choice, root substitution, or successful-root
rerun is allowed. A failed execution is retained and requires a separate protocol
before any replacement study.

## Exact oracle

Initial state is (a,b)=(0,1) over F2. Token IDs 0,1,2,3 denote X,S,C,R:

| Token | State update |
| --- | --- |
| X | (a xor 1,b) |
| S | (b,a) |
| C | (a,b xor a) |
| R | (a xor b,b) |

Execute six token positions left to right. Target class is `2*a6+b6` with
classes 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1). Enumerate all 4096 sequences in
lexicographic order. Exact counts must be [781,1146,1024,1145]. The public
arithmetic certificate and an independently written transition-table verifier
must agree before training. Preserve an identical-token-count order witness.

For any retained feature f, group the exact sequences by f; the maximum number
of correct predictions is the sum of the largest class count in each group.
Report conditional entropy from those same four-class counts. Compute prefix
lengths 0 through 6, last token, and unordered token counts. Population correct
counts must include prefix2=1232, prefix4=1556, last=1188, counts=1603.
These bounds concern missing input information. They do not establish a
computational-depth lower bound for an architecture that sees the entire input.

## Data and exposure

Use NumPy `default_rng(246810).permutation(4096)` with fixed slices:
train[0:2048], validation[2048:2560], reserved_dose[2560:3072],
heldout[3072:4096]. This preserves the published SUB-07 sequence partition.
No sequence occurs in two slices. Labels are a public deterministic oracle.
The unexposed readout is prospective model behavior, rather than secret labels.

Pilots may train only on train and score only train/validation. They must not
evaluate model predictions on heldout or reserved_dose. Validation is explicitly
exposed to architecture selection and cannot support heldout claims. The exact
task certificate may enumerate all labels and split-specific feature bounds.
Reserved_dose is unused by this competence phase.

Every prospective root uses the same partition and scores heldout only after
its fixed final update. Retain all heldout IDs, labels, four logits, prediction,
and prefix2/prefix4 head predictions. The state after t tokens is a deterministic
function of that prefix; exact heldout ceilings therefore bound these diagnostic
predictions too. Do not apply the population ceiling as a hard finite-split cap.
The independently calculated heldout ceilings are prefix2=347/1024,
prefix4=531/1024, last=304/1024, counts=449/1024. Executable enumeration must
reproduce those counts before training.

## Bounded excluded design pilots

Use seeds 1890,1891,1892 for each width48 and width64: six training trajectories.
Record checkpoints at exactly 2400 and 3600 updates from each trajectory, for
four candidate configurations and twelve candidate/seed measurements. Reusing
one trajectory at two checkpoints does not add an independent root. Smoke seed
1889 is reserved for at most sixteen updates and is always excluded.

All candidates use a PyTorch GRUCell(input_size=4, hidden_size=width), initial
hidden state zero, shared cell parameters at all six positions, and one linear
width-to-four head. Input is a four-way one-hot token. Train final-class cross
entropy only; do not supervise intermediate states or add token-position labels.
Use the default PyTorch parameter initialization after `manual_seed(seed)`.
CPU float32, one torch thread, deterministic algorithms, no augmentation.
Adam: lr=.003, betas=(.9,.999), eps=1e-8, weight_decay=0, amsgrad=False,
foreach=False, fused=False. Batch size128 sampled with replacement using a
dedicated CPU torch.Generator seeded with seed+44. No clipping, learning-rate
schedule, early stopping, retries, or checkpoint selection within a candidate.

Pin Python3.12, torch2.5.1+cpu and NumPy2.3.5 for both phases; record the exact
Python patch, CPU/platform and numerical versions in every run. A runtime change
requires an explicit amendment before fresh model training.

A candidate passes the design screen only if all three excluded pilot seeds
have validation exact-state accuracy >=.95 and each final-state-class recall
>=.90. Select the passing candidate with lowest
`updates * (3*width**2 + 22*width + 4)`; break ties by smaller width then fewer
updates. The fixed ordering is (48,2400), (48,3600), (64,2400), (64,3600).
Save all candidate scores, including failed configurations. If none passes,
return `DESIGN_HOLD` and do not release roots1900..1907. Further architecture
search requires a separately registered design study and fresh excluded seeds.

## Prospective source gates

For each of roots1900..1907, train the one frozen candidate from scratch for
the exact locked update count. Evaluate each checkpoint once after training.
The root passes only when all of the following hold:

* Heldout exact-state accuracy >=.90.
* Heldout recall for each of classes0,1,2,3 >=.85; all four classes present.
* Finite weights and logits, exact requested update count, correct split hashes,
  and a complete per-sequence receipt.
* Full six-token accuracy exceeds the exact heldout unordered-count ceiling
  by at least .40 and the prefix4 ceiling by at least .35.
* Prefix2 and prefix4 head accuracy cannot exceed their exact split-specific
  feature ceilings; prefix-consistent logits are verified to tolerance1e-6.
* Full-width rollout and an independently iterated cell/head continuation agree
  to maximum absolute logit error1e-6. This checks the task implementation only.

Report train/validation/heldout accuracy and cross entropy, per-class confusion
matrix and recall, per-last-token recall as a diagnostic, prefix-head accuracy,
all exact feature ceilings and full-order margins. Save each checkpoint and its
SHA256, initialization-state SHA256, final training loss, update trace every200
updates, training IDs hash, model parameter count and source hashes.

Do not fit subspaces, differentiate source Jacobians, evaluate wrong clocks,
apply hidden-state impulses, or score repair/corruption in this phase. A source
pass allows the design of a later causal study, whose model roots and complete
intervention protocol must be registered separately.

## Independent audit and uncertainty

An auditor that never trains a model must reconstruct the transducer using
literal transition tables, verify complete unique sequence IDs and targets,
recompute accuracy/confusion/recall/margins from saved logits, check finite
values, checkpoint hashes and source locks, and reconstruct the eight-root
decision. It must validate every registered root, including failures. Replaying
saved checkpoints for parity is read-only and cannot change the original score.

Report a two-sided exact95% binomial interval for the fraction of model roots
passing the gate, conditional on this fixed training setup. This interval does
not certify that population reliability is at least7/8. No token, sequence,
checkpoint, class or alternative readout is counted as an independent model.

## EvalMath and scientific boundary

Use EvalMath skill and preflight.py pinned at
MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a.
Before SUB-08 training, rerun strict preflight on the actual archived SUB-07
development pilot and matched source-fitted wrong clocks. Verify original
artifact ZIP SHA256 063fdbad7c9e592ac10f68ce944ac2e7f6affe31691e3bc8b997727574d42598
and raw original_eight.json SHA256
cc0d465aa5c75825180e5674f986f86ed8af8987ca12ec9c787d36e8e0ebb930.
That inherited result remains REDESIGN_OR_PILOT_ONLY. It provides provenance for
the redesign and cannot measure a SUB-08 causal effect.

SUB-08 has zero measured causal units, no physical intervention dose, no
causal effect floor, no matched-sham floor and no causal power estimate.
These fields remain null. Task accuracy must not be inserted as a causal
effect or paired with fabricated zero shams to make EvalMath accept a file.
Evidence classes: exact finite domain (oracle) and prospective observational
source competence (trained models). Scientific status remains
`CAUSAL_STUDY_NOT_RUN` regardless of the engineering result. All existing
ARG, AMM, CQ and SUB-01..07 protected data and confirmation seals remain intact.

## Required receipts and release order

Commit this protocol first, then exact oracle/source/tests/auditor and pinned
EvalMath input/output. Run and archive the excluded pilot grid only after the
oracle and inherited preflight checks. Commit complete pilot scores plus the
deterministic candidate decision. Freeze source-file and protocol hashes,
runtime, selected candidate, partition hashes and roots in COMPETENCE_LOCK.json
in a separate release commit before any prospective training. The runner must
refuse an existing output directory and any seed outside the appropriate roster.
Commit original scores, audit, provenance and the final report; preserve
checkpoints and raw rows in a durable artifact. Any source-only CI may run
oracle/auditor tests and the excluded smoke only. It must never auto-run the
prospective cohort on documentation changes.
