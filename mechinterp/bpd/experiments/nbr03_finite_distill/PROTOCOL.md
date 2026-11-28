# BPD-NBR-03: finite-risk distillation after local Taylor failure

Development registration, committed before implementation and fresh roots.
NBR-02 (PR #148, run 45542172466, artifact 14989557600) passed all
engineering checks, but every local derivative policy lost to physical
top-k on delayed logit MSE in all eight roots, for both pass tests.
Terminal finite oracle improved MSE 8.56505 -> 6.27775 and accuracy
2042/2048 -> 2048/2048. Preserve that negative result. This is a separate
development experiment, not a revision of NBR-02 or confirmatory inference.

## Source and fixed budget

Use unchanged CQ-06 teacher/factor/gate source blob
6f9e03268ea5c3d8a706ee547b8e76da296e86e7, unchanged NBR-02 implementation
at 109441b1aa7e4bcf8bc6921472edda23d4a8c68d. Fresh roots
202611250301..202611250308; excluded toy root 202611250300.
Each trains teacher 850, factors 600, existing CQ06 gate 500 updates.
Canonicalized rank-four factors, CPU Torch 2.5.1, Python 3.12, one thread.
For each root train two separate six-mask classifiers on 2048 new
source cases from offset 931000, paired-z. Eight hypothetical future
cues from offset 932000 paired-z (same probes for train and test, no
actual second-cue access at first-pass gate). Test 256 paired-z task
cases from offset 810000 and 128 genuine donor states with CQ06
donor_inputs. Seeds disjoint from NBR-02 and original CQ06 registered roots.
All roots and failures retained. First pass and second pass evaluated
separately with exactly two of four selected components and the other
pass fully active. Report cost counts; no runtime savings claim.

## Policies

Physical top-k, local joint Taylor (NBR-02), common-path joint,
finite-probe oracle (first pass), finite native oracle (second pass),
distilled finite-risk classifier, equal-cardinality random.
Second pass also original trained CQ06 gate scores top two.
All use current state and cue only. First-pass policies receive only
u0 and root-fixed independent future cues; require identical selected
masks on both examples in each paired-z task row.

Common path: p(t)=b+t sum_i w_i, t=0,1/2,1. At the terminal
pass use exact integrated per-coordinate tanh secant
(tanh(p)-tanh(b))/(p-b) (derivative at b when difference <=1e-6).
At the first pass integrate the current and hypothetical delayed
readout gradients along the same common path via Simpson weights
1/6,4/6,1/6, eight probes each. Multiply each component write by the
common integrated gradient and select a k=2 mask minimizing squared
omitted signature. This is an additive common-path attribution, not
the exact integration along each mask's deletion path. It does not
certify finite deletion effects; saturation and cancellation may
still cause failure. Report as diagnostic proxy.

Finite-probe oracle first pass: enumerate all six k=2 masks in actual
native BPD. Score squared current-logit difference plus average
squared difference across eight hypothetical future cues with full
second pass; full reference uses identical probes. The actual heldout
future cue is never used for selecting the first mask. Finite native
oracle second pass enumerates six real masked outputs and picks
minimal squared full-logit difference on actual reached state/cue.
Both oracles pay full forward plus six candidates; first probes also
pay eight full second-pass continuations per candidate. These are
reference policies, not zero-cost inference baselines.

Distillation: for each pass learn independent MLP 48-tanh-48-tanh-6
from permitted input (u0 or [s1,u1]) to six mask scores. Use only
finite candidate risks on calibration cases as targets, no behavioral
labels. Normalize each row by subtracting min and dividing by
max-min clamped at 1e-8. Minimize softmax expected normalized regret
plus 0.2 cross-entropy to lex-first minimum-risk mask. Adam lr .009,
600 updates, batch 256, seeded initialization and minibatches. Fixed
architecture and schedule, no validation-based selection or threshold
changes. Choose argmax at test. Record calibration input/risk hash,
policy parameter hash, calibration finite-choice agreement and regret.
Learned test-time policies do not compute full current preactivation,
gradients, probes or all candidates in order to choose a mask;
scoring still executes the full reference for comparison. Include
policy parameter count and operation counts rather than claiming speed.

## Outcomes and audit

Every scored mask executes the CQ06 native wrapper, with independent
state and logit parity <=1e-5. Task: both and per-readout correctness,
actual squared current/second logit error to full BPD, mean physical
policy contrasts by root. On genuine reached donor states score second
pass recomputed versus frozen clean-recipient mask; same cue, k=2,
actual full-donor continuation reference, independent donor target.
No projected/off-manifold edits in this stage. Full source competence
>=90% both-readout accuracy diagnostic; no exclusion.
Save per-case masks/logits/targets and finite candidate risks for both
test sets, policy training and source hashes. Independent arithmetic
auditor checks cardinality, first-pair invariance, finite oracle
ordering, task/donor outcomes and root-level summaries. The first
oracle is optimal only for its hypothetical probe objective; actual
test future error may be larger. Engineering tests include a tanh
saturation witness and native parity. No statistical significance
claim; development only. A separate EvalMath preflight and untouched
confirmation population are required before claiming learned gate
superiority. Protected original CQ and ARG-03-N data remain sealed.
