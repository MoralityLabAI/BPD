# BPD-RP-02: mechanistic steering during training

## Question and scope

Does a loss on causal component interventions improve ordinary delayed-memory
answers after training? The primary comparison is B minus A on the longer-delay
readout. B minus C tests whether the physical direction of the training signal
matters. Both contrasts belong to one two-comparison family.

This is a synthetic **DEVELOPMENT_ONLY** experiment. A root consists of a fresh
model initialization, input encoding and training stream. The three descendants
of its common warmup are paired treatments. Cases within that root are repeated
measurements. RP-01 and all historical protected populations remain unchanged.

## Task and data barrier

Four memory slots each hold one of four values. Each slot is written exactly
once. One to three distractor steps follow during training, and a final query
asks for the value of one slot. Distractors carry plausible addresses and values
under an explicit NOOP type. The source cannot reread a past write.

Each root fixes independent permutations of input address and value encodings.
Host-owned labels use the semantic values. Gaussian input noise has standard
deviation 0.03 on the address/value channels; operation-type bits stay exact.

Canonical memory tuples are encoded as base-four integers. Training uses
residues 2, 3 and 4 modulo 5; evaluation uses residues 0 and 1. These sets contain
153 and 103 memory tuples, respectively. Receiver, donor and hybrid tuples must
all satisfy the selected partition. A hybrid changes one receiver slot to the
donor's different value. The selected slot is queried in half the cases, while
the other half test preservation of an unaffected answer. Rejection sampling
conditions the pair distribution on these constraints; its value-change
distribution is part of this finite synthetic task.

All arms receive ordinary supervised examples for receiver, donor and hybrid.
Thus every counterfactual training label is also available through an ordinary
input in every arm. The neural forward interface accepts feature tensors only.
Final scored batches are generated after all updates finish. No scored labels
or evaluated tuple are used for component discovery or model selection.

## Shared architecture

The recurrent cell has width 32 and a GRU-style update. Its hidden-to-gate map
uses the repository's `TiedComponentLinear` with 32 components: one trainable
V/U pair is reused at every sequence step. Ordinary execution uses component
mode, all-open component gates and residual delta zero. The input map and output
head are trainable. There are 5,380 trainable parameters.

`W_target` is a stored initialization buffer and contributes zero to execution.
An invariance test checks this by changing that buffer. This gym trains the
component parameterization end to end. The original post-hoc BPD trainer and
its learned importance network are outside RP-02's training recipe.

## Mechanistic signal

After 200 common warmup updates, actual component interventions identify a group
for each semantic slot using 128 training-only receiver/donor pairs. The fixed
candidate family contains eight disjoint groups of four component indices.
Each group is evaluated for every slot, and the smallest counterfactual
cross-entropy selects that slot's group. Stable index order resolves exact ties.
The four chosen groups are frozen before the arms fork; overlap between slots
is allowed and recorded.

Let h_r and h_d be states after all four writes. For the frozen component group
G associated with the edited slot, the candidate recurrent write is

`d = ((stop_gradient(h_d) @ V - h_r @ V) restricted to G) @ U_G`.

Normalize a nonzero d to L2 norm 0.75 and add it to the recurrent preactivation
at the first distractor step. The donor state is detached; the model's V/U and
receiver trajectory remain differentiable. Zero directions produce a logged
zero intervention. Continuing the receiver toward the known hybrid label
defines the causal auxiliary loss. Group discovery is included in the common
cost of every treatment.

The control applies a fixed, root-seeded signed permutation to the same write
vector. It has identical physical L2 norm, while its coordinate correspondence
changes. Downstream state radius and all-head movement are measured separately.

## Training comparison

| Arm | Applied update direction |
|---|---|
| A_task | Ordinary task cross-entropy gradient |
| B_causal | Task gradient plus normalized causal-intervention gradient |
| C_rotated | Task gradient plus normalized rotated-intervention gradient |

Each arm starts from exactly the same warmup state and a fresh AdamW optimizer.
All receive 240 updates, identical batches of 48 pairs and delays cycling through
1, 2 and 3. Learning rate is 0.003, weight decay 0.0001, and global gradient
clipping is 1.0. All forward and gradient evaluations execute in every arm:
three ordinary trajectories and two intervention trajectories per pair, with
three gradient evaluations per update.

For B and C, the auxiliary gradient is scaled to 0.25 times the L2 norm of that
arm's task gradient. A computes both auxiliary gradients and applies neither.
This matches the auxiliary-to-task gradient ratio before AdamW. Actual task
gradients and optimizer updates can differ as the learned trajectories diverge;
their norms and cosines are recorded. Degenerate auxiliary gradients produce a
logged zero contribution. No checkpoint selection or early stopping occurs.

## Scoring and inference

Each trained arm receives 512 ordinary examples at delay 2 and 512 at delay 7.
Every ordinary scored pass has steering disabled. The primary endpoint is
ordinary delay-7 accuracy. The 256 additional causal pairs use unseen tuples and
delay 7, with targeted and unaffected queries reported separately. They measure
the trained intervention objective and collateral effects.

The analysis preserves per-case predictions and labels, then computes paired
root effects. B-versus-A recovery and corruption use the same ordinary inputs.
Root means are equally weighted. The fixed family consists of B-A and B-C on
ordinary delay-7 accuracy, with exact root sign-flip calculations under signed
exchangeability, Holm adjustment, and a seeded 10,000-draw descriptive root
bootstrap. The scientific reference floor is an absolute five-percentage-point
gain. A ten-point target is supplied only to the planning routine.

The small root count leaves population resolution and sham-tail calibration
limited. Every readout retains DEVELOPMENT_ONLY, including a positive nominal
test. A stronger result needs a separately designed prospective population and
adequate independent sham calibration. C may damage learning; B exceeding C
alone supplies no evidence of improvement over ordinary training.

## Execution order and provenance

Engineering roots are 26000 and 26001. Their outputs are kept even if source or
recipe changes are needed. The official pinned EvalMath preflight is run on
their root-level B-A effects and C-A control effects. The expected small-n
scope is pilot-only; no empirical 95/95 population sham bound is claimed.

The source and recipe are then published before running development roots
26101 through 26108. `SOURCE_FREEZE.json` identifies the remote commit, all
source hashes and the roster. The runner requires an unused output directory
and verifies the freeze for development. Failures remain in the record.

Saved checkpoints permit an independent implementation of the recurrent
equations to reproduce outputs. Receipts include common initialization hashes,
per-step data hashes, selected groups, gradient doses and all terminal outputs.
No new repair search or quantization sweep is part of RP-02.
