# BPD-COL-02 protocol

Date: 2026-11-25. Exposure: public development. Verdict class: `PILOT_ONLY`.

## Question and observer

Let `H` be the `[1, sequence, hidden]` output of Nanbeige's first native
`model.norm` call. Its last prompt position is `h_q`. Define the current
observer as the complete checkpoint vocabulary head `W h_q`. The prefix
`H[:, :-1, :]` becomes input to the second native pass, so changing it can
change later query representations through attention.

Every intervention preserves `h_q` exactly. Before applying an edit, the
runner requires the entire replayed first-pass state to equal the captured
recipient. After the native forward it compares the first-pass query vector
with the clean recipient again. Holding that vector and the checkpoint head
fixed certifies the observer equality at the declared position. Native tests
also apply the complete vocabulary head to these vectors.

Earlier prefix positions lie outside this observer. The experiment tests
context carried across a loop boundary; it leaves the answer vector itself
unchanged at the selected boundary. Interpolated prefix edits have a
constructed-state interpretation. Full-state natural reachability requires
additional evidence.

## Source cases

The exact split-Collatz oracle and its SHA are inherited from COL-01. The
manifest fixes all cases before model execution. Roots are admissible source
parameters `d <= 1025`; selection is deterministic and capped at 32 roots.
The first four are `13, 73, 17, 25`. Horizons are one or two arithmetic
transitions and remain distinct from the model's fixed two native passes.

Each triplet matches current quotient and current branch. The effect donor's
future branch differs from the recipient; the sham donor's future branch
agrees. The generator prefers same-basin triplets. The `d=13` calibration has
a cross-basin effect donor and receives its own summary stratum. For `d=73`,
the fixed seeds are recipient `1`, effect donor `9`, sham donor `37`.

The `trace` tier puts the computed trajectory in the prompt and tests routing
and answer readout. The `arithmetic` tier supplies the initial state and rules,
with oracle trajectories retained in the host manifest. Letter mappings
alternate deterministically across cases. Every matched prompt is unpadded,
has the same token count as its donors and fits the 128-token cap.

## Native execution

The registered checkpoint is `Nanbeige/Nanbeige4.2-3B` revision
`451ed48c3273ecef7ea8faaa43c31ce529763bb1`. Source, config, tokenizer, index and
both checkpoint shards have pinned hashes. Production execution uses BF16,
batch one, SDPA attention, disabled cache, and the trained two-loop schedule.

The actual model is constructed on `meta`. Physical decoder pre-hooks load
checkpoint tensors just before a layer executes; post-hooks return its
parameters to `meta`. The unchanged native forward chooses recurrence,
normalization and attention order. Required rotary buffers are reconstructed
in FP32 as in native initialization. Only requested token embeddings are
loaded. Scoring streams requested vocabulary rows through the native linear
head operation.

Torch deterministic algorithms are enabled, TF32 is disabled and the seed is
fixed at `20261125`. The run contract includes package versions, Python
version, CPU thread count and source hashes. Resume compares both the source
manifest and exact rendered tokenization.

## Dose and arms

Write the recipient prefix as `R`, effect donor prefix as `D`, and sham prefix
as `S`. The prospective dose is

\[
\epsilon = 0.1\min(\lVert D-R\rVert_F,\lVert S-R\rVert_F).
\]

The effect direction uses `D-R`; the sham uses `S-R`. A Gaussian random
direction comes from a CPU generator seeded by the SHA-256 of the case ID.
Each direction is normalized and scaled. A bounded search chooses a scale
whose **delivered BF16 displacement** is within 1% of the common target.
The scorer checks the three measured norms against a 2% relative allowance
plus `1e-6` absolute. A failed match stays in the record and makes the case
ineligible.

Identity and full donor prefix arms complete the control set. The identity
has zero dose. The full donor arm replaces all prefix positions with the
captured effect donor prefix and retains the recipient query position; its
larger dose is reported separately. All arms run the complete two native
passes and patch the selected boundary exactly once.

Direction selection and scaling use captured states, fixed geometry and the
case-derived random seed. Patched answer scores never influence that choice.

## Outcome and eligibility

The assistant prefix ends in `Answer:`. Appending `" A"` or `" B"` must add
exactly one token without altering any existing prompt token. Let their final
logits be `l_A` and `l_B`; the score is the two-answer conditional softmax.
Free-generation performance lies outside this outcome definition.

All three clean prompts must assign strictly greater than 0.5 probability to
their correct labels. A tie fails competence. Instrument checks require
identity replay, exact first-pass protection, positive and matched delivered
doses and all declared arms. Cases that fail remain visible in the summary.

The primary case statistic is the effect-donor label probability shift minus
the larger absolute probability shift from matched sham and random arms.
This estimates donor steering conditional on baseline competence. Recipient
task correctness is reported separately because the effect donor's label is
opposite to the recipient's target. See [SCORING.md](SCORING.md) for formulas.

The aggregation unit is `d`. A root contributes only when all planned cases
for that root are complete and eligible. Eligible root means receive equal
weight. The manifest limits overlap and case count explicitly; token count
and intervention count never increase the independent root sample size.

## Resolution and interpretation

The prospective effect scale is 0.05 conditional answer probability. Every
report retains `PILOT_ONLY`, including a large positive development mean.
The calibrated sham floor remains `null`. With zero exceedances, establishing
a nonparametric 95th-percentile sham bound at 95% confidence requires at least
59 independent sham roots since `0.95^59 < 0.05`. This gym's maximum of 32
already exposed roots cannot supply that design.

Competence filtering and prior exposure condition the pilot result. The
separate trace and arithmetic tasks help identify whether failure concerns
the task or the intervention. Neither tier trains BPD factors. An eligible
state effect can motivate a later factor-mediation experiment with a newly
declared population, independent controls and adequate resolution.

## Stop and persistence behavior

The allocator cap is 1.75 GiB and the added sampled-RSS cap is 2.75 GiB. CUDA
driver/display allocations and host peaks between samples have separate
measurement scope. A resource exception stops the session and is recorded.
The runtime requests eight available forwards before starting a case. Its
wall-time limit is checked between forwards.

Completed scores survive later arm failures in diagnostic partial-progress
records. Such cases remain excluded from primary estimates. Errors and skips
remain attempted cases when resuming; only unattempted cases continue.
Changing frozen source or runtime requires a new run directory. Local records
are appended and flushed, and the summary is atomically replaced. A truncated
JSONL record causes a parse failure that asks the operator to preserve the
file before repair.
