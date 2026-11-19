# DT05 development protocol

19 November 2026. This is an engineering and development successor to the
exposed DT04 campaign. DT04's original registered verdict remains
`INCOMPLETE_OR_INELIGIBLE`. Replaying its exact archived 56-root EvalMath input
returns `REDESIGN_OR_PILOT_ONLY`; see [the reproduction](evidence/REPRODUCTION.json).

## Scope and units

The independent unit for a training comparison is the model root. Repeated
boards, cells, depths, routes, and geometry probes within it remain repeated
measurements. New development roots are `2605000..2605058`. Excluded smoke
root `2605999` serves eight synthetic implementation fixtures. Resource probes
use seed `2605991` on random tokens and labels and perform zero optimizer steps.

DT05 uses the exposed Sudoku4 generator/catalog with a fresh per-root canonical
partition. It does not establish generalization to a new puzzle distribution.
The original report informed every new design choice. The two official 7M
checkpoints can be replayed only from user-supplied, SHA-verified local files
against explicit development/engineering case manifests.

No confirmation population is released by this protocol. DT04 confirmation
roots `2604100..2604355`, ARG-03-N's 168 cases, and every other historical
confirmation seal remain untouched. The runner has no confirmation command
and accepts only the declared development roots for new training.

## Freeze before running

`runner plan` records source SHA-256 values, profile, optimizer budget,
microbatch, logical batch, precision, EMA rate, factor fit budget, component
counts, and memory limits. Each development run checks these source hashes.
One exclusive marker per root per campaign preserves failed/interrupted
attempts. Additional campaigns retain their own source and selection history.
Reusing a root in another recipe does not increase the independent sample size.

Use resource calibration on engineering inputs before choosing microbatch.
Training/evaluation loops sample process and CUDA memory at work boundaries.
Those samples cannot prevent every instantaneous host allocation spike.
The cache stores only requested fit-depth teacher targets and remains bounded.
Changes to dtype or the logical batch are explicit recipe changes.

## Training and comparisons

The default teacher is the native 241,922-parameter Sudoku4 TRM, trained for
480 optimizer updates with AdamW, cosine scheduling, and fit-depth losses at
2/4/6. The logical batch remains 128. Blank and clue sums are divided by whole
logical-batch counts before microbatch backpropagation. Legal Sudoku symmetries
are sampled once for that logical batch.

EMA updates once per optimizer update. Final raw and EMA models are both
reported without evaluation-driven checkpoint selection. EMA is fixed in the
plan as the factor teacher. The factor model starts from SVD factors, fits
V/U on training-only teacher logits and zL at 2/4/6, and chooses its own live
component masks throughout training and inference. Inference stores no dense
teacher residual at replaced sites. Shared-factor fixed and seeded-random
subsets provide controls. They receive no additional fit or eval selection.

Every arm retains the same-case raw trajectory through depth 12. A separate
first-valid policy physically compacts unfinished cases after checking digit
range, public clues, rows, columns, and boxes. Gold solutions never enter that
policy. Unique-puzzle admission is checked at replay loading. Raw and compact
case outcomes remain separate so arithmetic changes from batch compaction are
visible.

## Evidence contract

| Field | Registered development interpretation |
|---|---|
| Unit | Trained model root |
| Planned units | Up to 59 development roots; no automatic full launch |
| Cases | 64 same-case eval boards per root; retain every root and failed attempt |
| Primary contrast | Online-factor exact boards minus shared-factor fixed control at depth 8 |
| Other comparisons | Raw versus EMA; seeded-random routing; first-valid versus raw final; all reported |
| Physical intervention | Factor selection at each recurrence, with the same fitted V/U; state transport remains diagnostic |
| Mechanistic effect floor | No new mechanism claim admitted; DT04's old 0.10 exact-board floor is historical context |
| Matched sham floor | New live-routing/state-transport dose calibration still required; DT04 signed shams do not certify new arms |
| Multiplicity | Five model arms, repeated depths, and stopping policies are disclosed; no significance gate is implemented |
| Resolution/power | Unestablished for the redesigned mechanism; frozen DT04 preflight remains negative |
| Selection exposure | DT04 report, known generator/catalog, and any prior DT05 development results |
| Causal controls | Shared-factor fixed/random routing are development controls; a future causal claim needs matched delivered dose and calibrated shams |
| Verdict | Engineering and development measurements only |
| Provenance | Source/recipe hashes, runtime, all case predictions, loss/memory traces, checkpoint hashes, and attempt records |

The JVP diagnostic transports a protected random thin basis on one training
example, checks rank loss, and reports local closure and projection escape.
It does not fit a task basis, apply a donor intervention, or test causal repair.
The next scientific design must choose those quantities on training data,
measure actual intervention dose at each depth, calibrate independently, and
run EvalMath before proposing a separately sealed confirmation study.

Hardware outcomes are stage-specific. The historical full-7M 82 MiB reserved
figure describes a small batch-16 inference pass. New forward/backward probes
include persistent optimizer/EMA storage and report their exact timing scope.
Active-component counts alone establish neither wall-clock speed nor total
memory savings.
