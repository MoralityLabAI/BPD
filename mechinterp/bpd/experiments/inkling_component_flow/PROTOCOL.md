# INK-BPD-06: component flow into loop readers

**26 November 2026. Status: source instrumentation and proposed development
design.** The implementation audits constructed native BPD writes and
two-block reader equations. Training arms and pretrained checkpoint
experiments below await implementation and a prospective source lock.

## Implemented instrument

The input unit is one captured invocation of a clean `TiedComponentLinear`.
Its execution identity records the checkpoint, example and token, as well
as the physical layer and loop. Recurrence through sequence time has a
separate label. Every reader carries its own identity.

The instrument accounts for the physical component writes, weight residual,
bias and other writes supplied for the completed block. It compares their
sum with separately observed tensors. Unsupported component schedules and
active native interventions fail before attribution.

Reader reports preserve signed score contributions and signed projection
shares. Epsilon and an optional learned RMS scale enter the raw-state
Jacobian. An explicit validity flag covers a near-zero projection
denominator. A finite patch recomputes normalization and all supplied
reader probabilities, with the second block held fixed. Its message change
is decomposed into route, content and interaction terms whose sum is
checked against the full change.

The caller chooses the component and dose. The instrument performs no
component ranking or task-dependent selection. It detaches observations
and does all diagnostic arithmetic on CPU in float64.

## Source verification contract

| Check | What it establishes |
|---|---|
| Native output and block closure | Every supplied additive term is accounted for within the declared tolerance |
| Clean wrapper state before/after extraction | The audit leaves call counters, parameters and stored inputs unchanged |
| Analytic Jacobian, autograd and finite differences | The stated reader equations agree at the tested vectors |
| Reciprocal factor scaling | Physical-write measurements survive inverse rescaling of a factor pair |
| Opposing activations at distinct executions | Signed component roles remain distinguishable across the supplied sites |
| Zero patch and exact finite message accounting | The boundary-edit report obeys its algebraic contract |
| Invalid inputs and unsupported modes | Unsupported observations fail explicitly |

These are constructed engineering fixtures. Their count is a test count.
They supply zero independent trained-model roots for scientific inference.
Any example receipt must identify that scope and preserve its source hashes.

## Proposed next development experiment

**Scientific question:** Does a training target conditioned on the
recipient and recurrent phase improve ordinary task performance while
preserving answers at unaffected memory slots?

Start from RP04's single-slot receiver/donor/hybrid construction. Selection
data supply candidate factors and phase labels. The ordinary evaluation
uses fresh task families and disables steering. For an Inkling adapter,
each selected source also names the later-loop reader receiving its state.

| Proposed arm | Treatment |
|---|---|
| A | Ordinary task training with matched diagnostic work |
| B | A global component-steering target following the RP03 reference |
| C | A receiver-conditioned target indexed by execution phase |
| D | The C machinery with phase labels permuted within the declared training strata |

The conditioner's capacity and training examples must be matched across
B, C and D. Every arm starts from the same warmup within a model root.
Diagnostic forward/backward work, ordinary task supervision and optimizer
updates must be accounted for. A prospective implementation must specify
how the permutation is generated and frozen, and how gradients through
refreshed directions are handled.

The proposed primary family contains C-A, C-B and C-D, with Holm adjustment.
An improvement claim requires the comparison against task-only A to pass
its prospective effect and uncertainty gates.
Ordinary steering-disabled task accuracy is the primary endpoint. Use
the existing five-percentage-point science floor as a planning reference;
freeze the final floor before new development generation. Report unaffected
corruption with its initially correct denominator and all recoveries.
An unaffected-accuracy guardrail also needs its own prospective margin.

A physical hidden-state dose of 0.75 and auxiliary-gradient ratio of 0.25
are inherited planning values from RP03. Their suitability for another
architecture must be calibrated. At every selected site, retain state
radius and delivered displacement, then measure the selected-read
amplitude and all other reader responses. Comparing the same state dose
does not imply equal reader amplitude, as the RP04 receipts demonstrate.

### Required empirical controls

Native full-vocabulary zero-hook parity precedes outcomes. Correct-cell
recovery/corruption and full native continuation accompany the reader
metrics. A matched sham receives the same intervention count and physical
state dose. A separate phase-shuffled control tests whether timing carries
the proposed benefit. For block readers, include boundary edits in one
pass and every pass with their intervention counts and work reported.

The complete control inventory must also retain a privileged natural
hybrid-state replacement with its measured dose. Its scope is a task
ceiling. Reader-only route/content counterfactuals remain diagnostic
interventions whose task effects require a corresponding native hook.

## Resolution and provenance contract

| Field | Current value or required decision |
|---|---|
| Unit | Independently trained model root for cross-model claims; task-generator roots nested within checkpoint for a fixed-model claim |
| New independent units / task cases | 0 / 0 |
| Primary contrasts | Proposed C-A, C-B and C-D; actual inferential comparisons: 0 |
| Physical dose | Proposed L2 0.75; report delivered dose and state radius |
| Scientific variation floor | Proposed +5 pp; freeze before development |
| Matched-sham floor | Pending independent calibration |
| Estimated resolution/power | Pending observed root effects and matched shams |
| Selection exposure | Public Goodfire candidates and already exposed RP03/RP04 results inform this design |
| Causal controls | Proposed phase shuffle, matched physical sham, zero hook, hybrid-state ceiling and native continuation |
| New-method verdict | SOURCE_INSTRUMENT_ONLY |
| Historical RP03 planning verdict | REDESIGN_OR_PILOT_ONLY, independently replayed in the attached receipt |

Run the pinned EvalMath mechanistic-resolution preflight after obtaining
independent development effects and matched sham variation. A 95th
percentile sham bound with 95% confidence needs at least 59 independent
sham roots under the sample-maximum rule. Extra prompts or directions on
one checkpoint retain their clustering.

The current update creates no confirmation population. All prior cohort
seals and frozen source manifests retain their existing meaning. A future
training run needs a new source lock containing the complete implementation,
selector, data split, control family and gate decisions.

## Source references

- [Review, current BPD evidence and derivation](../../docs/GOODFIRE_LOOPED_MECHINTERP_20261126.md)
- [Goodfire, Interpreting Inkling 2 Mini](https://www.goodfire.com/research/interpreting-inkling-2-mini/)
- [Existing pretrained hook protocol](../inkling_routing/INK_BPD05_PRETRAINED_PROTOCOL.md)
- [RP03 evidence snapshot](https://github.com/MoralityLabAI/BPD/blob/402ef7764e11bc7736873d66357b8dc24b996e94/mechinterp/bpd/experiments/training_steering_stability/RESULTS.md)
- [RP04 evidence snapshot](https://github.com/MoralityLabAI/BPD/blob/aa4b55d8c968084c1414846441cb539356ba7995/mechinterp/bpd/experiments/component_selectivity/RESULTS.md)
- [EvalMath skill at the reviewed commit](https://github.com/MoralityLabAI/EvalMath/blob/958b3fc797cdc8fda927eff153d056b5e65be861/skills/evalmath-mechinterp-power/SKILL.md)
