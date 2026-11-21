# Inkling Reader-01: causal cross-loop routing, source-only phase

Date: 2026-11-21. Status: source-only math and proposed native protocol.
No trained-model causal result. The NC09 Fixed12, AMM08/09 and all other
sealed BPD cohorts remain untouched, with their historical decisions intact.

## Literature and hypothesis

- Thinking Machines, Inkling 2 Mini (2026-11-20):
  https://thinkingmachines.ai/news/inkling-2-mini/
- Chen et al., Attention Residuals:
  https://arxiv.org/abs/2603.15031
- Goodfire, Interpreting Inkling 2 Mini:
  https://www.goodfire.com/research/interpreting-inkling-2-mini/
- Variable-loop model card:
  https://thinkingmachines.ai/model-card/tml-d24-varloop-research/

Inkling repeats two middle six-layer blocks with a learned gate admitting
the previous pass's block representation. Its four cross-loop readers
(query, key, value and MoE) select from the preceding block 2/3 outputs.
Goodfire reports a conspicuous outlier coordinate 2027 in the released
FIXED three-pass model, not in the variable-loop research checkpoint.
At the measured block boundary it explains a large fraction of the
normalized activation and several routing score gaps. Correlation with
reader preference alone is insufficient for a causal mechanism claim.

For block vectors v2, v3 and a frozen reader u, let

    s_i = dot(u, RMSNorm(v_i)) / sqrt(d)
    beta3 = sigmoid(s3 - s2)

The coordinate contribution u_j * (RMSNorm(v3)_j - RMSNorm(v2)_j)
divided by sqrt(d) is an exact additive decomposition of the frozen
score gap. A physical raw-coordinate patch changes RMSNorm, so
recompute the entire downstream state to measure an intervention.

## Rival hypotheses

H1: A selected routing direction causes a greater change in independently
verified serial-task success than dose-matched random interventions.

H2: Routing probabilities change while correctness remains within the
matched-sham effect envelope. The direction may encode phase or punctuation.

H3: A large attribution under a frozen-normalization decomposition
shrinks or reverses after physically patching the raw block vector and
recomputing RMSNorm.

H4: Coda/tuned-lens decodability rises with recurrence without a
corresponding increase in verified task success.

## Native experiment registration requirements

- Freeze the checkpoint revision, tokenizer, actual operator, pass count,
  cache-sharing mode, sampling seed, chosen layer and evaluation task root.
  Check that the target model solves a positive-control algorithm.
- Discover outlier directions in a TRAIN-only split. Coordinate 2027 is
  a hypothesis for the fixed three-pass checkpoint only.
- Select the reader and intervention direction on independent development
  roots. One primary reader and one prespecified dose must be frozen
  before any fresh confirmation readout.
- Patch the raw block 2 or 3 output at a named loop boundary (after
  layers 12 or 18). Recompute complete model execution thereafter,
  including normalization, all four readers, gates and the final head.
- Arms: untreated, zero patch, candidate direction, dose-matched random
  coordinate, dose-matched Haar direction, shuffled donor, matched
  unrelated donor, and clean/corrupted minimal-pair donor when available.
- Physical dose is ACTUAL raw-state L2 displacement, checked separately
  from intended dose. Record full-head parity, future logits, clean/patched
  KL, reader weights, layer state norms and per-token differences.
- Primary endpoint: independent verifier-confirmed task repair/corruption
  rate under a fixed resource budget, not a reader score or a linear probe.
  Secondary endpoints include first-valid visit and outlier propagation.
- Independent unit: source task root or separately trained checkpoint as
  registered. Tokens, loop passes, directions and donor variants are
  correlated measurements within one root.
- EvalMath preflight sets the minimum meaningful effect, shams, power and
  Holm multiplicity plan. Calibrating a nonparametric 95/95 sham threshold
  requires at least 59 independent sham roots even using the maximum.
- Evaluation uses a fresh sealed source-root population with explicit
  access authorization and release protocol. Never reuse the NC09 or
  AMM confirmation samples.

## Failure gates and scientific scope

Failure includes absent trained positive control, mismatched actual dose,
reconstruction/head-parity faults, source overlap, unregistered direction
selection, or a task effect within sham variability. All negative arms
and outcomes must be retained. Conclude observational feature detection
when a probe decodes state but native causal evidence is absent.

Variable-loop pass counts 7 and 10 are extrapolation probes. The model
was midtrained at 1-5 passes and post-trained at 3; this differential
exposure complicates depth interpretations.

## Existing source-only implementation

reader_probe.py evaluates the two-source reader score and recomputes
RMSNorm following a raw coordinate patch. The norm-matched control uses
a deterministic Gaussian direction. It is a synthetic algebra check,
with no activation extraction, GPU allocation, training or external
model access. Run:

    python -m unittest discover -s mechinterp/bpd/experiments/inkling_reader_v0 -p test_reader_probe.py -v

Development root-level depth audits belong in the EvalMath companion
branch, research/inkling-loop-monitorability-20261121.
A future native adapter and a fresh preregistered sample are prerequisites
for quantitative BPD mechanism claims.
