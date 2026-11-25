# COG-04: proof routing through shared factors

COG-04 fits a 225-parameter candidate scorer, decomposes its input matrix with
BPD's `TiedComponentLinear`, and measures how proof search changes when each
component activation travels in a short binary message. The same factors are
reused at every verifier step. The world stores the proved facts between steps.

The included run has 64 training roots, 64 calibration roots, 64 ordinary
readout roots and 64 deeper readout roots. Each root is a generated Horn proof
DAG. All four partitions are public development material. Provenance and
source hashes appear in `PROVENANCE.json`; `vendor/ledger.py` matches the
DataGoblin COG-02 source at commit
`ae0dbecffc1d246778ee47d8f3d8ec59b5948f2b`.

## What the experiment measures

A sender observes the public proof graph and the facts proved so far. For each
available rule, twelve local features describe its premises, conclusion and
nearby graph structure. A two-layer scorer learns whether applying that rule
would prove a new ancestor of the goal. Labels come from the exact checker and
public graph ancestry during training. The input allowlist contains graph and
state features; outcome labels and split metadata stay outside it.

At a search step, the sender computes z = xV. A receiver decodes the message,
computes `tanh(zU + bias)`, and applies the shared scoring head. Its reply names
one candidate position. The sender maps that position to the public rule index
and makes one verifier call. Rejected proposals consume the same one-call
budget as accepted proposals.

All policies have a maximum of 24 verifier calls per root and stop when the
goal is proved. The common public action support excludes rules with an
already-known conclusion. A rejection removes its rule until a new fact changes
the state. The random and symbolic controls use that support too. This shared
bookkeeping supplies an explicit baseline for the learned scorer.

The full reference sends twelve float32 input coordinates through an identity
read factor. Four ranks and four bit depths give sixteen SVD-based arms. A
channel-shuffled rank-four control and three random matrix perturbations test
the learned basis. The latter preserve the singular values and Frobenius dose
of the rank-four truncation residual. Their full-rank messages receive separate
costs. The public symbolic rule selector provides a structural ceiling.

## Reproduce

Python 3.12, NumPy 2.3.5 and PyTorch 2.5.1+cpu were used for the archived run.
The scorer trains on CPU with one thread and seed 1701; the source downloads
zero model weights.

```bash
python -m unittest mechinterp.bpd.experiments.cog04_ledger_communication.test_experiment -v
python -m mechinterp.bpd.experiments.cog04_ledger_communication.experiment \
  --out /tmp/cog04-replay --per-split 64 --budget 24
python mechinterp/bpd/experiments/cog04_ledger_communication/audit_results.py \
  --results /tmp/cog04-replay
```

The independent standard-library audit can verify the committed receipts
without PyTorch:

```bash
python mechinterp/bpd/experiments/cog04_ledger_communication/audit_results.py
```

`checkpoint.json` holds every learned tensor and the full recorded loss curve.
The runner fits on training roots once, fits quantizer scales on those same
roots, checks the calibration cohort, invokes the pinned EvalMath preflight,
and then evaluates the fixed set of public readout arms. Its output includes
per-root outcomes and every decision in `decisions.jsonl.gz`.

## Message accounting

The byte protocol has a seven-byte header containing candidate count, rank and
bit depth. Fixed-width unsigned values occupy `ceil(candidates * rank * bits /
8)` bytes. Float32 values occupy four bytes each. The receiver sends a two-byte
candidate-position reply. Padding bytes and both headers are charged.

Separate setup streams install the sender's feature normalization and V, then
the receiver's U, bias and scoring head. Each party receives its quantizer
scales. `setup_streams.json` retains the exact streams as hexadecimal strings;
`arms.json` records their binary sizes and hashes. The displayed online cost
charges every actual message through the final verifier call. Setup is charged
once per deployed arm, and remains separate from recurring bytes.

Quantizer bounds use maximum absolute component activations on training rows.
Clipping events on later roots are reported. The implementation sends packed
codes through the decoder before computing scores, so the budget describes an
executed transport protocol. The proof graph and candidate ordering are
sender-side information under the stated communication model. This gym fixes
where projection and readout execute. A sender given the whole scorer could
choose the action locally and transmit its two-byte index; the reported byte
ratios describe the chosen split-compute implementation. Evaluation
instrumentation can access the full scorer to calculate fidelity statistics;
its extra verifier calls appear separately from policy calls.

### Rank, precision and decisions

A rank-r real factorization introduces r real coordinates. A b-bit codebook
turns those coordinates into a finite message whose payload grows with r times
b and the number of candidates. The codebook, model parameters and input
normalization are part of the protocol's setup. Matrix rank alone leaves these
precision choices unspecified.

For an ideal real-valued hidden preactivation h and its decoded counterpart
h', a head vector a yields

\[
|a^\top\tanh(h')-a^\top\tanh(h)|
\leq \sum_j |a_j|\,|h'_j-h_j|.
\]

The inequality follows because the derivative of tanh has magnitude at most
one. If the full top candidate's lower score bound exceeds each competitor's
upper bound, the argmax agrees. The code records this sufficient-margin test,
adds a floating-point guard, and verifies every issued flag against the actual
stored score computation. It is an offline numerical certificate for the
archived finite decisions. Transfer to other inputs requires another bound.
The full scorer's weights are available to the evaluator for this check; the
receiver uses its declared message and setup only.

The Cogentic log-rank paper motivates the question about structure and
communication. Its theorem concerns Boolean matrices and deterministic
communication complexity. COG-04 measures an explicit lossy protocol for a
small real-valued scorer, so the experiment reports its byte costs directly.
No theorem from that paper is assumed by the code.

## Reading the evidence

`RESULTS.md` includes every arm. The principal outcome is roots solved at the
24-call cap, with useful-lemma counts and solve-per-call accounting alongside
it. Recovery/corruption counts compare each arm's action to the full scorer at
that arm's reached state. These local counts serve as diagnostics; final solve
fractions come from each policy's entire rollout.

A source root is the unit for the reported routing outcomes. The run uses one
trained checkpoint, which limits claims about model-seed variation. Ordinary
and deeper roots use the same public generator families, with depth shifted
from 2 through 5 to 6 through 9 for the latter. Earlier smoke exposure and the
dense arithmetic amendment are recorded in `NUMERIC_AMENDMENT.md`. The
`fresh_root_readouts` field excludes the eight smoke roots at the first full
readout; the complete report retains every root.

EvalMath returned `REDESIGN_OR_PILOT_ONLY`. Its 64-root sham calibration gives a
conservative unit-scale variation floor for the binary solve endpoint. The
maximum observed sham changes a root's entire solve outcome, which places the
0.10 planning target beneath that floor. The public readout and single trained
checkpoint also set a clear limit on scientific inference. The completed work
supports a finite engineering comparison of precision, transport and routing.
Historical BPD confirmation cohorts were never accessed.

## Files

| File | Purpose |
|---|---|
| `experiment.py` | Train the scorer and run the fixed transport arms |
| `transport.py` | Encode/decode actual bytes and action replies |
| `test_experiment.py` | Quantization, residual-bypass, leakage and budget tests |
| `audit_results.py` | Independently replay receipt outcomes and byte charges |
| `protocol.json` | Initial model, splits, budget and arm grid |
| `NUMERIC_AMENDMENT.md` | Dense reference added after observing SVD roundoff |
| `PROVENANCE.json` | Exact source revisions and hashes |
| `results/` | Checkpoint, setup streams, loss curve, preflight and receipts |
