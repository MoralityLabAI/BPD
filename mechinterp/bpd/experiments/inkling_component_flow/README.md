# INK-BPD-06: native BPD writes into loop readers

This instrument connects `TiedComponentLinear` writes to a named reader
at the next recurrent execution. It checks the captured writer output and
completed block state, then reports how each physical component write
affects the supplied reader.

**Evidence: source instrumentation with constructed CPU fixtures.** The
included example assigns its weights explicitly. Training updates and
scored task outcomes are both zero. Pretrained-model execution and the
proposed training experiment remain pending.

[Research review](../../docs/GOODFIRE_LOOPED_MECHINTERP_20261126.md) ·
[Development design](PROTOCOL.md) ·
[Source](factor_flow.py)

## Run the checked example

From the repository root, in an environment with CPU PyTorch and the
repository's NumPy dependency:

```sh
python -m unittest discover \
  -s mechinterp/bpd/experiments/inkling_component_flow -p 'test_*.py' -v

python -m mechinterp.bpd.experiments.inkling_component_flow.run_source_audit \
  --check mechinterp/bpd/experiments/inkling_component_flow/evidence/source_fixture.json \
  --out /tmp/inkbpd06_source_fixture.json
```

The example uses two explicit inputs with the same factor weights. Its
first component supplies opposing signed responses to the Q reader.
The input construction supplies that change of sign. The receipt tests
the instrument's ability to retain it and deterministically replays
each finite boundary edit. Separate tests compare the reader with autograd,
finite differences and a probability calculation from serialized inputs.

Replay requires identical recorded source hashes and checks every
non-runtime field. Float comparisons use absolute tolerance `1e-12`
and relative tolerance `1e-10`. Python and PyTorch version differences
remain recorded in the new receipt. Use `--out` alone to generate a new
fixture receipt after an intentional source change.

## API and provenance

Import from
`mechinterp.bpd.experiments.inkling_component_flow.factor_flow`.

| API | Inputs and output |
|---|---|
| `ExecutionSite(...)` | Root, checkpoint, example, token, loop, physical layer, module and recurrence kind |
| `ReaderSpec(site, reader_ids, queries, eps, rms_scale=None)` | A named set of queries at one reader execution |
| `clean_target_writes(module, x, observed_output, site)` | A clean native BPD invocation and its separately captured output; returns physical component writes, residual and bias |
| `component_reader_flow(ledger, reader, block2, block3, other_block_writes)` | Captured block states and an independently assembled remainder; returns signed attribution and reader derivatives |
| `component_direction_patch(ledger, component, dose, orientation=-1)` | An absolute-L2 displacement along a selected component write |
| `patch_reader_audit(reader, block2, block3, delta, max_dose=..., source_site=...)` | A finite previous-loop block-3 boundary edit with reconstructed routing and message changes |
| `to_jsonable(result)` | JSON-compatible values, preserving undefined shares as `null` |

Physical layers are one-based. Token and loop indices are zero-based.
`step_kind="depth"` requires matching source/reader token positions.
`step_kind="sequence_time"` requires a later reader token. Both require
the reader loop index to immediately follow the source loop index. These
checks validate the supplied metadata; the caller provides capture provenance.

Physical layer 13 maps to Inkling loop reader index 6. Physical layer 18
maps to index 11. A separate `ReaderSpec` represents each receiver layer,
with explicit Q/K/V/MoE IDs in the query tensor's order.

The clean extractor accepts a native `TiedComponentLinear` in target mode
with no active schedule or intervention. It performs no model forward
calls. Existing call counters, recorded inputs and statistics stay intact.
The output captured by the caller is compared with the full decomposition
under declared tolerances. If a low-precision capture fails closure, inspect
its arithmetic before changing those tolerances.

`other_block_writes` contains contributions that are outside the selected
linear invocation. Assemble it from captured writes. The API can verify
closure to the separately captured block state; it cannot establish how a
caller obtained that remainder.

## Interpreting the fields

For a component activation `a_c = (x @ V)_c`, its physical write is
`w_c = a_c * U_c`. The signed score contribution uses the observed
normalizer. `gap_derivative_per_write` also differentiates that normalizer
with respect to a change in the raw block state. Their difference matters
when a write changes the state's radius.

The finite audit recomputes normalization and softmax for the actual
displacement. Its output retains raw block states, reader parameters,
epsilon and normalization scale so another implementation can replay it.
The message change equals the sum of the route, content and interaction
terms. Larger edits can produce a substantial difference from the local
derivative prediction, which is retained in the report.

An additive patch records its changed state radius. Matching absolute L2
alone leaves that radius available to change. The existing
[INK-BPD-05 hook](../inkling_routing/INK_BPD05_PRETRAINED_PROTOCOL.md) uses
its own registered RMS-preserving treatment. Preserve each treatment's
physical definition when comparing results.

Component identities are local to a factorization. Inverse rescaling of
a factor pair preserves its physical write. Mixing factor coordinates can
redistribute the individual entries while preserving the total operator.

## Resource and empirical scope

Each call audits one vector at one execution. The implementation copies
the target weight and relevant factors to CPU float64 and constructs one
`[components, width]` write array. The dense target-weight copy can dominate
host memory. Capturing every token or materializing all sites at once
requires a separate resource budget; throughput remains unmeasured.

This is a previous-loop block-boundary calculation with the other block
held fixed. A writer intervention earlier in the network changes later
computations and needs a native hook. Keep/gain gates, full transformer
continuation and task effects remain outside this local reader audit.

The [archived RP03 planning recheck](evidence/rp03_preflight_recheck.json)
reproduces the prior `REDESIGN_OR_PILOT_ONLY` result exactly. Its eight
model roots supply historical context. New-method power planning awaits
independent development effects and matched shams.
