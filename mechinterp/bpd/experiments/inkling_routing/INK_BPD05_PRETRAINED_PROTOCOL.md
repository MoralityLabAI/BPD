# INK-BPD-05: pretrained Inkling 2 Mini cross-loop read-edge pilot

**Registered source/engineering plan: 22 November 2026.**
Status: source-pinned native hook + offline runner; **no pretrained
development observations have been scored at registration**.

This follows the merged INK-BPD-04 [tangent geometry protocol](INK_BPD04_TANGENT_PROTOCOL.md).
Historical NC09, AMM, ARG, INK-BPD-02/03 and all other confirmation cohorts
remain unchanged and sealed. INK-BPD-04's synthetic PyTorch reader parity
is an engineering control, not trained-model causal identification.

## Upstream and exact hook

Source:
- [Goodfire, *Interpreting Inkling 2 Mini*, 20 Nov 2026](https://www.goodfire.com/research/interpreting-inkling-2-mini/).
- [Thinking Machines, Inkling 2 Mini, 20 Nov 2026](https://thinkingmachines.ai/news/inkling-2-mini/).
- [Hugging Face Transformers, Inkling 2 Mini implementation](https://github.com/huggingface/transformers/blob/046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py).
- Upstream commit `046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3`, model source Git blob `78498834e2806e83a1bb96eea5cf5447d9b1531f` (enforced on the installed source by `verify_transformers_source`).
- Official checkpoint: [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0, ~20 GB mixed precision.

In the upstream `Inkling2LoopTextModel.forward`, looped layers 6–17
are executed in three passes. In passes 1 and 2 (zero-indexed),
`message_input(previous_block_states, previous_hidden_states,
loop_layer_idx)` is invoked once for each of the twelve looped layers.
The previous block collection contains exactly the completed block 2,
while `previous_hidden_states` is the previous pass's completed
block 3. The reader independently computes four messages (Q/K/V/MoE)
with a learned query and RMSNorm. The native layer's message gates then
mix these into the ongoing computation.

`pretrained_message_hook.py` intercepts this **read edge**, edits the
previous-pass block-3 representation at the final input token and
substitutes that same changed source into each of the twelve message
calls in the selected pass. All four message readers and remaining
transformer execution are run normally. This treatment changes which
representation the next pass can read; the original writer computation
from the preceding pass remains intact. The previous block 2 input
and all current-pass block accumulation are unchanged at the hook.

Strict invariants:
- installed modeling source matches the exact upstream blob;
- config matches 24 layers, 2304 width, blocks of 6, core layers 6–17,
  three passes; reader parameters have shape `[12,4,2304]`;
- 24 invocations (12 in each subsequent pass), ordered by layer;
- block collection contains one completed block and one partial state;
- all four gate probabilities recompute directly from the actual
  physical pre-normalized states;
- native message outputs match that recomputation (absolute tolerance
  0.003) and final zero-hook logits match unhooked full-head logits
  (absolute tolerance 0.003);
- the hook cannot silently skip a pass, a reader, a patch, or a control.

The model is run with `use_cache=False`, eager attention, deterministic
source-registered probes, fixed max 192 prompt tokens and single-item
batches. No KV-cache reuse enters the intervention comparison.

## Physical interventions and pre-data decisions

One hypothesized candidate was supplied externally by Goodfire: raw
block-state coordinate **2027**, so its selection status is exposed.
This is **not** an independently discovered BPD feature. The
candidate is the tangent projection of the coordinate basis direction
onto the sphere centered at the original block-3 state.

Every active intervention is an orthogonal sphere move with
`||v3' - v3|| / ||v3|| = 0.08` (8% relative L2), preserving
the raw block-state RMS up to casting tolerance. The eight registered
shams consist of four coordinate-basis directions at offsets
`97,194,291,388` from 2027 modulo 2304 and four seeded Gaussian
directions. Each receives the same per-prompt raw input and 8%
physical dose. These physical vectors differ in direction after
projection. Source-angle and delivered-dose metrics are recorded
after cast for each of the twelve repeated reader calls.

Arms: **unhooked**, **zero hook**, **candidate**, and eight shams.
The primary engineering treatment changes the previous block-3 state
at cross-loop pass index **1** only; pass index 2 recomputes from
the modified recurrent continuation. A separately marked all-pass
variant is available in the hook API for future diagnostics, but
is not part of the registered first-development runner. Repeated
reader calls and sham directions are dependent observations from
one prompt/root.

Probe the four Q/K/V/MoE beta3 routing weights at layer index **11**
(the last looped layer, physical layer 17), for each eligible pass.
Record their changes before task readout and ensure full native
message parity.

## Development data and outcomes

Use an explicit frozen UTF-8 JSONL task file, each line carrying:

```json
{"phase":"development","root_id":"dev-arith-001","family_id":"arithmetic","prompt":"Question: 2 + 2 =\nAnswer:","correct":" 4","distractor":" 5"}
```

The tokenizer must map each candidate answer to exactly one token.
One root is one complete independent prompt-generation root; shared
templates/generator families are tracked with `family_id`.
Prompt text is an exposed development artifact. Any invalid label,
duplicate root, out-of-range token ID or unadmitted physical dose
fails the execution rather than filtering cases. All assigned arms
and failures must be retained in the report/CI logs.

The runner scores:
1. clean full-vocabulary top-1 correctness and corresponding
   restricted correct-versus-distractor one-token margin;
2. candidate and eight sham margins, fullhead top-1 successes,
   answer corruption/repair and per-root direction effects;
3. Q/K/V/MoE gate probabilities, delivered physical L2 and RMS,
   native message parity and source hashes.

The initial engineering batch is **at most 16 development roots**,
one fixed checkpoint and eight within-root sham directions. The clean
full-head top-1 competence gate is **75%** on this batch, with every root
retained. Failure means `FAIL_CLEAN_COMPETENCE`, regardless of a
large gate or logit movement. The restricted two-token task may be
insufficiently difficult and is a preflight instrument only.

Scientific aim for a *separate* prospectively registered experiment:
a candidate-specific corruption increase of at least 5 percentage
points over the matched-sham family on independently competent,
verifier-backed serial tasks. Effective minimum is the maximum of
5pp, measured independent-root sham-tail floor and decision
resolution. Source pilot metrics here cannot establish that floor.
Run the pinned EvalMath mechinterp-power preflight on independent
development-unit effects and matched sham variation **before**
opening a fresh sealed confirmation cohort. Calibrating a 95th
percentile with 95% confidence requires at least 59 independent
sham roots even via the sample maximum. Use independently trained
weight samples if the claim is generalization across models.

**Current scientific verdict:** ENGINEERING_ONLY. At most 16 task
roots sharing one checkpoint cannot authorize source-to-confirmation
inference, equivalence, causal transfer, or generalization.

## Offline execution

Run only on a host where the official full checkpoint is already
present and where local device memory/offload supports native
Transformers forward hooks. A GGUF inference API does not expose
these internal raw block tensors. The published checkpoint is
approximately 20 GB; official guidance recommends 32 GB+ VRAM
for inference. Source code loads no weights during CI.

Install the pinned Transformers source and dependencies in a
dedicated environment, and independently provision the licensed
checkpoint files into `/path/to/Inkling-2-Mini`. Check that the
installed modeling file matches the registered Git blob. The
following command reads only local files and hashes every local
`*.safetensors` shard:

```sh
python -m pip install "git+https://github.com/huggingface/transformers.git@046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3" accelerate safetensors
python mechinterp/bpd/experiments/inkling_routing/run_pretrained_dev.py \
  --model-dir /path/to/Inkling-2-Mini \
  --tasks /path/to/frozen_development.jsonl \
  --out /path/to/inkbpd05_development.json \
  --max-roots 4
```

Checkpoint load uses `local_files_only=True`,
`trust_remote_code=False`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`. No credentials, remote inference,
GPU allocation, checkpoint download or training are triggered by
repository CI. All full checkpoint executions are user-hosted,
development-only and must preserve hardware/software receipts.

## Tests

```sh
python -m unittest discover -s mechinterp/bpd/experiments/inkling_routing -p 'test_pretrained_*.py' -v
```

This suite uses a tiny synthetic message-attention fixture, verifying
that a single previous block-3 write is patched consistently across
each pass's 12 reader invocations. It also verifies local source-hash
generation, sealed-phase rejection, per-file weight manifest
generation, and invalid-shape failure.

## Scientific caveat

The hook is a genuine potential *native-model causal intervention*
when executed on pretrained weights. The added code and synthetic
fixture constitute source/engineering validation until such an
execution supplies full-shard hashes, source identity, clean/zero
parity, complete control admissions, and native outputs. The
Goodfire observational outlier remains a candidate.
