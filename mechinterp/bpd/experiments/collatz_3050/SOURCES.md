# Model sources and validation scope

## Implemented checkpoint

- Repository: [Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B).
- Immutable revision: [`451ed48c3273ecef7ea8faaa43c31ce529763bb1`](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/tree/451ed48c3273ecef7ea8faaa43c31ce529763bb1).
- [Configuration](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/451ed48c3273ecef7ea8faaa43c31ce529763bb1/config.json).
- [Native recurrence](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/451ed48c3273ecef7ea8faaa43c31ce529763bb1/modeling_nanbeige.py).
- [Configuration class](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/451ed48c3273ecef7ea8faaa43c31ce529763bb1/configuration_nanbeige.py).
- [Checkpoint index](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/451ed48c3273ecef7ea8faaa43c31ce529763bb1/model.safetensors.index.json).

`streaming.py` contains the complete source, tokenizer and LFS weight hashes.
Production construction checks the architecture and exact parameter inventory,
verifies both complete weight shards with a bounded hashing buffer, and loads
the pinned executable source. Model weights stay outside Git. The public model
card supplies its usage and license terms.

The released configuration uses 22 physical decoder layers with hidden width
3072 and two recurrent passes. Its vocabulary has 166,144 rows, and the
embedding and head are untied. The checkpoint contains 4,169,800,704 parameters
including those tables. The decoder and final norm account for approximately
3.15 billion parameters; the two vocabulary tables contribute approximately
1.02 billion. The gym's resource estimate uses the complete inventory.

The native code applies the same final norm after each loop when
`skip_loop_final_norm=False`. That normalized tensor becomes the next loop's
input. The adapter observes that actual boundary through a hook. It executes
the checkpoint's trained recurrence with the native attention path and
normalization schedule.

## Streaming and runtime

[Safetensors partial tensor access](https://huggingface.co/docs/safetensors/index)
provides the source for row and layer reads. Short-lived mappings supply copied
rows or a single decoder's parameters, and completed layers return to `meta`.
The full embedding and head tables are never assembled in the running backend.

[PyTorch's official previous-version instructions](https://pytorch.org/get-started/previous-versions/)
list Torch 2.6.0 wheels for CUDA 12.4 on Linux and Windows. The Windows launcher
uses that release and the dependencies pinned in `requirements.txt`. CI uses
the corresponding Torch 2.6.0 CPU wheel for source validation.

`fetch-test-source` downloads the pinned architecture and tokenizer assets,
checks each SHA-256 and keeps every download below a 64-MiB asset limit. Those
assets allow the full test suite to run with tiny local random weights. It
downloads zero pretrained checkpoint shards.

The CPU suite checks the unchanged official implementation, including native
BF16 construction and FP32 rotary frequencies. It compares full and selected
vocabulary scores, verifies intervention continuation, runs all eight forwards
of a case, round-trips saved tensors and rejects altered scoring provenance.
Real-tokenizer tests verify continuation boundaries and matched donor lengths.

These tests establish code-path parity and instrument behavior on the tiny
fixture. The full checkpoint's task outcomes and CUDA resource behavior require
their own recorded run. The memory receipt reports its allocator and sampling
scope explicitly.

## Audited alternate: Dory

[NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
was also inspected at revision
`ee517dd388a9b348890ae1e00dacc5eb4d0c750d` using its released configuration and
the official [Transformers Dory documentation](https://huggingface.co/docs/transformers/main/en/model_doc/dory).
Its recurrence uses distinct layer pairing, input-state reinjection and
normalization, and its head applies a learned logit scale. Those features
require a dedicated adapter and native parity tests.

The shipped backend accepts the pinned Nanbeige checkpoint. Dory NVFP4
acceleration on the RTX 3050 is outside the implemented runtime contract.

## Exact task oracle

The finite split-Collatz core is the unchanged COL-01 copy from
[MathCan commit b3eb33ed10485264fe7a593e3b0ace418d9f9a6d](https://github.com/patrickdugan/MathCan/tree/b3eb33ed10485264fe7a593e3b0ace418d9f9a6d/research/collatz_resonance).
Its file SHA-256 is
`24ff52171aedf890c33439896ab5e71de65d9b05d557ba61d162f3623aa5dbca`.
`cases.py` verifies that source before generating or validating cases. This
finite family supplies exact task labels and donor relationships. Its
mathematical claims concern the declared finite maps.

The scoring design follows the independent-root and matched-sham checks in
[EvalMath's mechanistic-power skill](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power).
The 0.05 probability effect scale is a prospective design choice. The code
reports a null calibrated sham floor and a pilot-only verdict for this exposed
development population.
