# FLOOR-01 native collector: deterministic noise law and local-only protocol

Registration supplement 2026-11-17, after the separately frozen mathematical evaluator and before any native dither execution. This does not alter the B4 v2 registration.

Source: `mechinterp/bpd/v3/REGISTRATION.md`. Model implementation: `mechinterp/loopscope02/scripts/trm_sudoku.py` pure PyTorch `PureSudokuTRM`. Target checkpoint keys and SHA-256 hashes are those frozen in `TRM_CHECKPOINTS`.

## Data and source admission

Inputs must be provided as an explicit local `.npz` file, with encoded `input_ids [B,81]` (integer tokens 1..10). The intended full roster is the 183 fixed `WB-Stall` source boards; its creation and SHA-manifest are upstream and not part of the collector. Reject duplicate roots and require all 183 boards for cohort reporting. Development `--max-boards` runs are allowed with status `DEVELOPMENT_ONLY`. Do not read or mention the sealed ARG-03-N `test.jsonl`.

Each native checkpoint is an explicit local `--checkpoint-file`; validate the binary file SHA-256 against the checkpoint key before `torch.load`. Never invoke the Hugging Face auto-downloader.

## Arithmetic dither law

Run the exact `l_cycles=6` + one `z_H` call period of the pretrained `TRM` for 63 calls at one board per batch. Record digit readouts exactly at pass 20, 41 and 62. At every call, perturb **only the just-updated recurrent buffer**, in float32 before converting back to the model dtype:

`state_next = cast_dtype(state_next_fp32 + U[-sqrt(3)*sigma*rms, +sqrt(3)*sigma*rms])`

where `rms` is the per-token RMS of the updated 512-dimensional vector, and `sigma=2^-8/sqrt(12)` by default. This fixes a bounded uniform state-dither law, with the same standard deviation order as bfloat16 arithmetic rounding per unit-RMS coordinate; it is an *injected-noise model*, not a proof that GPU rounding errors are independent uniform random variables. The random stream is seeded deterministically from the global seed, board index and draw index, and recorded in the manifest.

Store `target_logits[B,K,3,81,9]` and `static_logits[B,3,81,9]`; static logits come from a clean, undithered run for each board and are independent of the dither realization. Always preserve the passed input roster SHA and checkpoint SHA. The evaluator computes the finite-K plug-in `J_delta`, board-cluster intervals and the noise-independent risk decomposition. Use K=8 as a developmental floor estimate, K=32 for precision sensitivity.

A `driven_logits` array requires an independently specified **paired driven rollout**; the native collector does not manufacture one from a static clean trajectory. No v2 leakage magnitude or causal-importance gate claim follows from these logits alone.

## Validation gates and execution

Source-only fake-model CPU tests before weights: exact shape, readout timing, reproducible dither seeds, output perturbation, zero-dose equivalence and refusal of nonlocal/unpinned weights. Native admission: verify checkpoint and board hashes, full cohort size, finite logits, digit-class and call-role parity, and record GPU batch size/dtype. A float32-to-float32 repeat baseline and unchanged-model clean rerun are required before numerical branch effects may be attributed to dither.
