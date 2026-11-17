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

## Commands from the BPD repository root

Prepare the exact public LOOPSCOPE-02 WB-Stall roster; the script asserts 183 unique roots and writes its SHA manifest:

```bash
python mechinterp/bpd/v3/prepare_native_roster.py \
  --bundle-json mechinterp/loopscope02/results/benchmark_bundle.json \
  --out runs/bpd_v3/wb_stall_inputs.npz
```

Source-level sample for the local RTX checkpoint, labeled DEVELOPMENT_ONLY and limited to two boards:

```bash
python mechinterp/bpd/v3/collect_native_dither.py \
  --checkpoint-file /LOCAL/PINNED/step_21700 --checkpoint-key gaoxin492 \
  --inputs-npz runs/bpd_v3/wb_stall_inputs.npz \
  --out runs/bpd_v3/gaoxin_dither_dev2.npz \
  --device cuda:0 --dtype bf16 --k 8 --max-boards 2
python mechinterp/bpd/v3/floor_and_basis.py \
  --target-npz runs/bpd_v3/gaoxin_dither_dev2.npz \
  --out runs/bpd_v3/gaoxin_dither_dev2_floor.json
```

After inspecting that manifest and the zero-dose and clean-repeat controls, remove `--max-boards` for the full 183-root development cohort. A full cohort now requires an externally pinned roster hash; pass `--expected-roster-sha256` with the `roster_npz_sha256` from the reviewed `wb_stall_inputs.manifest.json`. This proves that the roster bytes stayed fixed between review and collection. Retain the benchmark-bundle hash in the manifest to document the source.

For PowerShell on the RTX workstation:

```powershell
$rosterSHA = (Get-Content runs/bpd_v3/wb_stall_inputs.manifest.json | ConvertFrom-Json).roster_npz_sha256
python mechinterp/bpd/v3/collect_native_dither.py `
  --checkpoint-file "C:/PATH/TO/step_21700" --checkpoint-key gaoxin492 `
  --inputs-npz runs/bpd_v3/wb_stall_inputs.npz `
  --expected-roster-sha256 $rosterSHA `
  --out runs/bpd_v3/gaoxin_dither_183_k8.npz --device cuda:0 --dtype bf16 --k 8
``` Repeat with the independently pinned `yagizdevre` `model.pt` checkpoint. Retain each full tensor and sidecar SHA manifest. The `--inputs-npz` file must be copied locally into the user's compute environment; cloud CI does not fetch checkpoint binaries.

To compute board-cluster uncertainty from an emitted NPZ:

```python
import numpy as np
from mechinterp.bpd.v3.uncertainty import board_js, bootstrap_board_means

with np.load("runs/bpd_v3/gaoxin_dither_dev2.npz", allow_pickle=False) as d:
    result = bootstrap_board_means(board_js(d["target_logits"]))
print(result)  # Development only: B=2 is unpowered
```

The Gaussian/rounding-size calculation supplies a perturbation-scale prior. A completed K=8 injected-dither experiment must be reported as a stochastic intervention on state, with its law and scale explicitly recorded. It cannot serve as a measurement of the actual physical distribution of `bfloat16` GPU roundoff.
