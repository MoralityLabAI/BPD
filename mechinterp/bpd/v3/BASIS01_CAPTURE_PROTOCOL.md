# BASIS-01-MODULES: preregistered native input-capture instrument
Date: 2026-11-17, before any pretrained TRM activation capture.

## Purpose and evidence class

Exports fixed, nonadaptive input vectors to the four tied 7M-TRM modules
`layers.{0,1}.{o_proj,down_proj}` at *every one* of the 63 recurrent core
invocations (three outer ACT steps, period 7). This is the first native-input
step of [BPD-v3 BASIS-01](REGISTRATION.md). The previous synthetic basis result
has **no** evidential status for natural model activation sparsity.

The native exporter requires exact local `gaoxin492` or `yagizdevre`
checkpoint bytes verified against `TRM_CHECKPOINTS`. Never download model
weights automatically. Read only the explicit 183-WB-Stall local NPZ generated
by `prepare_native_roster.py`, which originates in the existing *public*
`LOOPSCOPE-02` benchmark. Require its externally pinned SHA-256. Do not open
the protected ARG-03-N test corpus.

## Capture, sampling, and data splits

For each independent board in order, execute clean target recurrence for
`tau=0..62`, with no gates, intervention, noise, future-state observer, or
learned dictionary. Forward **pre-hooks** on the four physical linear modules
capture the actual input activation `x_{b,tau,j}` before matrix multiplication,
with exactly one invocation per module per core call.

Sample `P=2` distinct *board-token* positions from the 81 cells per board and
call (prefix positions excluded). This is a fixed RNG schedule, seeded from
the registration seed `20261117`, board index and call index, independent of
the vector value and model outcome. Use the identical sampled positions for
all four modules, preserving `board_idx`, `pass_idx`, `call_type=tau mod 7`
and absolute `token_pos` for every row. All seven call phases occur equally
often within each 63-step run. A hook count or module shape mismatch aborts.

Write one NPZ per physical matrix, with `W [d_out,d_in]`, `inputs [B*63*2,d_in]`,
`board_idx`, `pass_idx`, `call_type`, `token_pos`, source-checkpoint SHA,
roster SHA and registry metadata. Export float32 copies of native pre-linear
inputs and float32 weights with original dtype documented. Include per-file
SHA-256 and a manifest. Do not claim that numeric exact zeros in bfloat16
have the same meaning as zeros in an ideal real-valued activation distribution.

The dataset is **DEVELOPMENT_ONLY** because these 183 roots have already
been used in other TRM audits. For future *learned* complete dictionary
claims, register a new independent board-level split and test on a fresh
roster prior to fitting. Tokens, passes and both checkpoints with the same
board root are repeated observations, not independent samples.

## Prospective gates

1. Exactly `63` pre-hook calls per physical matrix per board; `2` distinct
   board positions per call, and identical positions across four modules.
2. Each `inputs` matrix has the same second dimension as `W.shape[1]`.
   All arrays are finite; `tau mod 7` matches the recorded call type.
3. Checkpoint and roster SHA-256 pinned before the first rollout and written
   to every output manifest; full-on factorization must reconstruct `W`.
4. Report the natural exact-zero fraction and thresholded near-zero fractions
   separately by phase. State any basis advantage as a developmental finding
   until independently replicated.
5. Run the registered streaming `score_exported_basis.py --matrix-npz` on the exported arrays,
   comparing neuron/SVD/random at fixed C within each family and reporting
   module-output certificate accuracy before any downstream task claim.

Source-only CPU fake-module tests are authorized immediately after this commit.
Native model evaluation is still PENDING.

## Native PowerShell runbook

From the BPD repository root, first run the roster-preparation command in
`NATIVE_COLLECTION_PROTOCOL.md` and review its output manifest. Use its
pinned roster SHA for the complete clean-activation export:

```powershell
$rosterSHA = (Get-Content runs/bpd_v3/wb_stall_inputs.manifest.json | ConvertFrom-Json).roster_npz_sha256
python mechinterp/bpd/v3/collect_basis_inputs.py `
  --checkpoint-file "C:/PATH/TO/step_21700" --checkpoint-key gaoxin492 `
  --inputs-npz runs/bpd_v3/wb_stall_inputs.npz `
  --expected-roster-sha256 $rosterSHA `
  --out-dir runs/bpd_v3/gaoxin_basis_inputs `
  --device cuda:0 --dtype bf16 --positions-per-call 2
python mechinterp/bpd/v3/score_exported_basis.py `
  --matrix-npz runs/bpd_v3/gaoxin_basis_inputs/layers_0_down_proj.npz `
  --chunk-size 256 `
  --out runs/bpd_v3/gaoxin_basis0_baselines.json
```

For the pinned replication checkpoint run a second exporter with
`--checkpoint-key yagizdevre --checkpoint-file "C:/PATH/TO/model.pt"`
and a separate output directory. Inspect the `manifest.json` for the
per-call-type exact-zero fractions, file hashes and input matrix shapes.
A source-only fake-model test runs with:

```bash
python -m pytest -q mechinterp/bpd/tests/test_v3_basis_inputs.py
```

No checkpoint or test data are downloaded by either exporter.

## BASIS-01-SCALE scoring amendment

The source-only scorer from [BASIS01_SCALE_REGISTRATION.md](BASIS01_SCALE_REGISTRATION.md) retains the original `floor_and_basis.py` as an independent numerical oracle. For full `[512,1536]` down projections and 23,058 vectors per module, use `score_exported_basis.py`; it processes vectors in bounded chunks and scores a full SVD once per matrix. The full right-SVD basis has 1,024 numerical null-write directions for a full-row-rank `down_proj` matrix. Report near-null write count and mean active non-null writes alongside raw active-component count. Those directions are a property of the weight matrix and do not imply semantic feature sparsity.

The sampled vectors share 183 board roots. The scorer reports per-phase mean active counts, but those 23,058 vector observations are repeated measurements within boards.

## One-command RTX launch

The orchestrator handles the SHA-pinned roster, four physical module exports, and six basis comparisons per matrix. From a PowerShell prompt in the BPD checkout:

```powershell
pwsh -NoProfile -File mechinterp/bpd/v3/run_basis01.ps1 `
  -CheckpointKey gaoxin492 `
  -CheckpointFile "C:/PATH/TO/step_21700" `
  -MaxBoards 2
```

The two-board smoke is **development-only**. For the complete registered 183-board cohort, omit `-MaxBoards 2`. Repeat with `-CheckpointKey yagizdevre` and its SHA-pinned `model.pt`. The script reads only the public LOOPSCOPE-02 benchmark bundle and explicitly supplied local checkpoint. Its output remains in `runs/bpd_v3`.

The core scorer has a fixed `--chunk-size 256` default and can use `--family input` or `--family output` when independently auditing one factorization family. Full registered comparisons keep the random control and both families. The output's `mean_logit_kl` is a module-output softmax diagnostic; it is **not** the actual Sudoku digit-head KL after the recurrent model. Its `near_null_write_directions` field uses a documented numerical threshold, which is separate from exact-zero activation support.
