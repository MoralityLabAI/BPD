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
5. Run the existing `floor_and_basis.py --matrix-npz` on the exported arrays,
   comparing neuron/SVD/random at fixed C within each family and reporting
   module-output certificate accuracy before any downstream task claim.

Source-only CPU fake-module tests are authorized immediately after this commit.
Native model evaluation is still PENDING.
