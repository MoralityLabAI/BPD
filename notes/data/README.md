# BPD Round 2 (`v2`) Receipts, Weights, Diagnostics, and Consolidated Fable Review

All artifacts from the Round 2 (`v2`) gate runs (`registration_commit = 180f1bcbbc633a16bfd887a9593252cefd7a6a65`), the debugger-extracted `Gate B4 v2` Stage `1a–4` `(V, U)` factors, the zero-fit / zero-CI diagnostics, and the consolidated Fable review exchange.

## Contents

- **[FABLE_CONSOLIDATED_REVIEW_AND_DIAGNOSTICS_2026-11-17.md](./FABLE_CONSOLIDATED_REVIEW_AND_DIAGNOSTICS_2026-11-17.md)** — Complete chronological record of all five Fable review passes, mutual code/data corrections, and the `v3` toy-first design checklist.
- **[STATUS.md](./STATUS.md)** — `v2` gate verdict ledger (`B1: PASS`, `B2: FAIL`, `B3: FAIL`, `B4: INCOMPLETE`).
- **[registration_amendment_v2.md](./registration_amendment_v2.md)** — Locked Amendment `v2` pre-registration (`180f1bcbbc633a16bfd887a9593252cefd7a6a65`).
- **Gate Receipts (`v2`):**
  - `gate_b1_v2_receipt.json` — Gate B1 `v2` (`PASS`)
  - `gate_b2_v2_receipt.json` — Gate B2 `v2` (`FAIL`: 8-factor negative arm passed; `0.5x–4x` natural-scale `sigma` sweep failed under Haar rotation null)
  - `gate_b3_v2_receipt.json` — Gate B3 `v2` (`FAIL`: nested clock warm-start passed; full-rank `s1,s2 > s3` visibility gap and `sigma` sweep failed)
  - `gate_b4_v2_stage4_receipt.json` — Gate B4 `v2` (`INCOMPLETE`: live Stage `1a–4` metrics exported after Stage 4 completion)
  - `gate_b4_v2_stage4_telemetry.json` — Step-by-step `240`-step training logs and wall-clock stage durations (`41,255.7s`)
- **Saved `(V, U)` Factor Weights:**
  - `bpd_b4_v2_stage4_weights.pt` (`25,171,817` bytes = `24.01 MiB`, SHA-256 `65ecb2ab6c4a144b3563cecf49d6a55b157201fa98ef0b6dc652ff3effc33cbc`) — `float32` `V [d_in, 512]` and `U [512, d_out]` matrices for `bpd_ref` (`gaoxin492`) and `bpd_tgt` (`yagizdevre`), where `W_hat = (V @ U).T`; `ci_net` weights are not present.
- **Post-Hoc Zero-Fit / Zero-CI Diagnostic Receipts & Scripts:**
  - `gate_b4_v2_nofit_diagnostics.json` — Exact `||\Delta W||_F / ||W_target||_F`, `comp_mask = 1 / delta_mask = 0` cancellation check (`3.94` & `12.18` nats at Pass 20), shared-direction input variance, and 60% SVD baselines.
  - `gate_b4_v2_controls_and_ablation_split.json` — Per-board `float32`-vs-`bfloat16` and SVD-all-on numerical reproducibility controls (`Pass 20, 41, 62`), code verification of `BPDTrainer.evaluate()` clean-trajectory gate steering, and genuine `2×2` `(zero vs pass_mean) × (sphere_rescale False vs True)` factorial split.
  - `scripts/extract_stage4_from_pid.py`
  - `scripts/audit_b4_v2_nofit_checks.py`
  - `scripts/audit_b4_v2_controls_and_ablation_split.py`
