# Tesseract Desktop Runner Integration (`BPD-DT05`)

Because no external Tesseract Rust desktop runner repository or canonical JSON schema was present on this host or in the public `MoralityLabAI` GitHub catalog at sprint time, `BPD-DT05` ships an explicitly versioned **provisional adapter contract** (`Tesseract/gym-manifest-provisional/v1`) in [`tesseract.py`](tesseract.py) and [`tesseract_manifest.json`](tesseract_manifest.json).

## 1. Generating or Validating the Manifest

```powershell
python -m mechinterp.bpd.experiments.depth_steering_gym.dt05.runner tesseract-manifest `
  --device cuda:0 `
  --cpu-threads 8 `
  --dt04-train-microbatch 128 `
  --full-7m-infer-microbatch 64 `
  --full-7m-train-microbatch 32 `
  --out mechinterp/bpd/experiments/depth_steering_gym/dt05/tesseract_manifest.json
```

`tesseract.validate_tesseract_manifest(manifest)` fails closed if:
- `git_revision` is not a 40-character hexadecimal Git SHA or `source_lock_sha256` is missing;
- `HF_HUB_OFFLINE=1` is omitted or `network_egress_permitted` / `weight_autodownload_permitted` is enabled;
- `cohorts.reserved_development_roots.auto_launch_all_prohibited` is not `true`;
- any `"protected"` confirmation workload has `permitted: true` or a nonempty command;
- any `"development"` workload permits automatic retry (`auto_retry_permitted != false` or `max_attempts != 1`) or in-place resume.

## 2. Rust `serde` Field Mapping

| Tesseract Runner Concept | Provisional JSON Path | Semantics & Enforcement |
|---|---|---|
| Repository & Revision | `repository.{url, branch, git_revision, source_lock_sha256}` | Runner verifies `git rev-parse HEAD == git_revision` before launching a `"development"` job. |
| Working Directory | `repository.working_directory` (`"."`) | Commands execute from BPD repository root with `core.autocrlf=false` (`LF` line endings). |
| Command & Arguments | `workloads[*].{command, args}` | Tokenized argv array (no shell interpolation). Template placeholders `${RUN_ID}`, `${FROZEN_PLAN_JSON}`, `${DEV_ROOT}`, `${CAMPAIGN_DIR}` are substituted by the Rust runner. |
| CPU / System RAM / VRAM | `target_hardware_envelope.*` & `workloads[*].resources.*` | `MemoryBudget` enforces sampled host RSS ceiling (`12.0 GiB`), minimum host free reserve (`4.0 GiB`), CUDA allocator ceiling (`12.5 GiB` via `set_per_process_memory_fraction`), and minimum global CUDA free reserve (`1.0 GiB`). |
| Device & Precision | `workloads[*].{device, precision}` | `cuda:0` with `bf16` autocast (FP32 master weights, `allow_tf32=False`) or `fp32`. |
| Logical & Microbatch | `workloads[*].{logical_batch, microbatch, accumulation_steps}` | `logical_batch` is invariant; `microbatch` divides `logical_batch` (`128/128` for `dt04` graphed training, `256/64` for `full_7m` inference, `32/32` for `full_7m` checkpointed training). |
| Timeout & Cancellation | `workloads[*].timeout.{wall_seconds, graceful_cancel_signal, kill_escalation_seconds}` | Send `SIGINT` / `SIGTERM` first; `graceful_cancellation_guard` flushes `RECEIPT.json` with `"status": "CANCELLED"` and exits `1` before `kill_escalation_seconds` expires. |
| Artifact Locations | `workloads[*].artifacts.{output_dir, primary_receipt, attempt_lease, checkpoints}` | Atomic `.tmp` + `fsync` + `replace` JSON receipts; stdout emits `{"event": "TRAIN_UPDATE", ...}` JSONL progress lines followed by final `{"command": ..., "status": ...}`. |
| Retry & Resume Rules | `workloads[*].retry_policy.*` | `"engineering"` jobs may retry only with a fresh `--out` path. `"development"` roots (`2605000..2605058`) create `${CAMPAIGN_DIR}/attempted_${DEV_ROOT}.json` via exclusive `open("x")` before model execution and **must never** be auto-retried or resumed in place. |
| Evidence Scope | `workloads[*].evidence_scope` | `"engineering"` (excluded calibration/smoke), `"development"` (single explicit root from frozen plan after EvalMath preflight), or `"protected"` (`permitted: false`; `2604100..2604355`, `ARG-03-N`, `NC09`, `AMM-03..06` remain sealed). |
