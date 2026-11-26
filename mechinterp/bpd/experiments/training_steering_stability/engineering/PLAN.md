# RP03 engineering: fixed task-learning curve

This plan precedes execution. It uses synthetic training-partition examples to
check whether the RP02 recurrence learns address binding when ordinary task
training continues. The prior planning record is in `../planning/`; its verdict
is `REDESIGN_OR_PILOT_ONLY`. These runs are engineering diagnostics.

## Fixed execution

- Roots: **27000 and 27001**.
- Model and data implementation: the unchanged RP02 `ComponentMemory` and
  `make_pairs`, including native shared BPD V/U factors, residual delta zero,
  width 32, four slots, four values, and root-specific input encodings.
- Updates: **2,000 task-only AdamW updates** per root, with one optimizer carried
  continuously through the run. The loss averages recipient, donor and hybrid
  ordinary labels, as in the RP02 warmup.
- Batch: 48 pairs, giving 144 ordinary trajectories per update. Delays cycle
  through 1, 2 and 3. Noise is 0.03. AdamW uses learning rate 0.003 and weight
  decay 0.0001; gradient clipping uses global L2 threshold 1.0.
- One CPU thread and deterministic PyTorch algorithms.
- Checkpoints: save the model and optimizer after exactly **200, 800 and 2,000**
  updates. Keep all six checkpoints across the two roots.
- Record training cross-entropy every 50 updates and retain source hashes.
- Every random seed is derived from SHA-256 over an explicit
  `BPD-RP-03:engineering` domain, root, purpose, checkpoint/update and delay.

## Fixed diagnostics at each checkpoint

Use new training-partition pairs at delays **2 and 7**. Each ordinary/causal
diagnostic contains **512 pairs**. Record the cohort fingerprint and prediction
bits. Every recipient, donor and hybrid remains in the RP02 TRAIN partition.

The ordinary result runs with steering disabled. A query-blind modal-value
baseline reads the receiver's semantic values on the host and chooses their
most frequent value, resolving ties by the smallest value. Its privileged access
to those values is explicit; the model receives only sequence features.

For the causal diagnostic, reproduce RP02's existing discovery recipe on a
separate **128-pair training calibration batch at delay 2**: evaluate exactly the
eight fixed contiguous groups of four components, retain the complete group by
slot cross-entropy matrix, and select the minimum for each slot. Group choice is
recomputed separately at each checkpoint. This is the already defined RP02
candidate family. Apply each selected group at a recurrent-write L2 dose of
**0.75**, then score targeted and unaffected queries separately at both delays.
Record actual dose, invalid directions and physical replay checks.

### Address-binding diagnostic

At both delays, obtain **128 fresh admitted training receiver sequences** whose
four memory values are all distinct. Expand each sequence to all four late
queries, for **512 cases**. Only the final query-address one-hot changes; write
order, distractors and noise remain common within a four-query group. The root's
address encoding is preserved. Each query appears exactly 128 times and each
semantic answer appears exactly 128 times.

The modal-value predictor scores exactly **25%** on this diagnostic because all
values tie and every stored value is queried once. Ordinary model accuracy here
therefore tests binding values to addresses. These cases stay in the TRAIN
partition and provide no held-out estimate.

## Scope and preservation

The source and recipe stay fixed through both roots. Checkpoint results cannot
change the run length, candidate family, optimization settings or roster.
Retain failures, traces and every checkpoint. No evaluation-partition examples,
new steering training arm or hyperparameter sweep belongs to this run. RP02
sources and receipts remain unchanged.
