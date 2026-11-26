# RP03 engineering extension: continue the exposed learning curve to 6,000 updates

This plan was written after the two prescribed 2,000-update engineering runs.
Those results and sources remain unchanged. The extension uses the same exposed
roots **27000 and 27001** to investigate the ordinary-learning limitation found
there. The prior `REDESIGN_OR_PILOT_ONLY` scope still applies.

## Fixed continuation

Load each saved `results/root_<root>/checkpoint_2000.pt`, including its exact
model tensors and AdamW optimizer state. Continue the same task-only recipe
through **6,000 total updates**. Use the original RP03 engineering training seed
domain and the next stream indexes, 2000 through 5999. Preserve the delay cycle,
learning rate, weight decay, batch size and gradient clipping. No optimizer
reset, new model root or steering-training treatment belongs to this extension.

Save checkpoints at **3,000 and 6,000 total updates**. Both roots run through
6,000 regardless of the intermediate result. Record cross-entropy every 50
updates. Bind the parent checkpoint hashes and all executed sources in receipts.

## Fixed diagnostics

At both new checkpoints, repeat the 512-pair ordinary/causal diagnostics at
delays 2 and 7, together with the 128-base, four-query address-binding diagnostic.
The address-binding modal baseline stays at exactly 25%. Every example remains
in the TRAIN semantic partition. Diagnostic seed purpose strings begin with
`extension_`, keeping these draws distinct from the completed run.

At each new checkpoint, apply the unchanged RP02 eight-group discovery pattern
on 128 training calibration pairs at delay 2. Preserve the entire score matrix.
The subsequent causal diagnostics use recurrent-write L2 dose 0.75.

## Prespecified choice for a later recipe

Candidate budgets are **3,000 and 6,000** updates. Choose the earliest candidate
for which **both roots at both delays** meet both requirements:

- Ordinary TRAIN accuracy is at least **80%**.
- All-distinct address-binding accuracy is at least **75%**.

If neither candidate passes, retain 6,000 with an explicit
`WEAK_COMPETENCE` flag. Every checkpoint and every failed gate remains in the
record. This budget choice uses exposed engineering roots; it supplies no
held-out estimate or confirmation claim. Causal transport results remain
separate diagnostics and do not alter this fixed competence rule.
