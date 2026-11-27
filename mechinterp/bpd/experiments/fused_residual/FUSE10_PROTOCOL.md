# FUSE-10 | Clean pass-mean calibration residual reuse

Status: proposed, default off, CPU FP32 only. Stacked on FUSE-09 1ee5a870c25e88ba6cafa07f349297498daf945f.

## Source audit
Native TiedComponentLinear.observe_clean_pass calls weight_delta once per pass
for mean_delta_out. Target forward and override_read Haar rotations do not
request a residual; do not cache or count these rotations. Trainer calls
run_clean_rollout(update_means=True) both in calibrate_baselines and in the
ordinary training clean capture. Cache only inside that target_capture.
Preserve EMA ordering, buffer growth, rotations, target forward, pass offsets,
and independent gradient-bearing rollouts. Use original weight_delta with no
algebraic reassociation. Drop cache before leaving capture; no cross-capture
reuse, no overlap with training step pool. No mutation of W_target, V or U in
the scope, no raw storage writes, reentrancy or concurrent model users.

For m sites, T calls each and k selected, expected reconstruction count is
m*T-k*(T-1). Planner uses saved reconstruction MACs and byte cap, not timing.
Zero cap, single pass, unsupported precision, inference or grad mode, and
frontier overflow remain native. Selected storage only, not total RSS.
Mutation detection raises; finally restores native methods.

## Prospective gates
- CPU float32 exact torch.equal state, tolerance dictionaries, clean and
  evaluation readouts/boundaries, CI outputs, training metrics, optimizer
  state, PGD state, component/CI gradients, RNG state; finite values required.
- Four width-64 six-pass matched campaigns, two updates each, fresh roots
  4010001..4010004. Both closed loop and clean window; binding/full caps;
  calibration-only and same FUSE09 PGD cache enabled in both arms.
  Force tolerance calibration on both steps, one rotation; preserve native
  EMA updates. Verify two separate scopes per step, exact hit/miss counts,
  selected bytes, no outstanding storage or patched methods after scope.
- Direct lifecycle tests: grad/inference/autocast fallback, zero cap,
  one pass, parameter mutation rejection and exception cleanup.
- Record paired train-step wall times as descriptive, no performance claim
  until hosted gate and wider matched runs. Later widths 512/1024,
  passes 6/12, rotations 1/4, calibration cadence 1/50/off; separate clean
  timing from overall step timing. GPU, mixed precision, real checkpoints,
  total-memory claims and production promotion remain HOLD.

## FUSE-09 late receipt
All ten wide jobs and the native CPU gate completed successfully.
The two late width-1024, 12-pass closed-loop jobs add 16 campaigns, 32
paired steps, 10016 exact tensors, 4464 scalar checks, 128 greedy checks.
Totals across wide campaign: 76 campaigns, 152 paired updates,
45592 exact tensor checks, 20820 scalar checks, 480 greedy checks.
PGD ratios (sum of paired reference durations / sum of cached durations):
binding 15 MiB 1.209350x; full 30 MiB 1.492948x.
Source: run 46006681126, jobs 140532721207 and 140532721214.
See FUSE09_LATE_RECEIPT.json for original log-derived rows.
Raw artifact IDs 15202253279, 15201628788; archives not independently unpacked.
