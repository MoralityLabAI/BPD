# MMS native RP03 follow-on (development only)

This adds an opt-in memory lane beside the frozen RP03 protocol. It does not
change RP02/RP03 source locks or their historical results. The native shared
operator executes live V/U factors with all gates open and residual zero.
Cache detached BF16 physical output gradients of prior training observations,
using a quantized full-input signature as the key. Read and write factors
select current components. No extra rollout or auxiliary autograd pass.
This is historical cotangent feedback, not the prior causal state-patch loss.
Do not interpret a selected factor as a causal memory role without prospective
interventions and calibrated nulls.

Run one fresh development root and one arm per process:
```
python -m mechinterp.bpd.experiments.training_steering_stability.mms_run --out runs/mms_28101_bpd --root 28101 --threads 2 --mms-mib 4 --mms-mode bpd --cpu-seconds 60
```
Repeat with --mms-mib 0, and positive-budget observe/dense/sham controls.
Same warmup, fixed-delay training stream and independent dev readout. The CPU
limit covers the post-warmup update loop and is a soft stopping boundary;
account for warmup, evaluation, peak RSS and process CPU externally. Run
sequentially. The small symbolic input support saturates early: actual
occupancy, cache hits, drift and time must justify any larger budget. A
positive result requires the EvalMath mechanistic-resolution preflight and
new sealed confirmation protocol. The rotated-coordinate sham is a weak
engineering control. Never use the historical sealed cohorts.
