# AMM-02: cross-trained-recurrence BPD strategy-memory transfer

**Pre-data protocol registered 2026-11-18. Status: prospective source-only.**
This experiment extends AMM-01R's engineered positive instrument. AMM-02 is a
harder test of whether a strategy learned on *independently trained model
roots*, rather than a known analytic family routing law, transfers to new
trained recurrent BPD decompositions.

## Independent behavioral problem

Each independent root trains a 4D, rank-two, weight-tied physical recurrent
linear model `F_W(h)=h W^T` with `W=(I O)^T` and separately initialized
trainable factors I[4,2], O[2,4]. Inputs are `h0=(0,0,a,b)` with
a,b independently uniform [-0.8,0.8]. Two human-specified tasks, hidden
from the decomposition fitting loss except as teacher trajectories:

- family `wren`: `q(h1)=a`, `q(h2)=b`;
- family `basalt`: `q(h1)=b`, `q(h2)=a`.

Here `q(h)=h[0]`. Task labels and the donor's target value are defined from
a,b and the family **before** any model or BPD factor fitting. The exact
teacher uses 200 Adam steps at LR .035 with batch 96 and 0.0002 weight
penalty. Separate rank-two BPD factors V[4,2], U[2,4] train for 140
Adam steps on independent input trajectories and physical weight MSE at
LR .035. Freeze the learned V,U, then derive canonical rank-two component
coordinates from the SVD of VU in float64; report singular gap and residual.
All model initialization, data and factor fitting are independent by root.

## Donor assay and action budget

Every root receives 48 generated donor-recipient pairs, sharing the input
coordinate that is NOT the family's *second-pass* target, and displacing
that target coordinate by a signed 0.30..0.45. The external target is the
donor's new second-pass label. Independently generated test pairs are never
used to train teacher or factors.

An action is one of two gauge-fixed BPD component indices c=0,1. Compute
the native component contribution from the changed donor input
`z_c=(((h_d-h_r) V)_c) U_c`. Project it to the kernel of the *current*
readout (set coordinate zero to zero); normalize it to the same fixed
per-pair dose `D=||F_BPD(h_d)-F_BPD(h_r)||_2` for every nonzero-dose
arm. If projected z_c vanishes, the registered fallback is e1*D and the
degenerate pair is retained and flagged. Inject this current-dark dose
into the recipient after pass 1; run the **learned** shared BPD recurrent
operator for pass 2. Success is `|q(h2_patched)-donor_target|<=0.18`
for *every generated pair* (intent-to-treat). Record clean eligibility,
teacher/factor behavior, full donor intermediate replacement and
no-patch sham as diagnostics, without censoring denominators.

Action policies are frozen after cross-root training:
- `memory`: choose the component with greatest donor-label recovery on
  independent training roots sharing the opaque family label;
- `pooled`: choose one globally best component from exactly the same
  training observations, independent of family;
- `swapped`: apply the other family's learned component choice;
- `random`: choose 0/1 from a phase- and pair-specific hashed coin;
- `random_dark`: one physically equal-dose isotropic dark vector,
  independent of component choice (strong causal baseline);
- `sham`: no patch, logs incidental successes.

All active interventions except sham and full donor have exactly the
same `D` and current readout preserved, with max relative norm error
<=1e-5. The signed effect is memory-hit-rate minus pooled-hit-rate,
averaged across independently trained roots. Training-world case scores
may inspect both candidate actions to learn a policy; each test agent
selects exactly one component without test labels or feedback.

## Separation, fixed counts and hard admission

Use SHA256-derived phase-specific roots `AMM02/{phase}/{index}`.
Allocate 64 TRAIN roots, 64 development roots, and 128 SEALED
confirmation roots. Each cohort is family-balanced by index parity.
Eight excluded `engineering` roots are available for unit tests,
implementation smoke and runtime profiling. A test must never evaluate
any `confirmation` root: generators and confirmation scoring require
a successful external strict preflight receipt *before* any access.
Never access ARG-03-N or other protected splits.

Each root is independently initialized and trained. 48 repeated
donor pairs within a root count as repeated observations, not
independent worlds. Preserve all failures, zero-dose pairs and source
hashes. A teacher's loss, final independent-task MSE and factor
reconstruction residual are required diagnostics. The mechanism is
a trained **synthetic linear** recurrence, not a pretrained TRM/RMP.

## EvalMath gates

Freeze the primary family-keyed-memory minus pooled accuracy effect at
+0.12 absolute probability, target +0.20, resolution 1/48,
familywise alpha=.05 across **three** prespecified positive comparisons:
memory-pooled, memory-swapped, and memory-random-dark. The random
component arm is reported but excluded from the planned inferential
family. Use root-level one-sided exact sign tests, 2,000 seeded paired
root bootstrap intervals, and Holm adjustment across three comparisons.

On independent development roots, run the pinned EvalMath
`skills/evalmath-mechinterp-power/scripts/preflight.py --strict`,
using all 64 root-level memory-pooled differences and per-root
`pooled - random_dark` as the matched sham variation source. Use
q=.95, 95% confidence (sample-maximum rule possible with >=59 roots).
The scientific floor is max(0.12, 95/95 sham floor, 1/48).
Normal-power estimates are planning only. If preflight says
`REDESIGN_OR_PILOT_ONLY`, confirmation MUST NOT be evaluated; publish
a development-only negative/power-limited receipt. If accepted, score
128 new model roots once. A positive claim additionally requires
Holm-adjusted p<.05, root-bootstrap primary 95% lower bound above
the mechanistic floor, and physical-dose/teacher/factor integrity gates.

## Engineering/identifiability checks

Verify rank-two learned physical W, both teacher factors updated, both
BPD factors updated, independent task label source, paired dose
matching, zero current-readout displacement, and exact no-patch behavior.
Check gauge invariance of full VU under a fixed nonsingular GL2 shear
and SVD re-canonicalization, including component patch directions
when the singular gap exceeds .01. Provide BPDModel/TiedComponentLinear
PassMaskSchedule native parity test on one excluded engineering root.
Native and independent forward results must agree <=1e-5. Any
degenerate/gauge failures remain in receipt.

## Evidence and outputs

Commit this protocol first, then implementation/test source prior to
any development scoring. Freeze decisions before confirmation and bind
training-policy, sources, development root data and EvalMath preflight
to reproducible receipts. Keep the historical AMM-01R data intact.

A positive result establishes transfer of **fixed component-index
intervention strategy** across independent trained synthetic BPD
systems under these disclosed role-family cues. It does not establish
learned mechanistic concepts across model architectures, agent LLM
memory, generalization to nonlinear transformers or Buzz deployment.
