# BPD-AMM-04: memory under noisy, calibration-ambiguous causal feedback

Registered **2026-11-18 before any AMM04 training or development scoring**.
Prior AMM-03 (BPD PR #52) measured 7774/8192 correct for both memory+8-pair
local calibration and local-only, with ten posthoc oracle headroom hits
(0.12pp) and zero effect on all 64 independent development worlds. The
AMM03 128-root confirmation population remains sealed and unqueried.
AMM04 uses fresh namespaced roots and a harder *feedback uncertainty*
setting; this is a separate experiment, not a reanalysis of AMM03 test.

## Physical model, independent behavioral outcome and dose

Each independent world fits a new rank-two physical 4D weight-tied recurrent
transition `h -> (h@I)@O`, from random initial I[4,2], O[2,4],
using Adam LR=.035, 200 updates, batch 96 and weight penalty .0002.
External task labels are defined before fitting: family `wren` has
`q(h1)=a, q(h2)=b`, family `basalt` has `q(h1)=b, q(h2)=a`,
for input h0=(0,0,a,b). Freeze teacher W; independently fit V[4,2]
and U[2,4] on source-independent teacher trajectories for 140 Adam steps,
LR=.035 with weight MSE coefficient .5. Canonicalize the trained full VU
product with a double-precision rank-two SVD before any signatures or
interventions. Report teacher and BPD factor errors, singular gap, source SHA.

Donor-recipient pairs share the irrelevant input coordinate, displace the
second-pass label's source coordinate by signed 0.30..0.45 and keep
external donor y2 independent of learned factors. For both candidate
components c=0,1, form the native BPD donor write
`delta_c=(((h_d-h_r)@V)_c) U_c`. Project current readout (coordinate 0)
to zero, normalize to common physical dose
`D=||F_BPD(h_d)-F_BPD(h_r)||_2` per pair. Zero-dose and degeneracy
are counted and retained; fixed fallback is e1*D for a vanished dark
component. Patch after pass 1, continue using the shared trained
recurrent factor operator, and score `|q(h2patched)-donor_y2|<=.18`.
Each test case uses exactly one chosen component; candidate counterfactual
scores and equal-dose random-dark probes are generated offline for auditing
the frozen choices. All test cases enter the ITT denominator.

## Calibration ambiguity (the sole deliberate difficulty change)

Instead of AMM03's eight clean local pairs, give every new root exactly
**four independent local calibration pairs** with both components tested
(8 local labeled interventions). Each reported binary calibration
outcome is independently flipped with fixed probability **0.35** by a
root/arm/case SHA256-keyed judge-noise stream. The agent sees only these
noisy bits, opaque family label, and SVD-canonical geometric signatures.
The independent clean calibration labels may be retained as *audit-only*
truth, never supplied to either decision policy. Compare two strategies
using the *same four pairs and eight noisy feedback bits*, with the same
model-weight access and physical dose. No active data selection on
development or confirmation labels is permitted.

The `local_only` policy chooses the component with more noisy reported
hits; ties resolve toward higher signed one-step delayed causal gain
`r_c=q(F_BPD(direction_c))`, then component 0. The `memory_calibrated`
policy uses a fixed pseudo-count prior weight **4** and each component's
training-derived ridge predicted true success probability:
`score_c=(4 * p_train_c + (hits_c - 4*0.35)/(1-2*0.35)) / 8`.
Clipping `p_train_c` to [0,1] is allowed; the affine correction of
noisy calibration hits is not clipped before scoring. Ties use the same
signed gain then component zero. The judge error probability is known
to both policies; the noise correction preserves the ordering of local
hits. The prior adds amortized cross-root training information, disclosed
separately from the equal **eight-query local budget**.

## Geometry memory and frozen arms

Seven-dimensional action signature from the canonically fixed VU:
`[1,r_c,|r_c|,r_c²,f_c,(s0-s1)/(s0+s1+1e-8),1_basalt]`.
Here `direction_c=sign(V[j,c])*project_dark(U_c)/||project_dark(U_c)||`
and `f_c=|V[j,c]|*||project_dark(U_c)|| /
(||V[j,:]@U||+1e-8)`, j=3 for wren and j=2 for basalt.
Fit a single ridge model (slope penalty .01; free intercept) on both
candidate hit rates from **64 training model roots × 96 clean labeled
donor cases** each. Root weights are equal, not case-as-independent.
Freeze coefficients and pooled best component, all source hashes, and
the training-only memory before development.

Frozen test arms on the same per-root 128 new test pairs:
- `memory_calibrated` (primary) vs `local_only` (noisy local feedback);
- `geometry_rule` (signed r_c, no local feedback), `pooled` (train-best
  globally), `memory_only` (ridge prior alone), `random_component`
  (root-keyed coin), `random_dark` and `random_dark_b` (equal dose);
- `sham` (no patch) and `full_donor` (full donor intermediate state,
  different physical dose; contextual only).

Three prespecified one-sided positive comparison family at alpha=.05:
memory-calibrated minus local-only (primary), minus geometry-rule and
minus pooled. Holm correction and 2,000 seeded root-cluster paired
bootstrap replicates. Root is one independently fitted teacher/BPD
decomposition. Cases and two component patches within root are repeated
measurements, not independent roots.

## Fresh, disjoint roots and physical uncertainty envelope

Use SHA256 `AMM04/<phase>/<index>/...`, with counts engineering=8,
training=64, development=64, confirmation=128, balanced source families.
The engineering partition alone may be used in source unit tests.
The confirmation partition is *strictly sealed*: no code path, test,
or invalid preflight may fit or even generate a confirmation root.

Development has 64 independent roots, each 4 noisy local calibration
pairs and **128 primary heldout donor pairs**, plus **128 disjoint
repeatability donor pairs**. The repeatability arm fixes the *same*
local-only action and measures its success on the primary and new
repeatability cohorts. Independently measured within-root sample-rate
differences `|local_only_primary - local_only_repeatability|` across
64 worlds calibrate a nonparametric 95th-population-quantile
upper tolerance bound at 95% confidence. This assesses repeated-case
measurement resolution at the same dose and policy, and is explicitly
distinct from the across-model variability used for causal inference.

Scientific minimum +0.08 absolute accuracy, target +0.20, case
resolution 1/128, 3 comparisons. Use the **updated** EvalMath
mechinterp-power preflight (post PR #51, not its old pinned version)
on the 64 independent development root effects and matched
repeatability sham values, with q=.95, confidence=.95, 128 planned
independent confirmation roots. Require READY, observed mean pilot
effect above variation floor, positive pilot variation assessment,
and no zero-variance pseudo-power artifact. If it says
REDESIGN_OR_PILOT_ONLY, stop at development, publish all negatives
and do not run or access the sealed confirmation roots.

Before any confirmation run additionally require a documented human
check that the local-calibration ceiling has enough headroom: observed
development test-answer-only per-root best component minus local-only
average must be at least +0.08 (eight percentage points); this is a
post-development *conservative stop condition only*, never a reason
to reinterpret the registered results. A confirmation label additionally
requires primary root bootstrap lower 95% CI above max(0.08, sham
floor, measurement resolution), Holm p<.05, a positive secondary
memory-over-geometry direction and physical equal-dose parity.

## Receipts, integrity, and claim scope

Freeze this protocol in Git before any new source and before any new
training/dev data. Freeze implementation and tests before training.
Commit frozen training policy with its SHA before development scoring.
Use exact source/code SHA256 and full per-case bitstrings, including
noisy calibration observations and independent clean audit labels.
Retain family/root, all actions, calibration ties, teacher/factor MSE,
zero-dose count, singular gap, norm error and immediate visible drift.
Provide native BPDModel/TiedComponentLinear PassMaskSchedule parity
on excluded engineering roots plus GL2 full-product gauge identity,
recanoicalized rank-one parity for separated singular values.
Do not touch ARG-03-N or AMM03 sealed model roots.

Evidence is restricted to learned **synthetic linear** tied recurrent
BPD interventions. A result is not direct demonstration of actual
Goose/Codex model agents, deployed Buzz memory, neural language
model mechanism transfer, or cross-architecture generalization.
