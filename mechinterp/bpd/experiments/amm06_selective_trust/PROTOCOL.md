# BPD-AMM-06: selective trust of noisy calibration in case-conditioned causal memory

**Registered 2026-11-18, protocol-first, before AMM-06 source or model scoring.**
AMM-05's registered 64-root development found +5.87pp memory+local over
local ridge, with a 14.06pp independent-root repeatability variation
floor. Its training-only memory (3796/8192) exceeded always-updated
memory (3641/8192). AMM-05's 128 confirmation roots remain sealed.
AMM-06 tests a *fixed selective trust* gate that accepts a local update
only when source-disjoint calibration feedback gives heldout predictive
evidence supporting it.

## Independently trained recurrent causal task

Follow AMM-05's frozen scientific instrument with *fresh* SHA256
\`AMM06/<phase>/<index>/...\` streams. Every root independently
Adam-trains a 4D rank-two shared physical recurrent operator
\`h -> (h@I)@O\` (200 steps, batch 96, LR=.035, .0002 weight regularizer)
to external two-readout labels from sources (a,b), and trains a distinct
V[4,2],U[2,4] BPD factorization (140 steps, batch 96, LR=.035, full
teacher trajectory plus .5 reconstructed-weight MSE). Families remain
wren q(h1)=a,q(h2)=b and basalt q(h1)=b,q(h2)=a, with q(h)=h[0].
Canonicalize full VU by float64 SVD before any component-index
decision and check GL2 gauge-shear/recanonicalization parity.

Each donor pair changes the externally defined second-pass target
source coordinate by signed 0.30..0.45 and an independent nuisance
coordinate by signed 0.15..0.80. The true donor target is defined
directly from input data *before teacher/BPD fitting*. At a fixed
unit pair, choose one of two component patches from
\`z_c=(((h_d-h_r)@V)_c)U_c\`. Set the immediate readout coordinate
to zero and renormalize to the same per-pair physical dose
\`D=||F_BPD(h_d)-F_BPD(h_r)||_2\` for both candidates and equal-dose
random dark controls. A zero projection uses fixed e1*D; all zero-dose
and degenerate cases remain in the ITT denominator. Score
\`abs(q(F_BPD(h1_rec+patch_c))-donor_y2)<=0.18\`.

## Exactly matched local information and hidden reliability regimes

Each evaluated root has **eight** new, disjoint calibration donor pairs
(16 binary component feedback observations); no test pair reappears in
calibration. Judge reliability is a *hidden root-level regime*, balanced
with family using \`(index//2)%2\`: quality LOW_NOISE flips each
binary label independently with p=.10; quality HIGH_NOISE flips
independently with p=.45. Family is determined by index parity, so
each phase has exactly 1/4 of roots in each family×quality stratum.

The policy receives only the 16 noisy labels, the two component's
eleven-dimensional AMM-05 SVD-canonical causal-response features,
and three unlabeled continuation readouts per donor pair
(one unpatched reference, two candidate continuations). Both
strategies receive exactly the same physical input and local
observation budget. **The hidden judge regime, clean calibration
labels, donor future continuation, true donor target and all test
successes are audit-only, never passed into a decision function.**
Both sides know the *population mixture* (.10,.45), not the root's
regime. The cross-model memory has the additional disclosed amortized
64-training-root corpus.

## Frozen global memory and local update

Fit AMM-05's *exact* 11-feature ridge model with lambda=10, free
intercept on 64 *fresh* independent TRAIN roots, 128 clean labeled
donor cases/root, both candidate actions scored. Freeze the 11
coefficients, pooled best component and training receipt digest.
Clip the memory's predicted per-component true success probability
to [0,1] before any local feedback update.

For local estimation use the known *marginal* noise probability
\`eta=(.10+.45)/2=.275\`, with de-noised binary target
\`(noisy_y-eta)/(1-2eta)\`. A λ=20 ridge with all 11 coefficients
penalized fits the difference of this target from the memory
prediction; **always_update** fits all 16 noisy labels and
selects \`argmax_c(p_memory + residual)\` per test case.
**memory_only** selects \`argmax_c(p_memory)\` and ignores calibration.
A full λ=2 local ridge with free intercept fits the same 16 labels
without memory, giving the equal-local-budget **local_ridge** control.
All policies have the same 3 unlabeled candidate readout probes
per test case.

## Fixed heldout-likelihood trust gate (primary arm)

The **selective_trust** arm splits calibration into the *first four*
pairs (fit) and *last four* pairs (gate validation), identified by
the independent local-calibration stream before seeing labels.
It fits a λ=20 residual on the eight *fit* labels, with the same
marginal noise correction, then predicts the validation eight noisy
labels under two hypotheses: H0=memory probabilities and H1=
memory+fit residual (clipped to [0,1] for likelihood evaluation).

For each H, compute the joint calibration validation likelihood
under an unknown but *shared root-level* judge regime:
\`L(H)=.5*prod_j Bern(y_j| .10+.80*p_Hj)
     +.5*prod_j Bern(y_j|.45+.10*p_Hj)\`.
Use a numerically stable log-sum-exp with all eight validation bits;
clip predictive observation probabilities to [1e-9,1-1e-9].
Define log Bayes factor \`B=log L(H1)-log L(H0)\`.
The **fixed, theory-selected threshold is log(2)**.
If \`B>log(2)\`, choose the same **always_update** policy fitted
on *all 16* noisy calibration labels; otherwise use **memory_only**.
The gate cannot access confirmation, development outcomes, audit-only
clean calibration labels or quality labels. The threshold and
training/calibration split are frozen before AMM06 source scoring.

The source should retain the scalar B per root, accept/reject flag,
full per-case action traces, and hidden quality only in audit receipts.

## Arms, sample sizes and inference

Frozen arms: \`selective_trust\` (primary), \`always_update\`,
\`memory_only\`, \`local_ridge\`, \`pooled\`, \`random_component\`,
\`random_dark\`, \`random_dark_b\`, \`sham\`, \`full_donor\`.
The full donor intermediate replacement is a different-dose
contextual ceiling, not an equal-dose control.

Independent roots: 8 EXCLUDED engineering for tests only,
64 TRAIN (128 cases/root), 64 DEVELOPMENT and **128 protected
CONFIRMATION**. Each development/confirmation root has eight
calibration pairs, 256 fresh heldout test pairs and **256 additional
independent repeatability test pairs**. Pair scores are repeated
measurements nested in the one model root. Test root keys and
confirmation computations are forbidden from source tests and
invalid-preflight paths, including indirect calls.

Primary registered signed root effect:
\`selective_trust - always_update\` true heldout success rate.
Two secondary registered signed root effects:
\`selective_trust - memory_only\` and
\`selective_trust - local_ridge\`.
Three one-sided exact independent-root sign tests, familywise .05
Holm adjustment, plus 2000 seeded paired root-bootstrap 95% CIs.
Minimum meaningful gain **+0.08** probability, planning target
**+0.15**, measurement resolution 1/256. Negative and zero results
are all retained. Report performance separately in the two hidden
judge-reliability strata *as posthoc descriptive diagnostics*,
including gate acceptance and discrimination.

For each independent development root, evaluate the same frozen
\`always_update\` policy on 256 primary test pairs and on 256
independent repeatability pairs; record their absolute success-rate
difference. Over the 64 roots, the sample maximum calibrates the
95th population quantile with >=95% confidence (95/95 tolerance).
The variation floor is max(+.08, 1/256, upper tolerance).
Use **corrected EvalMath** preflight pinned to merged commit
\`6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a\`.
A \`REDESIGN_OR_PILOT_ONLY\` blocks all confirmation scoring,
regardless of repeated-case totals.

Additional preregistered **scientific admission vetoes** even if
EvalMath returns READY: (1) at least eight of 64 independent
development roots accept a local update, (2) gate acceptance
fraction in LOW_NOISE minus HIGH_NOISE >= .10, and
(3) development casewise oracle gap above always_update >= .08.
These are *veto-only*, never opportunities to optimize thresholds
on the development cohort.

A positive confirmation claim requires primary Holm p<.05,
its paired-root bootstrap lower 95% limit above the registered
variation floor, and no dose/parity/source integrity failure.
No preexisting source-locked confirmation population may be opened
without a separately committed, explicit human-approved release
manifest that binds its model roots, training-memory SHA, source SHA
and EvalMath receipt. In this conversation, any gate failure leaves
all 128 independent confirmation roots unopened.

## Evidence, provenance and security

Source and protocol must be committed **before** new training.
Freeze training-memory source and digest in Git **before** any
development world is trained or scored. Preserve every case-bit
trace, selected component, calibration feedback, heldout/repeat
outcomes, zero dose and gauge tests. Native BPDModel and
TiedComponentLinear PassMaskSchedule execution must agree with the
independent factors on engineering roots to <=1e-5.
No ARG-03-N protected case or AMM03/04/05 sealed confirmation root
may be accessed. Evidence remains **synthetic trained linear tied
recurrent causality with an explicitly modeled noisy feedback
channel**. It cannot establish agent learning between real LLMs,
nonlinear transformer mechanistic transfer, or live Buzz/Nostr
memory transport.
