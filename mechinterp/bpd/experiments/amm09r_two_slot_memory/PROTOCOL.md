# BPD-AMM-09R1: fresh development under a controlled CPU recipe

Prospective protocol dated 2026-11-18. Commit this document after the registered
AMM09R pinned32 engineering gate passes and before generating any AMM09R1 root.
Commit all study source before training, then commit the resulting training lock
before development. This work has no confirmation runner or release interface.

## Scope, prior exposure, and engineering prerequisite

The research question is whether a policy learned across independently trained
linear recurrent models can choose useful one- or two-component BPD writes on
new models, beyond the best fixed permitted mask learned from the same training
corpus. AMM09's exposed development result motivated this fresh study. Its source
and receipts remain unchanged, including its corrected primary analysis and its
unresolved historical CPU portability incident. AMM03 through AMM09 confirmation
populations and ARG-03-N protected data remain sealed and excluded.

The numerical recipe is selected by exact engineering equality, without using
scientific success rates. The engineering registration is commit
`ffc7143e3b597d840c5000a94a45b45d964931cf`. Before scientific training, preserve
its complete passing comparator report as
`results/ENGINEERING_ACCEPTANCE.json`. It must report every registered repeat,
matching executable source, bit-exact pinned32 payloads, matching registered
software and torch binaries, the specified runtime settings, and two distinct
workers. A descriptive or incomplete report fails admission. The report's
observed CPU models bound the hardware claim.

The new experiment ID is `BPD-AMM-09R1`, and every stream uses the prefix
`AMM09R1/`. It shares no streams with AMM09, earlier studies, or the excluded
`AMM09R-ENGINEERING-V1/` portability fixture. The numerical intervention is the
accepted pinned32 environment. Training losses, optimizer steps, random-input
distributions, masks, features, and the success threshold remain as declared
below. A failed development result cannot justify a posthoc numerical change.

## Independent units and protected populations

One independent unit is a separately trained 6D rank-three physical recurrent
model plus a separately initialized and optimized rank-three BPD decomposition.
The source seed is the first 15 hexadecimal digits of SHA256 applied to
`AMM09R1/` followed by slash-joined root and stream arguments. Root keys are
`<phase>/<index:04d>`.

| Population | Roots | Permitted role |
| --- | ---: | --- |
| engineering | 8 | Excluded implementation checks |
| training | 64 | Fit the global ridge and choose one pooled mask |
| development | 64 | Score once after committing the immutable training lock |
| confirmation | 128 | Reserved and sealed; generation is unavailable |

The model functions reject the confirmation namespace before RNG access. The
CLI offers engineering, training, and development only. A later confirmation
release would need separate explicit human authorization and a new committed
release protocol. An EvalMath READY verdict does not grant that access.

Cases, masks, and recurrent passes are nested observations within a model root.
Every finite poor-fit or degenerate root stays in the denominator. Numerical
failure that prevents a complete cohort blocks inference; preserve the failure
and partial receipts instead of silently replacing a root.

## Controlled numerical execution

`RUNTIME.json` freezes Python 3.12.12, torch 2.10.0+cpu, numpy 2.2.6, float32
training and assay tensors, and the existing float64 SVD followed by float32
canonical factors. Before importing torch, the direct `run.py` launcher sets:

```text
ATEN_CPU_CAPABILITY=default
MKL_CBWR=COMPATIBLE
MKL_NUM_THREADS=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_DYNAMIC=FALSE
OMP_DYNAMIC=FALSE
PYTHONHASHSEED=0
```

Use one intra-op thread and one inter-op thread. Enable deterministic algorithms
with hard errors. Require the effective default CPU capability and float32
default dtype. Record torch's source version and hashes of `libtorch_cpu.so`
and its Python extension. The training memory binds this runtime signature, and
development requires an exact match. This is the experimentally accepted CPU
recipe within the recorded hardware/software scope.

The source lock hashes the fixed executable file list in `source_sha256`, the
protocol, and `RUNTIME.json`. Finish executable files before training. The full engineering
acceptance file has a separate hash bound into the training memory and lock.

## Physical task and fitting

Use three weight-tied passes and readout `q(h)=h[0]`. Independently sample
`(a,b,c)` uniformly from `[-.8,.8]`, and set `h0=(0,0,0,a,b,c)`. Even-indexed
roots use the wren target order `(a,b,c)`; odd-indexed roots use basalt
`(c,a,b)`. These external targets precede either model fit.

Initialize physical parameters `I[6,3]` and `O[3,6]` with independent standard
normal draws scaled by .45. Optimize 400 Adam steps, learning rate .035, batch
96. The loss is the sum of the three readout MSEs plus .0002 times mean squared
physical operator weight. Independently initialize BPD `V[6,3]` and `U[3,6]`
with normal draws scaled by .35. Optimize 240 Adam steps at the same learning
rate and batch size. Its loss sums three teacher-trajectory MSEs and .5 times
mean squared operator reconstruction error. Every batch uses the named
per-root generator stream. Preserve the original arithmetic order.

Canonicalize the BPD product using float64 SVD, keep its leading three
components, split each singular value by its square root, and copy the factors
to float32. Record effective teacher/BPD weight hashes, update norms, held-out
task MSEs, relative reconstruction error, and singular gaps. A 256-case source
validation stream is used only for fidelity diagnostics.

## Donors, action set, and physical controls

Recipient source coordinates are independently uniform in `[-.3,.3]`.
Independently perturb the coordinate used by the first readout with signed
magnitude uniform in `[.15,.70]`; perturb each future-target coordinate with
signed magnitude uniform in `[.25,.45]`. The donor's external source values
define its two delayed targets.

The six permitted masks have immutable order
`(0), (1), (2), (0,1), (0,2), (1,2)`. Full `(0,1,2)` selection is excluded from
every scored action, baseline, training label, policy, and oracle. Compute
recipient and donor first-pass states `r1,d1`, and common physical dose
`D=||d1-r1||₂`. For each mask S, form
`((h_don-h_rec) @ V[:,S]) @ U[S,:]`, zero coordinate zero, and normalize to D.
A projection with norm at most `1e-8` uses the registered coordinate-one unit
fallback at dose D. Record every fallback and zero-dose case.

Apply one selected write after pass one, then continue the same recurrent
operator for two passes. Success requires both delayed readouts to be within
.18 of their externally defined donor targets. Keep the exact inclusive
threshold and retain every generated case.

Two independent isotropic dark controls receive the same D and zero immediate
readout write. Also record zero-patch sham and full donor-state replacement.
The full donor replacement has norm D and can change the immediate readout; it
is an unrestricted visible control and never a policy input. Its measured
success rate need not exceed the allowed-mask oracle.

The accepted engineering receipts provide native BPDModel/PassMaskSchedule
six-mask and delayed-readout parity, pass reset, and GL(3) recanonicalization
checks. Development retains dose and readout diagnostics for every case.

## Information, training, and immutable policies

The same 18 features are available to all policies: intercept, mask size/3,
recipient first readout, current donor-readout difference, D, two candidate
future shifts, their absolute values and squares, their product, two raw
candidate future readouts, patch/full-write cosine, normalized minimum singular
gap, immediate nuisance-dose fraction, and family bit. Their computation uses
the current states and permitted continuations. Source-defined target values,
scored success bits, and full donor future values stay in scoring code.

The per-case observation budget is 14 unlabeled readout values, consisting of
two common unpatched future readouts plus two per allowed candidate. All
comparators receive the same available features and physical action budget.

Training uses 64 roots, 128 donor pairs per root, and all six candidate labels.
Fit the shared 18-feature ridge with lambda 10 on slopes and an unpenalized
intercept. Round every coefficient to four decimal places. Choose the pooled
mask by aggregate training successes, breaking ties by lowest index. Preserve
the complete training features and labels, verify that they reproduce the
ridge coefficients, and bind all raw row contents into the memory digest.
Publish `results/TRAINING_LOCK.json` before generating development roots.

Each development root receives 16 separate calibration pairs with six labels
each. Flip each clean label independently with registered probability .20.
Every local comparator receives the same 96 noisy labels. Use the unbiased
target `(reported-.20)/.60`. The global corpus is an additional disclosed cost.

| Policy | Frozen rule |
| --- | --- |
| memory_only | Clip the shared ridge prediction to `[0,1]`, then choose its maximum |
| pooled | Always use the training-selected fixed permitted mask |
| memory_plus_local | Fit a residual ridge with lambda 20 on all coefficients to de-noised labels minus clipped global priors; add to those priors, clip the combined score to `[0,1]`, and maximize |
| local_ridge | Fit all 18 features from the same 96 noisy labels with lambda 2 on slopes and free intercept; maximize its raw score |
| local_count | Choose the mask with most reported local successes |
| random_mask | Use the registered SHA256-derived per-root/per-case action |

Every maximization breaks ties by lowest index. No held-out success bit enters
policy fitting or choice. Root-specific policies remain fixed across the
held-out and repeatability sets.

## Development endpoints and inference

After the training lock, each of 64 new development roots is scored on 512
held-out donor pairs and a separate 512-pair repeatability stream. The primary
effect is the root success-rate difference `memory_only - pooled`.

The four positive-direction contrasts are fixed as explicit treatment/control
triples throughout code and reporting:

| Name | Treatment | Control |
| --- | --- | --- |
| memory_only_vs_pooled | memory_only | pooled |
| memory_plus_local_vs_pooled | memory_plus_local | pooled |
| memory_plus_local_vs_local_ridge | memory_plus_local | local_ridge |
| memory_plus_local_vs_memory_only | memory_plus_local | memory_only |

For each contrast, use the exact one-sided sign test over nonzero independent
root differences. Report ties and apply Holm across the four tests at familywise
alpha .05. Use 2,000 paired-root bootstrap resamples with
`random.Random(seed('bootstrap', contrast_name))`, retaining sorted positions
50 and 1949 as the declared interval. Root effects divide each success-count
difference by 512, with all 64 roots weighted equally.

Before any statistics or pilot rows, reconstruct each root's selected outcome
bits from its candidate bits and admitted actions. Derive root counts from that
reconstruction and reject discrepancies with supplied counts, even when
aggregate totals agree. Validate every candidate/arm set and bit alphabet,
calibration counts and budget, repeatability counts, pooled actions, and source
fidelity flags. Preserve the historical AMM09 analysis untouched.

## Mechanistic floor, sham, planning, and vetoes

The physical minimum is an absolute primary gain of .08, with planning target
.16 and measurement resolution `1/512`. The matched sham is the same fixed
pooled policy's success-rate difference between the two independent 512-case
sets within each development root. Use one absolute difference per root. With
64 independent roots, use the sample maximum as the upper 95th-percentile
tolerance bound at at least 95% confidence. The variation floor is the maximum
of .08, `1/512`, and that sham bound.

Run corrected EvalMath at
`MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a`, using all
reconstructed primary effects and sham differences, alpha .05, four comparisons,
target .16, and 128 planned confirmation units. Rerun the pinned planner from
the reconstructed pilot rather than accepting a supplied READY label. Record
its source hash, inputs, and output. Any `REDESIGN_OR_PILOT_ONLY` verdict blocks
a future confirmation proposal for this instrument.

For positive development admission, the primary lower bootstrap bound must
exceed the calibrated floor, its Holm p must be below .05, and the planner must
return READY. Further vetoes require at least 56 of 64 roots to satisfy teacher
MSE at most `1e-4`, factor task MSE at most `.02`, relative operator error at
most `.10`, and minimum adjacent singular gap above `.01`. The audit-only
six-mask oracle must exceed pooled by at least .08. Maximum active dose error
must be at most `1e-5`, with exactly zero immediate readout displacement.

All outcomes, adverse secondary effects, and failed fidelity strata are
reported. Eight engineering roots cannot estimate the science sham tail.
Planning output remains distinct from evidence on the reserved confirmation
cohort, which this source cannot generate.

## Evidence and provenance

This study can provide prospective development evidence for synthetic 6D
rank-three linear recurrent BPD routing. It does not establish a mechanism in
nonlinear transformers or deployed agents. Engineering receipts establish the
observed numerical replay scope and do not add scientific sample size.

Preserve exact source, runtime and engineering-acceptance hashes, original
training and development JSON, paired bit outcomes, action choices, full
training features, all four contrasts, reconstructed pilot, and pinned planner
output. Historical AMM09 receipts keep their original identities. Future
scientific confirmation requires a separate authorized release after review of
this complete prospective result.
