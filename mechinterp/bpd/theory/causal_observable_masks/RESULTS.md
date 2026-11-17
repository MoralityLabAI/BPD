# BPD-CQ-01 exact-rational CPU results

**Source freeze:** BPD branch `research/bpd-cq01-delayed-observable-masks-20261117`, source/protocol commit `19fe6bcb6b6e242efb49a9bb1090a87573d246e1` before generating the new 384-world reference. All claims concern finite exact-rational linear systems. **No trained BPD network was run.** ARG-03-N's 168-case test remains sealed.

## Predata source and environment

- Exact arithmetic: Python `fractions.Fraction`; all determinant/Gramian/rank/zero comparisons exact.
- 13 local engineering unit tests passed before the extended 384-world run, including 256 deterministic old-seed metamorphic shears.
- New extended seed `2026111801`; source-locked protocol fixed 128 worlds per construction cohort, then generated the complete final reference.
- Committed JSON receipt: `results/cq01_cpu_reference.json` (158,918 UTF-8 bytes), SHA256 `bbaf1e521501de3f039756134c42c40887fc46d334ff69db5273351b8845ccbe`.
- The receipt pins `quotient.py`, `reference.py` and `PROTOCOL.md` SHA256 independently. Deterministic replay compares all bytes.

## Frozen primary mathematical checks

| Cohort | Independent seeded construction instances | Horizon-0 masks | Horizon-1 masks | Gramian/direct | CH stabilization |
| --- | ---: | ---: | ---: | --- | --- |
| 2D delayed shear | 128 | 4 | 2 | 128/128 exact | 128/128 |
| 4D restricted observer | 128 | 16 | 6 | 128/128 exact | 128/128 |
| Random 4D frames | 128 | variable 2–16 | variable 2–16 | 128/128 exact | 128/128 |

Each constructed delayed shear first makes its nontrivial singleton mask observable after one swap. In the 4D cohort, every one-step observer-visible mask family fails closure under coordinate intersection, with two masks `(1,1,0,0)` and `(1,0,1,0)` whose intersection `(1,0,0,0)` violates the invariant equation. The 128 randomized frames satisfy horizon-monotonicity and stabilize no later than recurrence dimension minus one, as the theorem requires.

Random 4D frame final observability ranks: rank 1 in **29**, rank 2 in **32**, rank 3 in **35**, rank 4 in **32** worlds. These counts describe one frozen random-generator run, not evidence about trained model rank distributions.

The exact two-pass linear capacity witness has observability rank 2. A phase-disjoint `(1,0)` source can be carried by top-1 mask `(1,0)`; the mixed `(1,1)` source has **zero** successful top-1 masks. This checks the limited-capacity mechanism proposed for BPD-PD-03's mixed-source negative control without rerunning or modifying its 24-seed experiment.

## Scientific interpretation

The original BPD all-observer mask algebra uses full read/write identifiability and a common centralizer of relative optimum bases. This experiment establishes that **observable** mask equivalence over a restricted input span and finite readout horizon has an extra identifiability obstruction. A mask can be invisible today and visible tomorrow. The observable mask set can also fail to be a Boolean block family because cancellations occur in the restricted readout.

The prior SCM-21 source ([merged PR #207](https://github.com/MoralityLabAI/Compactification-Theory/pull/207)) shows native rank-four VPD writes had smaller patch norms than calibrated difference-PCA patches. Those counts do not isolate basis geometry without norm matching, and they cannot be used as a learned BPD optimum-basis test. The new theorem supplies a way to distinguish delayed observable equivalence from whole-state sparsity when a future model experiment is registered.

## Limits and next gate

The experiment is synthetic exact algebra with known matrices and an analytic full-dimensional Gaussian law ensuring every read basis is L0 optimal. It cannot establish trained BPD optimum recovery, correct pass routing, hidden semantic variables, sampled topology, or native task mediation. Before a trained experiment, freeze physical parameter sites, actual nonlinear readout (including normalization), pass-specific derivative estimates, per-pair norm controls, and independent task-family blocks; use the existing EvalMath mechanistic-power skill for the expected detectable effect. Keep ARG-03-N test sealed.
