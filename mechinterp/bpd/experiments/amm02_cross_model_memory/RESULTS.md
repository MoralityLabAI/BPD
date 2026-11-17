# AMM-02: cross-trained BPD agent-memory transfer, development-only result

**2026-11-18. Evidence class: registered synthetic trained-model developmental null.**
Source registration `ce390c8`, pre-development sham amendment `b811ab1`,
and frozen source commit `604db2f` preceded development scoring.
The 128-root independent confirmation cohort remains unaccessed.

## Research question and intervention

Each independent world contains a separately Adam-trained rank-two 4D
weight-tied recurrent operator, and separately trained V,U BPD factors.
The task's two external behavioral targets were defined from input a,b
before any model training: wren has `q(h1)=a,q(h2)=b`, basalt
`q(h1)=b,q(h2)=a`. Donor-recipient pairs change the second-pass target
coordinate by 0.30..0.45. An agent selects one of two SVD-canonical BPD
components. Its projected current-dark donor write is normalized to
the same per-pair dose as all randomized dark controls, then scored
against **donor's independently defined task label**, after one more
learned recurrent pass.

Independent-root sample sizes: 64 cross-model *training* roots and
64 disjoint *development* roots, each with 48 scored donor pairs.
The confirmatory population of 128 new model roots was preregistered
but left sealed. Every pair, including failed-model and low-dose cases,
is retained in the development denominator.

## Training-only memory result

| Source family | Training comp 0 / 1536 | Comp 1 / 1536 | Frozen learned policy |
|---|---:|---:|---|
| wren | 1214 | 1002 | component 0 |
| basalt | 1196 | 1032 | component 0 |

The pooled best component is also **0**, so family-keyed memory,
pooled memory and swapped-family memory take **identical actions** on
every development pair. This is a structural failure of the proposed
family-index strategy and cannot be remedied with more repeated cases
under this frozen policy.

## Development results: 64 independent trained model roots

| Policy | Hits / 3072 | Accuracy |
|---|---:|---:|
| Family-indexed memory | **2379** | **77.44%** |
| Pooled best component | **2379** | **77.44%** |
| Swapped-family memory | **2379** | **77.44%** |
| Random component per pair | 2221 | 72.30% |
| Equal-dose random-dark A | 801 | 26.07% |
| Independent equal-dose random-dark B | 762 | 24.80% |
| No-patch sham | 5 | 0.16% |
| Full donor state (different intervention) | 3066 | 99.80% |

**Primary memory-minus-pooled root effect = 0/48 on every one of 64
independent development roots.** A formal sign test would be
uninformative: the two policies are identical by construction after
training; this is a policy-collapse observation.

Counterfactual component success varies across independently trained
models. In the wren development family (32 roots), comp0 scores 1119/1536,
comp1 scores 1173/1536; root preferences are 7 for comp0, 8 for comp1,
17 tied. In basalt (32 roots), comp0 scores 1260/1536 and comp1
909/1536; root preferences 13 vs 6, 13 tied.
A **posthoc impossible-to-deploy oracle** selecting each root's
component from its 48 test labels would score 2961/3072 (96.39%).
This is a diagnostic of model-level heterogeneity and an optimistic
in-sample upper bound, not a measured deployable strategy.

## Physical and learning checks

- All 3072 pairs included; zero zero-dose pairs, zero degenerate
  component projections in either action.
- Max active equal-dose norm discrepancy `2.3842e-7`;
  immediate readout displacement of every dark action `0.0`.
- Independent teacher behavioral MSE max `4.801e-6`;
  BPD factor behavioral MSE max `0.057892`.
- Learned BPD full-weight relative error median `0.001375`,
  maximum `0.30678`. The high-error roots remain in the receipt.
- Canonical rank-two singular gap minimum `0.10715`; the excluded
  engineering tests verify full-product GL(2) gauge invariance and
  gauge-recanonicalized rank-one components.
- Hosted unit tests include actual repository
  `BPDModel/TiedComponentLinear` forward parity under
  `PassMaskSchedule` for excluded roots.

## EvalMath strict preflight

Pinned planner:
`MoralityLabAI/EvalMath@2fdd5fe3f499070b63c474a2455f0e2c00fc55a8`.

The sham floor is the sample maximum across **64 independent
development roots** of the per-root difference between two separately
randomized physically equal-dose dark probes. This produces a
nonparametric 95th-quantile upper tolerance bound at >=95% confidence.

| Gate input / output | Value |
|---|---|
| Registered independent development worlds | 64 |
| Repeated cases per independent root | 48 |
| Science minimum effect | +0.12 |
| Planned target effect | +0.20 |
| Matched sham 95/95 tolerance bound | **0.208333** (10/48) |
| Observed primary root-mean effect | **0.000000** |
| Planned comparisons, Holm family | 3 |
| Planned independent confirmation worlds | 128 |
| **Strict decision** | **REDESIGN_OR_PILOT_ONLY** |

The registered target lies inside the calibrated sham-variation floor,
and memory's chosen action matches the pooled control across every
root. The preflight returned both
`target_below_mechanistic_variation_floor` and
`confirmation_underpowered` (the normal planning power calculation
has no finite answer for target below floor).
The study **stops at development**; no confirmatory p-value,
equivalence claim or cross-model memory benefit is asserted.

## Reproducibility and source locks

- Protocol-first commit: `ce390c8f0938a695dd0a73a92b53ab035e158763`.
- Sham amendment: `b811ab1fd6929335d5c8c817f66a356a979a8e90`.
- Frozen runner: `604db2fd0273d655fdd1ef5eeff32aa9cbe9f06b`.
- Runner SHA256: `670999874c2bc417bc2aefade1d116114d8280a4a1e9368d553385d59e331283`.
- Training policy SHA256: `0c85f01d8238e45d8676c6693229706bbe22abb2ec062d7d234f843c3aa2b587`.
- Full training canonical receipt SHA256:
  `ee17ffdd2ad4fd9fa903c19ae567a52039af52df3ce840333522624973fd3d7e`.
- Full development canonical receipt SHA256:
  `ec5851fc2f74a49e0a3326c98bcbcf905d7c7bfc2bd6651cd8e7a0305779b368`.
- EvalMath preflight canonical receipt SHA256:
  `4e6b3939fb4c0b8baba0b1a3c018f967f5a3d3e7e4472ab38652440f46b82aa7`.
- Environment: CPython 3.13.5, Torch 2.10.0+cpu. Deterministic
  SHA256-derived model and data generators.
- Compact locked receipts in `results/DEVELOPMENT_LOCK.json`
  and `results/PREFLIGHT.json`. Full per-root and per-case
  JSON records archived with the handoff artifact.

## Follow-on design implication

AMM-03 should test **portable, gauge-invariant model-conditioned
strategy memory**. Candidate signatures could summarize a local
observable Gramian, direction-specific causal response, SVD singular
gaps and component read/write overlap. Freeze the feature extractors
using separate training roots, select the strategy without donor
test labels, compare against pooled and **cost-matched per-model
calibration**, and re-run the EvalMath 59-plus independent-world sham
preflight. Treat the posthoc root-wise best component above as an
optimistic upper bound only. No actual Goose/Codex agent, live
Buzz/Nostr relay, nonlinear transformer or sealed ARG-03-N case was
accessed for AMM-02.
