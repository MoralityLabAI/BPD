# AMM-01R: audited synthetic agent-memory transfer (2026-11-18)

**Current status:** Corrected prospective synthetic positive control, [AMENDMENT_20261118.md](AMENDMENT_20261118.md). A source-exposure issue invalidated the original 512-root cohort for confirmatory inference; it remains in Git history and the original `results/CONFIRMATION_LOCK.json` as an **engineering-only** cohort.

## Chronology and provenance

- Original experiment registration, before initial code: `cd0ed97d147bcf54340233e9243abf1eaea6e02a`.
- Original source lock: `2adc180d5a65299d7dec1530b408587f34b778f8`. The original `test_confirmation_must_have_preflight` evaluated all "confirmation" worlds before rejecting an absent preflight. As a result, the previously publicized "prospective" numbers **94.138% / 73.678%, gain +20.460 points** were engineering measurements and should not be cited as independent confirmation.
- Pre-data corrective amendment: `f6471f97b58acbc23c52d8015b19c0f5816b2638`. Replaces only the confirmation root population with independently seeded `confirm_fresh/0000..0511` and bans missing-preflight code from touching the cohort.
- Amended source frozen before fresh scoring: `ae88737dfa302eb678ddd72f51f95bae6fbda323`. Exact experiment `sha256=cc79250ced0c1470b87c01ceb4d1c6e29720140486ae8167d3c189ccebeabe13`.
- New development receipt and gate committed before fresh scoring: `2eedbab547ed0689c03ddb12cc5ae96194c4c226`.
- Current full fresh-case canonical SHA-256: `fa29c744b7b92abfcc7f99b73ce410c540d8e727a767aab66bf305cfaf96c688`.
- Training-only memory canonical SHA-256: `2b7f42d00af7ff3c683fd4c39d38b1770b05fb2f5976e77704c2f14f3e0a1b94`.
- EvalMath preflight source: `MoralityLabAI/EvalMath@2fdd5fe3f499070b63c474a2455f0e2c00fc55a8`.

## Protocol

**Independent unit:** one 4D rank-at-most-two tied linear recurrent system, with independently drawn sign and gain. Two opaque labels (`wren`, `basalt`) determine whether a fixed unit-norm, current-dark causal probe is informative after one or two passes. The observable is the sign of a single noisy future readout. Identical dose `d=(0,1/√2,-1/√2,0)` and one query per case in every policy.

Training consists of 64 independent roots. Memory learns the best horizon per family, while a pooled arm learns one training-optimal horizon overall. Swapped-memory routing and seeded random-horizon policies are registered controls. Development uses 64 separate worlds with 256 repeated cases per world, and the corrective confirmation uses **512 new worlds with 256 repeated cases each**. All 3 signed root-level comparisons, root-bootstrap CIs, 15-point mechanistic floor, and 95th-quantile/95%-confidence sham gate were fixed before any new `confirm_fresh` query.

**EvalMath strict preflight:** `READY_FOR_CONFIRMATION`. Measured development sham tolerance floor 0.1015625 (10.156 points); declared scientific effect floor 0.15; target effect 0.25. The root-level normal planning approximation requested 46 confirmation roots; the frozen plan supplied 512.

## Corrective first-run confirmation (source-locked)

| Agent policy | Correct / 131,072 cases | Accuracy |
|---|---:|---:|
| Family-keyed memory | **123,581** | **94.2848%** |
| Training-derived pooled horizon | 96,903 | 73.9311% |
| Random horizon | 95,616 | 72.9492% |
| Swapped family-key memory | 67,762 | 51.6983% |
| Pooled horizon with independent sham noise | 97,001 | 74.0059% |

| Registered comparison | Root-mean gain | Root-bootstrap 95% CI | Holm-adjusted exact one-sided root sign p |
|---|---:|---:|---:|
| Memory vs pooled (primary) | **+0.20353699** | **[+0.18589783, +0.22167206]** | **8.6362e-78** |
| Memory vs swapped | +0.42586517 | [+0.42172241, +0.43033600] | 2.2375e-154 |
| Memory vs randomized | +0.21335602 | [+0.21051025, +0.21633911] | 2.2375e-154 |

Primary root directions: **256 positive, 256 tied, zero negative**; all independent roots and repeated-case failures included. The primary confidence lower bound (18.5898 points) exceeds the registered 15-point effect floor and the development sham tolerance floor (10.15625 points). The Holm-adjusted sign test meets the registered familywise alpha=.05. **The corrective prospective synthetic transfer gate passes.**

## Audit boundary

This is an intentionally constructed strategy-memory *positive instrument*. Memory learns a family-specific probe horizon; the timing structure is designed into the transition. The project has yet to evaluate trained neural agents, learned BPD mechanism discovery, nonlinear TRM/transformer causal effects, genuine cross-agent message transfer, or an authenticated Buzz relay. This scientific inference belongs to the 512 **independent worlds** rather than the 131,072 repeated cases.

The Buzz bridge exports only an **unsigned NIP-29 kind-9** provenance message draft. No model weights, protected ARG-03-N split, private conversations or signing keys were accessed.

## Reproduce

From the BPD root with CPU PyTorch available for package imports:

```sh
python -m unittest mechinterp.bpd.experiments.amm01_agent_memory.test_experiment -v
python -m mechinterp.bpd.experiments.amm01_agent_memory.verify --lock mechinterp/bpd/experiments/amm01_agent_memory/results/DEVELOPMENT_R_LOCK.json
python -m mechinterp.bpd.experiments.amm01_agent_memory.verify --lock mechinterp/bpd/experiments/amm01_agent_memory/results/CONFIRM_FRESH_LOCK.json --preflight mechinterp/bpd/experiments/amm01_agent_memory/results/PREFLIGHT.json
```

The verifier reconstructs every repeated-case bit and checks the committed canonical digest. The legacy engineering-only cohort can be reproduced from its historical source commit `2adc180d`.
