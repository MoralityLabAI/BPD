# AMM-04: memory transfer with noisy local calibration (development-only)

**November 18, 2026. Evidence: registered synthetic trained-model developmental positive direction, below preregistered scientific-resolution floor.** There is no confirmation claim. All **128 new confirmation trained-model roots are sealed**, and earlier AMM-03 confirmation roots remain sealed.

## Preregistration and source chronology

- Frozen scientific protocol: [PROTOCOL.md](PROTOCOL.md), committed first at `31e46035c42bb0345086e22e43c9f76de4263fb0`.
- AMM04 trained-model generator, fixed intervention scorer, local feedback corruption, tests, and CI frozen at `a1af66f00207d899307eb704d262f047e154454d` before train/development scoring.
- 64 source-independent training worlds: learning policy lock committed as `f5f7c5ceaeb823573b39c7a67e4bcfa03cde233d` **before scoring the 64 development worlds**.
- Source SHA-256: `models.py dc186b5b1fef4aa8b4562f5e927b69a13c5a346e95d4ba51e8cff37412bdc40f`; `experiment.py e55ca56d70d04b7d4ada5cec4d56a6a9c9da9b74914fee299964a5c9b9ca7bab`.
- Training memory canonical SHA-256: `f9983333551bd7a16d62ac0e2b5c855dd4d6b9e932d96534d38b20a30e0a1b14`.
- Complete development JSON canonical SHA-256: `79d6d5097d89fbe8cddda18f6a805444b02a2219c92b891bea1295a15c52381e`. The exact retained per-root candidate, noisy-calibration, policy and repeatability bit digest is `abae857059c929558b78132feb184079c7a7ebac0abc48e2a281cc6753745c29`.

## Experimental setup

Each independent root fits a new rank-two 4D tied recurrent physical teacher and independent BPD V/U factors on externally defined two-pass tasks. The donor's second-pass input coordinate supplies the independently defined target. Each active candidate component patch is projected into the current-readout kernel, with a per-pair L2 norm matched to the full learned first-pass donor write. The immediate readout displacement is zero. All 8,192 scored recipient/donor pairs, including failures, remain in the denominator.

AMM-04 changes the AMM-03 calibration instrument: every root has **four local donor pairs, both components scored, and each of the eight binary feedback labels independently flipped with known probability 0.35**. The two primary competing policies receive those *identical eight noisy observations* and exactly the same model-derived geometry. Memory is additionally trained from a declared amortized corpus of 64 independent models with 96 clean donor pairs/model; ridge coefficients and policy are locked before development.

## Source-only memory training

Training worlds: 64 (32 per family), 96 donor pairs/world.

| Candidate | Correct training donors out of 6,144 | Rate |
|---|---:|---:|
| SVD-canonical BPD component 0 | 4,864 | 79.17% |
| SVD-canonical BPD component 1 | 4,501 | 73.26% |

Pooled policy selects **component 0**. The seven-dimensional weight-derived causal-geometry memory uses ridge coefficient .01 with frozen fitted weights and a declared pseudo-count prior weight of four. The actual remembered action on new models varies with their full-product geometry. These model features contain neither development donor labels nor the clean four-pair calibration audit bits.

## New development results

**64 independent trained model roots, with 128 heldout donor-recipient cases/root (8,192 cases per policy)** and a separate 128-pair repeatability set per root.

| Frozen test policy | Successes / 8,192 | Accuracy |
|---|---:|---:|
| Memory plus four noisy local calibration pairs | **7,457** | **91.028%** |
| Training-only geometry memory, no local labels | 7,472 | 91.211% |
| Local noisy calibration alone (analytic geometry tie breaker) | 7,161 | 87.415% |
| Signed causal geometry rule | 7,102 | 86.694% |
| Training-global pooled component | 6,673 | 81.458% |
| Root-hashed random component | 6,458 | 78.833% |
| Equal-dose isotropic random-dark A | 2,242 | 27.368% |
| Equal-dose isotropic random-dark B | 2,206 | 26.929% |
| No patch | 41 | 0.500% |
| Full donor intermediate state (different dose) | 8,111 | 99.011% |

The registered primary memory-plus-calibration vs local-calibration effect is **+296 / 8,192 = +0.0361328125** (3.613 percentage points). Root directions: **3 positive, 0 negative, 61 tied**, yielding exact one-sided root sign p=0.125 and Holm p=0.25 across the three registered contrasts. The 2,000-draw root bootstrap development 95% percentile interval is **[0, +0.08301]**, supporting substantial uncertainty at the *independent root* level. This is a development estimate, not a confirmation interval.

| Registered comparison | Root-mean effect | Root sign directions (+/−/tie) | Holm p | Root-bootstrap 95% |
|---|---:|---|---:|---|
| Memory+local vs local-only (primary) | **+0.03613** | 3/0/61 | 0.250 | [0, +0.08301] |
| Memory+local vs geometry rule | +0.04333 | 7/4/53 | 0.274 | [−0.03979, +0.12915] |
| Memory+local vs pooled component | +0.09570 | 10/3/51 | 0.138 | [+0.00525, +0.18762] |

The learning policy and local-only policy chose **different components on 7/64 roots**. Three of those differences repaired 296 heldout donor outcomes; the other four root differences changed zero case scores. The two judge channels together flipped 184/512 observed calibration labels (35.94%), near the declared 35% error rate. Local feedback itself cost the memory-only policy 15 scored successes on this development sample (7,472 vs 7,457), a small, descriptive reduction.

An impossible-to-deploy posthoc root-wise test-label oracle obtains **7,891/8,192 = 96.33%**, providing **730 possible additional hits (8.91 percentage points)** above local-only. This larger headroom meets the experiment's conservative ceiling-screen, but the registered memory effect fails its science minimum. The test-label oracle is descriptive only and must never enter training, policy selection or a confirmation decision except as the predeclared *veto criterion*.

## Physical checks

- 64 separately trained teacher/decomposition roots; maximum independent teacher-task MSE `2.81e-8`, maximum BPD factor task MSE `0.10687` (all higher-error roots retained).
- Minimum canonical SVD singular-value gap across development roots `0.09418`, above the registered .01 degeneracy threshold.
- Maximum active equal-dose absolute norm discrepancy `2.38419e-7`; maximum immediate current-readout displacement `0`.
- Independent full donor-state replacement is a larger-dose ceiling, included as a separate control.
- All 64 roots retained, including 61 zero-effect roots. Twelve local engineering tests passed and one native check is performed only in the full BPD CI environment.

## Corrected EvalMath strict preflight: REDESIGN_OR_PILOT_ONLY

Pinned **merged** EvalMath source: `MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a` (fixed zero-effect/zero-variance gate in PR #51). Exact source blob `0ff062f76f5534d9f18425f986ae3e750397f50c`.

- Independent development roots: **64**; 128 repeated cases per root.
- Per-root paired effect: memory+local minus local-only.
- Matched sham: absolute difference of local-only *same selected action* measured on the primary 128-pair set and an independently sampled 128-pair repeatability set for each root. A repeatability contrast measures case-sampling noise at matched physical dose.
- Sample-maximum upper bound on population 95th quantile at >=95% confidence: **7/128 = 0.0546875 (5.47pp)** from 64 independent roots.
- Scientific minimum `0.08` (8pp) dominates the sham floor and 1/128 measurement resolution. Registered target is `0.20`.
- Observed pilot root-mean effect `0.0361328125`, root-world sigma `0.1788010831`. Corrected normal planning estimate is 24 independent confirmation roots to reach a hypothetical target under its assumptions, and **six** is the exact one-sided sign-test minimum for three registered comparisons. Both numbers are planning quantities, not proof of a positive effect.
- **Decision `REDESIGN_OR_PILOT_ONLY`**; explicit reason `pilot_effect_not_above_mechanistic_variation_floor`. The 128-root confirmation remains untouched.

The heldout study is therefore a directional, **power-limited development result**, with gains concentrated in three roots, below the physically meaningful effect minimum. Its repeated-case accuracy differences cannot substitute for independent-root support.

## Next questions

A follow-on should improve the policy's ability to recognize cases where noisy feedback overrides a correct geometric prior, or study strictly less-informed per-root local calibration under a fixed authorization budget. New candidate policies should be tuned on independent new development worlds and scored on a newly frozen population. Merely adding another 128 cases to the same model roots will not resolve cross-root variation. Preserve this entire negative study, the AMM03 negative receipts, and both sealed confirmation partitions.

## Reproduce

From the repository root, with CPU PyTorch 2.10 and Python 3.12/3.13:

```bash
python -m unittest mechinterp.bpd.experiments.amm04_noisy_calibration.test_experiment -v
python -m mechinterp.bpd.experiments.amm04_noisy_calibration.experiment --phase training --out /tmp/amm04_training.json
python -m mechinterp.bpd.experiments.amm04_noisy_calibration.experiment --phase development --memory /tmp/amm04_training.json --out /tmp/amm04_development.json
```

Run merged EvalMath `preflight.py --pilot <development-preflight-input> --strict` and expect exit 2 with `REDESIGN_OR_PILOT_ONLY`. The source-locked training, development and preflight compact receipts reside under `results/`, with full raw JSON supplied as a portable handoff ZIP. GitHub Actions reconstructs the training memory and every root's scored candidate, noisy calibration, chosen action and repeatability bits.

The result concerns **trained synthetic linear recurrences**; it makes no claim about pretrained TRM/RMP behavior, unique natural-language mechanisms, actual model agents, or a live Buzz/Nostr relay.
