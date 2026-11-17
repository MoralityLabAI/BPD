# AMM-01 prospective results (2026-11-18)

**Evidence:** Prospective positive-control result for deterministic strategy-memory transfer in engineered rank-at-most-two tied linear recurrent systems. The conclusion is confined to this synthetic instrument.

## Pre-registration and source provenance

- Protocol-first commit: `cd0ed97d147bcf54340233e9243abf1eaea6e02a`.
- Frozen implementation and verifier commit preceding test: `2adc180d5a65299d7dec1530b408587f34b778f8`.
- Python runner SHA-256: `f68468bcc6248ac42349aa8438362cb7cfa4eadecfc8e57a725ed83b31b4352b`.
- Training memory SHA-256: `2b7f42d00af7ff3c683fd4c39d38b1770b05fb2f5976e77704c2f14f3e0a1b94`.
- Complete development raw receipt canonical SHA-256: `ed47c75ee6e2e5a584957e2be623635df5056c8a263c7e7802cd4865b245c4a4`.
- Complete confirmation raw receipt canonical SHA-256: `5317c7ff6b2f512bc41556478f9439d3df7349dc06d0b14c9fbe951c1845cf4c`.
- The exact code reconstructs every scored binary case from phase-separated SHA-256 seeds. Committed `results/*_LOCK.json` bind each full raw receipt to the frozen source.
- EvalMath pinned source for strict preflight: `MoralityLabAI/EvalMath@2fdd5fe3f499070b63c474a2455f0e2c00fc55a8`. CI independently reruns the pinned preflight.

## Frozen design

The independent unit is a world-specific shared physical 4×4 transition with one random latent sign and gain. There are two opaque families: one transmits a current-dark unit-dose perturbation to the readout at t=1, the other at t=2. A policy can query exactly one horizon per scored case, with additive Gaussian noise SD 0.55. Family identity is available; latent sign, gain, transition coefficients and noiseless answers are withheld from the policy. The family-keyed table learns its probe horizon solely from 64 training worlds.

Matched arms: training-derived pooled best horizon, swapped family-key routing, deterministic random horizon, and an independently noised same-horizon pooled sham. The equal-dose query is `d=(0,1/√2,-1/√2,0)`, so `q(d)=0` and `||d||₂=1` in every arm.

## Development and strict EvalMath preflight

64 independent development roots × 256 repeated case observations = 16,384 scored cases per arm. Development memory hit rate 15,508/16,384 (94.65%); pooled 12,166/16,384 (74.26%). Root-mean effect +0.20398. The 64 matched sham-world resamples provide a 95th-quantile / 95%-confidence sample-maximum floor of 0.1015625; the preregistered scientific effect floor is +0.15. The exact EvalMath planner returned `READY_FOR_CONFIRMATION` with no stated blockers and an approximate 46 roots needed for 80% planning power under its normal approximation; 512 new roots were already registered.

## Prospective confirmation results

**512 independent worlds × 256 repeated one-probe cases = 131,072 scored cases per arm.**

| Frozen arm | Correct / 131,072 | Accuracy |
|---|---:|---:|
| Family-keyed strategy memory | 123,388 | 94.138% |
| Training-derived family-agnostic pooled horizon | 96,571 | 73.678% |
| Deterministic random horizon | 95,349 | 72.746% |
| Swapped family-keyed memory | 67,621 | 51.591% |
| Pooled horizon, new independent sham noise | 97,205 | 74.162% |

| Prespecified comparison (memory minus control) | Mean root effect | Root-cluster bootstrap 95% CI | Holm-adjusted one-sided exact root sign p |
|---|---:|---:|---:|
| Pooled (primary) | **+0.2045975** | **[+0.187363, +0.222488]** | 8.6362e-78 |
| Swapped | +0.4254684 | [+0.421165, +0.429710] | 2.2375e-154 |
| Random | +0.2139206 | [+0.210930, +0.217087] | 2.2375e-154 |

The primary comparison has **256 positive independent roots, zero negative roots, and 256 ties**. The ties correspond to roots whose family-keyed probe equals the pooled probe, and are retained in the reported effect and sign test. All cases and all negative-control failures are retained. The primary bootstrap lower bound (+0.18736) exceeds max(scientific floor +0.15, matched-sham floor +0.10156). The frozen primary gate therefore **passes**.

## Interpreting the result

This setup deliberately gives different probe horizons different information value. Training-only family memory recovers the correct horizon and transfers it to independent worlds with different signs and gains. The swapped-memory arm demonstrates that transporting the wrong family rule loses the advantage; the random arm places the gain against a one-probe baseline.

The timing structure and family key are engineered, and the policies contain no trained neural models or natural-language reasoning. Mechanistic discovery, BPD factor identifiability, model-to-model agent-memory transfer and a deployed Buzz system remain untested. Significance is assessed over independent worlds; 256 measurements from one world are repeatedly measured cases rather than independent roots. The exact finite synthetic instrument provides a rehearsal of the provenance and evaluation pipeline.

## Audit and next experiment

Reproduce with:

```bash
python -m unittest mechinterp.bpd.experiments.amm01_agent_memory.test_experiment -v
python -m mechinterp.bpd.experiments.amm01_agent_memory.verify \
  --lock mechinterp/bpd/experiments/amm01_agent_memory/results/DEVELOPMENT_LOCK.json
python -m mechinterp.bpd.experiments.amm01_agent_memory.verify \
  --lock mechinterp/bpd/experiments/amm01_agent_memory/results/CONFIRMATION_LOCK.json \
  --preflight mechinterp/bpd/experiments/amm01_agent_memory/results/PREFLIGHT.json
```

The next registered test should let independent Goose/Codex or local-model agents learn probing/repair policies from separate BPD-CQ-style trained worlds, transfer those memories to genuinely new recurrent model families, and evaluate independently defined donor-behavior outcomes against equal-dose shams. Agent policy quality and causal mechanism evidence will require separate gates.

Buzz export currently creates a privacy-minimized **unsigned** NIP-29 kind-9 message draft with the GitHub provenance digest. No event was signed, sent to a relay, or shared with other agents in this run.
