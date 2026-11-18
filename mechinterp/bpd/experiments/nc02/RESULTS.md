# BPD-NC-02: eight-root planted-path engineering results

**Date:** 2026-11-18 (UTC)  
**Evidence class:** registered synthetic positive control, **not scientific confirmation**.  
**Prospective protocol:** `0a8b0c1d676b24dbc657c9709792efb483f768f2` (committed before NC-02 code and new data)  
**Native GitHub Actions run:** [44230596985](https://github.com/MoralityLabAI/BPD/actions/runs/44230596985) — both `source-checks` and `development-roots` **success**.  
**Complete 8-root raw artifact:** [artifact 14394364560](https://github.com/MoralityLabAI/BPD/actions/runs/44230596985/artifacts/14394364560).

## Confirmed engineering findings

Eight independently initialized and trained shared-weight nonlinear four-pass model roots, seeds **200–207**, used 2,048 training, 1,024 graph-discovery and 1,024 separate evaluation examples **per model**. Two planted paths are activated by the exogenous context sign; two further lanes provide equal-gain physical-write distractors.

| Arm | Eligible independent roots | Communities detected per root | Planted exact-path recovery | Root-mean keep-only KL advantage vs. discovery-dose-matched shams | Root-mean ablation NLL increase |
|---|---:|---|---|---:|---:|
| Native input neurons | 8/8 | 2 in each root | Both paths, 8/8 roots | **+0.320608 nats** | +0.338347 nats |
| Full-rank BPD SVD initialization | 8/8 | 2 in each root | Not a native-coordinate comparison | **+0.320608 nats** | +0.338347 nats |
| Orthogonal factor rotation | 0/8 | 0 in every root | Not applicable | Not estimable | Not estimable |

The **root-resampled 95% bootstrap interval** for the native keep-only KL advantage is **[+0.319353, +0.321813] nats** (4,096 bootstrap resamples across eight trained models). The analogous ablation NLL increase interval is **[+0.337490, +0.339129] nats**. Model roots, rather than prompts or recurrent passes, are the bootstrap units.

Each native root passed exact target-versus-open-mask parity, planted-path recovery, the eight discovery-only dose-matched sham criterion, and the zero-readout intervention. Across 32 per-pass-independent graph shuffles per root, the observed mean count of cross-pass communities was **zero**. The readout-zero graph likewise returned **zero** communities in all eight native roots. All eight model roots met the task gate; source tests passed **10/10** on the native `BPDModel/TiedComponentLinear` code path.

### Per-root primary causal outcome

The mean advantage is calculated from the *first two discovery-ranked communities* in each root. No oracle-based outcome selection was used to rank them.

| Root | Native exact planted paths | Native keep-only KL advantage (mean of top two, nats) | Rotated discoveries |
|---|---:|---:|---:|
| 200 | 2/2 | +0.322876 | 0 |
| 201 | 2/2 | +0.319432 | 0 |
| 202 | 2/2 | +0.319298 | 0 |
| 203 | 2/2 | +0.321750 | 0 |
| 204 | 2/2 | +0.322521 | 0 |
| 205 | 2/2 | +0.317225 | 0 |
| 206 | 2/2 | +0.321422 | 0 |
| 207 | 2/2 | +0.320341 | 0 |

## Interpretation and limitations

**The planted-path instrument works on this constrained recurrence.** Exact native paths appear as coherent cross-pass co-attribution communities, produce causally important writes, and outperform equally sized/sham-dose alternatives under keep-only intervention. The negative readout and shuffled graph controls close.

The **rotated read-basis null is important**: it preserves the same physical weight and input-output behavior up to floating-point error while abolishing the qualifying graph communities. This demonstrates that the graph procedure is strongly **coordinate dependent**. The full-rank SVD basis retains the planted grouping because the block-structured shift is near axis-aligned under that factorization.

The test imposes a stage-shift skeleton, equalizes lane gains by projection and uses context gating with an externally defined label. Its narrow root bootstrap interval measures variability within this intentionally homogeneous model family. It cannot establish discovery in an unconstrained transformer or support cross-family generalization. Necessity-vs-primary-sham advantages are not the main claim because incoherent shams can accidentally remove a necessary link; the fixed primary statistic is the keep-only KL contrast.

**Scientific status:** `ENGINEERING_POSITIVE_CONTROL`; prospective confirmation remains **sealed**. The independent-root EvalMath power preflight and a new population/protocol are required before a confirmatory mechanism assertion. ARG-03-N and AMM-03 confirmation material were untouched.

## Reproduction and integrity

- Python source SHA-256: `83b5c1b789157223c4af94226e30cc3833fb0c346e460e4c53287bc93e752ea4`
- Protocol file SHA-256: `c6f2b16c7f2491bc273c679a933c8065a0a71d5717f06a712064d0f09ee8657e`
- Full zipped CI artifact SHA-256: `196acce882618aca82578c1f1488499b30c195e3a911bb259ab805267bf31b70`
- The artifact contains `NC02_ENGINEERING_SUMMARY.json` plus the eight complete `nc02_root_{seed}.json` receipts, including all matched-sham endpoints.
- Execute `python -m pytest -q mechinterp/bpd/experiments/nc02/test_experiment.py` and run `python -m mechinterp.bpd.experiments.nc02.experiment --seed 200 --steps 128 --discovery-size 1024 --evaluation-size 1024 --out /tmp/nc02_root_200.json` from the repository root to reproduce a development root.

**Next preregistered direction, not yet run:** randomize route identities or permit learned off-route coupling while preserving a fixed causal intervention budget. Test recovery beyond a fixed stage/lane coordinate chart; add independently trained nonlinear recurrent models or a compact TRM checkpoint before interpreting neuron communities in general models.
