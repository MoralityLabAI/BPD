# BPD-v3 FLOOR-01 / BASIS-01 source-only execution receipt

**17 November 2026. Evidence class:** SYNTHETIC_SOURCE_ONLY. Source registration `bddd0c37bc0ddaf83164dd387273b940ca134cd7` preceded the first synthetic run. The v2 B4 result is unchanged: **INCOMPLETE**.

## Executed environment

Local CPU Python 3.13, NumPy, pytest. `python -m pytest -q mechinterp/bpd/tests/test_v3_floor_and_basis.py mechinterp/bpd/tests/test_v3_uncertainty.py`: **11/11 passed**. The isolated GitHub Actions workflow also passed on the source-only branch. Command:

```bash
python mechinterp/bpd/v3/floor_and_basis.py --smoke --out runs/bpd_v3/synthetic_smoke.json
```

Full synthetic result is reproducibly emitted as a GitHub Actions artifact. Seed 711, 24 independent toy roots, K=8 bounded per-step dithers, 63 nonlinear tanh steps, and readouts at passes 20, 41 and 62. There are 2 toy readout cells and 2 possible classes per cell. This construction is an exact mathematical/instrument positive control, not a 7M TRM evaluation.

| Toy measure | Value |
|---|---:|
| Finite-K plug-in JS `J_delta`, nats/cell | **0.465054747829** |
| JS at readouts 20 / 41 / 62 | 0.465054747901 / 0.465054747689 / 0.465054747897 |
| Toy roots with JS > 1e-8 | 18 / 24 |
| Risk for the static ensemble-barycentre predictor | 0.465054747829 |
| KL between barycentre and itself | 1.48e-17 |
| KL risk-identity residual | **-1.11e-16** |
| Paired perfectly driven reference KL | 0 |
| Conditional leak bound, for the perfect reference drive | 0.465054747829 |
| Independent surrogate-ensemble mean KL | 0.095596820622 |

The perfectly driven toy saturates the information inequality. A static surrogate matching one selected target dither has zero KL on that *one* realization even when the target ensemble has positive JS. This is a counterexample to treating a single-reference KL as proof of leakage. The driven comparison, unlike a post-hoc single-run match, is paired across the realized dithers.

## Finite-K sensitivity and independent-board uncertainty

Using the same synthetic model and a declared deterministic RNG, followed by 5,000 bootstrap replicates over **boards** (seed 20261117):

| Toy dithers K | Plug-in JS (nats/cell) | Board-bootstrap 95% percentile interval |
|---:|---:|---:|
| 8 | 0.465055 | [0.344741, 0.573458] |
| 32 | 0.507674 | [0.371409, 0.619603] |
| 128 (exploratory post-protocol check) | 0.517981 | [0.375106, 0.633021] |

Finite-K plug-in bias persists. Bootstrap samples independent toy roots, not readout cells, per-step visits or dither draws. These intervals do not account for all uncertainty in choosing the toy model. The K values represent synthetic sample-size sensitivity and say nothing about the native `J_delta`.

## Exact matrix and off-budget audit

One random `W in R^(6×8)`, 512 synthetic activation vectors (72% coordinate-zero probability), seed 711, `G = I`. Six bases; each factorization satisfies `(VU)^T = W`. All 18 basis-budget combinations at beta0 in {0.01, 0.03, 0.10} certified every observed module deletion within numerical roundoff. Max relative full-on residual: **2.46e-16**.

| Basis family | C | Mean active at beta0=0.03 | Mean module-output softmax KL |
|---|---:|---:|---:|
| Input-complete neuron | 8 | **2.0957** | 2.44e-6 |
| Input-complete SVD | 8 | 4.8848 | 1.38e-5 |
| Input-complete random | 8 | 6.9961 | 1.56e-5 |
| Output-complete coordinate | 6 | 5.3457 | 5.06e-6 |
| Output-complete SVD | 6 | 4.8848 | 1.38e-5 |
| Output-complete random | 6 | 5.2988 | 6.39e-6 |

The input-complete and output-complete families have different C; comparisons across those two rows are descriptive only. Module-output softmax KL is a *diagnostic proxy* and not downstream digit-head KL. The certificate applies exactly at the linear module under the chosen seminorm; recursive nonlinear transport needs measured Lipschitz or Taylor controls.

In the separate two-dimensional support-identification witness, a full-support pass type has exactly two active coordinates under all rotations. Adding complementary one-axis pass types makes the coordinate basis recover a pooled mean of 4/3 active terms, versus 2 under 45-degree mixing. Pooled advantage **2/3**, with the coordinate basis and signed permutation as grid minima. This demonstrates the exact-zero-atom requirement; it does not recover a learned basis from native activations.

## Native outstanding evidence

The real `gaoxin492` and `yagizdevre` checkpoints are absent from the CPU environment; repository agent instructions require local SHA-pinned models and prohibit automatic weight downloading. The full `B4 v2` causal-importance network is absent from the saved Stage-4 artifact. As a result: no native dithered `J_delta`, no native zero-fit neuron/SVD comparison, no v2 information-leak magnitude, no Sudoku task-recovery estimate, and no native component-level mechanistic attribution have been measured here.

Next native artifact: checkpoint-verified 183-board K=8 `target_logits[B,K,R,81,9]` at passes 20/41/62, with manifest, paired seeds, arithmetic controls and board-cluster uncertainty. Then export pass-pooled clean module inputs `[N,d_in]` from the chosen `down_proj` and `o_proj` matrices, and run the registered input/output complete baselines before trying an offline learned basis.
