# SCCG-04 first local development receipt

Source registration b37877f4b92c32bb3338b2ef47f972a2a36f2426; pre-scoring
batched tangent-index correction 2479ae4749584c81275d2c6c5355095e9c634d1b.
Local executed support_gym.py Git blob c992defb9f5319786303b11f6daccdabeb2f6f01
matches the remote source. SHA256
1a67297a2dae76da8183250e237a42750492c6ddbede6825eacdd2f1a005162f.
Python 3.13.5, torch 2.10.0+cpu, numpy 2.3.5, one torch thread.
First local output is preserved as receipt.json (serialized aggregate and
per-root outcomes). Local output SHA256
af8c106f901a2d1a5c8cffbb40f3368e4392e0e9862888031aad1dda2ddf1348.
No native model or checkpoint used. No post-result model tuning.

| Arm | Top 1/48 | By 4/48 | Replayed exact checks |
|---|---:|---:|---:|
| Analytic finite cap | 48 | 48 | 48 |
| Tangent | 42 | 46 | 60 |
| Random | 40 | 47 | 61 |
| Magnitude | 41 | 45 | 62 |
| Recurrent 41 | 38 | 46 | 67 |
| Recurrent 43 | 41 | 45 | 63 |
| Recurrent 47 | 40 | 46 | 67 |

Each of 96 training and 48 development roots has at least one positive
support. Exactly 4,029/5,376 training supports and 1,995/2,688 development
supports are positive. The analytic baseline is at ceiling; the learned
policy shows no advantage on this constructed family. Positive support
density (74.2% on development) makes random ordering strong, and the
shared finite cap geometry permits a fast vectorized analytic ranking.
There were no exact/float disagreements outside 1e-8; one near-boundary
decision retains its exact rational label. All exact verification calls
were independently recomputed by the same exact geometry function, not
accepted on learned scores. Per-root aggregate arithmetic is checked
during receipt publication.

96x56 + 48x56 = 8064 exact offline labels; each ranker receives 160
updates with eight sampled complete roots per update, each of 56
supports, or 71680 candidate-example visits per fit. All arms score
all candidates. Timings in receipt are single-process measurements
with different batch organization (analytic includes all three
classical scores; ranker timing excludes feature preparation and
training). No hardware or wall-clock speedup claim follows. The
analytic baseline needs no training labels at deployment.

Next: design a higher-rank, rare-success, nonorthogonal finite-response
family with both solvable and infeasible roots, fixed by construction
rather than selected on evaluation results. Include exact circuit
screening, branch-and-bound, and analytic cap controls. Only use learning
where certified search leaves meaningful candidate-work headroom.
Separate proposal cost, proof/verification cost and amortization.
Require new roots and prospective EvalMath preflight before any
statistical claim. All existing protected cohorts remain untouched.
