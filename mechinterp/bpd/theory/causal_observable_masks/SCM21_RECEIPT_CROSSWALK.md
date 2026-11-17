# SCM-21 causal failure as a BPD observability question

Sources: [`Compactification-Theory` PR #207](https://github.com/MoralityLabAI/Compactification-Theory/pull/207), merged at commit `64d29001923329759508407a5dd649cddf3a872a`. Its published `research/dimensional_atlas/gym/scm21_conservative_causal/results/summary.json` has Git blob SHA `a765043e0837268d52c0548e26d477ba1e813da9`. The following retrospective figures concern the **causal model** at zero-based visit 1, on the identical common-eligible 128+128 pairs in each of three seed blocks, ID/OOD.

| Method | Correct donor transfers / 768 | Mean per-pair patch L2 (weighted) |
| --- | ---: | ---: |
| Full CLS donor delta | 432 | 11.4545 |
| Head-readable state delta | 413 | 10.7871 |
| Difference-PCA rank four | 147 | 7.7328 |
| Ordinary PCA rank four | 101 | 7.2865 |
| Native VPD write | 0 | 0.7017 |
| Native write span projected delta | 0 | 1.4048 |
| Random fiber rank four, norm-matched to native write | 0 | 0.7017 |

SCM-21's conservative training preserved clean accuracy in all six seed/split comparisons, with 0/768 trained native-write successes. The full rank-four alternatives act at different magnitudes; their success counts alone cannot identify a favorable direction at a common dose. SCM-22/23 subsequently explored per-pair norm-matching in a **new synthetic model**, with stronger head-readable effects and smaller, uneven delayed fiber effects. Those later results are a design guide, not tests of SCM-21's private weights.

The BPD-CQ-01 theorem says an optimal expected-L0 basis may admit many equivalent component coordinates, while only a subset of **readout- and horizon-conditioned interventions** is stable. Sparse reconstruction cannot determine causal intervention identity. The previous BPD-PD-02/03 evidence similarly separates recovery of planted read axes from correct pass routing.

## Applied protocol for trained BPD (new registration required)

1. Select candidate BPD component axes and an optimal-basis equivalence class using calibration only. Freeze the physical write site, times, source hashes and readout mapping.
2. Compute native present and delayed response matrices / local Jacobian-vector products for the actual head. Group masks by delayed observable effect, using the certified equivalence criterion where available.
3. Evaluate patch directions at identical **per-pair** intervention norms, with independent sham, random-orthogonal and same-subspace direction-shuffle controls; avoid double-counting seeds and tasks.
4. Report readout current/future mismatch, named task output correctness, loss of clean math, projection rank/condition, pass variation, and feature-splitting/merging across equivalent dictionaries.
5. Use EvalMath mechanistic-power preflight before sampling independent generator worlds. ARG-03-N remains strictly sealed.

The current BPD-CQ-01 code proves finite linear statements and exact CPU fixtures. Its observability quotient is not a trained dark-fiber causal measurement.
