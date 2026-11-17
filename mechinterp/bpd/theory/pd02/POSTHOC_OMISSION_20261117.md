# BPD-PD-02: posthoc error localization after registered FAIL

**Date:** 2026-11-17. **Evidence type:** posthoc analysis of the **original unchanged** registered run [44067120071](https://github.com/MoralityLabAI/BPD/actions/runs/44067120071). Its frozen result remains 6/24 hard-gate successes versus 10/24 soft, paired McNemar p=0.34375.

[Full original JSON receipt](https://github.com/MoralityLabAI/BPD/actions/runs/44067120071/artifacts/14321006811), SHA256 \`b28dd8feecabc7f934185f6f88d38ed1fbc5a0e8c7bbb7169c12da3615220917\`. The stdlib script [omission_audit.py](omission_audit.py) reads that receipt without fitting or evaluating any model.

## Directional gate error census

The trained hard-disjoint arm emitted masks for 24 seeds × 128 held-out boards × four tied passes. Each pass has one planted active coordinate and one planted inactive coordinate. After matching the global learned component permutation:

| Outcome | Count |
|---|---:|
| Required component gate invocations | 12,288 |
| Missing required component (false-off) | **498 (4.0527%)** |
| Inactive component mistakenly switched on | **0** |
| Boards with all four pass masks correct | **2,614 / 3,072 (85.09%)** |

These are held-out **invocation-level**, not seed-independent sample counts. The single-seed scientific unit remains the model seed.

All 24 hard-gated dictionaries exceeded read-axis cosine 0.90 (mean 0.99925), so this experiment's failures primarily concern gate threshold crossings on lower-amplitude active inputs. For seed 197, both first-phase gates fire on every board while second-phase required gates fire on only 68.0% and 73.4% of boards; board-level correct routing is 50.0%.

## Four-pass product prediction

The generator samples the signed activation amplitude independently at each pass. For a fixed dictionary/gate and the true phase-specific active component, let \(q_t\) be the chance of firing that component at pass \(t\). With the observed zero false-on rate and independent per-pass inputs, expected complete-board routing is

\[
P(\mathrm{all\ four\ correct})=\prod_{t=0}^3 q_t.
\]

Across the 24 seeds, the product predicts **0.850946 mean** correct-board fraction; measured mean is **0.850911**. The per-seed product/measured correlation is **0.998725**, mean absolute difference **0.003895**, and RMSE **0.006177**. The agreement is a retrospective consistency check, conditional on independence of the generator's per-pass inputs and the observed lack of wrong-component activations.

A mean per-invocation success of \(1-498/12288 = 0.959473\) gives the approximate four-pass probability \(0.959473^4=0.84748\). This closely matches the more accurate per-seed pass-wise product and shows how a small false-off rate compounds across four compulsory decisions. An exact per-pass 100% success predictor was *not* recovered in 18/24 seeds under the registered 95% board-level route requirement.

## Diagnosis and next falsification

The optimized hard-gated factors learned the planted axes even in many scientific failures; gate threshold calibration was unstable across the amplitude range. The SVD-frozen hard baseline attained routing >=0.95 in just 5/24 seeds, supporting a gate-optimization obstacle that survives perfect dictionary initialization.

This diagnostic was discovered after unsealing the registered result. Any next experiment aimed at changing training coefficients, gate initialization, one-of-two routing, margin constraints, or threshold calibration requires its own prospective registration and fresh seed cohort. The BPD-PD-02 full JSON and negative verdict remain frozen.
