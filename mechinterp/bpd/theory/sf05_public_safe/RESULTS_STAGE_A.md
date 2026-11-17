# BPD-SF05 Stage A: public-tangent safety breaks finite nonlinear public preservation

**2026-11-18. Development-only, excluded roots 901 and 902.** No new independent scientific or source-family result is claimed. [Source-locked hosted workflow](https://github.com/MoralityLabAI/BPD/actions/runs/44135773574) passed all **12 unit tests** and the trained excluded-seed assay. [Raw source-locked JSON](results/stage_a_excluded_hosted.json) and [CI artifact](https://github.com/MoralityLabAI/BPD/actions/runs/44135773574/artifacts/14352556350). The experiment verified every Git blob recorded in [LOCK.json](LOCK.json) before running. Parent SF04 source blob remains unchanged. PyTorch version: `2.5.1+cu124` (CPU).

## The important counterexample

Projecting an entire BPD physical write into `K=ker Dq∩ker D(q∘F_U)` **did preserve the actual prequery current readout and cancel first-order public sensitivity**. The finite nonlinear *public* result changed much more than the original raw physical donor write, and authorized donor retrieval dropped.

| Excluded seed | Eligible auth | Full donor raw recovery | Full donor after tangent-safe projection | Public raw absolute Δ | Public after tangent-safe projection absolute Δ | L2 dose removed | 14-factor after projection | Rank-16 public absolute Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 901 | 242 | 100.0% | 26.9% | 0.0111 | 0.3541 | 15.4% | 34.7% | 0.3506 |
| 902 | 263 | 100.0% | 42.6% | 0.0160 | 0.0908 | 14.9% | 32.7% | 0.0875 |

The strongest empirical issue: on seed 901, removing **15.4%** of the full physical write's L2 norm worsened average public-output deviation from **0.0111 to 0.3541** and authorized recovery from **100% to 26.9%**. Seed 902 shows a smaller yet substantial deterioration.

These two roots have max first-order public gradient residual < `5e-16` and **exact** prequery current-logit conservation. Their full BPD physical operators agree with the sheared-gauge versions within `6e-8`; the rank-restricted output-SVD writes agree within `1.7e-6`. The failure is therefore downstream nonlinear geometry, rather than a numerical failure of the tangent intersection or a singleton BPD coordinate artifact.

## Nulls and privileged references

Across the 16 matched random K-directions per root, signed-authorized-gradient-matched mean donor recovery for the **full physical projected write** was 6.1% on 901 and 13.8% on 902; the corresponding BPD projected donor recovery was 26.9% and 42.6%. These comparisons are **development diagnostics**, not independently calibrated statistics, and the matched nulls often produce appreciable finite public deviation too (≈0.107 and 0.079, respectively).

Rank-16 BPD and teacher-C SVD, exact dark donor-state, and full BPD physical write all exhibit similar finite effects after tangent projection. The teacher-C baseline confirms that the phenomenon is tied to this trained physical transition and the off-manifold projection, with no supported claim of a uniquely trained BPD component role.

## Interpretation

For the original full physical source-label donor change, the public output is naturally close to its baseline under a clean source-tag swap. The Euclidean tangent correction changes the full physical displacement, potentially leaving a curved reachable manifold or changing higher-order cancellations. Being orthogonal to the public Jacobian at the **recipient's unedited state** does not ensure finite public equality after the edit.

The follow-up is a **finite-output** continuation/projection on the nonlinear public level set, starting with the raw physical donor write and correcting only the residual public-output discrepancy. This requires a separate post-result engineering protocol and must report its privileged public-readout oracle, correction size, success rate, stability and performance against the same correction applied to null controls.

The SF04 public-safe partial-coalition gate remains FAIL 3/8, and SF03's strong mechanism gate remains FAIL 0/32. This Stage A observation does not revise either prior verdict or touch the sealed ARG-03-N 168-case test.
