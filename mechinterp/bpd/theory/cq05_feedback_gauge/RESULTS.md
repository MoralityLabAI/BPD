# BPD-CQ-05: first prospectively registered feedback / gauge pilot

**Date: 2026-11-17.** [Protocol committed first](PROTOCOL.md), commit \`3faa36c7\`; [frozen experiment source](experiment.py), commit \`2ec53750\`. Independent model seeds: **1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249**. Engineering-only excluded seeds 997/1009/1013/1021. Two distinct trained rank-two stages: learned weight-tied physical recurrence, then trained full-capacity BPD V/U. SVD of the *full reconstructed physical operator* fixes the execution-coordinate gauge before two stage-specific hard-mask controllers learn.

**Evidence class: prospective synthetic pilot**. There are eight independently trained model roots and 256 independent task examples + 128 fresh donor pairs per root. The repeated cases are *not* 2,048 and 1,024 independent models. The [strict EvalMath preflight](PREFLIGHT.md) returned \`REDESIGN_OR_PILOT_ONLY\`, including an independent 59+ sham-world requirement and a normal-model planning estimate of **26 confirmation roots** at the registered target.

## Task performance and component budget

Both behavioral outputs are scored jointly: first y1=a and then y2=b, with max absolute error ≤.12. All arms see the same 256 cases per model root. Support rule is the calibration-selected best deterministic pattern per exact-zero source class, subject to the learned feedback controller's **calibration** slot budget. It realizes lower cost on the fresh test cohort, so its lower accuracy is *not a causal estimate of policy quality at identical realized test-time slots*.

| Seed | Feedback correct / 256 | Support-rule correct | Input-blind static correct | Feedback slots / 4 | Random-slot correct | Full |
|---|---:|---:|---:|---:|---:|---:|
| 1201 | 244 | 183 | 159 | 2.473 | 144 | 256 |
| 1213 | 228 | 185 | 162 | 2.297 | 101 | 256 |
| 1217 | 255 | 188 | 172 | 2.398 | 150 | 256 |
| 1223 | 239 | 179 | 170 | 2.184 | 81 | 256 |
| 1229 | 232 | 183 | 151 | 1.664 | 86 | 256 |
| 1231 | 246 | 189 | 169 | 2.270 | 142 | 256 |
| 1237 | 231 | 167 | 154 | 1.766 | 65 | 256 |
| 1249 | 253 | 194 | 188 | 1.957 | 74 | 256 |
| **Total** | **1928 (94.14%)** | **1468 (71.68%)** | **1325 (64.70%)** | **2.126** | **843 (41.16%)** | **2048 (100%)** |

The full four-slot budget is reduced by **46.85%** under feedback. Input-blind static actual mean cost = **2.1719**; the support-rule actual mean cost = **1.7856**, below feedback's 2.126. Matched-slot permutation has the same 2.126 slots per case by construction. Seven of eight model roots meet the prespecified exploratory ≥90% feedback task hit criterion with average slots ≤3; root **1213** has 228/256=89.06%.

The initial first-pass gate directly observes a and b, and it was trained from an exhaustive task-label oracle. Pass two observes only the **reached recurrent state** h1, including after interventions. The model is a synthetic linear recurrence with explicit low-dimensional input variables. The outcome establishes performance of supervised *execution allocation*, not recovery of a hidden natural language mechanism.

## Primary causal test: all donor pairs counted

Recipients and donors share a, independently vary b, and have behavioral donor target b_d set *before* any BPD decomposition or patch. Each donor score is defined over **all 128 generated pairs per root**, including unclean examples, changed masks and zero-dose component writes. No pair was replaced.

| Seed | Feedback current-dark BPD | Feedback random-dark, dose matched | Frozen pass-2 BPD | Full donor state |
|---|---:|---:|---:|---:|
| 1201 | 70 | 18 | 12 | 112 |
| 1213 | 89 | 17 | 36 | 107 |
| 1217 | 44 | 6 | 20 | 126 |
| 1223 | 74 | 18 | 19 | 114 |
| 1229 | **19** | 9 | 13 | 83 |
| 1231 | 31 | 9 | 4 | 119 |
| 1237 | 51 | 10 | 41 | 96 |
| 1249 | 68 | 7 | 44 | 128 |
| **Total /1024** | **446 (43.55%)** | **94 (9.18%)** | **189 (18.46%)** | **885 (86.43%)** |

The raw native component write arm, which may disturb the immediate readout, gives **351/1024** with recomputed feedback. Its projected current-dark and norm-restored form gives **446/1024**. Sham is **1/1024**. Actual patch norm-matching error ≤**5.96e-8** for every generated pair.

The prespecified primary difference, feedback current-dark BPD minus same-dose random current-dark, is **+352/1024 = +34.375 percentage points**. Per-root differences in the seed order above are \`[.40625, .5625, .296875, .4375, .078125, .171875, .3203125, .4765625]\`. Seven roots exceed the registered +15-point practical threshold; **seed 1229 is the retained scientific failure**. The registered direction gate passes its ≥6/8 exploratory rule.

The second prespecified comparison, recomputing pass-two feedback after the current-dark patch versus freezing the recipient's original pass-two mask, gives **+257/1024 = +25.098 percentage points**. Every model root has positive difference, with the weakest +0.046875. The feedback continuation actively changes a pass-two mask on hundreds of patched donor states; a post-hoc descriptive rate is stored per arm and root in the locked receipt.

Under a symmetric root-sign null, both comparisons are positive in **8/8 roots**, each having two-sided exact sign p=**0.0078125**; a conservative Holm two-comparison bound is ≤**0.015625**. An illustrative Student-t 95% interval for the mean *root* primary effect is **[0.209,0.478]**, and for feedback vs frozen is **[0.117,0.385]**, conditional on a normal root-effect approximation. These are descriptive pilot statistics with an uncalibrated independent-world sham tail, not confirmation.

**Diagnostic eligibility, never an exclusion:** Across the full 1,024 donor pairs, 726 had both original trajectories clean, 180 shared the same four-bit gate plan, only 100 were simultaneously clean and same-plan, and 106 native patches had zero physical dose. The reported 446/1024 primary numerator therefore measures an uncensored source population rather than the favorable conditional minority in CQ-04.

## Gauge and native execution

For each model and both registered GL(2) shears, the full represented W agrees within approximately **2.23e-16** (double-precision arithmetic). Reusing trained hard mask identities in the **raw sheared component coordinates** can move a future readout by as much as **2.016**. Canonicalizing the complete physical product by float64 SVD before routing reduces the maximum two-readout drift to **1.6234e-7**; there are **zero** recomputed pass-two mask switches across the eight model roots and two shears. Minimum trained nonzero singular-value gap among prospective roots is **0.16665**. This is a finite-precision experiment on two prescribed shears, with a mathematical unique-basis condition requiring distinct singular values.

## Reproducibility and hosted closure

- Frozen protocol, source and verifier Git blob hashes are stored in [results/LOCKED_COUNTS.json](results/LOCKED_COUNTS.json); the field \`source_sha256\` is a legacy key name, while its recorded digest *algorithm* is explicitly Git blob SHA-1.
- The original independent CPU pilot used Python 3.13.5, PyTorch 2.10.0+cpu. Every per-seed arm, by-source count, mask cost, gauge switch and donor-success count is locked.
- [GitHub Actions run 44115709120](https://github.com/MoralityLabAI/BPD/actions/runs/44115709120) completed **PASS** for engineering, independent source-locked eight-root replay and actual native \`BPDModel/TiedComponentLinear\` execution with \`PassMaskSchedule\`. [Both native and independent full raw JSON receipts](https://github.com/MoralityLabAI/BPD/actions/runs/44115709120/artifacts/14343106421) are archived.
- The corresponding old CQ-04 and CQ-03 results, ARG-03-N protected material and other project sources remain unchanged.

A confirmation study requires a **fresh** separately sealed root population and independent sham calibration of the 95%/95% nonparametric tolerance floor. Training and intervention work on nonlinear looped-transformer checkpoints would be a distinct evidence class.

