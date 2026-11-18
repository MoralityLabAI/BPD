# AMM10 after-first-audit numerical tie correction

**2026-11-18. This is a retrospective engineering correction of the already registered tie rule, not a new research hypothesis.** No physical models were fitted, no new root or success label was generated, and all original AMM09R1 TRAINING/DEVELOPMENT receipts and confirmation seals remained unchanged.

## What failed

The original source-only AMM10 audit on [GitHub Actions 44305663753](https://github.com/MoralityLabAI/BPD/actions/runs/44305663753) passed seven tests and scored all 19 exposed-root arms. Its output SHA256 was \`88d7e2f6bd4c04097196c723af9667686b3e3cbcc55c37dbd99392eb7c59c648\`. A later independent runner on [GitHub Actions 44306562013](https://github.com/MoralityLabAI/BPD/actions/runs/44306562013) reverified the exact original file SHA256, all training feature roots, and all developer action outcomes. Every primary mechanistic and geometric analysis agreed, **except** \`cardinality_only\`, which gave 1,896 instead of 1,822 successes out of 8,192 exposed training cases.

The mismatch affected **all eight heldout training model roots in fold index 3** (training indices 3,11,19,27,35,43,51,59), and **zero roots of every other arm**. The original training data, development data and coefficients of the actual frozen AMM09R1 memory were identical across these audits.

## Mechanism identified

The cardinality-only refit uses just two columns: an intercept and mask size \`|S|/3\`. All three allowed two-component masks \`(0,1)\`, \`(0,2)\` and \`(1,2)\` have **bitwise-identical feature rows**. They therefore have precisely the same analytical ridge score. Nonetheless, vectorized floating-point matrix multiplication produced results differing at roughly \`2.8e-17\` in fold 3 on one CPU, so ordinary \`numpy.argmax\` sometimes selected mask 4 instead of the **registered lowest-index tie mask 3**.

The result is a **floating-point implementation violation of an exact feature-identity tie**. It is not an actual different model, different causal signal, or meaningful variance in the physical experiment. It illustrates why an unqualified argmax is insufficient to implement mathematical equality under differently ordered BLAS kernels.

## Source amendment

Implement a source-only tie repair for the *new exploratory ridge refits*:

1. Compute each candidate's score by the frozen ridge formula and clipping contract.
2. On a given donor case, when two candidate rows are **exactly equal in all columns selected for that fitted variant**, make their computed scores exactly equal by copying the earliest matching mask's score to the later mask.
3. Apply the original \`numpy.argmax\` to these corrected, clipped scores, retaining the **lowest mask index** for exact ties.
4. For candidates whose selected feature rows are different, leave their independently computed scores untouched. There is no arbitrary near-equality epsilon, score rounding, outcome-conditioned action selection, or new hyperparameter.
5. Apply the rule identically to **all nine** registered ridge ablations. The six geometric heuristics and the frozen original AMM09R1 model/data are unchanged.

This repair implements the **pre-data \`ANALYSIS_PLAN.md\` tie-to-lowest requirement**, with no numerical accuracy target used in its construction. All 19 experimental arms, exposure labels, case outcome hashes and policies remain reported. Preserve the first-run result receipt and second-run failure archive; mark \`RESULTS_LOCK.json\` as superseded by the corrected deterministic audit once that audit independently passes.

**Evidence class remains exploratory and post-selection.** This correction does not generate an independent EvalMath confirmation claim and cannot authorize AMM09R1 or historical confirmation-root access.
