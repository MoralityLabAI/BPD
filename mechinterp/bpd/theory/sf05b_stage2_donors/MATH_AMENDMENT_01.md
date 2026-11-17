# Stage 2 source-only math amendment 01: donor's prior same-source write

**2026-11-18. Prior source lock Git blob:** \`d6efe40ca0cad30dd550800e72654b025d5fed5d\`. This is a **pre-model** mathematical correction; no teacher, BPD factor, corrective policy, calibration root or confirmation root was run.

The initial source-only [GitHub Actions run #44146966540](https://github.com/MoralityLabAI/BPD/actions/runs/44146966540) passed the source hash check and 9/11 tests but failed both closed-form blind-mean tests. An independent review identified that when the query matches the recipient's original final source, flipping the final donor tag causes the donor target to revert to that source's **previous** write. The initial simplified expression incorrectly treated both recipient and donor means as identical, and the high-contrast formula ignored the previous-old-source gap.

The source generator and its exact 24-configuration conditional enumeration were already correct at Git blob \`6c08787a2a08daf3c601265decbba6d8cfc7f930\` and remain unchanged. The revisions affect only [THEORY.md](THEORY.md) and the independent [closed-form unit tests](tests/test_donor_cohorts.py). The corrected unconditional formulas are

\[
\mathbb E[Y_r]=\tfrac12v_m+\tfrac14(v_{\ell_a}+v_{\ell_b}),\qquad
\mathbb E[Y_d]=\tfrac12v_{j_o}+\tfrac12v_m,
\]

conditional on numeric values, history length, observed role and the declared sampling law. In the high-contrast stratum the weights are proportional to the exact source-oracle admission indicators for the original source's prior value and the other two sources, as documented in THE0RY (see [THEORY.md](THEORY.md)).

The model-root reservation remains \`6101..6159\` calibration and \`7101..7132\` confirmation, with **zero new-root evaluations**. The prior failure is preserved; no threshold, teacher result, source cohort or frozen donor generator was replaced.
