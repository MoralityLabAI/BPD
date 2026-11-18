# BPD-NC09-FC01: 59-root source-matched sham calibration and EvalMath preflight

**Prospective registration, 2026-11-18.** This document is committed before implementing FC01 or training calibration model roots. First run records all failed roots and outcomes. **This is a calibration and resolution experiment, not scientific confirmation.**

## Previous evidence and separate research

NC09 Fixed12 merged [PR #88](https://github.com/MoralityLabAI/BPD/pull/88) with a first-run DEVELOPMENT_POSITIVE on independently trained roots 900–915. The source-derived total-12 passwise rank allocation beat the equal-rank flat-three control by mean +0.027179 NLL nats, dose-matched flat-three by +0.028017 and eight physically matched same-tuple local shams by +0.053406. The intact-model task NLL cost was +0.014932 and extra cost versus sixteen-slot flat-four +0.007012. One root 903 had an individual negative sham effect. The original development cohort has only sixteen independent roots and cannot supply a distribution-free 95th-percentile/95%-confidence sham-tolerance bound.

Separately, [NC10](https://github.com/MoralityLabAI/BPD/pull/93) was a cross-model transfer null; it supplies no samples to this study.

## Independent calibration population, model and controls

- Exactly **59 NEW independently trained recurrent model roots 12000 through 12058**, each one independent unit. Engineering smoke uses excluded root 11999 only. No model root is drawn from NC04–NC10 or any future confirmation population. The source release is a fixed one-time roster, with no replacement for any failed or ineligible model.
- Exact NC09 source law and model: 20D fully trainable dense shared physical mix, four tied nonlinear recurrent passes, learned encoder/preweight/final head. Exactly 384 Adam updates on 2,048 training inputs, batch 128, learning rate .005, minibatch RNG 1,737,701 plus model root. True task labels may be used during supervised model training.
- Each root separately samples 1,024 derivative-discovery inputs (split 2), 512 model-only diagnostic inputs (split 4) and 1,024 externally labeled evaluation inputs (split 3). **No task evaluation label enters projector discovery, gain selection or sham matching.**
- Original NC09 Exact Fixed12 is unchanged, including top-r pass eigenvectors, original 625-candidate first-order omitted model-score dynamic program at total rank twelve, source-only lexicographic ties, full Native BPDModel/TiedComponentLinear four-pass override-read interventions, native/rotated orthogonal factor gauge tests, and the physical task gate. Reuse its source methods. Only replace the protected train-root guard with a byte-equivalent training wrapper for new calibration roots. Protected first-run NC09 original experiment and protocol SHA256: experiment 8d1b88eea8518a83669f03f9126136ca3d56ac5bac5fe041e2e03fb12922d380; protocol dc9423f1fd9e308221c41ed2af83e7398a59dd0eb0beefa8394091e373156e02.
- Fixed evaluated projectors: original source-derived DP12 gain one; flat 3+3+3+3 gain one; flat three with discovery physical-write RMS matched through the frozen gain ratio in [0.5,1.5]; flat 4+4+4+4 gain one; and reversed passwise ranks as descriptive placebo. Exactly eight local physical same-tuple shams from the **first eight eligible among 4,096** independent seed-9090000-plus-root angle/QR proposals with principal angles from {0.45,0.65,0.85,1.05} radians, retained clean physical-write RMS within 15% of the original DP12 source.
- Full technical admission per root: two context-specific clean task accuracies >=0.85, physical mix density >=0.90, all four full-mask BPD readouts close to the intact model at logit error <=5e-5 and Bernoulli KL<=1e-7, original eigen/gauge/physical gain/rank checks, all four primary arm outputs present, eight source-matched shams, native/orthogonal physical operator discrepancy <=1e-4 relative and retained closed-loop logit discrepancy <=5e-5. Retain every failed root.

## One root, one matched-sham null

For root i, evaluate each of its eight prospectively source-matched local sham projectors on exactly the same 1,024 independently labeled evaluation cases and define L_ij as its external-label task NLL. Put M_i = median of the eight L_ij. Define the **rootwise worst matched-sham orientation variation**:

    T_i = max(j=0..7) | L_ij - M_i |.

The eight correlated sham orientations are observations nested within a single independent trained root. Only the **59 root-level T_i** count as independent null calibration units. This null is deliberately conservative: the model-root maximum of absolute deviations among eight shams selected without task-label information.

Predeclare the exact upper one-sided **nonparametric 95/95 tolerance bound** T_max=max_i T_i. For 59 independent identically distributed model-root nulls, confidence that their sample maximum covers at least the 95th percentile of the rootwise null distribution is 1-(0.95)^59 = 0.951505..., above the required 0.95. This is a rootwise tolerance bound and **does not automatically provide an exact significance threshold for a 32-root mean**.

Also record the five signed independent-root causal effects, each in final external-label NLL nats:
- A_i = NLL(flat3) - NLL(DP12), equal physical 12-slot rank.
- D_i = NLL(source-dose-matched flat3) - NLL(DP12), equal physical rank plus matched source write RMS.
- S_i = median_j NLL(sham_j) - NLL(DP12), same-tuple physical sham.
- C_i = NLL(DP12) - NLL(intact model), absolute task fidelity cost.
- F_i = NLL(DP12) - NLL(flat4), additional cost versus rank sixteen.

Calibration uses all 59 roots to estimate the variation floor, prospective power and task source eligibility. It makes **no confirmation p-value or positive causal identification claim**.

## Mandatory pinned EvalMath mechinterp-power script

Use the exact public [EvalMath script](https://github.com/MoralityLabAI/EvalMath/blob/6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a/skills/evalmath-mechinterp-power/scripts/preflight.py) pinned to EvalMath commit 6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a and original Git blob SHA 0ff062f76f5534d9f18425f986ae3e750397f50c. Vendor a byte-identical copy or check out precisely that commit in CI; if unavailable or mismatched, report EVALMATH_NOT_RUN and hold confirmation.

The source-frozen EvalMath version-one JSON specification is:
- experiment_id: BPD-NC09-FC01.
- independent_unit: independently trained dense 20D four-pass model root.
- pilot: exactly 59 rows, one unique unit_id per 12000–12058 model root; paired_effect equal to A_i as a signed decimal string; sham_effects equal to eight signed differences L_ij - M_i, never individual examples promoted to additional roots.
- alpha 1/20; scientific_min_effect 1/125 (0.008 nats); target_effect 27/1000 (0.027 nats, historical development planning); measurement_resolution 1/1000; sham_quantile 19/20; sham_confidence 19/20; planned_comparisons 3; confirm_units_planned 32; pilot_from_development_only true; heldout_sealed true; units_independent true after audit.

Run this script **without strict mode** and archive both the original pilot JSON and resulting preflight JSON. Respect its exact READY_FOR_CONFIRMATION versus REDESIGN_OR_PILOT_ONLY classification. The target is a planning value, not a substitute for a measured A_i mean above the calibrated floor.

## Frozen calibration and future confirmation admission

A positive calibration **handoff** requires all simultaneously:
1. **59/59** fresh independent trained model roots technically admitted with eight sham effects each. No replacement or post-hoc exclusion.
2. All nulls/effects finite, each root distinct and T_max supported by the original 59 complete root-receipts (95/95 order statistic rank 59).
3. The pinned exact EvalMath script returns **READY_FOR_CONFIRMATION** on those original roots with no unresolved sham variation or power concerns. No alternate script or retuned target effect can rescue a negative result.
4. Even after a positive handoff, confirmation is still sealed until a **separate prospective confirmation protocol** fixes familywise testing, physical dose and new independent root identifiers and commits before any scientific model training.

If a source root is missing, preserve **CALIBRATION_INCOMPLETE**. If all are admitted but EvalMath returns negative, preserve **CALIBRATION_REDESIGN_OR_PILOT_ONLY** as a named scientific hold, with the raw floor, reasons and original outcome. If EvalMath fails to run/verify, preserve **EVALMATH_NOT_RUN**. Distinguish engineering failure from negative scientific planning.

No independent >=32-root confirmation model, no protected BPD-CQ-08, ARG-03-N, AMM-03, or NC10 cross-model source is generated, trained or evaluated in FC01. The experiment is restricted to the same synthetic 20D nonlinear recurrent task family as NC09 Fixed12. Historical NC09 original first-run data and source-lock are unchanged.
