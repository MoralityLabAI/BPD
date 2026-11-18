# NC09-FC02: gradient-conditioned physical sham calibration

**Prospectively registered November 18, 2026.** The [original source protocol](PROTOCOL.md) precedes implementation and all new calibration model roots.

FC01 found a positive mean passwise rank-allocation effect on 59 independent newly trained recurrent models, but its sham orientation null had a **rootwise nonparametric 95/95 sample maximum of 0.294179 NLL nats**. This was far wider than the allocation benefit and caused the exact EvalMath mechanistic-variation preflight to return REDESIGN_OR_PILOT_ONLY. The confirmation hold remains in force.

FC02 tests a more informative physical null on **59 entirely new 20D four-pass learned recurrent model roots 13000–13058**, engineering smoke 12999. It reuses NC09 Fixed12's frozen source-only DP12 allocation, same-rank/dose-matched controls, stronger rank-four reference and actual closed-loop BPDModel/TiedComponentLinear override-read task evaluation.

The new [source-only sham family](tangent_shams.py) requires each same-tuple local orthogonal projector to satisfy **three predeclared matching gates** against DP12: clean retained physical write RMS within 15%, signed first-order final-model-score change within max(.005,25% of target magnitude), and first-order model-score RMS within max(.005,25% of target). It chooses the first eight matches from exactly 4096 independent angle/QR proposals with minimum principal-angle 0.20 rad. Shams are fixed before final task labels are accessed.

The [registered aggregate](aggregate.py) uses one independent model root per sample and explicitly calibrates the **future 32-root mean of a pseudo-selected sham causal contrast**. For any set of eight source-selected matched sham task NLL values, randomly designate one as a pseudo-target and compare it with the mean of the other seven. The pseudo-target contrast has **exact conditional expectation zero**, and its nested root-bootstrap aggregate threshold preserves dependence among the eight within-model shams.

The frozen 256-outer/4096-inner nested bootstrap is an **approximate planning bound** on a hypothetical 32-root mean. An independent nonparametric 59-root sample-maximum **95/95 rootwise tolerance bound** is also retained as a separate diagnostic. This experimental statistic cannot silently replace FC01's source-registered EvalMath conclusion.

A developmental positive result requires 59/59 full physical/sham source admission, positive independent-root DP12 task benefits over fixed same-rank controls, absolute task fidelity, task superiority over matched shams exceeding the aggregate null with an added 0.020-nat practical margin, and estimated 32-root familywise power. **A positive calibration is still not scientific confirmation.** New validation of this null methodology, a separate EvalMath power contract and independently registered confirmation models remain mandatory before release.

[Mathematical analysis](THEORY.md) · [excluded12999 source tests](test_experiment.py) · [synthetic mean-null tests](test_aggregate.py) · source-first registered 59-root CI workflow


## First registered original result: CALIBRATION_INCOMPLETE

[Source-locked GitHub Actions #44281207145](https://github.com/MoralityLabAI/BPD/actions/runs/44281207145) completed the original full 59-model training and evidence verification with engineering SUCCESS. **40/59 models** obtained the required eight same-tuple shams simultaneously matching retained physical write RMS, signed model-self first-order score mean and RMS. The other **19/59** had fewer than eight triple-matched candidate shams from the fixed 4096-proposal source pool; eight found none. The original full root-mean null and 32-root power planning consequently remain **unscored**, with scientific confirmation sealed.

All 59 roots passed original model/physical full-mask admission; every original result was retained. The first excluded engineering seed 12999 obtained 93/4096 matches, showing the gate was executable yet variable across independently trained model geometries. The full-model source distribution had a median **49** qualifying shams per root and first quartile **3.5**, causing nonrandom calibration missingness.

Predeclared DP12/flat3 task comparisons were evaluated on all 59 roots regardless of local-sham admission: mean DP12 NLL advantage **+0.019325** over equal 12-slot flat3 and **+0.019377** over discovery physical-dose-matched flat3, favoring DP12 in 51/59 and 53/59 roots respectively. These are descriptive after the joint calibration hold. In the selected 40 roots with eight stronger shams, the **mean matched-sham-minus-DP12 NLL** was **+0.010241**, substantially below the previous FC01 local sham advantage, while direct cross-cohort causal comparison remains unavailable.

[Full frozen report](RESULTS_20261118.md) · [original all-59-root CSV](PER_ROOT_20261118.csv) · [original receipt SHA manifest](ORIGINAL_ROOT_SHA256.csv) · [independent original archive auditor](test_published_results.py) · [original 60-file immutable artifact #14417745672](https://github.com/MoralityLabAI/BPD/actions/runs/44281207145/artifacts/14417745672).

The original FC01 REDESIGN_OR_PILOT_ONLY decision remains intact. A further prospective geometry-constrained sham generator would require fresh models and a new named protocol; existing 59-root FC02 results are exposed development data and do not qualify for selection or confirmation.
