# NC09-FC04: full tangent-Jacobian source-matched physical shams

**Prospectively registered November 18, 2026.** [PROTOCOL.md](PROTOCOL.md) was committed **before** the full-tangent solver, model code, or training any new engineering model root. Its first source commit is e51938e4cf96ac4d34ffeaca3afdb52b84969db9.

The original [NC09 Fixed12](../RESULTS_20261118.md) positive engineering result motivates source-derived causal rank allocation. FC01's independent 59-root matched-sham calibration produced EvalMath REDESIGN_OR_PILOT_ONLY with a rootwise sham variation floor +0.294179 NLL nats. FC02 strengthened source matching with signed first-order score moments, but 19 of 59 trained models failed eight-sham admission. FC03 replaced random rejection with an eight-dimensional constrained Grassmann solver, recovering 13/16 model roots versus 11/16 for random FC02, while still failing its frozen 16/16 gate.

FC04 enlarges each pass's local complementary-frame search to the **full approximately 202-dimensional rank-preserving tangent chart** and uses a deterministic damped Gauss-Newton correction based on the **three source-observable matching metrics**. The fixed 0.25-radian original projector displacement and minimum pairwise separation remain independent audit requirements. All original FC02 physical retained-write RMS, signed model-self score mean and RMS tolerances are unchanged.

The engineering cohort is **16 fresh trained 20D four-pass recurrent models 15000–15015**, with excluded source smoke **14999**. Original FC03's 32-chart/64-step solver and original FC02's 4,096-proposal rejection sampler are executed as frozen controls on the same new source data. Each model's native four-pass BPDModel/TiedComponentLinear intervention and external task NLL are scored on independent heldout labels only after all source-only sham choices are fixed.

FC04's success criterion is full eight-sham source admission on 16/16 new trained models and at least two more fully admitted roots than FC03's unchanged eight-coordinate solver, with independent projector SHA/geometry, source metrics and future-task receipts. This is a **feasibility engineering gate** with a fail-closed first-run verdict. It cannot replace FC01's original negative EvalMath calibration, the FC02 incomplete null, or the still-sealed 59-root recalibration and 32-root scientific confirmation populations.

[Full tangent solver](solver.py) · [native source and original controls](experiment.py) · [independent root/projector audit](aggregate.py) · [source geometry and model smoke](test_solver.py) · [synthetic fail-closed decision tests](test_aggregate.py) · [derivation](THEORY.md) · [Lean source arithmetic](TangentBudget.lean).


## Frozen first original sixteen-root result: FULL_TANGENT_ENGINEERING_POSITIVE

The original [source-locked 16-model Actions run #44307332891](https://github.com/MoralityLabAI/BPD/actions/runs/44307332891) completed all 16 independently trained 20D four-pass recurrent models, all three original source-matched physical sham methods, every native task evaluation and independent projector geometry check. Every preregistered engineering source gate passed.

| Same fresh independent model population | Models with eight full matching source shams |
| --- | ---: |
| **FC04 full complementary-tangent constraint Jacobian** | **16/16** |
| Original FC03 eight-coordinate constrained Grassmann solver | 12/16 |
| Original FC02 local random source sampler | 11/16 |

The frozen FC04 improvement gate required at least two additional fully source-admitted models over FC03; **four** were recovered, including roots **15013 and 15015** where both old source samplers found zero candidates. Across **128 accepted full-tangent projectors**, the fixed original-projector distance was sin(.25)=.247404 and minimum archived pairwise distance **.297767**, above source-registered .18. The solver made 348 total full Jacobian corrections, with all evaluated source Jacobians having numerical row rank three.

On independent external-label task examples, the descriptive mean-of-eight stronger-sham-minus-DP12 true-label NLL effect was **+0.009626 nats**, original whole-root bootstrap 95% **[+.003057,+.015251]**, positive **15/16**, with root **15014** a retained negative counterexample. This **engineering feasibility success** does not pass the earlier +.020 practical causal margin or substitute for independent scientific calibration.

[Original full prospective report](RESULTS_20261118.md), [sixteen-root task/feasibility table](PER_ROOT_20261118.csv), [original model JSON SHA manifest](ORIGINAL_16_ROOT_SHA256.csv), [post-result independent first-run archive auditor](test_published_results.py) and [immutable original 17-file GitHub Actions artifact #14428573240](https://github.com/MoralityLabAI/BPD/actions/runs/44307332891/artifacts/14428573240) preserve the evidence. Original ZIP SHA256 **477e40d3e2d90005d7ab60739db2f263f5e91c4ca60656b0834c7811769ee5ac**.

No independent 59-root recalibration or scientific 32-root confirmation model was evaluated. FC01 REDESIGN_OR_PILOT_ONLY, FC02 CALIBRATION_INCOMPLETE and FC03 ENGINEERING_NULL_OR_INELIGIBLE original outcomes are unchanged.
