# NC-13 bridge to native recurrent-model studies

**Prospective source-stage note, 18 November 2026.** The [NC-13 protocol](PROTOCOL.md) was registered at [069b355e8f37ec780dc6c46b82176b6f32f8e39b](https://github.com/MoralityLabAI/BPD/commit/069b355e8f37ec780dc6c46b82176b6f32f8e39b). This note identifies where its diagnostic can improve existing studies. Choosing a native follow-up arm depends on the completed NC-13 results and a separately frozen protocol.

**Evidence checked on 18 November 2026 at 22:08 Singapore time.** The current cross-thread sources are BPD [b343609706596d00088a110cd2e644a1b6e45ea2](https://github.com/MoralityLabAI/BPD/commit/b343609706596d00088a110cd2e644a1b6e45ea2) and LoopedTransformers [6b4ecbbcced768b60596a8138b210250aa81d33e](https://github.com/MoralityLabAI/LoopedTransformers/commit/6b4ecbbcced768b60596a8138b210250aa81d33e). Section 3 records LGA-09's original completed experiment and the subsequent LGA-10 feasibility result. NC-13 remains prospective throughout this note.

## 1. Carry the complete target-access budget forward

NC-13 compares mixed-context and shared-context source discovery at eight original-model B gradient inputs. Both source projectors use raw gradients, and one fixed alignment serves both conditions. Its two calibration policies share the mixed-source transfer at gain 1 as their anchor. Policy E matches retained write RMS on clean dose trajectories. Policy D searches for a scalar gain that matches the actual write displacement through a complete intervened rollout. The D tolerance applies to the complete dose pool, which includes both task contexts. Displacement within each context and each pass remains an explicit diagnostic.

Policy D receives substantial additional target access: up to 41 forward evaluations on the 1,024-input dose pool for each non-anchor projector. Three main projectors and eight fixed shams allow at most **451 search rollouts per model pair**, with other calls counted separately. A future claim about gradient efficiency must disclose this forward-computation cost. The target-native comparator receives the same search rule and dose inputs. Its gain depends on the common transfer reference, even though its subspace estimate consumes B gradients alone. [NC-13 protocol](PROTOCOL.md)

The registered gap-change endpoint measures the effect of recalibration. A smaller gap needs to be read alongside the absolute losses of both arms: changing the comparator's gain can increase its loss. Every gain-query trace and failed admission belongs in the evidence. The sixteen trained pairs provide engineering evidence; estimating a calibrated sham tail still requires a new independent population.

## 2. BPD-DT03 supplies the native state contracts

[BPD-DT03, merged PR #113](https://github.com/MoralityLabAI/BPD/pull/113), repairs the depth-steering gym's source contracts. Its [review at BPD snapshot 03959ad](https://github.com/MoralityLabAI/BPD/blob/03959ad558c02ed7f1bfbfdef306cb3417a28d03/mechinterp/bpd/experiments/depth_steering_gym/DT03_REVIEW.md) records zero whole-grid successes in the original Sudoku4 and synthetic ARC pilot. Storyworld terminal success was also zero. Native competence must therefore precede interpretation of transfer on those tasks.

The repaired [dt03_causal.py](https://github.com/MoralityLabAI/BPD/blob/03959ad558c02ed7f1bfbfdef306cb3417a28d03/mechinterp/bpd/experiments/depth_steering_gym/dt03_causal.py) differentiates every output coordinate, packs the complete H/L state, and binds continuation to the receiving input. Its sham checker validates delivered full-state L2 dose and the signed response of the complete task Jacobian. It also records whether the candidate and sham remain distinct.

An NC-13-derived native study should use those contracts to compare a source-assisted subspace with a target-native subspace under identical target access. Register same-case competence at the fitted depth and every evaluated continuation depth. Whole-case recovery is the task endpoint; coordinate accuracy remains a diagnostic. Preserve the original DT01/DT02 source locks and failed receipts when adding that study.

## 3. Use the original full-response result to choose the next control

[LOOPSCOPE-12/LGA-09's original report](https://github.com/MoralityLabAI/LoopedTransformers/blob/6b4ecbbcced768b60596a8138b210250aa81d33e/mechinterp/loopscope12/RESULTS_FIRST_NATIVE_20261118.md) records `DEVELOPMENT_STRONG_FULL_J_PASS` in all three historically exposed teachers, seeds 103, 211 and 307. The [original native run and independent audit](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44315883529) executed source commit `f19db86154a2e990ebf34853603a9bac33db5e19` on 18 November 2026. The BPD [EFD-02 addendum](https://github.com/MoralityLabAI/BPD/blob/b343609706596d00088a110cd2e644a1b6e45ea2/mechinterp/bpd/experiments/efd01/LOOPSCOPE12_FULL_J_NATIVE_20261118.md) records the same original result. This updates the earlier source-stage description of LGA-09.

At native h2, the registered edit changes a physical predecessor/goal pair of two 48D token states. All arms continue through the trained k2-to-k4 horizon. A 75D subspace satisfies the current-head and scalar-observer tangent constraints. The complete ten-logit future Jacobian restricted to that subspace has numerical rank 10 on all 1,536 admitted source-goal cases, leaving 65 dimensions with zero first-order future response. The eight full-J shams keep the source-selected edit's complete ten-logit first-order response and physical L2 norm fixed while rotating its component in this remaining subspace. The actual applied float32 response mismatch stays below 4.69e-7 across the three teachers. [Original geometry and matching audit](https://github.com/MoralityLabAI/LoopedTransformers/blob/6b4ecbbcced768b60596a8138b210250aa81d33e/mechinterp/loopscope12/RESULTS_FIRST_NATIVE_20261118.md)

| Exposed teacher | Selected donor correct | Advantage over median eight full-J shams | 95% whole-base interval, percentage points |
|---|---:|---:|---:|
| 103 | 181/512 | +17.68 percentage points | [+13.87,+21.68] |
| 211 | 219/512 | +21.88 percentage points | [+18.07,+25.88] |
| 307 | 241/512 | +23.83 percentage points | [+20.21,+27.34] |

Each teacher uses 64 new base strings from source seed 31063, with eight correlated goal variants per base. The intervals resample whole bases 2,000 times within each teacher. Original-model generalization therefore retains a sample of three previously examined checkpoints. The preregistered standard-library audit checked 47,616 applied physical edits, the 1,536 full-J source records, final task scores and the original decision. All three teachers passed the registered full-J, single-margin and random-K behavioral gates. [Original results and immutable receipts](https://github.com/MoralityLabAI/LoopedTransformers/blob/6b4ecbbcced768b60596a8138b210250aa81d33e/mechinterp/loopscope12/RESULTS_FIRST_NATIVE_20261118.md)

This establishes a finite nonlinear donor-specific effect beyond the complete first derivative at the clean state in this cyclic-copy TRM experiment. Its attribution still spans curvature, higher-order response and the states reached during recurrence. A native BPD study would need its own physical mediator, source registration and fresh model population. The LGA-09 controls also leave the learned scalar observer's explanatory role unresolved: the head-only donor recovered more correct targets in every teacher, and every selected natural-radius edit increased that observer's finite value. [Mechanistic controls and scope](https://github.com/MoralityLabAI/LoopedTransformers/blob/6b4ecbbcced768b60596a8138b210250aa81d33e/mechinterp/loopscope12/RESULTS_FIRST_NATIVE_20261118.md)

For NC-13, the practical next question concerns which response constraints a later control should preserve after source context and delivered displacement have been measured. Its current shams leave downstream signed derivatives free. LGA-09 supplies an executed template for matching the full future output derivative at a fixed physical radius. Importing that template would add target derivative access to a separately registered study. The eight-input NC-13 gradient budget remains fixed. For an observer q_M and a declared continuation horizon k, define the finite response

$$
\Phi^M_{t,k}(\delta)
=q_M(F_M^k(h_t+\delta))-q_M(F_M^k(h_t)).
$$

A proposed correspondence would be evaluated through these finite responses on held-out inputs, including the externally graded task outcome. Any target response queries used to fit it must enter the same access ledger as target gradients. Matching a derivative supplies a local control; the complete rollout measures what follows at the registered edit radius.

The next curvature control already has a documented admission problem. [LOOPSCOPE-13/LGA-10's source-only result](https://github.com/MoralityLabAI/LoopedTransformers/blob/6b4ecbbcced768b60596a8138b210250aa81d33e/mechinterp/loopscope13/SOURCE_FEASIBILITY_RESULT_20261118.md) admitted all eight separated curvature-matched shams in **28/64 cases, or 43.75%**, against its registered 95% gate. The other 36 cases exhausted the fixed random-plane search. This excluded test used an untrained model, synthetic quadratic observer and source 61064; prospective trained-model source 31064 remains held. The source-only CI retained `FAIL_SOURCE_FEASIBILITY`. A later BPD response control should establish geometric admission under its frozen search and separation rules before opening a behavioral cohort.

Horizon competence matters here. [LOOPSCOPE-09](https://github.com/MoralityLabAI/LoopedTransformers/blob/f19db86154a2e990ebf34853603a9bac33db5e19/mechinterp/loopscope09/README.md) reports mean IID accuracy falling from 98.6% at four passes to 10.2% at twelve. A transfer test beyond the trained horizon must measure that native decline on the same cases before attributing loss to the intervention.

## 4. Nanbeige prerequisites and compute limits

[NAN-COMP-04](https://github.com/MoralityLabAI/LoopedTransformers/blob/7d415c41c4cc0bc7aba965028c3a00d2a3a40d16/experiments/nanbeige/nan_comp04/PROTOCOL.md) remains an open, stacked source-stage study in [PR #194](https://github.com/MoralityLabAI/LoopedTransformers/pull/194). It explicitly keeps BPD transfer disabled until a separately registered competence-gated experiment. Its actual tokenizer audit and source-lock checks precede pretrained execution. The sixteen development roots have an exploratory role; forty future confirmation identifiers remain ungenerated.

The proposed **NAN-DEC-01** comparison in the research conversation has a separate resource contract: **one H100 at a time, four cumulative H100-hours, BF16, and no NF4 run**. A repository registration for that name was unavailable in the inspected main tree, branch list and PR search. Keep those limits in its future registration. The BF16/NF4 options in COMP-04 do not expand the DEC plan's authorized backend scope.

For DEC, add the direct target-native comparator to the same physical-dimension and reconstruction-fidelity comparisons. Record same-dtype identity replay before causal scoring, then distinguish clean-state evaluation from the states actually reached by the intervened model. An unconverged fit retains its incomplete status when the compute cap is reached.

## 5. Release and evidence conditions

Use [EvalMath's mechanistic-resolution guidance](https://github.com/MoralityLabAI/EvalMath/blob/6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a/skills/evalmath-mechinterp-power/SKILL.md) before selecting a new native experiment. Register the independent unit, minimum behavioral effect, comparison family and sham calibration, with all target access and selection exposure visible. Repeated depths and interventions remain clustered within their source or checkpoint.

LOOPSCOPE-12's three historically examined teachers remain development models. Its source 31063 was generated after the original pretest lock; reserved 31059 stays unopened. LOOPSCOPE-13 source 31064 remains held after the failed excluded feasibility gate. Preserve the ARG-03-N 168-case seal and all AMM, CQ and SUB protected populations. Additional cohort release or pretrained execution requires its own authorization and registered protocol.
