# BPD-PD-01: phase-disjoint dictionary-learning control

**Registered:** 2026-11-17. **Status:** source-only prospective control before the BPD-PD-01 training execution. This is a newly specified, two-dimensional BPD toy, distinct from the preexisting B1–B4 gates and from the historical BPD-v2 receipts. This file is a frozen registration; amendments must be dated before outcomes are generated.

## Question

Does the actual shared BPD factor pair (V,U), when trained jointly with phase-sensitive gates, discover the planted phase-disjoint read axes with sparse execution? The experiment tests a concrete negative-control alternative in which both input axes are active in every phase. Training the dictionary is essential: a gate-only fit with SVD-frozen identity factors is a positive calibration arm.

## Frozen generator and design

- Physical target: one shared bias-free linear module W = I_2, called at four tied passes on independent input slices. Exactly two factor components. Use the existing BPDModel / TiedComponentLinear wrapper and PassMaskSchedule with mode=zero, sphere_rescale=false, delta_mask=0. This disables residual shortcuts in the masked output; report full-matrix faithfulness separately.
- Sparse phase-disjoint arm: at passes 0,1, the input is (signed amplitude,0); at passes 2,3, (0,signed amplitude). Each active amplitude is uniformly drawn from [0.75,1.25] with an independent equiprobable sign, always nonzero. No explicit phase or label is supplied to the gates; the gate sees |x| through a shared two-input/two-output affine-sigmoid network.
- Matched mixed-source negative arm: both coordinates are independently sampled from the same signed-amplitude law on every pass. Target is W x in all arms.
- Factor-trainable arms: run the same 100 Adam steps, batch 16, seed 71 or 89, learning rate 0.04, on freshly drawn training batches. initialize the rank-2 factor pair with the BPDModel random initializer (init_svd=false) and train V,U and gate network **throughout every stage-1 optimizer step**.
- Gate-only positive-control arm: identical disjoint input source and budget, but SVD-initialize factors, freeze both factor matrices during stage 1, and train gates only. Its recovery is a calibration benchmark, not a competing mechanism-discovery result.
- Reconstruction loss: mean squared prediction error over B,T,P,output dimensions. Add 0.3 × mean squared physical weight residual and 0.05 × mean gate value. Report every term, held-out values, row-to-axis read cosine, mean gate by pass, and gate-switch magnitude. This is a declared differentiable **surrogate**, not exact expected-L0 minimization.
- Stage 2: freeze trained V,U, retain gate capacity, sweep switch coefficients {0.01,0.05,0.20} with 25 gate-only Adam updates per candidate, LR 0.01, while maintaining the stage-1 reconstruction and gate-level metrics within +0.005 absolute of their stage-1 **calibration** baselines. Select only among calibration-feasible candidates by lowest calibration switch score; 0 is the admissible reference. Restore chosen gate weights before evaluation. Verify V,U bitwise unchanged across stage 2.
- Reproducibility: torch.manual_seed(seed); fresh train samples from generator 100000+seed; *independent* fixed calibration batch 128 from generator 300000+seed and held-out batch 128 from generator 200000+seed. Seeds 71 and 89 are the only registered model seeds.

## Hypotheses, diagnostics and reporting

The primary source-lock result is a per-seed and pooled table of **trained dictionary** axis recovery and held-out reconstruction, compared with gate-only SVD calibration and mixed-source controls. Report assignment-invariant mean absolute cosine of two normalized read vectors to the coordinate axes, reconstruction MSE, factor update norm, duty cycles by phase, and the inferred binary-gate pattern at the 0.5 threshold.

A positive directional result requires, in both registered seeds:
- trainable disjoint arm axis matching >= 0.90 and held-out reconstruction MSE <= 0.02;
- exactly one of two mean gates above 0.5 in each phase, and opposite winning gates between phases;
- trainable factors have nonzero parameter-update norm;
- any apparent stage-2 reduction in switching must be calibration-feasible, and stage-2 reconstruction cost and sparsity tolerance must be explicitly reported.

These fixed thresholds are diagnostic and not a population-level inference with only two seeds. If a criterion fails, preserve the result as a failure; the experiment cannot establish BPD mechanism identification. The mixed-source arm is a stress control whose learned axes and duty cycles are reported without a stipulated success threshold, because its sparse-execution optimum may differ.

Use deterministic CPU and torch.set_num_threads(1). Run one training job at a time. Training does not use any 7M TRM checkpoint, external service, private split or B4 process. Execution is a small two-dimensional source-only engineering and science smoke. No weights, gates, seeds, metrics, thresholds or labels from this run may retune B1–B4 or be counted as the phase-disjoint full-dictionary neural positive control without an additional preregistered higher-dimensional replication.

## Next step

Publish one JSON receipt with per-seed outcomes and unit tests verifying that the shared physical BPD wrapper is used, factor gradients flow in the trainable arm, phase support is correct, the calibration and held-out seeds differ, and stage-2 factor freezing holds. Publish package/software versions and runtime.
