# BPD-CQ-02: trained factors and delayed observation (pre-data protocol)

Registered 2026-11-17 before the new CPU cohort. Builds on BPD-CQ-01's exact observable-mask quotient. This is a synthetic four-dimensional BPD-style model whose shared two-component read/write factors and hard top-1 gate are trained. It uses a separately specified linear recurrent transport, so its delayed effects cannot be interpreted as a learned Transformer recurrence. The 168-case ARG-03-N test stays sealed. PD-01/02/03 and SCM results remain frozen.

## Source, seeds, and training
- Four visits of R4 states: visits 0,1 carry only input channel 0; visits 2,3 only channel 1. Signed amplitudes uniform on [0.75,1.25]; remaining two channels zero.
- Target linear map W=diag(1,1,0,0). Two physical shared read/write factors V (4x2), U (2x4) use masked component execution y=((x V) * gate) U with residual mask zero. A single shared abs(x)-based affine gate selects exactly one component; forward one-hot argmax with softmax straight-through derivative. Confirm native BPDModel TiedComponentLinear masked-component parity where supported.
- Eight independent model seeds 503,509,521,523,541,547,557,563. Train with Torch FP32 CPU, one thread, 200 Adam updates per seed, lr=.04, batch=16, train RNG seed+100000. Initialization factor normal std 1/sqrt(4), gate normal std .05, zero gate biases. Loss MSE output versus W input + .30 weight reconstruction MSE + .05 mean gate activity. Both factors and gate update.
- Calibrate component ordering using 128 independent boards, RNG seed+300000, by cosine with first channel. Test on 128 heldout boards, RNG seed+200000. Report axis recovery, true hard-gate four-pass routing, read/write factor updates, train/heldout reconstruction MSE.

## Declared observer and intervention
- Readout C=(0,1,0,0), fixed transport T swaps channels 0 and 1 and fixes channels 2,3. Identity T is the no-transport control. T is hand-designed and frozen, not trained.
- After calibration freeze, draw 64 new phase-0 donor/recipient pairs per seed using RNG seed+400000. Recipient positive amplitudes in [.75,.95], donor in [1.05,1.25].
- Only swap the calibrated native first-component write on the recipient. Let delta = donor native component write minus recipient native component write; record immediate |C delta| / ||delta|| and one-step delayed |C T delta| / ||delta||. A delayed-dark witness has immediate <=.10 and delayed >=.75. Also project delta into ker C and rescale to ||delta||, ensuring an exact instantaneously invisible fiber patch.
- Compare the native, projected-native fiber, sham, and random ker(C) patches; random directions are 3D, independently sampled per case, with ||patch|| matched to ||delta|| to 1e-5. Grade donor scalar transfer |C T(w_rec + patch) - C T(w_don)| <= .05. Report pair-level success with seed-cluster unit. No threshold tuning on final pairs.

## Gates and limits
- Instrument validity: both trained factors move, outputs finite, hard gate exactly one component, official wrapper parity if available, and all random/projected doses individually norm-matched.
- Prospective pilot success: >=6/8 independent trained seeds meet >=.90 complete-board routing and median immediate <=.10 and median delayed >=.90. Report failures and matched-control scores regardless.
- This is a positive/negative instrument for delayed observability in **trained sparse factors with an engineered recurrence**. It does not identify canonical BPD components, a pretrained recurrent mechanism, a causal mathematical-reasoning variable, or any topological surface.
- Publish source, source hashes, per-seed and aggregate receipts, CPU cost, torch/Python versions, tests, and the null/mixed-source controls. Never access or unseal ARG-03-N, nor the private SCM checkpoint artifacts. Preserve the original contract even if the pilot fails.
