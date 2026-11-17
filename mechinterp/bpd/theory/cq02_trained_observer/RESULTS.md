# BPD-CQ-02 CPU pilot result (2026-11-17)

**Registered instrument gate PASSED for the independent shared-factor implementation.** This is a positive control on a source designed to make a rank-one component dark now and visible after one specified channel swap. The original 8-seed cohort is in `results/cq02_cpu_reference.json`; no target dates, seeds, thresholds or CPU settings were changed after preregistration.

- Eight fresh trained 4D two-component dictionaries; both trainable factors changed in every seed. 200 optimizer steps each (1,600 total), exact forward top-one mask; all 8/8 seeds have 128/128 correct four-pass masks on independently sampled phase-disjoint states.
- For each trained dictionary, the BPD two-component observability matrix on its learned write factors grows from rank 1 at horizon 0 to rank 2 at horizon 1. Minimum two-horizon singular value across seeds: see frozen JSON. This is because the readout and transition deliberately expose complementary output channels.
- Immediate native component-write sensitivity median at most 0.013 per seed; delayed median at least 0.9999 per seed. All 512 heldout donor-recipient pairs meet the fixed delayed-dark diagnostic (immediate <=0.10, delayed >=0.75).
- Native write swaps and their exact fiber-projected equivalents achieve 512/512 donor-scalar transfers. Separately sampled random 3D head-fiber controls: 54/512; sham 0/512, instant-visible channel control 0/512. Per-pair doses satisfy the frozen 1e-5 norm tolerance.
- Mixed-source negative control MSE stays approximately 0.25: one rank-one gate cannot reconstruct two simultaneously required coordinates. This is a task-capacity sanity check, not a statement about trained language models.
- Local CPU Python 3.13.5 / PyTorch 2.10.0, single-thread, approximately 1.8–2.0 seconds for the 1,600 updates plus eval, no CUDA.

**Scientific caveat:** The native donor-scalar target is constructed from donor component writes and the visible transition is a preselected swap; the large transfer advantage mainly validates geometry and patch plumbing. A real trained recurrent-parameter mechanism needs actual learned transition dynamics and a behavioral endpoint independent of the intervention construction. The `BPDModel/TiedComponentLinear`-native integration is separately checked in hosted CI; no local native-backend result is claimed where that package is unavailable.

Original BPD-CQ-01, BPD PD-01/02/03, SCM receipts, and sealed ARG-03-N remain unchanged.
