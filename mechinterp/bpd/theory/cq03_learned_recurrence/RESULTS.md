# BPD-CQ-03: frozen independent CPU result

**Prospective result (2026-11-17): Registered scientific gate passes 6/8 seeds, exactly the declared minimum.** The mathematical and native-equivalence evidence is for a newly trained synthetic rank-two linear recurrence. The source, protocol and test suite were committed to the BPD research branch before generating the eight-seed heldout set. The excluded engineering seed 997 does not enter these totals. Original PD-01/02/03, BPD-CQ-01/02 and sealed ARG-03-N evidence remain untouched.

The task supplies `h0=(0,0,a,b)` and requires one shared physical operator to produce `y1=a` and `y2=b`. The physical transition is trained from those independent labels, then frozen while a second factor pair V,U is trained on its trajectories. The source-selected component uses the frozen delayed response derivative; the donor test goal comes from the input's **b**, never from the component write.

Eight independent physical-model seeds, each with 520 teacher updates and 360 factor updates, two recurrent visits, 128 fresh paired tests/seed. Every seed passes clean task eligibility for all 128/128 pairs. Reconstruction errors are tiny, with max per-perturbation norm error **1.20e-7**, and full two-component gauge write error **1.05e-7**. Each learned full-observer component Gramian has rank 2 after one additional transition.

## Registered causal outcome

| Intervention after first visit | Donor-b transfers / 1024 | Rate |
|---|---:|---:|
| Full intermediate state replacement (larger-budget positive) | 1024 | 100.0% |
| Native selected component write | 798 | 77.9% |
| Selected component projected into current-dark fiber, dose-matched | **770** | **75.2%** |
| Random current-dark direction, per-pair dose-matched | **186** | 18.2% |
| Unselected component, dose-matched | 15 | 1.5% |
| Current-visible-only, dose-matched | 0 | 0.0% |
| Sham | 0 | 0.0% |

**Seed-level dark transfer:** `607:0/128`, `613:128/128`, `617:128/128`, `619:128/128`, `631:128/128`, `641:128/128`, `643:2/128`, `647:128/128`. All six successful seeds exceed their own norm-matched random-dark arm by at least 20 percentage points. Seed 607 and 643 fail the registered gate and are retained without replacement. Native writes can disturb the current visible readout (seed 607 0/128 current-preserving; seed 641 57/128), so the projected-dark comparison is the primary delayed-observability control.

Each learned BPD factor pair also has an unfixed GL(2) representation gauge. The fixed triangular shear preserves the full represented weight product within 1.05e-7 but changes singleton causal interventions. Across the eight trained seeds, the mean singleton patch-drift norms range 0.237 to 0.534; this is evidence of the gauge ambiguity, not of a canonical learned feature.

## Provenance and limits

Source-locked, write-once JSON: [results/cq03_independent.json](results/cq03_independent.json), with per-seed clean and patch outcomes, component choice, norm checks, and training environment. `reference --check` reruns all eight seeds and checks their complete integer outcome counts against that frozen receipt. The local standalone PyTorch 2.10.0+cpu run passed the reference replay; native BPDModel execution is separately checked in hosted CI.

The physical recurrence is **learned**, while its supervised task, rank-two capacity and readout coordinates are deliberately constructed. There is no trained sparse gate here: both components execute in every pass. The strong endpoint reports a one-task causal component that is calibration selected and representation dependent. It does not prove latent reasoning in an RMP checkpoint, nonlinear manifold topology, or robustness to arbitrary equivalent BPD bases. The two failing seeds show that read/write identifiability is material even when clean model function is near-perfect.
