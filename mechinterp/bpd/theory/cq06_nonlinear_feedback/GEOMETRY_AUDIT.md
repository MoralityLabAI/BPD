# CQ-06 post-confirmation geometry audit (interpretation only)

**Status:** Retrospective descriptive analysis, performed after both heldout source-locked stages. It changes no registered endpoint, intervention, seed, familywise threshold or receipt.

A tanh recurrent state has every coordinate inside [-1,1]. The registered intervention takes a donor-minus-recipient BPD read/write patch, projects it into the current linear-readout kernel and normalizes it to its original Euclidean magnitude. The stronger null follows the same procedure but starts with an unrelated clean first-pass source of matching first-step parity. This physically matches norm and current readout, while **projection and rescaling generally leave the state box**.

| Arm | Calibration out of range (n=7,552) | Confirmation out of range (n=3,328) |
|---|---:|---:|
| Current-dark BPD | 7,263 (96.17%) | 3,143 (94.44%) |
| Matched clean-source #1 | 6,620 (87.66%) | 2,979 (89.51%) |
| Matched clean-source #2 | 6,597 (87.35%) | 2,946 (88.52%) |
| Gaussian dark #1 | 7,552 (100%) | 3,326 (99.94%) |
| Unprojected BPD | 0 | 0 |
| Sham | 0 | 0 |
| Full donor replacement | 0 | 0 |

The comparison is best described as **equal-dose current-dark interventions seeded from different directions**, with clean recurrent states furnishing the manifold-inspired null direction. It does not certify that the patched state stays on the reachable-state manifold. The exact registered tests remain unchanged and apply to this intervention distribution. The causal advantage might be a combination of useful hidden-variable transport, differential off-manifold response and dose geometry. Root-level corrected confirmation comparisons cannot separate those causes by themselves.

Future prospectively registered gym should (1) generate a nonlinear transition with explicit reachability constraints; (2) compare an in-range projected patch, an equal-task-repair interpolated patch and genuinely reachable clean-donor or intervention-state manifold projections; (3) compare identical dose distributions after clipping/transport and report out-of-support mass and head perturbations; (4) use 59+ fresh independent sham-world calibration, fresh power planning, and new sealed confirmation worlds; (5) test BPD block invariance and learned 7M TRM states only after synthetic geometric controls close.
