# BPD-CQ-04: adaptive two-pass masks on a trained recurrence

Prospective registration: [PROTOCOL.md](PROTOCOL.md). Independent unit is the learned recurrence seed. Eight prospective roots constitute a **pilot** under the EvalMath mechanistic-resolution skill, since independent sham-tail calibration and a separately sealed confirmation population remain open.

The experiment retrains CQ-03's learned two-pass rank-two operator and BPD read/write factors. A controller then imitates the smallest successful hard 0/1 component-mask plan on separate development examples whose behavioral labels are known independently from the BPD decomposition. It uses four logits for two components at each of two passes. At evaluation it only observes initial input state, and both masks are hard. This is **input-adaptive planning**, not a hidden-state feedback gate.

The strongest input-blind control is the best calibration-selected mixture of up to two constant patterns, with expected active slots at most the controller's calibration mean. Equal-cardinality randomized slots, full mask, and held-out dual-source cases are also scored. A separate donor-b assay patches the learned intermediate state using actual factor writes, with norm-matched dark controls and independent donor labels.

From the repository root:

~~~sh
python -m unittest discover -s mechinterp/bpd/theory/cq04_adaptive_masks/tests -v
python -m mechinterp.bpd.theory.cq04_adaptive_masks.experiment --engineering
python -m mechinterp.bpd.theory.cq04_adaptive_masks.reference --backend independent --out mechinterp/bpd/theory/cq04_adaptive_masks/results/cq04_independent.json
python -m mechinterp.bpd.theory.cq04_adaptive_masks.reference --backend native --out /tmp/cq04_native.json
~~~

The reference writer is write-once. Use \`--check\` to replay the original source-locked result. Native execution uses \`BPDModel/TiedComponentLinear\` with \`PassMaskSchedule\` in both steps and donor continuation. A native receipt must match the frozen independent per-seed integer counts.

Interpretation restrictions: the controller knows (a,b) at inference in the hidden input coordinates. It is therefore not discovering those variables. It is learning a *compute allocation* over trained components. With a full \`GL(2)\` change of basis, fixed singleton masks can change meaning. The rank-two synthetic result is not a nonlinear RMP result.
