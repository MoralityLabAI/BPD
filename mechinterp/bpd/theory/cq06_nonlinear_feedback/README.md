# BPD-CQ-06: nonlinear delayed-state causal quotient

See [prospective protocol](PROTOCOL.md), [pre-test amendments](AMENDMENTS.md), [machine registration](REGISTRATION.json), and [mathematics](THEORY.md).

The two-visit nonlinear, rank-four weight-tied tanh recurrence receives x,y on its first visit, and a newly revealed z on its second. It predicts sign(xy) then sign(xz). The delayed target is independent of the first-visit observation under paired signs of z. A donor that flips both x and y preserves its first visible target while changing its second target.

A separately trained BPD factorization reconstructs the learned physical recurrent projection. Its full product is canonicalized by SVD before component selection and gate training. Four components execute on the first visit; a hard second-stage gate sees the actually reached state and fresh cue. Gate-training targets are *full-model behavior and trajectory*, and do not use independent task-label targets. The causal intervention uses the two leading singular components and preserves the immediate linear readout after projection into the current-dark kernel.

The **primary null** is a physically matched, unrelated clean intermediate state with the same first parity, dark-projected to the same physical Euclidean dose. Gaussian dark directions, sham, visible-only, and a full-donor-state ceiling are retained. The registered comparison counts **all 128 donor pairs per independent root**. Native execution uses BPDModel/TiedComponentLinear under PassMaskSchedule with residual delta off.

## Run stages (from the BPD checkout root)

~~~sh
python -m unittest discover -s mechinterp/bpd/theory/cq06_nonlinear_feedback/tests -v
python -m mechinterp.bpd.theory.cq06_nonlinear_feedback.reference --phase development
python -m mechinterp.bpd.theory.cq06_nonlinear_feedback.reference --phase sham
python -m mechinterp.bpd.theory.cq06_nonlinear_feedback.reference --phase preflight
# Runs only when the EvalMath preflight reports READY_FOR_CONFIRMATION:
python -m mechinterp.bpd.theory.cq06_nonlinear_feedback.reference --phase confirmation
python -m mechinterp.bpd.theory.cq06_nonlinear_feedback.reference --phase confirmation --backend native --check --out /tmp/cq06_native.json
~~~

The cohorts consist of eight excluded development roots, 59 independent sham-calibration roots, and 26 separately withheld confirmation roots. Each root includes a separate random input rotation, trained physical recurrence, trained BPD factor pair and learned gate. Strict EvalMath mechanistic-resolution planning uses 59 independently fitted null-world outcomes before opening confirmation. Source hashes and raw per-pair receipts must be kept; negative and inconclusive outcomes count.

All evidence pertains to this deliberately synthetic parity task family. Natural-language TRM/RMP checkpoints and the sealed ARG-03-N corpus are outside the experiment.
