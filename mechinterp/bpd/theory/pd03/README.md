# BPD-PD-03: deterministic top-1 versus margin-calibrated gates

The [prospective protocol](PROTOCOL.md) and [machine-readable registration](CONFIG.json) were committed on November 17, 2026, before any PD-03 training. PD-02's original 6/24 hard-gate result, 10/24 soft comparison, and 498 false-off events are preserved on main.

[experiment.py](experiment.py) implements seven arms across 24 fresh model seeds using the actual BPDModel shared physical weight wrapper. Both read and write factors receive gradients throughout every trainable arm. Exact one-hot and threshold-binary forward masks use explicitly implemented straight-through backward derivatives. Calibration chooses a margin arm's threshold using separate calibration examples, before held-out inference.

[test_experiment.py](test_experiment.py) runs an excluded-seed engineering smoke (seed 1009), verifying mask exactness, top-1 one-hot behavior, gradients through the real wrapper, identical initial parameters, split isolation, and threshold selection. The one-time full registered run is triggered by adding EXECUTE_LOCK.json only after these tests pass. CI archives a full original JSON receipt and prints the scientific pass/fail result without changing its thresholds.

The study is a controlled two-dimensional optimizer and gating experiment, with no claims about neural mechanistic identification or exact L0 minimization.
