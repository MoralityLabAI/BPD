# BPD-PD-08: freeze the dictionary, then refit gates without labels

The prospective [registration](PROTOCOL.md) and [machine-readable CONFIG](CONFIG.json) were committed before any PD-08 training. This study follows [PD-07's source-locked negative result](../pd07/RESULTS_20261117.md): read-conditioning a jointly learned dictionary succeeded in 1/24 seeds, while read-conditioning a planted frozen dictionary succeeded in 24/24.

A count-supervised PD-06 factorizer learns one rotated 4D shared BPD physical dictionary per fresh seed. Its V,U matrices are then frozen bitwise. Label-free read-conditioned and raw-observation binary gates receive identical frozen factors and matched stage-two data/compute. The source has the same forbidden pairs, triple and compositional four-pass transition as PD-06, evaluated under a single per-seed global component alignment.

Six second-stage arms include read_refit (primary), raw_refit (control), read_nofit, read_shuffled_refit, the original stage1 count-gate without refitting, and privileged oracle support. Stage-two optimization has no cardinality or support labels. The initial count-supervision is explicitly privileged, so any success is conditional on that first-stage representation training.

[test_experiment.py](test_experiment.py) runs excluded seed 3001 and checks factor pretraining, bitwise V,U freezing during gate-only backprop, zero latent-label leakage, cyclic-shuffled read negative control, matched gate initializations, and deterministic support source. The dedicated first-run workflow only runs after CI passes and an EXECUTE_LOCK.json commit pins source.

The registered primary success gate requires >=18/24 read-refit seed recoveries across all six families, net >=6 seed wins over raw_refit, and exact paired McNemar p<=0.05. The original full JSON scientific receipt is preserved whether those criteria pass or fail.
