# BPD-PD-08: freeze the dictionary, then refit gates without labels

The prospective [registration](PROTOCOL.md) and [machine-readable CONFIG](CONFIG.json) were committed before any PD-08 training. This study follows [PD-07's source-locked negative result](../pd07/RESULTS_20261117.md): read-conditioning a jointly learned dictionary succeeded in 1/24 seeds, while read-conditioning a planted frozen dictionary succeeded in 24/24.

A count-supervised PD-06 factorizer learns one rotated 4D shared BPD physical dictionary per fresh seed. Its V,U matrices are then frozen bitwise. Label-free read-conditioned and raw-observation binary gates receive identical frozen factors and matched stage-two data/compute. The source has the same forbidden pairs, triple and compositional four-pass transition as PD-06, evaluated under a single per-seed global component alignment.

Six second-stage arms include read_refit (primary), raw_refit (control), read_nofit, read_shuffled_refit, the original stage1 count-gate without refitting, and privileged oracle support. Stage-two optimization has no cardinality or support labels. The initial count-supervision is explicitly privileged, so any success is conditional on that first-stage representation training.

[test_experiment.py](test_experiment.py) runs excluded seed 3001 and checks factor pretraining, bitwise V,U freezing during gate-only backprop, zero latent-label leakage, cyclic-shuffled read negative control, matched gate initializations, and deterministic support source. The dedicated first-run workflow only runs after CI passes and an EXECUTE_LOCK.json commit pins source.

The registered primary success gate requires >=18/24 read-refit seed recoveries across all six families, net >=6 seed wins over raw_refit, and exact paired McNemar p<=0.05. The original full JSON scientific receipt is preserved whether those criteria pass or fail.


## First source-locked cohort result (2026-11-17)

[RESULTS_20261117.md](RESULTS_20261117.md) records a **preregistered PASS**: count-supervised dictionary pretraining followed by frozen-factor, label-free componentwise read refitting recovered mechanisms on **23/24** independent seed–rotation pairs, versus **0/24** for an equal-budget raw-input independent gate fit to the same physical factors. All 23 recovered models also passed with the **unrefitted read threshold**, and the cyclic-shuffled read/write correspondence sham recovered 0/24. The sole failed count-pretrained basis was seed 1237 (read cosine 0.794). The original stage-one raw count-top-k gate reached only 2/24.

The exact paired McNemar p-value is **2.384185791015625e−7**, with 23 read-only seed successes and zero raw-only. The full original 144-arm JSON is preserved in [GitHub Actions artifact 14340839503](https://github.com/MoralityLabAI/BPD/actions/runs/44114011511/artifacts/14340839503), SHA256 40133664fd05b71edfb9999b1885085125a0372ddafca0350b94fa3756aa544b. [PER_SEED_20261117.csv](PER_SEED_20261117.csv) and [test_published_results.py](test_published_results.py) replay the registered outcome without training.

This is **conditional compositional routing after privileged count-supervised factor discovery**. It does not establish unsupervised recovery of the factor dictionary or an interpretable mechanism in a pretrained transformer.
