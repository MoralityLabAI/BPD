# BPD-PD-08: frozen read/write dictionary followed by label-free gate fitting

The [prospective protocol](PROTOCOL.md) and [machine-readable CONFIG](CONFIG.json) were committed before PD-08 training. The study is a direct follow-up to [PD-07's registered result](../pd07/RESULTS_20261117.md), where planted read gates passed 24/24 but jointly trained read gates passed only 1/24, with mean read cosine 0.805.

PD-08 uses 24 *new* orthogonally rotated 4D model seeds with the exact same training source and six heldout support families as PD-06/07. Stage one trains a single physical BPDModel factor dictionary per seed, using PD-06's **count-supervised** top-k objective for 360 updates. Count labels enter only this first-stage dictionary-training procedure. Stage two **freezes both V and U**, creates five matched arms from that exact snapshot, and trains independent read-threshold or raw-input gates on identical new source batches, without count/support labels.

The five arms are a trained read-coefficient threshold, a trained raw-input independent MLP, a fixed 0.5 read threshold, the full-support oracle executed through the learned factors, and a planted-frozen read positive control. [experiment.py](experiment.py) implements this with the original BPD physical component wrapper; [test_experiment.py](test_experiment.py) verifies the factor freeze and label isolation on excluded seed 2017.

Primary success requires at least 18/24 seed recoveries with a single component alignment across all six heldout families, at least six net wins of the read gate over the raw gate, and exact McNemar p<=0.05. A separate strict output MSE and four-pass exact support gate applies per seed.

The first scientific cohort is released only after source CI passes and an `EXECUTE_LOCK.json` pins the audited experiment source. The dedicated workflow writes the original 120-arm JSON receipt and preserves a scientific failure unchanged. These remain controlled synthetic and partly count-supervised experiments, without a claim of complete unsupervised mechanistic identification.


## First registered result, 17 November 2026

[RESULTS_20261117.md](RESULTS_20261117.md) preserves a scientific **PASS** for the frozen primary hypothesis: trained read-threshold gating recovered 23/24 seed-level six-family mechanisms after count-supervised factor discovery, while a matched raw-input independent MLP recovered 0/24. There were 23 read-only paired successes and no raw-only successes; exact two-sided McNemar p=2.384185791015625e-7.

The [complete unmodified 120-arm JSON](https://github.com/MoralityLabAI/BPD/actions/runs/44114726057/artifacts/14342451949), SHA256 73704fd7eeef67930b7913ebdcf81d0fc748039898f7e797d86bc410ae683483, and [24-seed CSV](PER_SEED_20261117.csv) are archived with [independent source-only result replay](test_published_results.py).

A fixed read threshold 0.5, using the same learned frozen factors, also recovered 23/24 seeds and executed every held-out support perfectly in the 23 sufficiently aligned dictionaries. Trained thresholds approached 0.75 and added some false-offs. Seed 1237 failed stage-one factor discovery, which even an oracle-support gate could not rescue. These findings are conditional on the stage-one cardinality supervision, not evidence of a wholly unsupervised neural mechanism.
