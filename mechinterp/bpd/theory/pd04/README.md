# BPD-PD-04: learned adaptive top-1/top-2 routing

[PROTOCOL.md](PROTOCOL.md) and [CONFIG.json](CONFIG.json) are the original prospective registration from November 17, 2026, committed before any PD-04 model training. This study uses three balanced support families: disjoint single-factor phases, full two-factor overlap, and within-trajectory nested support.

[experiment.py](experiment.py) implements the real shared BPDModel wrapper and an adaptive count gate. Its exact binary mask selects either one learned factor or both. A differentiated count head learns the choice from the input, with no family labels. Baselines include calibrated independent gates, fixed independent gates, fixed top-1, always-two, oracle count, and the adaptive gate with frozen SVD factors. All non-frozen arms update V and U throughout 240 optimizer steps.

[test_experiment.py](test_experiment.py) is an engineering smoke using excluded seed 1009. It validates generator support, binary mask exactness, separate assignment/count gradients, matched initialization, and calibration separation. Only after it passes is EXECUTE_LOCK.json written, causing the dedicated [first-run GitHub Actions workflow](../../../../../.github/workflows/bpd-pd04-registered.yml) to execute the fixed 24 × 7 cohort.

Scientific success requires all three support families to reach 90% exact four-pass routing per seed, sufficient reconstruction quality and learned read-axis alignment. The primary comparative gate requires >=18 adaptive successes, >=6 net seed wins over calibrated independent gating and exact two-sided McNemar p<=0.05. The original PD-02/03 reports remain unchanged.

**Evidence boundary:** controlled two-dimensional parameter decomposition. Neither a pass nor a fail establishes trained neural transformer mechanism identification, and exact-L0 global optimality is a separate mathematical claim.


## Registered first-run result and proof follow-up

[RESULTS_20261117.md](RESULTS_20261117.md) preserves the **scientific FAIL** of the locked 24-seed × seven-arm cohort: learned adaptive 0/24, calibrated independent 6/24, and oracle cardinality 22/24. The paired McNemar test is p=0.03125 in the comparator's direction. No scientific thresholds were changed after opening the held-out source.

The immutable 439,372-byte [full first-run JSON artifact](https://github.com/MoralityLabAI/BPD/actions/runs/44084196892/artifacts/14327829889), SHA256 4c3d8655d71b86329d715b61a6d43de0bf61efba377ee5f7ca2dbe1f9a27fc9e, contains all 168 arms. [PER_SEED_20261117.csv](PER_SEED_20261117.csv) extracts the paired recovery and per-family support values, audited by [test_published_results.py](test_published_results.py).

[PlantedCapacity.lean](PlantedCapacity.lean) is a Lean 4.19 kernel-verified, limited-scope proof that planted coordinate one-hot writes cannot reconstruct a two-coordinate overlap and that switching on an inactive-coordinate factor does not change the represented output. This accounts for the task-capacity negative controls and the difference between low MSE and correct support.
