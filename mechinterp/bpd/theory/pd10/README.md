# BPD-PD-10: cold-start smoothing against the mixed-pair local optimum

The [frozen prospective protocol](PROTOCOL.md) and [machine-readable CONFIG](CONFIG.json) were committed on 17 November 2026 before any PD10 model training. The experiment is a registered continuation of [PD09's first-run failure](../pd09/RESULTS_20261117.md), in which an initial smoothing width 0.6 led to two source-aligned read axes and one near-45-degree mixed pair in every seed.

[experiment.py](experiment.py) tests 24 new orthogonal-rotation seeds across six arms. Four unlabelled orthogonal QR arms start from identical random BPD factor matrices and draw identical training batches: cold sigma 0.15, cold sigma 0.30, warm 0.30 to 0.15, and the original 0.60 to 0.30 to 0.15 annealing schedule. All use the same 360 Adam updates and final fixed **hard 0/1 read gate at threshold 0.5**, with no post-training gate refitting.

A separate 42-degree pair-mixed control deliberately initializes the factor basis using true Q and is **privileged**, not evidence of unlabelled recovery. The count-supervised control repeats PD06's labelled optimizer and is also explicitly privileged. The underlying six PD06 support families, forbidden diagonal pairs and triple, and true shared BPD physical interventions are unchanged.

[test_experiment.py](test_experiment.py) checks the QR gradients, initialized factor matching, source-label isolation, the 42-degree stress initializer, exact binary physical outputs and training stream split on excluded seed 2011. The one-time [registered Actions workflow](https://github.com/MoralityLabAI/BPD/actions) runs the full 24×6 cohort only after these checks pass and EXECUTE_LOCK.json pins the source SHA.

**Primary scientific gate:** cold sigma 0.15 must recover at least 18 of 24 entire six-family mechanisms, earn six net paired seed wins over the original annealing schedule, and pass the exact two-sided McNemar test at p<=0.05. The raw first-run receipt and every failure are retained even if this gate fails. These are synthetic 4D source-specific optimizer tests with no pretrained TRM mechanism claim.


## First registered result

[RESULTS_20261117.md](RESULTS_20261117.md) records the preregistered **scientific FAIL**: cold sigma 0.15 recovered 2/24 new rotated four-dimensional mechanisms, original sigma 0.60→0.30→0.15 recovered 0/24, net +2 paired seeds, exact two-sided McNemar p=0.5. Cold 0.30 and warm 0.30→0.15 also recovered 2/24; the 42-degree privileged mixed-pair initialization stress arm recovered 0/24; the true-count-supervised positive control recovered 24/24.

The [original 144-arm JSON artifact](https://github.com/MoralityLabAI/BPD/actions/runs/44124926867/artifacts/14347856096) has 505,574 bytes, SHA256 cfbac7207a7e354769bffa2620f94ef7d3379c7018de0d737b7e4209a9649b81. [PER_SEED_20261117.csv](PER_SEED_20261117.csv) and [test_published_results.py](test_published_results.py) preserve and replay every seed-level first-run outcome. No source hyperparameter, seed, or threshold was changed after opening the held-out evaluation.
