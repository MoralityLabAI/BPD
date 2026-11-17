# BPD-PD-06: rotated four-dimensional compositional sparse-support transfer

The [prospective registration](PROTOCOL.md) and [machine-readable CONFIG](CONFIG.json) were committed on November 17, 2026, **before any PD-06 training**. The experiment samples an independent 4×4 orthogonal rotation per registered seed. Every observed coordinate is almost surely dense, while the latent source activates one, two or three of four basis directions.

The frozen training source contains four singleton supports, four adjacent coordinate pairs and three coordinate triples. **Two diagonal pairs and one triple are held out combinatorially.** The held-out evaluation also includes a four-pass transition sequence using unseen combinations.

[experiment.py](experiment.py) executes real BPDModel/TiedComponentLinear mask interventions through 4 shared rank-one factors. Its seven arms test unsupervised top-k, directly supervised count, privileged supervised support, a true-count oracle, true-support oracle, dense-factor reconstruction, and count supervision with a planted frozen read/write dictionary. All other arms train V and U throughout each 360-step fit.

[test_experiment.py](test_experiment.py) verifies QR orthogonality, dense observed coordinates, absence of held-out support combinations during fitting, exact 0/1 forward mask execution, factor/assignment/count gradients, matched initialization, and an explicit poisoned-label test showing unsupervised output is unchanged if latent labels are altered. Its sole engineering seed is 1009, outside the 32 registered model/rotation seeds.

The [dedicated first-run workflow](https://github.com/MoralityLabAI/BPD/actions) executes 32 seeds × 7 arms only after the excluded-seed CI is green and EXECUTE_LOCK.json pins the protected source commit. A complete original JSON receipt is archived even when the scientific success criterion fails.

**Primary registered gate:** at least 24/32 count-supervised models must recover the rotated basis, achieve adequate reconstruction, and exactly route >=80% seen-support and >=70% unseen-support four-pass trajectories, with at least eight paired seed wins over unsupervised top-k and exact two-sided McNemar p<=0.05.

The idealized expected-L0 source optimum is the unknown rotated coordinate basis, with 2 active components per pass under the balanced training support distribution. The binary-forward STE optimizer is a separate experiment, with no claim about real 7M mechanistic interpretability until independently evaluated.


## First registered run and audit

[RESULTS_20261117.md](RESULTS_20261117.md) records the preregistered **scientific FAIL** in all 224 model arms: count-supervised 1/32, unsupervised 0/32 (exact paired p=1.0), support-supervised 17/32, oracle count 0/32, oracle support 32/32, count-supervised planted frozen 6/32, and dense-all 0/32.

The original [978,682-byte full JSON receipt](https://github.com/MoralityLabAI/BPD/actions/runs/44103217529/artifacts/14338021049), SHA256 6d33689b84ac40f169afc04d1cae50d35bc74ff319d820a56d02143d14fb9d0b, is immutable in GitHub Actions. The [32-seed paired CSV](PER_SEED_20261117.csv) and [independent source-only replay](test_published_results.py) preserve seed-level outcomes and exact correct-board counts in all six support strata.

The count-supervised read dictionary aligned with the rotated latent axes in **32/32 seeds**, but support routing fell from 93.47% on seen pairs to **31.18% on withheld pairs**, with 56.20% on the withheld triple and 42.07% on unseen-support transitions. The stronger support-supervised arm improved these values to 71.08%, 82.24% and 80.26%, demonstrating that count alone did not transfer component assignment to new support combinations.

[THEORY.md](THEORY.md) explains the source-law expected-L0 strict basis optimum. [RotatedSupportMass.lean](RotatedSupportMass.lean) verifies the finite support weights and the structural absence of withheld combinations from the training distribution. It does not kernel-prove convergence of learned gate training.
