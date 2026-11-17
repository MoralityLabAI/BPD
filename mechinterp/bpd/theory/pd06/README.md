# BPD-PD-06: rotated four-dimensional compositional sparse-support transfer

The [prospective registration](PROTOCOL.md) and [machine-readable CONFIG](CONFIG.json) were committed on November 17, 2026, **before any PD-06 training**. The experiment samples an independent 4×4 orthogonal rotation per registered seed. Every observed coordinate is almost surely dense, while the latent source activates one, two or three of four basis directions.

The frozen training source contains four singleton supports, four adjacent coordinate pairs and three coordinate triples. **Two diagonal pairs and one triple are held out combinatorially.** The held-out evaluation also includes a four-pass transition sequence using unseen combinations.

[experiment.py](experiment.py) executes real BPDModel/TiedComponentLinear mask interventions through 4 shared rank-one factors. Its seven arms test unsupervised top-k, directly supervised count, privileged supervised support, a true-count oracle, true-support oracle, dense-factor reconstruction, and count supervision with a planted frozen read/write dictionary. All other arms train V and U throughout each 360-step fit.

[test_experiment.py](test_experiment.py) verifies QR orthogonality, dense observed coordinates, absence of held-out support combinations during fitting, exact 0/1 forward mask execution, factor/assignment/count gradients, matched initialization, and an explicit poisoned-label test showing unsupervised output is unchanged if latent labels are altered. Its sole engineering seed is 1009, outside the 32 registered model/rotation seeds.

The [dedicated first-run workflow](https://github.com/MoralityLabAI/BPD/actions) executes 32 seeds × 7 arms only after the excluded-seed CI is green and EXECUTE_LOCK.json pins the protected source commit. A complete original JSON receipt is archived even when the scientific success criterion fails.

**Primary registered gate:** at least 24/32 count-supervised models must recover the rotated basis, achieve adequate reconstruction, and exactly route >=80% seen-support and >=70% unseen-support four-pass trajectories, with at least eight paired seed wins over unsupervised top-k and exact two-sided McNemar p<=0.05.

The idealized expected-L0 source optimum is the unknown rotated coordinate basis, with 2 active components per pass under the balanced training support distribution. The binary-forward STE optimizer is a separate experiment, with no claim about real 7M mechanistic interpretability until independently evaluated.
