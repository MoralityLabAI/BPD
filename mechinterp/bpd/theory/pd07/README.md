# BPD-PD-07: factor-read conditioned component routing

The prospective [CONFIG.json](CONFIG.json) and [PROTOCOL.md](PROTOCOL.md) were committed before any BPD-PD-07 training. This study retains PD-06's rotated 4D sparse-latent source and prohibited pair/triple support combinations, while using **24 fresh seed–rotation pairs**.

[experiment.py](experiment.py) trains six matched BPD arms: fully coupled gates reading normalized `x @ V`, an unlabelled raw-observation MLP, a stop-gradient read-feature control, a labelled count-supervised raw top-k baseline, planted frozen factors with the read gate, and a full-support oracle. Every non-frozen arm trains read and write factors throughout 240 Adam steps. Primary and raw controls receive no latent support/count labels.

[The excluded-seed test suite](test_experiment.py) uses seed 2003 to verify the actual shared physical wrapper, exact binary masks, matched init across trainable arms, deterministic QR, prohibition of heldout support shapes during training, poisoned latent labels unable to affect either unlabelled arm, and gradient flow from read-feature path into V. A dedicated first-run GitHub Actions workflow runs the full cohort only after these engineering checks pass and a one-time EXECUTE_LOCK.json pins the exact source commit.

The original [PD-06 result](../pd06/RESULTS_20261117.md) is preserved: count supervision recovers the rotated read directions but failed the strict compositional transfer test in 31/32 seeds. This experiment asks whether gates tied to each read coefficient supply the missing inductive bias without explicit count labels.

Preregistered primary success requires at least 18/24 read-coupled seed recoveries, six net paired seed wins over the raw independent-gate control, and exact two-sided McNemar p ≤ 0.05. All six held-out support families contribute to a seed's success gate. Results are stored as a full original JSON artifact even if the scientific hypothesis fails.


## Original first-run finding

[RESULTS_20261117.md](RESULTS_20261117.md) preserves the source-locked **scientific FAIL**: jointly trained read-conditioned gates recovered the whole six-stratum mechanism in **1/24** seeds, versus **0/24** raw independent gates (exact paired McNemar p=1.0). Read-feature detached recovered 1/24; raw-input count-supervised top-k 0/24.

The planted-Q frozen read gate and oracle-support factor control each succeeded **24/24** seeds. The learned primary arm's mean read-axis cosine was 0.8050 (only four seeds above 0.90), while its mean binary reconstruction MSE was just 0.00639. The error is dominated by incorrect factor directions and redundant gate execution, including on source combinations seen during training.

The [original GitHub Actions run](https://github.com/MoralityLabAI/BPD/actions/runs/44109446626) and [immutable 617,159-byte JSON receipt](https://github.com/MoralityLabAI/BPD/actions/runs/44109446626/artifacts/14340351331) remain linked. SHA256: 6d6b08374b635c7473a1fc9f3f03437bd584c4a3f55f28438f11fa5b2a8e6b6c. [PER_SEED_20261117.csv](PER_SEED_20261117.csv) and [test_published_results.py](test_published_results.py) allow independent seed-level replay of the failed primary gate.

[READGATE_THEORY.md](READGATE_THEORY.md) gives a sufficient robust read-alignment margin, and [ReadGateSeparation.lean](ReadGateSeparation.lean) kernel-checks exact threshold support on arbitrary 4-coordinate supports in the planted quantized case. Neither result establishes that the joint surrogate optimizer finds the aligned read basis.
