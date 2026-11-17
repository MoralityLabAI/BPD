# BPD-PD-07: factor-read conditioned component routing

The prospective [CONFIG.json](CONFIG.json) and [PROTOCOL.md](PROTOCOL.md) were committed before any BPD-PD-07 training. This study retains PD-06's rotated 4D sparse-latent source and prohibited pair/triple support combinations, while using **24 fresh seed–rotation pairs**.

[experiment.py](experiment.py) trains six matched BPD arms: fully coupled gates reading normalized `x @ V`, an unlabelled raw-observation MLP, a stop-gradient read-feature control, a labelled count-supervised raw top-k baseline, planted frozen factors with the read gate, and a full-support oracle. Every non-frozen arm trains read and write factors throughout 240 Adam steps. Primary and raw controls receive no latent support/count labels.

[The excluded-seed test suite](test_experiment.py) uses seed 2003 to verify the actual shared physical wrapper, exact binary masks, matched init across trainable arms, deterministic QR, prohibition of heldout support shapes during training, poisoned latent labels unable to affect either unlabelled arm, and gradient flow from read-feature path into V. A dedicated first-run GitHub Actions workflow runs the full cohort only after these engineering checks pass and a one-time EXECUTE_LOCK.json pins the exact source commit.

The original [PD-06 result](../pd06/RESULTS_20261117.md) is preserved: count supervision recovers the rotated read directions but failed the strict compositional transfer test in 31/32 seeds. This experiment asks whether gates tied to each read coefficient supply the missing inductive bias without explicit count labels.

Preregistered primary success requires at least 18/24 read-coupled seed recoveries, six net paired seed wins over the raw independent-gate control, and exact two-sided McNemar p ≤ 0.05. All six held-out support families contribute to a seed's success gate. Results are stored as a full original JSON artifact even if the scientific hypothesis fails.
