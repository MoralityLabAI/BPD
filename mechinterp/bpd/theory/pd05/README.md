# BPD-PD-05: count supervision versus STE gradients

The [prospective registration](PROTOCOL.md) and [frozen JSON configuration](CONFIG.json) were committed on November 17, 2026, before any PD-05 model training or generation.

Eight arms across 24 fresh seeds compare unsupervised count STEs (original sigmoid, identity-logit and temperature-2), count-supervised sigmoid/identity variants, a supervised frozen-dictionary positive control, the original count oracle and independent calibrated binary gates. The physical shared BPD `TiedComponentLinear` calls and trainable read/write factors are retained.

[test_experiment.py](test_experiment.py) executes source-only checks and excluded seed 1009 to verify exactly binary forward masks, distinguish backward derivatives, match initialization, preserve source support and isolate count labels to supervised arms. Registered science is released only by committing `EXECUTE_LOCK.json` with the final source SHA. The [dedicated CI](https://github.com/MoralityLabAI/BPD/actions) then runs 24×8 = 192 arms and stores one original JSON receipt with seed-level statistics and per-family diagnostics.

The primary gate compares `supervised_sigmoid` against `unsupervised_sigmoid` and requires at least 18/24 recovery successes, at least six net paired seed wins and exact McNemar p≤0.05. Independent count supervision is a deliberate positive control on a toy where active-coordinate count is directly observable from the input. The previous PD-04 failure remains intact and no neural BPD identification claim is made.


## Registered result and audit (17 November 2026)

[RESULTS_20261117.md](RESULTS_20261117.md) records the **first-source-locked scientific PASS**: supervised count with the original sigmoid-logit STE recovered 21/24 seeds, while its unsupervised counterpart recovered 0/24 (21 paired wins, exact McNemar p=9.5367e-7). Both originally planned source and outcome gates passed. Other arms retained their actual outcomes: supervised identity 14/24, supervised frozen-factor 23/24, oracle count 24/24, calibrated independent 6/24, and unsupervised identity/tempered2 each 0/24.

The [original 494,962-byte JSON receipt](https://github.com/MoralityLabAI/BPD/actions/runs/44096809925/artifacts/14335015466) is preserved at GitHub Actions run 44096809925, SHA256 `51b91e9cd3f6911cd853c9444d51ed2a06697b384f0f7464ef17b9c0285ed6dc`. The [24-seed paired CSV](PER_SEED_20261117.csv) is verified with a separate [source-only replay test](test_published_results.py).

[CountSeparation.lean](CountSeparation.lean) kernel-verifies a finite, quantized margin witness: magnitudes 75–125 for one active coordinate and the sum of two active coordinates at least 150 can be separated by a threshold of 138. It formalizes the source's directly accessible cardinality signal while leaving continuous source and optimizer claims appropriately scoped.
