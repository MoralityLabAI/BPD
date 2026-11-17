# BPD-PD-05: count supervision versus STE gradients

The [prospective registration](PROTOCOL.md) and [frozen JSON configuration](CONFIG.json) were committed on November 17, 2026, before any PD-05 model training or generation.

Eight arms across 24 fresh seeds compare unsupervised count STEs (original sigmoid, identity-logit and temperature-2), count-supervised sigmoid/identity variants, a supervised frozen-dictionary positive control, the original count oracle and independent calibrated binary gates. The physical shared BPD `TiedComponentLinear` calls and trainable read/write factors are retained.

[test_experiment.py](test_experiment.py) executes source-only checks and excluded seed 1009 to verify exactly binary forward masks, distinguish backward derivatives, match initialization, preserve source support and isolate count labels to supervised arms. Registered science is released only by committing `EXECUTE_LOCK.json` with the final source SHA. The [dedicated CI](https://github.com/MoralityLabAI/BPD/actions) then runs 24×8 = 192 arms and stores one original JSON receipt with seed-level statistics and per-family diagnostics.

The primary gate compares `supervised_sigmoid` against `unsupervised_sigmoid` and requires at least 18/24 recovery successes, at least six net paired seed wins and exact McNemar p≤0.05. Independent count supervision is a deliberate positive control on a toy where active-coordinate count is directly observable from the input. The previous PD-04 failure remains intact and no neural BPD identification claim is made.
