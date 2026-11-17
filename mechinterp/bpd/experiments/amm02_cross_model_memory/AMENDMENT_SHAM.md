# AMM-02 pre-development amendment: sham-variation calibration

Registered 2026-11-18 after inspecting only the eight explicitly excluded
`engineering` roots, **before training-memory / 64-world development scoring**.
Original protocol `ce390c8f0938a695dd0a73a92b53ab035e158763`
remains unchanged in history.

The original phrase "pooled - random_dark as the matched sham variation"
confounds genuine systematic performance differences between a BPD component
strategy and isotropic random-dark intervention with sampling variation.
For the strict EvalMath 95th-quantile / 95%-confidence sham *tolerance* floor,
replace the per-world sham value with the absolute difference in accuracy
between two independently seeded **equal-dose random-dark** interventions
`random_dark_a` and `random_dark_b` on the identical generated 48 pairs.
The original `random_dark` is renamed `random_dark_a`. Both are projected
into the current-readout kernel and use the same per-pair physical dose
`D`. The primary memory-pooled effect, the other two registered
comparisons, scientific floor +0.12, target +0.20, Holm family size 3,
n_train=64/n_dev=64/n_confirm=128, training scheme, donor pairs and sealed
confirmation seeds remain unchanged.

Report pooled-vs-random_dark_a explicitly as a *causal baseline contrast*;
the random-dark_a-vs-random-dark_b variation calibrates *noise* in the
null arm. This pre-development change prevents the null envelope from
silently counting a systematic causal difference as variation. Both random
arms are committed to every root's raw outcomes. No confirmation-world
generator may run absent successful strict EvalMath preflight.
