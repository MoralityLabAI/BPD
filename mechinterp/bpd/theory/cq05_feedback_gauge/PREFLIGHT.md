# CQ-05 EvalMath mechanistic-resolution preflight

The [EvalMath mechanistic-resolution skill](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power) standard-library \`scripts/preflight.py --strict\` ran on **four excluded development roots**, 997, 1009, 1013, 1021, using the same synthetic recurrence and 32 *development* donor pairs per root. The registered prospective roots 1201…1249 were absent from this planning input.

- Independent unit: freshly trained weight-tied physical W, BPD V/U and feedback controller for one seed.
- Effect: difference in intention-to-treat donor-b success between projected current-dark native component writes and per-pair Euclidean-dose-matched random-dark directions.
- Development paired effects: **17/32**, **2/32**, **14/32**, **20/32**. Matched baseline sham rates (random-dark minus zero-sham): 4/32, 1/32, 3/32, 3/32. A single baseline random direction per donor pair is far short of an independent-world sham-tail estimate.
- Scientific minimum: +15 percentage points. Target planning effect +30 points. Measurement resolution 1/32 per development root. Family: 2 registered donor comparisons, α=.05.
- Proposed independent confirmation roots: 8, current sealed confirmation: **absent**.
- Required sham-tolerance criterion: upper bound for the population 95th sham percentile with 95% confidence; the sample-maximum construction requires **59 independent sham roots**, separately generated from the confirmation set.

**Strict result: \`REDESIGN_OR_PILOT_ONLY\`, exit code 2.**

The official preflight found: \`heldout_unsealed\`, \`sham_quantile_uncalibrated\`, \`too_few_pilot_worlds\`, and \`confirmation_underpowered\`.

Development pilot SD across the four independent root effects: 0.24656; maximum observed matched baseline value: 0.125; nonparametric upper sham-tail floor unavailable. Normal-based approximation gave **26 independent model roots** for 80% nominal power at target effect .30 above the .15 mechanism floor, conditional on the development SD, and estimated resolvable effect .41875 with eight roots.

The registered eight-root CQ-05 prospective data are retained as a **synthetic pilot**. Root-direction statistics and parametric root-level intervals describe the pilot population, not independent confirmatory evidence. Train once on new model seeds and register a new experiment after 59+ sham calibration roots and an independent confirmation cohort are specified.
