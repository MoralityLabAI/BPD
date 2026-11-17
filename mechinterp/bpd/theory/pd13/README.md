# BPD-PD-13: clustered 16D recovery with pair priority and eightfold sample volume

[CONFIG.json](CONFIG.json) and the [prospective scientific protocol](PROTOCOL.md) were frozen before any PD-13 training on November 18, 2026. The source imports the unchanged [PD-12 16D clustered generator and physical BPD evaluator](../pd12/experiment.py), which had recovered 0/8 clustered strong-margin models and 0/8 clustered weak-margin models under the original 37-angle global Givens procedure.

The experiment has 32 fresh model/rotation seeds, with 16 in each clustered margin condition. Each seed draws **one fixed pool of 2048 training trajectories**. All four unlabelled search arms share the same initial orthogonal read dictionary; the smaller 256-board arms use exactly the first 256 boards of the large pool. The seeded hidden rotation Q and source support masks never enter the search objective.

The primary comparison trains read directions using either a priority order driven by the *lowest current per-read Gaussian zero mass* or the original lexicographic pair order. Both use 3 sweeps over 120 unordered pairs and 37 global angles, exactly **13,320 candidate-angle score evaluations** per source. Pair priorities are recomputed only once at the start of each sweep and require extra marginal-score computations that are recorded separately.

The secondary sample-volume comparison uses 2048 versus 256 training boards at identical angle budgets. The high-volume condition scores candidates over 8192 four-pass vectors, at **eight times the per-candidate sample cost**. Physical evaluation uses the original PD-12 BPDModel/TiedComponentLinear wrapper with exact binary read gates on eight seen/withheld support families.

A planted-Q positive control is explicitly privileged. The primary preregistration requires at least **18/32** priority2048 recoveries, **eight net paired priority-only seed wins** over lex2048 and exact two-sided McNemar p≤0.05. Every model seed and failure is retained.

[Excluded-seed engineering tests](test_experiment.py) use seed 4001 to check the unchanged correlated support source, nested sample pool, label-free priority calculation, budget equality, basis orthogonality and exact physical binary interventions. The full 160-arm cohort runs only after these pass and a one-time execution lock fixes the model source SHA. The original scientific receipt is archived regardless of PASS or FAIL.

The study is a linear, orthogonally rotated sparse-latent toy and cannot by itself establish identification in a trained recurrent transformer.
