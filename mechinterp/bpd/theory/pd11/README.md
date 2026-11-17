# BPD-PD-11: global-versus-local pairwise angular optimization

The [prospective protocol](PROTOCOL.md) and [machine-readable registration](CONFIG.json) were committed on November 18, 2026, before any registered PD-11 training.

PD-09/PD-10 found that smooth near-zero mass favors a mixed read pair at a broad initial bandwidth and retains a near-45-degree local optimum even after narrowing the bandwidth. PD-11 changes the optimization geometry: it explicitly evaluates a grid of angles when rotating pairs of read directions, allowing candidate moves across a local objective barrier.

[experiment.py](experiment.py) compares two unlabelled arms over 24 fresh seed–rotation pairs: global `[-90°,90°]` Givens searches and local `[-5°,5°]` searches. Both receive identical fixed observed training samples, identical random orthogonal starting factors, the same six ordered pairs and six passes over pairs, and exactly **2,628 candidate-angle score evaluations per seed**. They optimize Gaussian near-zero mass at sigma 0.15.

Two additional arms deliberately initialize a source-aligned read basis with a 44-degree mixed pair using true Q, providing separately labeled privileged stress controls. A random orthogonal negative control performs zero updates, and a count-supervised PD-06 positive control measures pipeline solvability.

Evaluation uses the real shared BPDModel/TiedComponentLinear identity target and fixed binary read threshold 0.5 across six original PD-06 support families, including withheld pairs/triples. The preregistered primary gate requires at least 18/24 full six-family recoveries, six net paired wins over local-angle search and exact two-sided McNemar p<=0.05.

[Excluded-seed tests](test_experiment.py) use seed 2029 to verify label isolation, deterministic angle grids, pair-score monotonicity, orthogonal factors, source splits and exact physical execution. A source-pinned one-time `EXECUTE_LOCK.json` releases the registered 24×6 cohort only after the tests pass. Full first-run evidence is archived even if the scientific hypothesis fails.

This is a four-dimensional synthetic source-specific optimization test. The nonlocal angle grid does not establish efficient learning in arbitrary high dimensions or mechanism identification in a 7M recurrent transformer.
