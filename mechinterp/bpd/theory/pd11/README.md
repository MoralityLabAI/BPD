# BPD-PD-11: global-versus-local pairwise angular optimization

The [prospective protocol](PROTOCOL.md) and [machine-readable registration](CONFIG.json) were committed on November 18, 2026, before any registered PD-11 training.

PD-09/PD-10 found that smooth near-zero mass favors a mixed read pair at a broad initial bandwidth and retains a near-45-degree local optimum even after narrowing the bandwidth. PD-11 changes the optimization geometry: it explicitly evaluates a grid of angles when rotating pairs of read directions, allowing candidate moves across a local objective barrier.

[experiment.py](experiment.py) compares two unlabelled arms over 24 fresh seed–rotation pairs: global `[-90°,90°]` Givens searches and local `[-5°,5°]` searches. Both receive identical fixed observed training samples, identical random orthogonal starting factors, the same six ordered pairs and six passes over pairs, and exactly **2,628 candidate-angle score evaluations per seed**. They optimize Gaussian near-zero mass at sigma 0.15.

Two additional arms deliberately initialize a source-aligned read basis with a 44-degree mixed pair using true Q, providing separately labeled privileged stress controls. A random orthogonal negative control performs zero updates, and a count-supervised PD-06 positive control measures pipeline solvability.

Evaluation uses the real shared BPDModel/TiedComponentLinear identity target and fixed binary read threshold 0.5 across six original PD-06 support families, including withheld pairs/triples. The preregistered primary gate requires at least 18/24 full six-family recoveries, six net paired wins over local-angle search and exact two-sided McNemar p<=0.05.

[Excluded-seed tests](test_experiment.py) use seed 2029 to verify label isolation, deterministic angle grids, pair-score monotonicity, orthogonal factors, source splits and exact physical execution. A source-pinned one-time `EXECUTE_LOCK.json` releases the registered 24×6 cohort only after the tests pass. Full first-run evidence is archived even if the scientific hypothesis fails.

This is a four-dimensional synthetic source-specific optimization test. The nonlocal angle grid does not establish efficient learning in arbitrary high dimensions or mechanism identification in a 7M recurrent transformer.


## Registered first-run outcome

[RESULTS_20261118.md](RESULTS_20261118.md) preserves the original preregistered **PASS**: unlabelled global full-angle Givens search recovered 24/24 six-family mechanisms, while an equal-candidate-budget local-angle search recovered 1/24. The 23 global-only paired successes yielded exact two-sided McNemar p=2.384185791015625e-7, above every preregistered improvement requirement.

The [complete 144-arm first-run JSON artifact](https://github.com/MoralityLabAI/BPD/actions/runs/44130218981/artifacts/14350115809) contains 1,233,189 bytes (SHA256 4d2b7a75b2f917254c141a828253557f801552cf8f655c7ed0bd4323d91a5df3). [PER_SEED_20261118.csv](PER_SEED_20261118.csv) and [independent replay](test_published_results.py) preserve all 24 paired seed outcomes, factor-alignment cosines, objective gains, exact withheld-support scores and control success flags.

The global arm achieved 100% exact component masks across every seen and unseen support family, with zero false-on and zero false-off activations. Privileged 44° source-aligned stress initialization also recovered 24/24 with global-angle search, while local-angle stress recovered 0/24. Those privileged controls are separate from the unlabelled primary comparison.

The synthetic orthogonal 4D source and exhaustive angular candidate grid are substantial design assumptions; this result does not establish recovery of causal mechanisms in a trained recurrent transformer.
