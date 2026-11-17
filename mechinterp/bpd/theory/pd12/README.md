# BPD-PD-12: dimensional, support-correlation and amplitude-margin scaling

[CONFIG.json](CONFIG.json) and the [prospective protocol](PROTOCOL.md) were committed before any PD-12 model data. The study extends [PD-11's 24/24 four-dimensional global Givens result](../pd11/RESULTS_20261118.md) to a fixed factorial over 8 and 16 dimensions, uniform versus clustered latent support distributions, and strong versus weak active-magnitude margins.

Each of eight cells has eight fresh model seeds with an independent hidden orthogonal rotation. A held-out set of antipodal pairs, stride-two triples and stride-two quartets is excluded from fitting and tested alongside seen supports and an unseen-support four-pass transition.

[experiment.py](experiment.py) compares the unlabelled global full-angle search with a local-angle control at exactly matched candidate budgets, plus a lower-budget global search and an explicitly privileged planted-Q oracle. All learned factors execute through the real shared BPDModel/TiedComponentLinear module with exact binary read-threshold masks. A deterministic optimal read assignment uses the SciPy Hungarian algorithm to scale factor comparisons to 16 dimensions.

The two primary arms evaluate 3,108 candidate pair-angle scores per 8D seed and 13,320 per 16D seed. Global13 uses 35.1% of that candidate budget. Both optimizer costs and recovery accuracy are recorded. Candidate evaluations are distinguished from sample-weighted score operations and CPU wall-clock time.

[Excluded-seed engineering tests](test_experiment.py) use only seed 3001 to check source support banks, correlation frequency, active amplitude floors, exact physical intervention, one global factor alignment, source-label isolation, orthogonality and budget parity.

The preregistered scientific gate requires 40/64 global successes, with dimension-specific floors of 24/32 in 8D and 12/32 in 16D, at least 16 net paired global wins versus local and exact two-sided McNemar p≤0.05. The original complete 256-arm result will be archived even when this gate fails. The study concerns a synthetic orthogonally rotated sparse source with an identity physical teacher, not identification in a trained recurrent transformer.
