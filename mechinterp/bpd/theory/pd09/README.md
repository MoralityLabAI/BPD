# BPD-PD-09: unsupervised smooth hyperplane mass and orthogonal basis discovery

[CONFIG.json](CONFIG.json) and [PROTOCOL.md](PROTOCOL.md) were committed on 17 November 2026 before any PD-09 training. The experiment uses 24 fresh seed–rotation pairs under the exact four-dimensional PD-06 sparse latent source and its six held-out support families.

[experiment.py](experiment.py) compares five arms: an unlabelled smooth expected-L0 surrogate with differentiable orthogonal QR factors and an annealed bandwidth, an identical fixed-bandwidth QR control, an untied physical BPD read/write sparse-penalty control, an untrained random orthogonal negative control, and the original PD-06 true-cardinality-supervised positive control. Only the last arm accesses latent cardinality labels. Orthogonal QR arms train a parameterized rotation A with physical V=QR(A), U=V.T; this is a **restricted factor family**, distinct from independent untied physical BPD factors.

All arms are scored with the same fixed read-magnitude gate at threshold 0.5 and true BPDModel/TiedComponentLinear rank-one physical output interventions. No gate is refitted after dictionary training. Evaluation requires read-axis alignment, low MSE and exact four-pass source-support masks across six seen and withheld families.

[test_experiment.py](test_experiment.py) uses excluded seed 2009 to verify QR gradients, the exact physical weight from tied factors, source-only unlabeled objectives, independent train/holdout support sets, and physical binary-mask execution. The dedicated source-locked full study runs 24×5 arms only after these tests pass. All original PD01–PD08 results remain intact, including [canonical PD08's 23/24 conditional success](../pd08/RESULTS_20261117.md).

**Registered primary gate:** annealed QR source-only learning must recover at least 18/24 seeds, win by at least six net paired seeds over fixed-bandwidth QR, and reach exact two-sided McNemar p≤0.05. Preserve every outcome, including scientific failure, in the first-run full JSON.
