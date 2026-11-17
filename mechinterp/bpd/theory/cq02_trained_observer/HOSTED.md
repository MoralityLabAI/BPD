# BPD-CQ-02 native BPDModel result (hosted CPU evidence)

GitHub Actions run: https://github.com/MoralityLabAI/BPD/actions/runs/44098525983
Workflow artifact: https://github.com/MoralityLabAI/BPD/actions/runs/44098525983/artifacts/14335515944

Native source: actual repository `BPDModel` / `TiedComponentLinear` with shared physical V,U matrices, exact hard top-one gates and zero residual mask on all four visits. All shared factors and the gate were trained with the same locked hyperparameters as the independent BPD-style pilot. Output parity is checked on separately sampled calibration vectors after training and each native model uses the official component execution in its training forward graph.

The native job completed successfully on Python 3.12.16 / PyTorch 2.5.1+cpu. Both 8-seed cohorts have 8/8 complete 128/128 phase routing, 512/512 native donor scalar transfer, 512/512 projected-native fiber transfer, and 54/512 randomly oriented per-case norm-matched fiber transfer. Sham and instantaneous-visible controls each produced 0/512 donor-scalar transfers. In every native seed, masked-execution versus explicit V,U component-output parity maximum absolute difference is exactly zero, the current component observability rank is 1, and the rank after the specified one-step transport is 2. The independent and native cohorts agree on the transfer counts. Their reconstruction MSEs differ only at tiny floating-point levels.

These outcomes establish that the instrument works with trained native BPD factors. The synthetic channel swap T is constructed to reveal a direction that the chosen readout hides, and the donor-scalar target is derived from the same component write. They therefore do not establish a new BPD causal mechanism in a naturally trained recurrent network, nor a claim on the protected ARG-03-N held-out split.

Hosted raw receipts (not model weights) are committed under `results/cq02_native_hosted.json` and `results/cq02_independent_hosted.json`. The preregistered protocol commit `957e23f` predates the cohort execution, and the original local independent reference is unchanged.
