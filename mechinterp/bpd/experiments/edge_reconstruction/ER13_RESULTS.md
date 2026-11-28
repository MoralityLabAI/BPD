# ER-13: trained nonlinear recurrent component-fault reconstruction

Reproduce locally with PyTorch 2.10.0+cpu, NumPy 2.3.5:
```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python neural_bridge.py > neural_bridge.json
```
Native parity gate (run from repository root, unexecuted locally because the repository cannot be cloned in the offline container):
```sh
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python mechinterp/bpd/experiments/edge_reconstruction/neural_bridge.py --native
```
Native path uses TiedComponentLinear in target mode with zero ablation and sphere_rescale=False, SVD init, reset_pass_state at each rollout, and explicit ComponentIntervention at passes 1 and 2 (zero-indexed). It asserts exact and additive response-bank parity within 1e-6 against reference equations. Native parity pending. No trained BPD factor decomposition or real LLM ablation claim.

Eight independent model roots. Each trains a 72-weight tied tanh recurrence/readout on outputs of a randomly initialized teacher for 350 Adam updates, batch 128. Four passes, six hidden dimensions; median test MSE 0.00010699345. Candidate intervention pairs: one SVD component removed at pass 1 and one at pass 2, 36 possible pairs. 16 separate probe inputs used to set Gaussian noise sigma = 0.1 RMS joint intervention effect, with floor 1e-5. 32 new probe/readout pairs for evaluation. For each model 24 hidden pair episodes, uniform draw with replacement, 12 randomly ordered distinct probes with common noise across arms. The exact bank provides ground-truth simulated fault response. The additive bank sums two single-component effects around clean output. This isolates predictive approximation error under matched observations; no learned intervention scheduler compared.

| probes | exact MAP | additive MAP | exact mean truth NLL (bits) | additive mean truth NLL (bits) |
|---|---:|---:|---:|---:|
|4|153/192|122/192|0.6083|15.9568|
|8|173/192|147/192|0.3423|18.5724|
|12|184/192|158/192|0.1811|21.6389|

Exact vs additive final hits by independent model root: 21/18, 24/19, 24/22, 24/19, 24/23, 22/18, 24/22, 21/17 (out of 24). Episodes within a model are correlated; eight independent model roots, not 192 independent models. Median additive prediction RMSE 0.01252054; median sigma 0.00389637; root ratios 1.855..4.262. NLL uses a numerical floor of 1e-300. No preregistration, timing or scaling claim. Known model weights/factors, known candidate fault family and true noise scale, no recovery of unknown full network. Exact bank construction is more expensive than additive bank construction; compare at matched rollout cost in follow-up. 

Next: execute native parity on repo checkout; compare selective pairwise interaction caches, rank and sparsity controls, development-only selection, held-out component pairs, pass-mean ablation and actual learned BPD factors; include runtime and peak RSS. For the interaction residual use R_ij(x)=f_ij(x)-f_i(x)-f_j(x)+f_0(x), then quantify recovery relative to additive and exact oracle arms under a matched budget.
