# BPD v3: dither-floor and exact-basis source-only gym

This directory implements the separately frozen **FLOOR-01 / BASIS-01** protocol in [REGISTRATION.md](REGISTRATION.md). The two checkpoint experiments remain **PENDING**. The Python module is NumPy-only, makes no network calls, and cannot load model weights. It does not inspect the protected ARG-03-N test cohort.

## Reproduce the synthetic CPU controls

From the BPD repository root:

```bash
python -m pip install numpy pytest
python mechinterp/bpd/v3/floor_and_basis.py --smoke --out runs/bpd_v3/synthetic_smoke.json
# In the standard BPD neural environment (which also has torch):
python -m pytest -q mechinterp/bpd/tests/test_v3_floor_and_basis.py mechinterp/bpd/tests/test_v3_uncertainty.py
```

The source-only GitHub Actions workflow runs the eleven tests in an isolated loader because BPD's existing package initializer imports torch. The deterministic smoke exercises 63-step bistability with K=8 bounded dithers, readouts at 20/41/62, an independently sampled second ensemble, six exact matrix bases, 3% and other off-contribution budgets, and a phase-disjoint hyperplane-mass witness. Its data are **synthetic**, including the estimated JS floor. The 11 unit tests cover the KL identity, a single-reference counterexample, a PSD output seminorm and the full-on and partial-off certificates.

## Load externally recorded TRM readouts

Create a local NPZ whose `target_logits` are shaped `[N_boards,K_dithers,N_readouts,N_cells,N_classes]`, in the same realization order at each readout. The intended TRM shape is `[183,8,3,81,9]` for the complete original WB-Stall cohort and passes 20, 41, 62. Optional arrays:

- `static_logits`: `[B,R,P,V]`, strictly independent of the realized target dither.
- `driven_logits`: `[B,K,R,P,V]`, matched realization by realization; the returned difference `J_delta - KL` is a **conditional-information bound only when the Markov/drive assumptions hold**. A zero KL against one selected reference trajectory is outside that setting.
- `independent_logits`: `[B,K,R,P,V]`, a separately dithered surrogate ensemble. Reported `ensemble_mean_kl` evaluates KL between the two empirical mean distributions.

```bash
python mechinterp/bpd/v3/floor_and_basis.py \
  --target-npz /local/gaoxin492_dither_logits.npz \
  --out runs/bpd_v3/gaoxin492_floor.json
```

Log the checkpoint file SHA-256, tensor SHA, independent board IDs, seed schedule, arithmetic mode, noise amplitudes and placements, call phase, readout and device details in a sidecar manifest. Reproduce the native target under the same implementation and arithmetic before sampling. The empirical K-rollout `J_delta` estimate uses its own sampled barycentre and is **downward biased** at finite K. Treat K=8 as a pilot; report uncertainty over independent boards and a K-sensitivity check on ambiguous boards before using numerical leakage inequalities.

The saved B4 v2 archive has factors `(V,U)` but lacks the causal-importance network. It therefore does not provide the gates required for a valid paired replay of the v2 `0.010` KL receipt.

## Load module vectors for the zero-fit basis comparison

Create another local NPZ with `W` shaped `[d_out,d_in]` and independent clean/dither-pooled module inputs `inputs` shaped `[N,d_in]` for one selected matrix. Run:

```bash
python mechinterp/bpd/v3/score_exported_basis.py \
  --matrix-npz /local/down_proj_0_inputs.npz \
  --chunk-size 256 \
  --out runs/bpd_v3/down_proj_0_basis.json
```

Each basis is an **exact** decomposition `(V U)^T=W`. The current gate reads its own input, switches off an ascending contribution prefix, and certifies all partial ablations of those switched-off components under the specified output seminorm. The three budgets are fractions `{0.01,0.03,0.10}` of the true module output norm.

The previous `floor_and_basis.py` matrix scorer remains as an independent small-instance oracle. The streaming scorer verifies matrix-file SHA-256 when supplied, uses one full SVD per module, checks the period-7 phase tags, and reports near-null write directions separately. For a full-row-rank `down_proj` of shape 512×1536, the input-complete right-SVD contains 1,024 directions in the numerical kernel of W, which must be disclosed when comparing active counts. These are structural null writes rather than learned sparse features.

For `down_proj`, the **input-complete** neuron basis uses `C=d_in` (typically 1,536), whereas the **output-complete** SVD uses `C=d_out` (typically 512). Compare active counts within equal C first and report both component counts for cross-form comparisons. `mean_logit_kl` in this source-only module is a softmax diagnostic of the **module output vector**. It is not the native digit-head or behavioral KL through the recursive model. Delayed Gramian norm arguments are local linear certificates; nonlinear transport needs a curvature remainder.

**Status:** CPU theorem/instrument checks PASS. Native dither floor, real module-vector sparsity, basis learning and held-out Sudoku causal improvements PENDING.
