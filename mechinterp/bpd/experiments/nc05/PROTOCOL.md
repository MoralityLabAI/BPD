# BPD-NC-05: prospective rank–dose frontier in a dense learned recurrence

**Source-only registration, 2026-11-18.** Commit this document on a new NC-05 branch before writing experiment code or generating NC-05 model/data. This follows the scientific fidelity failure of [NC-04](../nc04/RESULTS.md): its rank-two projector beat matched physical shams by 0.524 nats of external-label NLL, while increasing clean-model NLL by 0.1234 nats, above the frozen 0.05-nat fidelity ceiling. NC-04 roots 400–407 are development context, **not NC-05 results**.

## Question and fixed comparison family

Does increasing retained projector rank recover the fidelity of a **freely trained, dense four-pass recurrent model**, and how much retained write gain is required? Hold training distribution/model family and discovery method fixed and vary only:

- Ranks **r ∈ {2,4,6,8}** (the first r eigenvectors of *uncentered final-model-logit gradient covariance* at each pass).
- Physical write gains **α ∈ {0.75,1.0}**. At each pass, the retained physical write is `α a_t P_{r,t} Wᵀ` (using the repository's `V P U` convention). The removed complementary write is suppressed, with no residual bypass.
- **Eight fixed cells**, a single selection/comparison family. Primary final endpoint is absolute true-label NLL cost relative to the clean model, with sham-adjusted true-label NLL as the causal discrimination check. Report all cells even when they fail, including ranks or gains with negative effects.

A root-independent engineering rank–dose frontier is the **smallest r with α=1.0** satisfying the joint gate below. All other rank/gain cells are fully reported with the same multiplicity family and may be called descriptive engineering points; no post hoc reranking or rescue.

## Independent roots and sample separation

- New **engineering roots 500–507**. Each independently initializes and trains a free/dense `DenseLooper` from NC-04 on **2,048** independent N(0,1) inputs (split 1) for exactly **384** Adam updates, batch 128, learning rate 0.005, unmodified physical shared `mix` 20×20 weight reused over four nonlinear passes.
- The external target is NC-04's fixed context-selected sign task. Hold out **1,024 independent discovery inputs** (split 2) and **1,024 independent evaluation inputs** (split 3) per root. The task oracle may only be used in training and held-out scoring, not discovery, projector rank selection, sham admission or physical dose matching. All generators are root/split seeded as in NC-04.
- Task admission: ≥85% clean accuracy on *each* context and physical `mix` density ≥90% of entries with absolute magnitude ≥1e-4. Preserve failed roots in raw receipts. Training roots, not their prompts/passes/dose arms, are independent units.
- A one-root engineering smoke uses seed 500, same training steps and smaller **256 discovery, 256 evaluation** cases, **128 proposals**. It cannot be treated as independent scientific evidence.

## Discovery and exact physical parity

- Freeze the trained network before discovery. Score = final model logit, with detached model-prediction sign to compute positive co-attributions. For the primary projector, the uncentered gradient covariance is exactly invariant to arbitrary per-example ±1 re-signing, hence **external labels provide no projector information**.
- Use the existing native `BPDModel/TiedComponentLinear` wrapper for all physical interventions. In native basis `V=I,U=Wᵀ`; in rotated basis `V=Q,U=QᵀWᵀ` for a root-seeded Haar orthogonal `Q`. The same physical model and inputs are used in both; zero gradient/sham lookups on evaluated labels.
- Compute passwise second moments `C_t=E[g_t g_tᵀ]` in float64, top-r eigenvectors and exact orthogonal projectors. **Report** λ_r, λ_{r+1}, their ratios, projector idempotence/symmetry and relative downstream physical operator discrepancy at each rank. A per-rank projector is numerically identifiable only if λ_r >1e-12 and λ_r/max(λ_{r+1},1e-12) ≥1.001. A failure marks that rank `UNIDENTIFIABLE`; preserve per-rank diagnostic and do not choose a replacement rank. No observed NC-05 outcome may change these thresholds.
- Full-ones native and rotated BPD parity before causal interpretation: at all 4 readouts max logit difference ≤5e-5, KL(clean||full) ≤1e-7 nats computed in float64, and physical residual relative norm ≤1e-4. For each rank, physical operator `V P U` relative difference across bases ≤1e-4, and retained-model logit differences ≤5e-5. The BPD `override_read` counterfactual must use all 20 factor coordinates and all four recurrent passes.

## Rank-specific matched shams and gain

- Fix one rank-r **model-blind physical sham population** before evaluation. For each of 1,024 proposals, form a candidate rank-r projector with principal angle θ picked uniformly from **{0.45,0.65,0.85,1.05} radians** relative to the discovered rank-r subspace. At every pass, generate a fresh Gaussian r-dimensional orthogonal complement and orthonormalize by QR against the discovered eigenspace. Candidate `P_sham,t = (cosθ E_t + sinθ F_t)(cosθ E_t + sinθ F_t)ᵀ`. This is a deliberately *local* difficult comparator, rather than NC-04's activation-PCA Haar null.
- Candidate RNG is deterministic from (root, rank); generate all proposals in declared order. Select the **first eight distinct proposals** with retained **discovery clean-context physical write RMS within ±15%** of the discovered rank-r projector. Matching uses the complete native physical `mix` write, not factor coordinate norm. Because gain α multiplies both candidate and discovered write equally, the **same eight physical shams** serve both gains, with gains applied identically. If <8 candidates qualify, mark that rank and both gains `BLOCKED_SHAMS` and record eligible proposal counts; no replacement controls chosen by held-out outcomes.
- Physical write RMS controls retained energy, **not downstream influence or Jacobian**, and the local sham mixture makes the null qualitatively different from NC-04's global Haar comparison. Record selected angles, relative dose errors, each sham's true-label NLL, clean-to-arm KL, task accuracy and realized closed-loop write displacement. Preserve every candidate admission count and its root/rank derivation.
- Zero-head negative control on a discovery-only batch: zero all learned output-head weights/bias; the gradient must vanish and all projector ranks must reject as `UNIDENTIFIABLE`.

## Outcome and resolution gates

For each eligible root, rank and gain, score the full closed-loop target, selected retention, and eight dose-matched sham retentions on exactly the same untouched evaluation cases.

- **Primary root-level fidelity cost:** `Δ_fid = NLL(keep_{r,α}, true_labels)-NLL(clean,true_labels)` (nats); a smaller value is better.
- **Causal discrimination:** `Δ_sham = median_8 NLL(sham_keep,true_labels)-NLL(keep_{r,α},true_labels)` (nats); larger positive value means better selection than matched alternatives.
- Secondary: clean-vs-kept KL at four readouts, task accuracy and correct-cell count, realized write displacement, clean model pass accuracy, absolute true-label NLL, eigenvalue ratios, graph diagnostic if present, and native-vs-rotated equivalence.
- Bootstrap **entire independent trained roots**, 4,096 resamples and deterministic RNG, and compute **99.375% two-sided familywise Bonferroni intervals** for each of the 8 cells for both Δ_fid and Δ_sham. Multiplicity is **8 rank–dose cells**, not 64 outcomes or the two basis representations. Results from the two bases are paired numerical checks, not independent model roots.
- **Joint engineering fidelity-positive gate for a fixed rank at α=1.0:** all 8 task/closure/dose/eigenspectrum/zero-head/rotation-qualified roots; **upper 99.375% root-bootstrap CI of Δ_fid ≤0.05 nats**; **lower 99.375% CI of Δ_sham >+0.02 nats**. Declare the smallest passing rank from the frozen ordered list only if one passes, or `NO_ELIGIBLE_RANK`. A reported pass is a synthetic engineering finding, never a confirmatory cross-family mechanistic result.
- CI must exit successfully for an eligible negative/null scientific result; use `NOT_RESOLVED_OR_FAILED` rather than rewriting any gate. Keep complete source SHA-256, protocol SHA-256, all 8×8 cells, sham distributions and failures.

## Scientific seal and limitations

NC-05 tests a **single learned dense recurrent synthetic task**; other model layers remain intact, and the projector lives at only the shared physical `mix` site. No learned SAE dictionary, natural neuron semantics, Lean theorem, general LLM mechanism or hidden reasoning pathway is established. Orthogonal BPD factor transformations are weaker than arbitrary nonlinear neuron-chart changes.

The NC-05 confirmation cohort remains **unregistered, untouched, and sealed**. A later release needs an independent EvalMath mechanistic-power preflight, a separately frozen ≥64-root population and full multiplicity accounting. A nonparametric 95th-percentile sham bound with 95% confidence requires ≥59 independent sham roots; eight engineering roots cannot calibrate it. The protected ARG-03-N and AMM-03 cohorts remain sealed throughout.
