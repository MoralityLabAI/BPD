# BPD-NC-09: cross-task recurrent subspace transport between independently trained models

**Prospective source-only registration, 2026-11-18.** Commit this protocol as the **first NC-09 commit**, before code, training, discovery inputs, or new observations from roots 900–907. NC-05/06/07/08 results are historical design inputs. This experiment concerns transfer between *distinctly trained weights* rather than NC-03's exact factor-gauge rotation, and is **engineering only**.

## Hypothesis and task relationship

Two independently initialized 20D four-pass nonlinear `DenseLooper` networks share an architecture and Gaussian five-dimensional input format but train on related targets:

- Task A: sign of `x0` when `x4>0`, and `x1` when `x4<=0`.
- Task B: sign of `x0` when `x4>0`, and `x2` when `x4<=0`.
- Hence the **positive-context branch shares a true input signal**, and the **nonpositive-context branch uses a different external signal**. This is a designed form of partial transfer, not an assertion that hidden dimensions align natively.
- Train **separate randomly initialized models** A and B, never share parameters, gradients, optimizers or checkpoints. Each root ID defines one **paired independently trained model family**. Different roots are the independent inference units, not the two models within a pair, their samples, the four passes, or eight shams.

Primary question: after aligning model A's pre-`mix` activation chart to model B's on **paired, unlabeled positive-context examples**, can an A-discovered rank-four projector retain B's *shared-branch* behavior nearly as well as B's own rank-four projector, with a larger transfer deficit on the task-specific branch?

## Training, roots and untouched splits

- New independent development pairs: **900–907**, using new training randomness distinct from NC-04..08. Each pair initializes A at `root` and B at `root+100000`. The source code must enforce these roots and never sample any confirmation population. A separate **engineering smoke root 899** is permitted and is excluded from every development aggregate.
- Each model trains 384 Adam steps, batch size 128, LR .005, on its own 2048 Gaussian inputs. A data stream uses `sample_x(root,1,2048)`, B uses `sample_x(root,11,2048)`; distinct splits, shared input law.
- The model is exactly NC-04's freely learned dense `DenseLooper` with frozen `encode,pre,mix,head` after training, four genuine shared-`mix` closed-loop passes. No planted stage support. Root-specific training initializations are disjoint.
- Separate paired input streams per root: **1024 gradient-discovery** examples (`split=2`), **1024 unlabeled chart-alignment** examples (`split=4`), and **1024 externally labeled evaluation** examples (`split=3`). Both models see the same discovery and alignment input vectors, but their training examples are distinct. Alignment uses only the `x4>0` cases from split 4, with at least 300 admitted cases, and uses no task labels or external held-out information.
- Task admission requires each of the **four model×context combinations** to achieve ≥85% clean true-label accuracy on independent evaluation, and both physical `mix` matrices ≥90% dense at absolute weight ≥1e-4. A failing pair stays visible in the receipt.

## Frozen discovery and cross-model alignment

- Discover per-pass `rank=4` top eigenspaces of the **uncentered second moment of the frozen model's final-logit score gradients with respect to its own physical pre-`mix` reads**, computed in float64 over 1024 discovery examples. The detached sign of each model's prediction may multiply gradients; outer products are sign-invariant. Both A and B use their own self-logits. **No external task labels** enter discovery or alignment. One eigengap threshold, λ4>1e-12 and λ4/max(λ5,1e-12) ≥1.001 at each pass; report all spectral diagnostics. Failure remains `UNIDENTIFIABLE`.
- Given clean pre-`mix` read activations `Z_A(t)`, `Z_B(t)` from the independently trained models on *paired positive-context unlabeled alignment inputs*, compute orthogonal Procrustes `R_t = argmin_{R^TR=I} ||Z_A(t) R-Z_B(t)||_F` from `svd(Z_A^T Z_B)=UΣV^T`, with `R=UV^T`. **Do not** learn any cross-model map from target labels, intervention scores, held-out data, gradients, or model outputs.
- A's rank-four projector `P_A(t)` transports into B's physical input chart as `P_A→B(t)=R_t^T P_A(t)R_t`. Also form **identity transport** `P_A` and **permuted-pair Procrustes transport**: fit `R_t^{perm}` after a fixed seeded permutation of the B alignment rows (same permutation across all passes), preserving source/target sample marginals. Compare all transports at equal rank and gain one.
- Report alignment residual `||Z_A R-Z_B||_F/||Z_B||_F` at each pass (fit on positive-context source only), Procrustes orthogonality, projection idempotence, and transported-vs-target-owned projector overlap `tr(P_A→B P_B)/4`. There is **no post hoc orthogonal-map threshold** beyond finite values and orthogonality ≤1e-5; a bad fit is an informative transfer failure.
- Four passes use independent `R_t` because representations can rotate by pass despite shared physical weights. This is an *alignment from paired data*, not an exact function-preserving BPD gauge symmetry.

## Native BPD interventions and physical shams

- Wrap model B's physical shared 20×20 `mix` with the actual `BPDModel/TiedComponentLinear`; verify full-ones target parity for all readouts: max abs logit ≤5e-5, float64 Bernoulli KL ≤1e-7 nats, exact factor residual relative ≤1e-4. Same for model A. With native basis `V=I,U=W^T`, apply `P_B`, `P_A→B`, `P_A` and `P_A→B^{perm}` using full four-pass `BPDModel.intervene(mode="override_read",all_components,scale=0,read_override=P)` at gain 1.0, with fresh closed-loop hidden states.
- **One matched physical sham population per target B root**: exactly 2048 Gaussian angle-local rank-four proposals around `P_A→B` with angles `{0.45,0.65,0.85,1.05}` radians; choose first eight with **discovery-only B prewrite physical retained-write RMS** within ±15% of transported P. Apply the same target B model and held-out inputs for all eight, with exact rank and gain matched. Selection cannot query evaluation data/labels; never enlarge proposal budget after seeing outcomes. Record qualifying counts, selected proposal indices, dose and actual held-out write displacement.
- Check exact *within-B factor-gauge invariance* using a separately seeded orthogonal reparameterization of B's **same** weight, `P'=Q^T P Q`, requiring <1e-4 relative physical operator error and <5e-5 maximum kept logits difference. This is a numerical control independent of cross-model transfer success.
- **Negative control**: setting model B's output head weight and bias to zero must make all source score gradients exactly zero and return `UNIDENTIFIABLE` on discovery. No alternate rank/dose or alignment methods are screened in this study.

## Held-out external-label endpoints and fixed decision

After the complete projector, alignment and sham allocation is frozen, compute labels on the untouched 1024 `split=3` evaluation inputs, and score B's true task NLL under clean, B-owned P, A→B aligned P, raw A P, permuted-pair P, eight aligned local shams, and A's own self-projector (external Task A fidelity positive control). Retain all passwise clean-versus-keep KL and external true-label correct counts. Determine the root's shared branch `S:x4>0` and task-specific branch `D:x4<=0`.

On each **independent paired-model root** define:
- **Primary transfer deficit** `d_shared = NLL_B,S(A→B) - NLL_B,S(B-owned)`. A 4D transported subspace is noninferior to B's own 4D subspace when the root-bootstrap **95% upper bound of mean d_shared is ≤0.05 nats**.
- **Branch-specificity** `d_specific - d_shared` where `d_specific=NLL_B,D(A→B) - NLL_B,D(B-owned)`. Predeclared expectation is larger deficit on disjoint-input task branch; the root-bootstrap **95% lower bound must exceed +0.01 nats**.
- **Alignment usefulness** `NLL_B(permuted-P)-NLL_B(aligned-P)` on the shared branch: bootstrap lower bound must exceed +0.01 nats. The unaligned A projector is a separate reported control, not used for cherry-picking the primary claim.
- **Matched sham specificity** `median_8 NLL_B,S(local_sham)-NLL_B,S(aligned-P)`: root-bootstrap lower bound >+0.02 nats.
- Root-bootstrap intervals: 4096 whole-root resamples, seed **909009**. There are **four declared directional criteria**, use Bonferroni **98.75% two-sided percentile intervals for each** to control the four-comparison family at 5%. For every criterion require 8/8 independent root admission; never treat within-root cases, task contexts or shams as independent roots.
- A registered `ENGINEERING_CROSS_TASK_TRANSFER` requires 8/8 task/closure/eigengap/sham/negative-head/gauge-admitted roots and all four confidence gates above. Otherwise preserve `ENGINEERING_NULL_OR_INELIGIBLE` and all original per-root outcomes. Any scientific failure leaves CI green and original thresholds untouched. We also report whether own source A projectors preserve A's task, as a necessary descriptive diagnostic; no retrospective exclusion based on those scores.

## EvalMath boundary and protected cohorts

The 8-root engineering population can reveal whether cross-model alignment is usable and estimate paired effect variance. It cannot calibrate the **95th percentile of the sham variation with 95% confidence**, which requires at least **59 independent sham roots** for a sample-maximum rule. Future calibration/confirmation roots are **unregistered and sealed**; any scientific confirmation first requires fresh populations, exact EvalMath `evalmath-mechinterp-power` preflight and retained original failed roots.

Never touch sealed ARG-03-N, AMM-03 or other protected cohorts. This study tests **cross-model synthetic-task projector transfer** across independently trained networks with paired inputs and a known overlapping input branch. It does not establish semantic shared neurons in pretrained language models, universal gauge equivalence between trained models, or learned SAE identifiable components.
