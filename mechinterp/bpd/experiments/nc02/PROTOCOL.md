# BPD-NC-02: planted recurrent paths and causal-community recovery

**Prospective source registration, 18 November 2026.** Commit this protocol before implementing NC-02 or generating its model/data. The previous NC-01 engineering observation is exploratory context only (BPD PR #66 / merge `8e4501d`).

## Scientific question

Can a Goodfire-style co-attribution community algorithm recover the **known pass-specific causal paths** of a trained, shared-weight nonlinear recurrence, and do recovered paths preserve the target behavior better than equally sized, discovery-dose-matched sham paths?

NC-01 found communities in all three full-rank coordinate systems on one student root but had mixed, small causal advantages. NC-02 is a dedicated **instrument/positive-control benchmark**; planted routes make recovery verifiable, at the expense of generality. NC-02 gives no evidence of native language-model neuron communities by itself.

## Task and frozen target (determined from exogenous input)

* Inputs: four independent N(0,1) lane signals and one independent N(0,1) context signal. Context sign selects **lane 0** or **lane 1**; target class is the sign of the corresponding input lane. Two unused lanes (2/3) are active distractors.
* A shared `20x20 nn.Linear` has one admissible shift edge per lane and stage (five stages of four lanes). The same physical weight is executed **four times**, each followed by `tanh`. All 16 stage-shift weights start positive; stage-wise gains are projected to equality across lanes after each optimizer update, preserving a physically matched decoy dose.
* The model's readout is a trained, context-conditioned signed logit on the active output stage, with one learned scale per routed lane. BCE against the *external* input-defined class trains the shared stage gains and readout. Training is confined to a disjoint stream; no oracle selectors are used at inference.
* Exact native-coordinate planted path for lane `l` (0 or 1): `(pass t, input-read coordinate 4*t + l)` for t in 0,1,2,3. Both paths' **union** is the reference truth. This oracle is used for retrospective scoring only; discovering groups and allocating primary shams cannot consult it.
* The four-pass problem is architectural: only the currently active stage is carried by the recurrence; later-stage values arise after repeated shared-matrix application. This is a synthetic shift-register, not a trained general-purpose transformer.

## Discovery and intervention

Use NC-01's fixed discovery constants independently of outcomes: signed final-logit margin attribution `z * d(s)/dz`, positive-only top 8 per sample, >=12 co-selections, phi >=0.15, mutual top-4 neighbors, connected components with 2–12 nodes, rank by total phi mass, examine top 4 components. Nodes are `(pass, factor)`. For primary native-basis recovery report Jaccard, precision, recall and exact-match rate against the two four-node planted paths (maximize bipartite matching between discovered groups and planted paths, without using the oracle during discovery).

Arms:
1. **native**: `V=I`, `U=W^T`.
2. **bpd_svd**: full-rank exact BPD SVD initialization; a non-native factor-basis control, not a learned BPD dictionary.
3. **rotated**: seeded orthogonal factor reparameterization.

All arms use identical trained physical weights and held-out inputs. Deny outcome interpretation unless target vs full-ones BPD reconstruction passes all 4 readout steps (max abs logits <=5e-5; max KL <=1e-7 nats; W residual relative norm <=1e-4). Compute near-zero KL in float64. Ablation and keep-only interventions must be native `BPDModel.masked_components`, physical shared linear, complete 4-pass closed-loop, factor residual mask zero.

For each detected group, collect:
* **Necessity**: teacher-label NLL increase under removing the group across its discovered passes.
* **Selective sufficiency**: KL(clean || keep-only) at final and each intermediate readout. NLL and teacher-label accuracy also reported.
* **Dose**: discovery clean-context RMS physical write removed/retained, as well as realized closed-loop write delta RMS on held-out inputs.
* **Path recovery**: only native arm has a comparable planted-coordinate truth set.
* **Null graph**: independently shuffle per-pass co-selection rows (same per-pass marginals) on discovery examples, deterministically seeded, and report the number of detected cross-pass components across 32 shuffles.

## Controls and selection budget

* **Primary shams, oracle-free:** eight unique discovery-RMS-matched sham node groups per detected community, each preserving the selected count per pass. Native candidate pools are all four nonzero stage input reads at that pass; rotated and SVD candidate pools are all 20 read coordinates. 2048 fixed proposal draws with PRNG initialized from (root, basis, rank), sort qualified candidates by absolute dose error, select first eight. A valid sham requires RMS relative mismatch <=15%, defined from discovery only, and candidate distinct from the discovered group. Report candidate count. Deny the comparative causal claim if fewer than eight qualify. This setup controls physical write energy and per-pass count; it does **not** equalize effective downstream Jacobians or causal strength by construction.
* **Planted-decoy controls:** an additional oracle-tagged set of eight distinct lane-2/3 stage-local groups, selected by discovery RMS only from the 16 possible combinations. These are positive-control checks and are not used for the primary model-blind comparison. Keep outcomes separate.
* **Known-site zero-positive control:** set both readout scales to 0 on a copy of the trained model; the attribution graph should find no positive cross-pass communities.
* **Counterfactual direction:** ablation NLL should increase for recovered causal routes, and keep-only KL should be lower for recovered routes than primary matched shams. Incoherent random shams may themselves ablate at least one true node, making the **necessity-vs-sham** comparison conservative. Primary causal contrast is final keep-only KL benefit against median matched sham.

## Units, data and analysis

* Independent root = fresh initialized and trained weight-tied model. Within-root examples, context arms, four recurrent passes, shams and community ranks are correlated measurements.
* Registered engineering seeds: **200–207**; 2048 training samples, 1024 discovery, 1024 evaluation per root. Separate deterministic PRNG streams. One-seed CI smoke uses seed 200, 128 training steps, 256 discovery and 256 evaluation examples, for engineering only.
* Root effects calculated from the two best discovery-ranked communities (before oracle matching); report the primary highest-ranked group separately. Oracle path F1 is a separate validation metric.
* Development label: `ENGINEERING` even if all planted routes recovered. Report paired root-level differences, no p-value from within-root examples.
* **Confirmation sealed**: no confirmation seeds specified/generated by this source. A later registration must fix a completely new cohort of >=64 independent roots, multiplicity correction across arms/paths, and an EvalMath preflight based on independent engineering-root effects and matched-sham variance. If the measurable effect falls below the 0.02-nat floor or roots fail sham matching, retain the negative development result and design a separately registered test.

## Predeclared engineering checks

1. All physical weights outside the 16 planted edges remain **exactly zero**; within a stage, four lanes have identical positive gains after training. All other trainable readout parameters are tracked in the receipt.
2. Teacher-label accuracy must exceed 95% in both contexts, or mark the root `FAILED_TASK`.
3. Native graph recovers both planted four-node paths exactly, with no oracle supplied to discovery (engineering positive-control target).
4. The known-site zero-readout negative control yields **zero** recovered cross-pass communities.
5. Every scored arm has complete four-pass parity and eight dose-eligible primary shams. Retain all failed controls in the JSON.
6. Run root-level variance calibration on 8 engineering roots; assign `NOT_RESOLVED` if 95% root bootstrap lower confidence for the primary causal contrast fails to exceed zero.

Scientific restrictions: follow repository `AGENTS.md` and the EvalMath mechinterp-power skill. Do not touch the sealed ARG-03-N heldout or AMM-03 confirmation data. No pretrained weights are downloaded. Persist full raw receipt and exact source SHA-256. This protocol precedes implementation and cannot be retrospectively rewritten without an explicit amendment.
