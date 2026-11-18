# BPD-SF05B Stage 2-A: source-aware model-level donor admission, excluded roots only

**Prospective model-scoring protocol, 2026-11-18.** Parent: [issue #58](https://github.com/MoralityLabAI/BPD/issues/58). The source-only [Stage 2 donor generator](../sf05b_stage2_donors/PROTOCOL_SOURCE_ONLY.md) and [Bayes audit](../sf05b_stage2_donors/THEORY.md) were merged in [PR #59](https://github.com/MoralityLabAI/BPD/pull/59). The previously executed [Stage 1 teachers/policy](../sf05b_stage1/RESULTS_STAGE1.md) gave only 18 and 13 eligible donor pairs per 512 heldout records. This experiment tests whether the source-aware generator improves **model-level eligibility**, separately from public safety. Stage 2-A is a **development-only, excluded-root** pilot. Any new scientific root, new source-family training or calibration/confirmation is out of scope.

## Immutable upstream inputs

Use original \`sf05b_source_families\`, \`sf05b_stage2_donors\`, \`sf05b_stage1.teacher\`, \`sf05b_stage1.policy\`, \`sf05b_stage1.evaluate._score_arm/_shams\` and privileged finite-repair helper exactly as merged. Train only excluded roots **901,902**, with Stage 1 frozen schedules: 2600 teacher steps, 450 full-rank BPD V/U reconstruction steps and 240 oracle-label-supervised nonprivileged-policy steps, batch 128. All training labels come from \`F_train\` (2–4 writes). The source-aware model assay uses the structurally heldout \`F_heldout\` (5–6 writes). No parameter updates, intervention selection or posttraining tuning are allowed based on these new cohorts.

## Locked source cohort inputs, generated before any neural scoring

Per excluded root, draw **three disjoint cohorts**, each with 512 retained records:

- \`authorized_unconditional\`: same-query source-aware donor pairs; one last-valid-write source tag changes; common authorized query role is independently drawn from old/new source IDs; no source-level target threshold. Record original pair denominator 512.
- \`authorized_high_contrast\`: same construction, with source **oracle** target gap ≥0.30 chosen as the first 512 qualifying among **2048** independent candidates. Record attempted, qualified, selected and rejected counts. Do not select on teacher or BPD output.
- \`public_U\`: independent 512-record public-role cohort whose recipient and donor have exactly the same public oracle target. Score all 512 records without filtering by model competence, future answer change or authorized eligibility.

Seed each cohort deterministically from root and non-overlapping offsets \`530000,540000,550000\`, using \`root*430000+offset\`. Preserve sequence length, source identity metadata and the recipient/donor exact conditional Bayes means.

## Native trained model and eligibility

Run the **same trained physical teacher** on recipient and changed-last-source donor events, with all earlier writes identical. The model's current prequery readout is \`q(h)=h_0\`. Define the prequery-state edit from \`(phi_d-phi_r) W_hat\`, where \`W_hat=VU\` is the frozen full-rank 32-component physical reconstruction of the shared teacher matrix. Project only the first state coordinate to zero for the intervention, asserting exact current readout equality. Include rank-24 output physical SVD, full teacher-C write, exact hidden donor-state swap and unchanged recipient controls. Record GL(32) gauge invariance of the **full physical operator**.

Report the following **separate, sequential** authorized eligibility counts against 512 retained records:

1. exact source-oracle label changes, with full exact label-gap distribution (mean and fraction ≥0.20 and ≥0.30).
2. both clean recipient/donor model outputs within ±0.10 of their *own independent* exact source-oracle labels.
3. prequery scalar readouts differ by ≤0.05 (the original Stage 1 boundary).
4. the clean model's future donor-recipient output difference is ≥0.20.
5. full intersection of the three model-level filters (primary **eligible** denominator), retaining all exclusions and 0-eligibility as reportable failures.

These are **development diagnostics**, not a new inferential success gate. Only the exact 512 generated records per cohort and the independent 901/902 teacher roots count for population descriptors. Any eligible-count gain relative to Stage 1 (18/512 and 13/512 on different heldout draws) must be described as a **cross-cohort comparison**, not a randomized causal or independent teacher replication.

## Independent public role U and correction controls

The public cohort's sole diagnostic public outcome is the final model role-U query, on every generated record. Score its clean recipient and clean donor public-target MAE, and each intervention's absolute public-output change relative to the same recipient public forward pass. Candidate interventions: physical BPD full write, gauge-stable rank-24 output-state SVD, teacher-C full physical and exact hidden donor delta, plus zero-edit. Apply the **frozen** Stage 1 learned policy to BPD full/rank-24 writes with the strict four-tensor inference signature. Include a separately identified privileged finite-output Newton oracle **as a ceiling**, never as an input to the policy at inference.

On the authorized cohorts also report full/rank-24 raw and learned-corrected donor retrieval within ±0.10 of the **clean donor model answer**, with all eligible denominators. Run **16** norm-and-dark-matched random controls and **16** signed authorized-future-gradient and norm-matched controls, then apply the same learned correction to each. This privileged gradient matching is applied **before** correction and is not a calibrated null. Record public deviation and controls for every cohort and model success separately.

Reference descriptive public safety threshold: average absolute public-output change ≤0.02, reported over **all 512 independent public-role records**, even if a model fails public-label competence. An oracle exceeding this tolerance is a correction failure. Teacher OOD admission uses Stage 1 descriptive limits: heldout teacher MAE ≤0.12, public-role MAE ≤0.05, heldout MAE ≤0.8×source-blind conditional-mean MAE. Do not replace failure roots.

## Engineering admission and protected inference boundary

All 3 cohorts per root must be deterministic; donor changes only one valid source tag; current prequery scalar readout stays equal to numerical exactness; rank-24 singular gap and physical gauge error are recorded. Check data generator source lock and unchanged Stage 1 teacher/policy source Git blobs before training. Reject root IDs except \`901,902\`. The \`NonprivilegedPolicy\` inference method accepts only \`[state, candidate_edit, public_query, current_readout]\` and must pass the poisoned-untrusted-metadata test. Record every generated, eligible, failed-correctness and failed-current-boundary count. No denominators may be hidden by a conditional "successful" sample.

Keep Stage 1 and SF03/04/05 results unchanged. Do not access the unrelated ARG-03-N sealed 168-case data or scientific root IDs **6101–6159** and **7101–7132**. A source/seed-disjoint confirmatory Stage 2 program and 59-root null calibration must be independently preregistered after development. **No language-model privacy or newly discovered canonical BPD feature is claimed here.**
