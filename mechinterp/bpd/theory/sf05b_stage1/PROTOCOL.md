# BPD-SF05B Stage 1: nonprivileged recurrent source-binding correction

**Prospective engineering specification, November 18, 2026.** Parent: [SF05B issue #48](https://github.com/MoralityLabAI/BPD/issues/48), [Stage 0 source family PR #50](https://github.com/MoralityLabAI/BPD/pull/50), [SF05 A2 PR #44](https://github.com/MoralityLabAI/BPD/pull/44).

## Split and evidence class

This is **Stage 1 engineering on excluded roots 901/902**, not registered scientific confirmation. The Stage 0 family generator is frozen at Git blob \`525cfb28d40cc5f58c0e8141489a8e744d1778f4\`. All training samples and correction-policy supervision come only from \`F_train\` with 2–4 writes. The structurally disjoint \`F_heldout\` has 5–6 writes and abstract overwrite patterns with a per-record permutation of three source labels. Source alphabets, scalar distribution and event encoding are shared across families. First posttraining evaluation of \`F_heldout\` must follow commitment of the complete Stage 1 implementation, measurement definitions and source blobs. Engineer decisions made before this protocol used training-family samples only; no model-derived \`F_heldout\` outcomes were inspected.

Fresh roots **6101–6159** (59 calibration) and **7101–7132** (32 confirmation) are reserved and excluded from Stage 1 engineering. They cannot be used until an independent Stage 2 confirmatory registration fixes all decision thresholds and source hashes. Keep the unrelated ARG-03-N 168-case sealed holdout untouched.

## Shared physical teacher

On the Stage 0 11-dimensional event encoding, use the same physical recurrent transition for each write and the final query:

\[
F_\theta(h,x)=h+\tanh(hA+xB+b)\,C,
\quad h\in\mathbb R^{32},\quad
A\in\mathbb R^{32\times64},\quad B\in\mathbb R^{11\times64},
\quad C\in\mathbb R^{64\times32}.
\]

Initial \`h=0\`. Skip the padded write events exactly using the Stage 0 event mask; the valid events still run through the shared weights. Actual observable readout is \`q(h)=h_0\`, including the intermediate prequery observation. Train all physical teacher parameters from scratch per root using Adam LR \`0.012\`, **2600 updates, batch 128** on independent \`F_train\` seeds. Optimize \`4*MSE(q(h_final), target) + 0.15*mean_valid_write(q(h_t)^2) + 1e-4*(mean(A^2)+mean(C^2))\`. Clip teacher gradients at 5. Training input seeds are \`seed*100000+update\`, with a fixed separate initialization offset +1741. Frozen teacher competence is measured on independently generated \`F_train\` and once on \`F_heldout\` at distinct seed offsets. Compare to the exact source-blind *squared-error-optimal* conditional-mean oracle; report both MAE and MSE, public-role MAE and all sample counts. The source-blind oracle's MAE is a descriptive comparison rather than the minimum blind MAE.

## Physical BPD factorization and source-tag counterfactual

Freeze teacher and independently fit one full-rank BPD physical \`C_hat=VU\` with \`V∈R^{64×32}\`, \`U∈R^{32×32}\`. All 32 components remain active. Train both with Adam LR \`0.035\`, **450 updates, batch 128**, minimizing \`1.5*MSE(C_hat,C)+MSE(phi C_hat,phi C)\` on frozen teacher write/query features from a separate \`F_train\` stream. This tests *physical factor geometry*, not sparse gate discovery or canonical singleton feature recovery. Check full-mask parity with actual \`BPDModel/TiedComponentLinear\` at all 7 possible physical invocations. Report \`GL(32)\` shear invariance and output-SVD physical rank-24 and rank-32 writes; rank selection is frozen before evaluation. Include the teacher \`C\` SVD rank-24 control.

For recipient and donor matched source histories, only the **last valid write's source tag changes**. All previous writes, scalar values, public query, authorized query and event order remain fixed. At the shared pre-last-write state \`h_pre\`, form the physical BPD donor write

\[
d=(\phi(h_{\rm pre},x_{\rm donor})-\phi(h_{\rm pre},x_{\rm recipient})) C_{\rm hat}.
\]

After the recipient's last write, remove only coordinate zero to preserve \`q(h)\` exactly, and continue with the same physical teacher final query. Compare the unmodified physical write, rank-24 projection, teacher physical write, exact dark donor-state replacement and original zero edit. Audit all generated matched pairs and eligible proportions. Principal eligibility: authorized query role 0/1/2, both clean model outputs within \`0.10\` of the independently generated recipient/donor label, prequery recipient-donor readout difference \`≤0.05\`, and clean future difference \`≥0.20\`. Never replace an ineligible sample.

## Privileged correction teacher (training supervision only)

For each \`F_train\` matched donor history, compute the same physical donor write and separate role-U public query with the original public scalar. On a balanced training stream, also generate norm-matched random-dark perturbations. The privileged target is the **finite public level-set correction** from SF05 A2: eight damped Newton steps, absolute public residual tolerance \`1e-4\`, step cap \`min(0.25*||d||,0.5)\`, and line-search rates \`[1,0.5,0.25,0.125]\`. The teacher's true public output and gradient may be used **only to generate training labels and a separately labeled oracle ceiling**. The target correction is \`d_repaired-d\`, with all converged and failed instances retained and counted. The original A2 Newton oracle is a privileged reference and is never used to construct the test-time nonprivileged edit.

## Nonprivileged bounded predictor

Train a single MLP per root with input \`[h_prequery, d, public_query, q(h_prequery)]\`, dimensionality \`32+32+11+1=76\`. Network: 76→96→96→32 with \`tanh\` activations. Adam LR \`0.003\`, **240 updates**, batch 128, supervised correction MSE, independent \`F_train\` seeds. The output coordinate 0 is hard-zeroed; the correction vector is L2-clipped per example to \`min(0.25*||d||,0.5)\`. The inference API accepts exactly those four tensors and **no model, future public score, derivative, donor label, oracle label, or future trace**. A poison test must show that altering any separately supplied forbidden metadata does not affect the predictor's output. Test that the correction never changes the current readout or exceeds its radius.

## Stage 1 excluded-root pilot

On roots **901 and 902 only**, evaluate 512 independently seeded \`F_train\` pairs and 512 \`F_heldout\` pairs after code and readout definitions are frozen. Include complete generated/eligible denominators. Record, separately for all rows and authorized-eligible rows:

- Teacher final output MAE/MSE versus independently known last-write labels, clean public-role error, source-blind MAE/MSE and the source-family OOD performance gap.
- BPD reconstruction MSE, native physical parity, full-block \`GL(32)\` invariance, actual current-readout equality, rank-24 singular gap.
- Donor recovery at ±0.10 tolerance for raw full physical, raw rank-24, teacher physical, exact dark donor state, zero edit, learned corrected full and rank-24, and privileged oracle-corrected versions.
- Finite public-output absolute change against the unmodified recipient public query; number and fraction below 0.02. Correction norm/dose and actual public-output error on all cases, plus the authorized-eligible subset.
- Sixteen independent raw random-dark and signed-authorized-future-gradient-and-norm matched sham arms. Apply the *same frozen predictor* to each null and label this matching as **pre-correction**. Record oracle-corrected nulls separately when computational budget permits.
- Full-source identity metadata, no-secret role distribution, and single-tag counterfactual correctness. Mark any trained teacher that fails OOD competence independently of the intervention outcome.

**Engineering acceptance** requires finite source/model outputs, exact donor tag-only edits, full-mask native parity ≤1e-5, factor physical MSE ≤1e-4, current-readout equality ≤1e-6, predictor norm-bound verification ≤1e-6, and explicit ineligible/null denominators. A *descriptive* OOD teacher admission is MAE ≤0.12, public-role MAE ≤0.05, and MAE ≤0.80× heldout source-blind conditional-mean MAE; these are not scientific claims or license to replace roots. The OOD intervention result remains exploratory even if every numeric check passes.

## Future Stage 2 statistical boundaries

After Stage 1, author and commit a separate \`STAGE2_PROTOCOL.md\` and root/threshold source lock. It must preregister a specific inferential null statistic, same-correction matched shams, finite public-output margin, authorized recovery floor, teacher OOD competence, gauge-stable intervention and genuinely disjoint heldout histories, with ≥59 **new** independent calibration roots to establish a 95/95 upper-tail tolerance and then ≥32 independently trained confirmation roots. Retain negative and underpowered outcomes. This work alone establishes neither an LLM privacy mechanism nor a discovered secret-bearing canonical BPD coordinate.
