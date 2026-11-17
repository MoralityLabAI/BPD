# BPD-SF05B Stage 1: excluded-root OOD and nonprivileged correction results

**Executed November 18, 2026. Evidence class: engineering/development only.** The complete [hosted workflow](https://github.com/MoralityLabAI/BPD/actions/runs/44144164744) verified the SHA-locked model, evaluation, Stage 0 generator and privileged training-only oracle before running the excluded roots 901 and 902. Its complete [raw two-root JSON receipt](https://github.com/MoralityLabAI/BPD/actions/runs/44144164744/artifacts/14356166513) is archived as a GitHub Actions artifact (raw JSON SHA-256 \`fb21f7f956826df71acd06a1377934fd0e58b8e681dcca64dab30734f7afd194\`); [repository summary JSON](results/excluded_2roots_summary.json) preserves the principal exact values and links to the full artifact.

The first workflow run [#44143648755](https://github.com/MoralityLabAI/BPD/actions/runs/44143648755) failed its training-only smoke test because in-place norm clipping corrupted autograd. [ENGINEERING_AMENDMENT_01.md](ENGINEERING_AMENDMENT_01.md) records the failed source and corrected Git blobs. No heldout model scoring occurred until that one-line engineering repair was applied and the amended lock passed.

## Hosted code and native closure

- **9/9 engineering tests passed**, including forbidden-future-input poisoning, correction norm cap, exact current-readout conservation, reproducible source-tag-only donors and refusal to run reserved scientific roots.
- Native \`BPDModel/TiedComponentLinear\` full-mask execution over **seven physical invocations** matched the independently trained factor path with maximum absolute error \`1.1920928955078125e-6\`. Native source-label donor-write parity had maximum absolute error \`7.748603820800781e-7\`.
- Both 32-component physical factor fits met the frozen MSE ≤1e-4 limit: seed 901 \`2.3024067985488728e-8\`, seed 902 \`3.934756023227237e-6\`. Complete physical operator GL(32) shear errors were \`5.96e-8\`; rank-24 output-SVD singular gaps were 0.0292 and 0.0145.
- The policy was trained on 30,720 privileged oracle labels **per excluded root**, all produced by \`F_train\`. The training-only Newton oracle converged on the generated training records. The deployed policy method accepts only prequery state, proposed edit, known public query and current visible scalar. It has no access to any public future-model call, gradient, true donor target or training oracle label.

## Teacher generalization and admission

The teacher trained exclusively on source histories with 2–4 writes. The heldout history family contains 5–6 writes with alternating overwrite patterns and per-example randomized source-name permutations.

| Independent development root | Train MAE | Heldout MAE | Heldout source-blind MAE | Heldout public-role MAE | Frozen descriptive teacher admission |
|---:|---:|---:|---:|---:|:---:|
| 901 | 0.0571 | 0.0745 | 0.2201 | 0.0381 | PASS |
| 902 | 0.0703 | 0.0960 | 0.2273 | 0.0483 | PASS |

Both pass the **descriptive** heldout thresholds (MAE ≤0.12, public-role MAE ≤0.05, and heldout MAE ≤0.80×source-blind conditional-mean MAE). The source-blind conditional mean is Bayes optimal under squared error; its MAE is a descriptive comparator. The OOD MAE increases by ≈0.0174 and ≈0.0257 between families, so source-length generalization is incomplete. The average prequery visible magnitude is 0.1411/0.2125 on heldout, and claims of complete natural secret invisibility would be unwarranted.

## Causal donor cohorts are small

| Development root | Train authorized eligible / 512 | Heldout authorized eligible / 512 | Heldout label-changing pairs / 512 | Full physical BPD donor retrieval on heldout eligible |
|---:|---:|---:|---:|---:|
| 901 | 35 | **18** | 278 | 18/18 |
| 902 | 18 | **13** | 240 | 13/13 |

Eligibility requires authorized role, a correct clean recipient and donor answer within ±0.10, recipient/donor prequery readout proximity ≤0.05 and a clean future answer difference ≥0.20. Those conditions yield only **31 eligible heldout pairs among 1,024 generated**, with the two teacher roots as the independent units. The 100% recovered donor outputs are primarily a **full-physical-reconstruction positive control**: a near-exact C_hat naturally reproduces the physical last-tag donor difference, once its immediate current-visible component is removed. This does not demonstrate sparse BPD feature discovery or a statistically robust secrecy mechanism.

## Learned correction versus privileged oracle on the heldout family

All public quantities below are average absolute differences in the actual **separate unauthorized U future readout** when the same prequery state receives the candidate edit. They are evaluated over every heldout row, independent of donor eligibility.

| Development root | Full physical public Δ, raw | After learned oracle-free correction | After privileged Newton correction | Full physical authorized donor recovery |
|---:|---:|---:|---:|---:|
| 901 | 0.04139 | **0.03413** | 0.0000139 | 100% of 18 |
| 902 | 0.06885 | **0.06364** | 0.0000104 | 100% of 13 |

The nonprivileged correction reduces average public deviation by approximately **17.6%** on root 901 and **7.6%** on root 902, retaining authorized donor recovery on the small eligible cohort. This still falls short of the potential Stage 2 public tolerance 0.02; only 37.7% and 19.3% of heldout rows fall below that per-row cutoff after learned full-factor correction. The privileged Newton oracle achieves residuals around 1e-5 because it explicitly queries and differentiates the true future public output, and should be regarded as a high-information ceiling.

The physical output-SVD rank-24 write behaves similarly to the full operator: heldout learned public Δ 0.03430 / 0.06407 with full authorized recovery on the same eligible cohorts. Teacher-C and exact donor-hidden-state positive controls give near-identical results to reconstructed full physical BPD weights. A rank-24 SVD is gauge invariant with respect to BPD component coordinates, while singleton components remain noncanonical.

## Stronger nulls and limitations

The **same frozen learned policy** corrected 16 independent random current-dark perturbations and 16 signed-authorized-gradient-and-norm-matched perturbations per root (matching done *before* correction). On heldout:

| Root | Corrected physical BPD donor recovery | Corrected random-dark donor recovery mean | Corrected signed-gradient donor recovery mean | Public Δ under corrected signed nulls |
|---:|---:|---:|---:|---:|
| 901 | 100% of 18 | 9.7% | 23.3% | 0.0640 |
| 902 | 100% of 13 | 13.0% | 7.2% | 0.1097 |

These are small, source-dependent descriptive comparisons. A same-root collection of sixteen null vectors provides neither sixteen independent teacher roots nor the 59-root independent upper-tail calibration. The signed-gradient null uses the BPD candidate's privileged initial future Jacobian, and the learned correction may alter its final tangent; report it as **pre-correction matched**. Public error may arise from off-manifold edits in the synthetic model.

The analytical [global dark-output Lipschitz bound](THEORY.md) states that, for an oracle repair \`d_*\` and learned edit \`\hat d\` both conserving \`q\`, the final public residual is bounded by the oracle residual plus \`\|A\|op\|C[:,0]\|_2 \|\hat d-d_*\|\`. No model-weight spectral certificate or independent maximum-error guarantee has been evaluated for these roots yet.

## Scientific decision

**Stage 1 engineering: PASS. Heldout teacher competence: PASS on 2/2 excluded roots. Strong public-safe nonprivileged mechanism: UNDETERMINED / not admitted.** The policy shows modest improvement on the two engineering roots, retains authorized recovery on a selection-limited donor cohort, and remains far from the privileged finite-output oracle.

The planned ≥59 fresh independent calibration roots 6101–6159 and ≥32 confirmation roots 7101–7132 have **not been run**. Before authorizing them, freeze a new Stage 2 protocol with many more eligible donor pairs, task-family-independent heldout source histories, an explicit source-competence admission, and a calibrated same-policy causal/null effect floor. Preserve SF03's 0/32 strong incremental failure and SF04's 3/8 public-safe routing failure. All 168 ARG-03-N sealed rows remain untouched.
