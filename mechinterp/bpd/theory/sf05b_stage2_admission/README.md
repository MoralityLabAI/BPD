# SF05B Stage 2-A: excluded-root model-level admission

[Protocol](PROTOCOL.md) · [Source-only donor generator](../sf05b_stage2_donors/RESULTS_SOURCE_ONLY.md) · [Prior Stage 1 result](../sf05b_stage1/RESULTS_STAGE1.md) · [Issue #58](https://github.com/MoralityLabAI/BPD/issues/58)

This is a new **development-only** model-level evaluation using the same preexisting excluded teacher roots 901 and 902. It keeps the frozen Stage 1 recurrent training and source-only Stage 2 donor generator unchanged.

Each root processes three distinct structurally heldout 5–6-write populations: 512 authorized unconditional pairs, 512 authorized high-contrast pairs admitted by independently defined source-oracle labels from 2,048 candidates, and 512 public-role U pairs. In the public cohort, **all generated records** count toward public-output deviation. In the authorized cohorts, the numerical model admission retains independent and sequential denominators for clean accuracy, prequery readout proximity and donor/recipient future output difference.

The intervention is a readout-dark physical source-tag donor write from the full 32-component BPD V/U reconstruction. Controls include unchanged state, physical output-SVD rank 24, teacher-C physical write, exact dark donor-state transfer, privileged finite-output Newton repair, and norm-/signed-future-gradient-matched random shams with the same frozen nonprivileged policy applied. The policy inference API receives only the prequery state, candidate physical edit, known public query and current scalar readout.

The pilot re-trains the exact deterministic previously excluded teachers, factorization and policy under frozen Stage 1 hyperparameters. It evaluates only those two roots and **does not run** the 59 root-level calibration IDs or the 32 confirmation IDs.

Reproduce with PyTorch 2.5.1 in BPD root:

\`\`\`bash
python -m unittest discover -s mechinterp/bpd/theory/sf05b_stage2_admission/tests -v
python -m mechinterp.bpd.theory.sf05b_stage2_admission.evaluate \
  --roots 901 902 --samples 512 --shams 16 --out /tmp/sf05b_stage2a.json
\`\`\`

This work can establish improved eligibility on a deliberately different source-conditioned task population, and audit finite synthetic public-output changes. It does not by itself establish an independent-root effect, an LLM privacy property or an identified gauge-canonical BPD feature. Do not access ARG-03-N's sealed cases.
