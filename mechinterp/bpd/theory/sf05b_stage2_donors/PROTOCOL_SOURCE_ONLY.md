# BPD-SF05B Stage 2 donor admission: source-only preflight

**Prospective source-only registration, November 18, 2026.** Parent [SF05B issue #48](https://github.com/MoralityLabAI/BPD/issues/48), [donor power issue #58](https://github.com/MoralityLabAI/BPD/issues/58), [Stage 1 excluded-root report](../sf05b_stage1/RESULTS_STAGE1.md). No new teacher, BPD factor, correction policy, calibration root or confirmation root is trained or scored in this PR.

## Motivation and immutable upstream source

Stage 1's heldout donor eligibility was 18/512 and 13/512 for the excluded 901/902 teachers, despite adequate descriptive source-query competence. Its former uniform 4-way query allocation included unauthorized role U and often asked about a source unaffected by the last-write source swap. These synthetic generator decisions reduce the number of eligible counterfactuals. A public-role query remains an essential **independent** safety test.

Reuse the exact Stage 0 source generator from \`mechinterp/bpd/theory/sf05b_source_families/source_families.py\` with the same 3 source IDs, 11 input coordinates, last-write-wins semantics, independent continuous values, 2–4-write train and 5–6-write heldout source families.

## Authorized source-aware donor cohort

For a fully generated recipient sequence, the source tag on its last **valid** write is \`s\`. Draw a distinct donor source \`t\` uniformly from the two remaining IDs, and replace only that last write's one-hot source tag \`s→t\`, preserving every numeric payload and all earlier write events. Using an independent seeded bit, select the common authorized query source \`r=s\` or \`r=t\`. Set both recipient and donor **query role one-hots** to r; preserve the public scalar and query marker. All source-specific targets are recomputed with the unchanged exact target oracle.

This authorized-role stratification is based solely on the **synthetic source labels**, before any model call; it is separate from model correctness or from hidden-state/readout interventions. It gives a chance for the last source reassignment to change the label while maintaining equal payload content.

Define two explicit test strata:

- \`authorized_unconditional\`: keep all n generated cases, including those whose exact target is unchanged. Report the exact target-change fraction and any later model-based admission separately.
- \`authorized_high_contrast\`: draw **4n candidate cases** from one deterministic source seed; after the above role and donor-tag choice, retain the first n examples whose **exact source-oracle target difference is at least 0.30**, without looking at model outputs. Report every candidate denominator, accepted example and rejected count; raise an explicit admission failure if fewer than n qualify. This is a **different generator population** and must be labeled accordingly.

A fixed threshold of 0.30 is a data-generation criterion, not an empirical effect gate. The high-contrast stratum can increase causal test efficiency while conditioning on the source's true label; unconditional results must remain visible.

## Public role-U cohort

Draw an independent base record seed and a distinct donor source-switch seed. Set the shared query role to U (code index 3), with the **same public scalar** in recipient and donor. Recompute the two exact targets, which must equal the shared public scalar for every case. No authorized query, donor-eligibility threshold or model correctness filter may be used to discard public cohorts when measuring public output change. Report public-output deviation on **all generated public cases** and on any separately labeled model-competence subset.

## Required source assertions

- Only the last valid write tag differs between recipient and donor events; earlier source labels, all values, masks, event order, history length, public scalar and query marker agree.
- Authorized query role belongs to \`{s,t}\`; public cohort always queries U.
- Query role and source tag choices use deterministic local seeded RNG streams independent of teacher/factor/policy parameters.
- The original Stage 0 last-write-wins target oracle recomputes both outputs exactly, including overwrite and absent-source cases.
- The source-blind squared-error conditional means must be **recomputed under this source-aware sampling law**, because selecting the query role from {s,t} correlates the visible role with the otherwise hidden source-label assignment. The Stage 0 uniform-query blind mean must not be reused as a Bayes-optimal reference. For the heldout template, enumerate all six label permutations, both distinct donor-source choices, and both authorized role selections, conditional on the observed query role. In the high-contrast stratum also condition on the exact acceptance event |recipient target - donor target|≥0.30 before averaging. Report recipient and donor blind conditional means independently, and prove their transformation behavior under the donor-source flip using exact enumeration. For the independent U-query cohort both Bayes means equal the public scalar.
- Train/heldout structural source families remain disjoint in event counts and overwrite-template support, while source-ID alphabet remains shared.

## Evidence and future science boundary

This PR validates generator logic only. An excluded-root model admission study using the new generator requires **its own pre-run frozen source and controls**; the already scored Stage 1 root 901/902 findings remain intact. Calibration roots 6101–6159 (59) and confirmation roots 7101–7132 (32) remain reserved and unexecuted. A future confirmatory protocol must freeze an authorized utility floor, public-output tolerance, matched-sham model, factor subspace rank, and independent-root null calibration before new teacher training.

No test may access ARG-03-N's 168 sealed cases or assert real language-model privacy. A source-blind conditional mean is optimal for squared error and its MAE is descriptive.
