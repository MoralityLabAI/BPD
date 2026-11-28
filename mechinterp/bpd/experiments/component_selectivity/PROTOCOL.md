# BPD-RP-04: slot selectivity through native component reads and writes

## Question and scope

RP03's fixed and tracked projectors produced closely aligned edits, with weak selective transport. RP04 examines donor contamination, the selected read span and the physical writes made by BPD components. It executes the saved native models without parameter updates. Every result is **DEVELOPMENT_ONLY**.

The parent is commit `402ef7764e11bc7736873d66357b8dc24b996e94` in MoralityLabAI/BPD. Roots 27101 through 27108 are already exposed development model roots. Each contributes its common warmup checkpoint and all four final training arms. These five snapshots share a model root and are repeated measurements. Fresh input draws use the same finite evaluation-tuple domain as RP03. Their purpose-specific noise, pair draws and query expansion are new; semantic tuples can recur.

The primary snapshot is **B_causal_fixed**. A, C, D and warmup readouts provide descriptive context and before/after measurements on the same input cases. RP04 trains zero models, chooses zero new component groups and conducts no adaptive dose or architecture search.

## Prior evidence and planning

The archived RP03 preflight returns REDESIGN_OR_PILOT_ONLY: observed B minus A is +0.1221 percentage points, with an uncalibrated population sham tail. It supplies a prior planning warning for a bounded follow-up. Its ordinary-answer endpoint differs from RP04's targeted transport endpoint and supplies no RP04 power guarantee.

Two exposed engineering roots, 27010 and 27011, execute the same diagnostic with 16 base pairs each. Their role is source and replay verification. Every arm, metric and failure is retained. After this smoke check, publish the complete RP04 source before generating the eight-root development input batches. `SOURCE_FREEZE.json` binds that remote commit, the full recipe, source hashes and parent checkpoint hashes. Development requires this lock and a fresh output directory.

The scientific reference is an absolute five-percentage-point targeted-accuracy gain. An increase of five points in unaffected-answer corruption is a separate descriptive harm reference. The source-pinned EvalMath preflight receives independent-root effects and the corresponding rotated-control variation. Its population sham bound remains uncalibrated at this root count. The checkpoint population is exposed throughout; this study allocates no confirmation cohort.

## Data contract

Each development root supplies **256 base receiver/donor/hybrid pairs**, balanced at 64 per edited slot. The generator retains RP03's four slots and four values, root-specific channel encodings, Gaussian input noise of 0.03 and 31 distractor steps. Receiver, donor and hybrid must all have semantic tuple codes with residue 0 or 1 modulo 5. The hybrid changes exactly the selected receiver slot to the donor's different value.

Every base pair is queried at **all four addresses**, preserving its prefix, distractors and noise. The resulting 1,024 query cases contain 256 targeted queries and 768 unaffected queries per root. Queries are repeated observations of their pair, and pairs remain clustered within the model root. All snapshots and interventions receive identical expanded inputs.

Sampling uses SHA-256 seed domains beginning `BPD-RP-04`, disjoint from RP03's domains. Exact batch hashes, semantic tuples and every prediction are retained. The selected component groups and signed-permutation rotation come from the parent geometry checkpoint. Selection exposure from RP03 remains part of this study.

## Native BPD operator

The recurrent gate write is `w(h) = (h V) U`, using the saved `TiedComponentLinear` with all component masks open and zero residual contribution. `W_target` remains unused. Each physical V/U pair is shared across sequence positions.

For a selected group G, define current read-space projector P_live onto span(V_G) and the saved warmup projector P_fixed. The six state-edit directions are:

| Name | Raw row-vector state displacement |
|---|---|
| donor_span | (h_donor - h_receiver) P_fixed |
| hybrid_span | (h_hybrid - h_receiver) P_fixed |
| hybrid_live_span | (h_hybrid - h_receiver) P_live |
| donor_dual | (((h_donor - h_receiver) V) mask_G) V_plus |
| hybrid_dual | (((h_hybrid - h_receiver) V) mask_G) V_plus |
| hybrid_full | h_hybrid - h_receiver |

`V_plus` is the float64 Moore-Penrose pseudoinverse with relative singular-value cutoff 1e-10 and zero absolute cutoff. Direction construction uses float64. Valid raw directions are normalized to state L2 **0.75**, then cast to the model's float32 execution dtype. Raw norm at or below 1e-9 produces an explicit zero edit with its validity flag retained. The dual recipe preserves the minimum-norm least-squares solution if exact read isolation is infeasible; numerical rank and residuals expose that case.

Each direction also receives the same root-specific signed-permutation rotation, giving six physically matched direction controls. Together with clean continuation and natural full-hybrid-state replacement, these produce 14 full-recurrence conditions. Hybrid-derived edits use privileged counterfactual states. The unnormalized hybrid-state replacement supplies a continuation check at its natural dose.

The live-span edit is the minimum-norm state change that matches the requested selected reads. Exact suppression of other reads adds constraints and can increase the raw norm. Normalizing both policies to 0.75 therefore changes their delivered selected-read amplitudes. The dual/live-span comparison estimates the full isolation policy including this physical cost.

## BPD measurements

For each parent displacement delta, compute the induced component reads and physical writes:

```text
reads_j = delta v_j
atom_j = reads_j u_j
g = sum(atom_j for j in G)
o = sum(atom_j for j outside G)
```

Retain group-summed outside-write fraction `||o||²/(||g||²+||o||²)`, the cosine of g and o, and their cancellation ratio. These quantities survive invertible changes of basis within the selected and outside blocks. Atom-energy fractions and per-component averages survive reciprocal factor rescaling; within-block mixing can redistribute atom energy. Raw coordinate-read norms have their gauge dependence labeled.

The geometry receipt also records the hybrid displacement left outside the fixed/current V spans, the cosine between projected donor and hybrid displacements, and the desired hybrid gate write left outside the selected U-row span. For each dual edit, retain feasibility residuals, raw norm amplification, its norm ratio relative to the current read-span edit and the selected-read scale after normalization. The full-rank Pythagorean norm identity receives a numeric check.

Factor identities concern this finite linear operator. Task success and answer preservation remain empirical outcomes of the nonlinear recurrence. See THEORY.md for the algebra and the proposed training objectives it motivates.

## Carry and recurrent-write paths

For **hybrid_live_span** and **hybrid_dual**, keep each normalized parent state delta and evaluate six additional routes at the first distractor:

| Route | Carry state | Added recurrent gate write |
|---|---|---|
| carry_only | h + delta | 0 |
| all_write_only | h | g + o |
| selected_write_only | h | g |
| outside_write_only | h | o |
| carry_selected | h + delta | g |
| carry_outside | h + delta | o |

The full direction condition already supplies carry `h+delta` and write `g+o`. A temporary output hook alters the executed native recurrent output at that token, and subsequent steps continue normally. Hooks are removed through `finally`. Existing ComponentIntervention behavior in a different execution mode is outside this implementation.

These route controls inherit the same parent delta. Their gate-write norms vary by route and are recorded; a 0.75 state dose supplies no claim of equal write-site dose. Each snapshot has **26 total conditions**, all retained. Exact finite carry/gate identities, including interaction `(z_new-z_old)*delta`, receive direct numeric checks.

## Outcomes and statistical family

Record requested hybrid-value accuracy on targeted queries. On unaffected queries, report the initially correct denominator, corruptions, initially wrong denominator and recoveries. The condition's unedited receiver prediction supplies its baseline. Every prediction also supports a breakdown by edited slot.

Four contrasts on the final B checkpoint form the entire inferential family:

| Candidate minus baseline | Interpretation |
|---|---|
| hybrid_span minus donor_span | Contamination from other donor slots |
| hybrid_dual minus hybrid_live_span | Policy that isolates selected component reads |
| hybrid_full minus hybrid_live_span | Coverage lost by restricting the state direction |
| hybrid_dual minus hybrid_dual_rotated | Physical direction control for the dual policy |

Use equally weighted root effects, exact two-sided root sign flips under signed exchangeability and Holm correction across four comparisons. Report descriptive percentile intervals from 10,000 root bootstrap draws with seed 20261126. Those intervals have unestablished small-root coverage. All other snapshots, route controls, slot breakdowns and geometry summaries remain descriptive. Every assigned root stays in the record, including failed gates and numerical infeasibility.

The total development inventory is eight exposed roots, 40 saved snapshots, 2,048 base pairs, 8,192 query cases per condition across roots, and 1,064,960 predictions across all snapshots and conditions. The independent sample size remains eight.

## Verification and custody

Check original checkpoint hashes before inference and verify parameter-state hashes remain unchanged. Unit tests exercise query/label semantics, equal state dose, gauge transformations, BPD write sums, dual feasibility and cost, route execution and hybrid-continuation parity. Synthetic tests have their own engineering scope.

The independent audit reconstructs the recurrence and intervention directions from saved tensors, without importing RP04's intervention or execution modules. Shared input generation is followed by independent semantic decoding. Prediction bits, scalar physical receipts, component identities and aggregate statistics are replayed, with an explicit list of scope limits. Source hashes bind the diagnostic and the unchanged 26-file RP03 source dependency set. Runtime receipts include elapsed time, process CPU and Linux RSS high-water measurements.

Any implementation repair after source publication requires a visible amendment and a new purpose-specific input domain before rescoring development. Results already generated retain their original source and input receipts.
