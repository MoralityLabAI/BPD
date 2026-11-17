# BPD-SF05B Stage 0: source-family-disjoint provenance generator

**Prospective source-only registration, 2026-11-18.** Motivated by [SF05B issue #48](https://github.com/MoralityLabAI/BPD/issues/48) and [SF05 Stage A/A2](https://github.com/MoralityLabAI/BPD/pull/44). No new model, gradient correction, calibration root or confirmation root has been trained or evaluated for this stage.

## Testable source-binding task

Three independent user labels A/B/C are represented as one-hot source tags. An event is a write of a uniformly random scalar in \`[-0.8,0.8]\`, tagged by its source and timestamp. A final query asks for the last value written by one designated source or, for role U, an independent public scalar. If no write for the queried source exists, the target is 0. Every event is a synthetic 11-dimensional vector: three source bits, scalar value, write flag, four query-role bits, public scalar, query flag. The same physical future recurrent operator must process the events sequentially.

**Training source family F_train:** exactly 2, 3, or 4 writes per record, each write's source label iid uniform in \`{A,B,C}\`, independent scalar values. Source-name repeats and overwrites are permitted. The query source/unauthorized role is independent and uniform over four choices.

**Structurally disjoint heldout F_heldout:** exactly 5 or 6 writes with fixed abstract overwrite templates \`[0,1,0,2,1]\` or \`[0,1,2,0,1,2]\`. Independently permute the three source names on every record using a random permutation, preserving the template's semantic overwrite structure. Both families use the same value distribution and query alphabet. Train/test source histories have disjoint lengths and template supports; this is an **event-family** distribution shift rather than an unseen user-ID alphabet or language modality.

Paired donor source counterfactuals change **only the source tag on the last valid write** to one of the other two user labels, keeping every numeric payload, first writes, query role, public scalar and event count fixed. The target is recomputed by the same last-write-wins oracle. No pair selection uses the model's response.

## Exact source-blind conditional-mean controls

For F_train, conditioning on the event values, history length, role and public scalar **without seeing source labels**, the final authorized answer has

\[
\mathbb E[Y^*\mid v_{0:L-1},L,R\in\{A,B,C\}]
=\sum_{j=0}^{L-1}\frac13\left(\frac23\right)^{L-j-1}v_j.
\]

The probability that the source never appears is \`(2/3)^L\`, contributing 0. For role U the optimal prediction is the known public scalar.

For F_heldout, the source names are assigned by a fresh uniform permutation of three template identities. Conditional on ordered values, length and queried source while withholding name assignment, the source's last write is uniformly one of the three abstract-template last-write positions. Thus the squared-error Bayes predictor is the **average of those three last values**. The generator calculates and returns this exact source-blind conditional mean on every record. It is optimal for squared error; its MAE is only a fixed descriptive baseline.

## Split, independence and evidence scope

Generator seeds are deterministic \`torch.Generator\` streams and fully public. Unit tests enumerate all \`3^L\` source assignments on F_train and the six source permutations on F_heldout to confirm the oracle. Additional tests verify last-write semantics, no-public-label contamination, donor tag-only changes, independent seeds and family length separation. A passing generator establishes only source/target correctness.

New teacher code, input boundary hooks, policy architecture, rank/subspace choice, per-root calibration, future oracle-access prohibition and signed-gradient null definitions must be committed in a **separate Stage 1 protocol and lock before fresh roots are trained**. Reserved independent calibration roots: \`6101..6159\` (59); reserved confirmation roots: \`7101..7132\` (32). These are reservations, **not executed data**. Engineering seeds 901/902 and all previously scored SF03–SF05 roots are excluded from confirmation.

No posttraining model scoring on F_heldout is authorized until the final source family, model and numerical readout protocol are frozen. The 168-case ARG-03-N heldout split is unrelated and remains sealed. Source-family disjointness alone does not establish real model privacy, hidden secret representations or secure generated transcripts.
