# RP04 Amendment 01: replacement development results

**Verdict: DEVELOPMENT_ONLY / REDESIGN_OR_PILOT_ONLY.** This report records the
new, separately seeded replacement eight-root run following the arithmetic
replay amendment. The results belong to exposed RP03 checkpoints and do not
constitute a fresh-model confirmation.

Source and parent checkpoint freeze:
[`58a9c3cd`](https://github.com/MoralityLabAI/BPD/commit/58a9c3cd9969e6fdb6f074cea6756b3e55e6aaa6).
Independent run and audit:
[GitHub Actions #45806599564](https://github.com/MoralityLabAI/BPD/actions/runs/45806599564),
artifact **`bpd-rp04-development-7cfcd843e89e3622bd17df6978c937bb00557fca`**
(GitHub artifact ID `15109687211`, retained for 90 days).
Original arithmetic failure, original first-draw scope and revised seed domain:
[AMENDMENT_01_AUDIT_ARITHMETIC.md](AMENDMENT_01_AUDIT_ARITHMETIC.md).

## Protocol and custody

All eight exposed model roots 27101–27108 are retained. Each provides one
shared warmup checkpoint and four final arm checkpoints, giving 40 snapshots.
The frozen models receive zero parameter updates. Per root, 256 base
receiver/donor/hybrid pairs are expanded to four address queries, producing
256 targeted queries and 768 unaffected queries. There are 26 intervention
conditions per checkpoint and 1,064,960 saved prediction bits overall.
Every state-edit direction is normalized to hidden-state L2 **0.75**;
the full unnormalized hybrid-state replacement is explicitly an unmatched
privileged positive control. All conditions use 31 distractor steps.
The development draw uses the separately frozen
`slot-selectivity-v2-audit-arithmetic` domain.

## Final B checkpoint: selective transport

The unaffected denominator for corruption is the 4,892 initially correct
unedited receiver answers out of 6,144 unaffected cases. Requested-value
accuracy uses all 2,048 targeted cases.

| Direction or route | Requested correct | Target accuracy | Unaffected correct corrupted | Corruption rate |
|---|---:|---:|---:|---:|
| Clean, judged against hybrid target | 119/2,048 | 5.81% | 0/4,892 | 0.00% |
| Donor displacement, fixed span | 458/2,048 | 22.36% | 1,098/4,892 | 22.45% |
| True single-slot hybrid, fixed span | 636/2,048 | 31.05% | 816/4,892 | 16.68% |
| True single-slot hybrid, current span | 633/2,048 | 30.91% | 806/4,892 | 16.48% |
| Hybrid dual, isolated selected reads | 208/2,048 | 10.16% | 286/4,892 | 5.85% |
| Full-space hybrid direction, equal L2 | 799/2,048 | 39.01% | 474/4,892 | 9.69% |
| Natural full hybrid-state replacement | 1,628/2,048 | 79.49% | 698/4,892 | 14.27% |

The fixed-span hybrid direction obtains 178 more targeted correct answers
than the donor direction, while introducing 282 fewer corruptions. It also
changes unaffected recovery counts; those counts are included in the full
machine-readable results.

## Registered root-level comparisons

Each paired effect uses the independent model root as its unit. Exact
two-sided root sign flips use the stated symmetry assumption. All four
contrasts receive Holm adjustment together. Percentile bootstrap bounds
are descriptive, with unestablished small-root coverage.

| Final B contrast | Mean effect | Root-bootstrap 95% interval | Holm p |
|---|---:|---:|---:|
| Hybrid fixed span minus donor fixed span | **+8.691 pp** | +6.934 to +10.498 pp | 0.03125 |
| Hybrid dual minus hybrid current span | **−20.752 pp** | −23.486 to −17.773 pp | 0.03125 |
| Full-space hybrid minus hybrid current span | **+8.105 pp** | +5.664 to +10.596 pp | 0.03125 |
| Hybrid dual minus rotated hybrid dual | **+3.076 pp** | +1.514 to +4.492 pp | 0.03125 |

The dual-vs-rotation mean falls below the prespecified 5 pp scientific effect
reference. Eight exposed roots cannot calibrate the population sham tail.
The source-pinned EvalMath planner returns `REDESIGN_OR_PILOT_ONLY`,
with `heldout_unsealed` and `sham_quantile_uncalibrated` reasons.

The main donor-to-hybrid comparison is positive on every root:

| Root | Donor correct / 256 | Hybrid correct / 256 | Fewer unaffected corruptions |
|---|---:|---:|---:|
| 27101 | 53 | 78 | 40 |
| 27102 | 71 | 88 | 33 |
| 27103 | 36 | 50 | 35 |
| 27104 | 50 | 80 | 45 |
| 27105 | 59 | 79 | 36 |
| 27106 | 60 | 92 | 41 |
| 27107 | 71 | 97 | 13 |
| 27108 | 58 | 72 | 39 |

Roots 27106–27108 reuse a selected component group for slots 0 and 2.
Subgroup comparisons remain descriptive and share the RP03 selection exposure.

## Native BPD geometry and carry/write routes

On final B, the mean fixed-span hybrid displacement residual is **0.86660**
of its full norm, with current-span residual **0.87106**. The desired
hybrid gate write has a **0.77109** residual outside the selected U-row span.
The mean cosine between projected donor and hybrid directions is **0.60888**.

A dual edit isolates selected reads and drives outside-component write norm
to about **9.51e-8**. Its raw norm is **1.7611×** the live-span norm. At the
same final 0.75 dose it delivers **58.23%** of the live-span selected-read
amplitude. Mean selected physical write norms are **2.4305** (dual) and
**4.1944** (live span).

| Route for current-span hybrid | Target correct / 2,048 |
|---|---:|
| Carry alone | 526 (25.68%) |
| All recurrent writes alone | 163 (7.96%) |
| Selected writes alone | 130 (6.35%) |
| Outside writes alone | 133 (6.49%) |
| Carry plus selected writes | 572 (27.93%) |
| Carry plus outside writes | 572 (27.93%) |
| Full carry and recurrent write | 633 (30.91%) |

The intervention routes are physically traceable controls, with
write-site doses inherited from their parent edits. Their nonlinear
interactions should not be interpreted as additive attribution.

## Independent audit and compute

The replacement eight-root diagnostic took **41.94 elapsed seconds** and
**36.13 summed root CPU seconds**, with a recorded peak of **560,740 KiB RSS**
on GitHub Actions CPU. The independent audit took **39.35 seconds**.

| Verification | Observed |
|---|---:|
| Final verdict | **PASS** |
| Frozen roots and snapshots | **8 roots / 40 checkpoints** |
| Prediction bits replayed | **1,064,960; exact match** |
| Independent checks | **5,713,396 passed** |
| Source hashes | **43 matched** |
| Parent input hashes | **56 matched** |
| Artifact hashes covered by completion receipt | **61** |
| Maximum scalar comparison error | **4.29153e-6** |
| Scalar tolerance | **2e-5, unchanged** |
| Maximum tensor comparison error | **5.96046e-8** |
| Statistical and diagnostic aggregates | **Exact match** |
| Preflight input/output | **Verified and matched** |

The amended engineering replay separately passed **510,515 checks** across
16,640 prediction bits, with 3.09944e-6 maximum scalar error.

### Raw result custody

The complete GitHub Actions ZIP stores all root prediction records, case
metadata, per-root result hashes, development summary, diagnostic summary,
preflight input/output, run manifest, completion hashes, and the independent
audit, together with the amended engineering receipts.

- Downloaded ZIP SHA-256:
  `c7ca40ae722e58e367198b129351df9b182bfe6fc15dc62dcc8fca39ccb853a9`
- Development `summary.json` SHA-256:
  `8842885bbe1b49912402a845a8f8d1d61f3af41a531253996c92cba199e32815`
- Development `diagnostic_summary.json` SHA-256:
  `db2e3484a50843c95ca538ed351c119dd64fa0df25eb9060ec4194622d9b452b`
- Development `independent_audit.json` SHA-256:
  `cc94a88095477070062b9b55c7cfa56ac383d1e21a7b18a9793e4eb502559b6e`
- Development `run_manifest.json` SHA-256:
  `7de596bebc803bf0073b6e44649c5a415382f3fed31ef7f1d1971b35b91e6e32`

The audit independently reconstructs saved-tensor recurrence and verified
physical receipts. It excludes inherited RP03 optimizer trajectories and
uses shared helper code for bootstrap and diagnostic aggregation after
verifying raw arrays. The source lock checks local files and records the
published remote SHA without querying GitHub during replay.

## Implication

Matched-dose slot-specific hybrid displacement consistently improves
transport, supporting construction of donors that agree with receivers
at every unedited slot. Constraining every outside BPD read at fixed state
dose substantially reduces the delivered selected-read amplitude and
targeted success. The follow-up training experiment should test a learned,
receiver-conditioned slot displacement with explicit unaffected-answer
preservation and the same physical dose and matched controls. The present
result remains an exposed-model development diagnostic.
