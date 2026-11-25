# COG-04 results

The learned 225-parameter scorer completed 180 CPU training updates on 64 source roots. Weighted binary cross entropy fell from 1.218756 to 0.153489. The receipt bundle records every rank, bit depth and control.

## Full public readout

| Arm | Ordinary solved / 64 | Ordinary calls | Ordinary mean online bytes/root | Deeper solved / 64 | Deeper calls | Deeper mean online bytes/root | Setup bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| dose_matched_sham_0 | 64 | 651 | 7813.55 | 56 | 1140 | 20047.31 | 1682 |
| dose_matched_sham_1 | 63 | 673 | 7996.64 | 56 | 1141 | 19787.95 | 1682 |
| dose_matched_sham_2 | 64 | 683 | 8029.55 | 52 | 1157 | 20162.20 | 1682 |
| full_rank_float32 | 64 | 664 | 7852.12 | 55 | 1137 | 19802.39 | 1682 |
| public_structural_heuristic | 64 | 487 | 0.00 | 64 | 962 | 0.00 | 0 |
| rank12_2bit | 52 | 1007 | 807.00 | 21 | 1376 | 1668.42 | 1682 |
| rank12_32bit | 64 | 662 | 7843.59 | 55 | 1137 | 19802.39 | 1682 |
| rank12_4bit | 64 | 654 | 1066.88 | 56 | 1153 | 2652.80 | 1682 |
| rank12_8bit | 64 | 668 | 2042.81 | 55 | 1140 | 5081.44 | 1682 |
| rank1_2bit | 41 | 1209 | 248.05 | 15 | 1396 | 349.31 | 362 |
| rank1_32bit | 64 | 692 | 784.00 | 52 | 1222 | 1932.59 | 362 |
| rank1_4bit | 64 | 719 | 192.19 | 43 | 1253 | 406.25 | 362 |
| rank1_8bit | 64 | 692 | 268.91 | 51 | 1223 | 612.41 | 362 |
| rank2_2bit | 42 | 1138 | 299.62 | 16 | 1400 | 482.11 | 482 |
| rank2_32bit | 63 | 765 | 1536.08 | 52 | 1196 | 3631.56 | 482 |
| rank2_4bit | 63 | 740 | 280.00 | 48 | 1219 | 613.42 | 482 |
| rank2_8bit | 63 | 764 | 465.44 | 52 | 1182 | 1022.97 | 482 |
| rank4_2bit | 51 | 1044 | 374.73 | 25 | 1343 | 664.94 | 722 |
| rank4_32bit | 64 | 669 | 2694.08 | 55 | 1119 | 6663.11 | 722 |
| rank4_4bit | 64 | 665 | 422.83 | 57 | 1150 | 991.22 | 722 |
| rank4_8bit | 64 | 665 | 741.08 | 56 | 1112 | 1776.56 | 722 |
| shuffled_rank4_float32 | 34 | 1306 | 5115.41 | 8 | 1505 | 9725.64 | 722 |
| uniform_random | 9 | 1507 | 0.00 | 0 | 1536 | 0.00 | 0 |

Online bytes include all request headers, packed activations, padding and action replies. The ratios refer to the registered split-compute implementation, whose sender projects features and whose receiver scores them. Giving the sender the full scorer would support an action-index-only message. The twelve-coordinate feature map also contains affine redundancies, which a different protocol could compress. Zero-cost symbolic and random controls run locally. Their verifier calls still count. Setup streams are charged once for each learned protocol.

## Fixed rank-four, four-bit arm

ordinary_test: 64/64 solved with 665 calls, versus 64/64 and 664 calls for the full reference. Online bytes fall by 94.62%. The per-step usefulness audit found 40 full-useful choices corrupted and 29 full-unhelpful choices repaired at the compressed policy's own reached states. Clipping affected 0.02372% of transmitted component values.

ood_test: 57/64 solved with 1150 calls, versus 55/64 and 1137 calls for the full reference. Online bytes fall by 94.99%. The per-step usefulness audit found 65 full-useful choices corrupted and 33 full-unhelpful choices repaired at the compressed policy's own reached states. Clipping affected 0.01884% of transmitted component values.

The deeper solve-count difference is an observed engineering result. The preflight and public readout support a pilot evidence class. Rank-four four-bit transport issued zero sufficient-margin flags on both partitions, so its retained solve rate has direct replay evidence while its conservative local margin bound remains inconclusive.

## Root exposure and accounting

The four-root transport smoke exposed the first four ordinary and first four deeper roots before the full report. The initial full run then exposed the public cohorts before the identity-factor dense control was added. Every SVD arm remains archived. The additional control retains the trained checkpoint and tests arithmetic parity; its ordinary rollout uses two more calls than the full SVD arm.

| Arm | Ordinary roots fresh before first full readout | Deeper roots fresh before first full readout |
|---|---:|---:|
| full_rank_float32 | 60/60 | 51/60 |
| rank4_4bit | 60/60 | 54/60 |
| rank1_4bit | 60/60 | 40/60 |
| shuffled_rank4_float32 | 30/60 | 8/60 |

The independent unit is a generated source DAG, conditional on the fixed checkpoint. All reached states and candidate scores from a root stay in that root. The retained file lists every root identifier and split. The source schema binds presentation IDs and world hashes.

The run charged 12350 exact verifier calls to collecting training labels and teacher transitions. Calibration spent 3224 policy calls plus 3224 audit calls. The full public readout used 46923 policy calls and 42431 additional full-action audit calls. Local scoring statistics never draw on the heldout usefulness labels to choose an action.

## Preflight and checks

The pinned EvalMath preflight returned `REDESIGN_OR_PILOT_ONLY`. It used 64 independent calibration roots and three equal-Frobenius, equal-residual-singular-value random matrix shams. The observed primary solve effect was 0.015625; the conservative 95/95 sham floor was 1.0. The registered scientific floor was 0.05 and the planned target was 0.10. With 22 comparisons, the planner could provide no finite sample-size target that clears the chosen variation floor.

The transport tests cover every representable code at the selected bit depths, padding, malformed messages, clipping, actual byte accounting, native BPD reconstruction, absence of a residual bypass in the compressed path, metadata exclusion and verifier budgets. The separate replay auditor reconstructs each root from initial facts and checks each result against the exact Horn checker.

The source and checkpoint hashes are in `results/summary.json`; file hashes bind the retained receipts. `results/checkpoint.json` contains the learned weights. Native Lean replay of the source COG-02 worlds is recorded by DataGoblin at the pinned upstream commit. COG-04 checks finite proof steps through that exact replayable environment.

The local verification completed 10 focused tests and independently replayed 46,923 decisions from 2,944 root-arm rollouts. The audit checked 42 setup streams and all 10 receipt-file hashes; see `AUDIT.json`.
