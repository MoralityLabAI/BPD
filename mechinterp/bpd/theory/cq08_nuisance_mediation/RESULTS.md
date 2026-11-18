# BPD-CQ-08 excluded-root engineering development

**Evidence class:** independently retrained synthetic development worlds. Eight exposed engineering seeds, 128 donor pairs per world, 256 paired-z behavioral cases per world. The prospective 59 sham worlds and 32 confirmation roots remain outside this receipt.

The teacher is a nonlinear rank-eight, 36-dimensional recurrent network with an explicit 12-dimensional task core and 24-dimensional nuisance carrier, trained jointly. The physical recurrence is learned from behavioral labels and auxiliary nuisance targets. The first head is symmetry-supervised under simultaneous x,y inversion, preserving first-answer logits across opposite delayed-answer branches. The independent BPD factor pair is trained and then SVD canonicalized. The source-level nuisance-match control comes from actual legal first cues and never changes x,y.

## Full root outcomes

| Seed | Jointly matched / 128 | Selected block damage | Random block 1 damage | Random block 2 damage | Primary effect |
|---|---:|---:|---:|---:|---:|
| 1301 | 126 | 54 | 22 | 26 | +25.00 pp |
| 1303 | 127 | 60 | 23 | 22 | +28.91 pp |
| 1307 | 123 | 47 | 30 | 20 | +13.28 pp |
| 1319 | 118 | 63 | 30 | 20 | +25.78 pp |
| 1321 | 128 | 60 | 30 | 26 | +23.44 pp |
| 1327 | 127 | 69 | 34 | 27 | +27.34 pp |
| 1361 | 128 | 62 | 26 | 26 | +28.13 pp |
| 1373 | 126 | 66 | 36 | 28 | +23.44 pp |
| **Total** | **1003 / 1024 (97.95%)** | **481** | **231** | **195** | **+24.41 pp** |

The selected rank-two block produces 481/1,024 delayed correct-cell corruptions on the naturally reachable donor state. Uniformly selected equal-cardinality two-component ablations produce 231/1,024 and 195/1,024 corruptions, corresponding to two development contrasts of +24.41 and +27.93 percentage points. One development root (1307) falls below the +15-point prospective practical floor for the primary comparison and is retained. The selected block is *supervised calibration-selected* from a separate sample and depends on the canonical representation basis. Physical ablation write norms are **not matched per pair** by the cardinality comparison and are preserved in the full JSON receipt.

The full donor model predicts all 1,024 independent donor L2 labels. All eight model roots have source-level nuisance matches on at least 92.2% of their generated pairs (the registered minimum is 80%). All 1,024 mismatches and successes are included in the ITT accounting. Source candidates preserve x,y and change only nuisance inputs. The nuisance control's L2 target therefore remains the recipient label, despite the matched donor state-distance. Actual intermediate states have **zero tanh-box violations** and zero replay difference from legal teacher first-pass execution.

Across 2,048 paired-z behavioral cases, full teacher and full BPD execution each score **2048/2048**, while a deterministic one-pass predictor holding the first cue fixed necessarily scores **1024/2048**. This is a constructed information-separation ceiling, not a language-reasoning result.

## EvalMath scope and next gate

The independent-model root is the unit for inference. The eight engineering roots are development data, not sealed confirmation. Matched random-vs-random sham differences range from 0 to 7.8125 percentage points, but eight development sham worlds **cannot** calibrate the required 95th-percentile/95%-confidence population tail. The registered **59 new sham roots** must be trained and source locked before confirmation. No 32-root confirmation results are claimed. The strict preflight correctly blocks a confirmation interpretation until independent-sham calibration and power checks finish.

The planted task/nuisance block separation makes the source matching feasible in this controlled generator. It is an explicit architecture constraint and cannot certify a similar decomposition in an unconstrained transformer or a pretrained TRM. Only the first-pass input states are guaranteed natural reachable; second-pass masked outputs are interventions.
