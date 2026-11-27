# RP03 engineering extension: the fixed rule selects 6,000 updates

The separately planned continuation completed both exposed roots in 40.58
seconds. It restored the saved model and AdamW state at update 2,000, then
continued the same training stream through update 6,000. The completed
[2,000-update study](RESULTS_2000.md) remains unchanged.

The [prespecified rule](EXTENSION_PLAN.md) requires both roots at both delays
to reach 80% ordinary TRAIN accuracy and 75% address-binding accuracy. The
3,000-update candidate fails on root 27000. Every required endpoint passes at
6,000, giving `selected_total_updates=6000` and `weak_competence=false` in
[the selection receipt](extension_results/selection.json).

| Root | Total updates | Delay | Ordinary correct | Ordinary accuracy | Address-binding correct | Address-binding accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 27000 | 3,000 | 2 | 405/512 | 79.10% | 407/512 | 79.49% |
| 27000 | 3,000 | 7 | 372/512 | 72.66% | 341/512 | 66.60% |
| 27001 | 3,000 | 2 | 456/512 | 89.06% | 408/512 | 79.69% |
| 27001 | 3,000 | 7 | 445/512 | 86.91% | 388/512 | 75.78% |
| 27000 | 6,000 | 2 | 512/512 | 100.00% | 512/512 | 100.00% |
| 27000 | 6,000 | 7 | 510/512 | 99.61% | 512/512 | 100.00% |
| 27001 | 6,000 | 2 | 512/512 | 100.00% | 512/512 | 100.00% |
| 27001 | 6,000 | 7 | 510/512 | 99.61% | 510/512 | 99.61% |

The modal-value baseline remains exactly 25% on every address-binding cohort.
Each underlying base sequence contributes four query variants, so these rows
remain repeated measurements within the two model roots. All tuples belong to
the TRAIN partition. Ordinary inference runs with steering disabled.

## The RP02 write edit has zero targeted successes at 6,000 updates

At the final checkpoint, the same fixed group-discovery recipe and L2-0.75
component write edit produce **0/1,024 targeted counterfactual answers** across
both roots and delays. Unaffected queries retain **1,021/1,024** correct answers.
Each checkpoint's complete eight-group calibration matrix is preserved.

Ordinary address memory has become reliable on this training diagnostic. The
selected component edit preserves the original answer on 1,023 of the 1,024
targeted cases where the requested counterfactual requires a different value.
Increasing the warmup budget resolves
the observed task-learning limitation, while the intervention objective needs
its own redesign or measurement before a mechanistic benefit can be claimed.

## Reproducibility and selection exposure

[The run manifest](extension_results/run_manifest.json) pins every executed
source and both parent checkpoint hashes. Each root retains its 3,000- and
6,000-update model/optimizer files, complete diagnostic predictions, and a
cross-entropy trace every 50 updates. Training indexes continue from 2000;
diagnostic seed purposes use a separate `extension_` prefix.

[Summary rows](extension_results/summary.json) include every endpoint and the
modal controls. [Completion hashes](extension_results/completion.json) bind
both result files. The saved optimizer step counters, source hashes and
checkpoint diagnostics passed a separate [replay audit](extension_results/replay_audit.json),
covering all four checkpoints and 12,288 diagnostic prediction bits.

The budget choice uses two engineering roots exposed before the extension.
Thresholds and candidate budgets were fixed in its plan before continuation.
This result supplies an engineering recipe choice with that exposure recorded.
