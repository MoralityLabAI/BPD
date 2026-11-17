# SAU-TRACE02: high decodability with a gated native read path

**Executed:** 2026-11-17, local Python 3.13, NumPy CPU. Independently authored synthetic gym, motivated by [SauersML/retained-reply-state](https://github.com/SauersML/retained-reply-state). Four deterministic unittest checks passed. New BPD code is at [sau_trace02.py](sau_trace02.py).

## Question

Can hidden memory remain externally decodable in a fixed visible reply while its native continuation has essentially no access to that memory? Can a physical route intervention turn that same memory into causally effective state, without changing what the external probe can decode?

**A constructed affirmative witness:** the synthetic recurrent system separates memory, nuisance and native decision coordinates, with 50 possible hidden items. The visible answer string is fixed to "I understand." A linear nearest-prototype readout is learned from 32 calibration examples per item and has access to the retained memory state. The native system starts with zero decision state and evolves a gated recurrent transition over four steps. In the gate-closed condition, the native logits are identical for any hidden item. The gate-open condition routes that memory into the decision coordinates.

There is no language-model checkpoint, Qwen run, pretrained probe, SGD training of the recurrent map, or inference about subjective experience.

## Frozen synthetic construction

Physical recurrence in the 50-dimensional memory and decision sectors, plus an inert 50-dimensional nuisance sector:

\`\`\`text
m[t+1] = 0.8 m[t]
d[t+1] = 0.5 d[t] + g m[t],     d[0] = 0
native_logits = 7 d[4]
m[0] = onehot(item) + normal(0,0.30) per memory coordinate
\`\`\`

Each condition uses the identical state, label and seed. The causal coefficient from memory at time zero to native logits at depth 4 is \`7*g*(0.5^3 + 0.5^2*0.8 + 0.5*0.8^2 + 0.8^3)\`. At g=0 its value is exactly zero. At g=0.6, the unscaled memory-to-decision factor is 0.6942.

Eight independent RNG seed blocks each contain 1,600 label-balanced calibration episodes and 1,000 separate, label-balanced test episodes. The permuted-calibration null relabels the *calibration training set only*. All test results are scored once with the same test roster per seed. Distinct test episodes share one synthetic recurrence and parameter choices, so the 8,000 cases are not evidence across independent trained models.

## Executed CPU results

| Endpoint | Mean over 8 seeds | Seedwise range |
|---|---:|---:|
| External 50-way linear-prototype top-1 accuracy | 0.793250 | 0.783–0.811 |
| Permuted-calibration top-1 | 0.021750 | 0.010–0.035 |
| Retained memory, gate closed, true-item native probability | 0.020000 | 0.020–0.020 |
| Partially open gate g=0.10, true-item native probability | 0.044019 | 0.043399–0.044727 |
| Open gate g=0.60, true-item native probability | 0.498811 | 0.487058–0.514323 |
| Open gate with different-item donor memory, original true-item probability | 0.009705 | 0.009336–0.010142 |
| Open gate with norm-matched nuisance-sector sham | 0.498811 | 0.487058–0.514323 |
| Visible-disclosure positive control | 0.957229 | 0.957229–0.957229 |

The stripped state with the gate closed has the same 0.020000 mean native true-item probability as the retained memory. Open-gate donor memory is a coordinate permutation by 17 items, guaranteeing different latent identities for all 50 classes, and the sham edit has the exact same per-case L2 norm in a causally inert nuisance sector. This sham construction is intentionally strong for identifying the controlled path, though it is not a substitute for the matched-rank empirical controls used in BPD.

### Seedwise heldout outcomes

| Seed | Probe top-1 | Permuted top-1 | Native open true-p | Donor open true-p |
|---|---:|---:|---:|---:|
| 103 | .786 | .035 | .500297 | .009860 |
| 131 | .795 | .018 | .507466 | .009595 |
| 167 | .789 | .011 | .497017 | .009792 |
| 211 | .794 | .024 | .505710 | .009424 |
| 307 | .791 | .010 | .487541 | .010142 |
| 401 | .811 | .028 | .514323 | .009595 |
| 503 | .783 | .023 | .491080 | .009336 |
| 601 | .797 | .025 | .487058 | .009896 |

Every seed's raw test/calibration feature SHA-256 is emitted by the runner, alongside the per-seed metrics. The logged data are generated from code deterministically. To produce the complete receipt:

\`\`\`sh
python -m mechinterp.dark_fiber_gyms.sau_trace02 > sau_trace02_full.json
python -m pytest -q mechinterp/dark_fiber_gyms/tests/test_sau_trace02.py
\`\`\`

## Scientific boundary and next experiment

This demonstrates that the *existence* of a high-accuracy external probe places no lower bound on native task use without conditions on the native transition and readout. The gate supplies those conditions explicitly, making an intervention in a readable but behaviorally silent subspace effective. The local linear operator has block form \`F_g=[[0.8I,0],[gI,0.5I]]\`, and \`C=[0,I]\`. Its finite-horizon causal sensitivity to initial memory is \`C F_g^k P_memory\`, which is zero at g=0 even though the identity external observer \`P_memory\` decodes memory.

For the next real-model trial, hold acquisition history and visible reply fixed as in SauersML; compare stripped/retained/revealed contexts with the source-model native head. Audit KV positional rewrites and arithmetic observation parity first. Measure the native logits of a 50-way item and exact donor-state swaps before attributing a dark-fiber mechanism to any externally decodable SAE/BPD feature. Preselect held-out item contexts and report both test-seed variation and within-family sampling uncertainty.

Do not use this synthetic outcome to amend ARG-03-N's frozen threshold or open its sealed 168 cases.
