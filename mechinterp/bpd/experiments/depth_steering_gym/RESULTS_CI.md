# BPD-DT02-A: first native Attention-TRM training on hosted CPU

**November 18, 2026.** [Source-locked GitHub Actions run #44277919962](https://github.com/MoralityLabAI/BPD/actions/runs/44277919962) passed all **6 unit tests** and trained six actual, tiny 14,082-parameter native \`PureSudokuTRM\` models (2 previously excluded teacher roots 901/902 × 3 independently graded families). Full [raw JSON artifact](https://github.com/MoralityLabAI/BPD/actions/runs/44277919962/artifacts/14415741976), extracted \`bpd_dt02a_native_training.json\` SHA-256 \`9d4f9463c0853c6cc2491711ebf3cb6c37021a61861f3947962dff67fa76b30b\`. Git-blob pinned \`LOCK_CI.json\` was verified **before** any model training. Torch 2.5.1 CPU, two threads; bounded 30-minute job completed successfully.

## Distinguish supervised loss, exact task performance and terminal utility

All six models received 64 AdamW updates on independent training-generated examples, with task CE supervised at recurrent ACT depths 2/4/6. Depths **8 and 12 were never used for training**. Test cohorts consist of 64 independent heldout items per root/task. No official ARC-1 dataset was used; ARC tasks here are **synthetic transformations**, with different programs in training and evaluation.

| Excluded root | Task | Heldout depth-8 exact | Heldout depth-12 exact | Last training CE |
|---:|---|---:|---:|---:|
| 901 | 4×4 Sudoku, unique host solution | **0/64 boards** | **0/64** | 0.45467 |
| 902 | 4×4 Sudoku | **0/64** | **0/64** | 0.46240 |
| 901 | Synthetic ARC-style heldout program families | **0/64 grids** | **0/64** | 0.79249 |
| 902 | Synthetic ARC-style | **0/64** | **0/64** | 0.99222 |
| 901 | Storyworld public-state expert action labels | **61/64 actions** | **61/64** | 0.14820 |
| 902 | Storyworld action labels | **59/64** | **59/64** | 0.07675 |

**Storyworld terminal success is 0/16 for each root at 12-step closed-loop evaluation**, despite 95.3% / 92.2% independent-state expert-action classification. The trained policy therefore fails the actual hidden-ending task. Action classification is a **shallow proxy**, with distribution shift from independently sampled public snapshots to recurrent closed-loop execution. Any BPD component evaluated on those model actions lacks a validated behavioral improvement endpoint.

Sudoku4 and synthetic ARC-style exact tasks also fail. Token accuracy, legal Sudoku boards, per-root checkpoint hashes and full logits are in the raw JSON. These negative scores must remain visible; no official ARC-AGI-1 result, 9×9 7M TRM performance or new learned BPD discovery follows from this experiment.

## Native recurrent steering and protected full head

For Sudoku4, the experiment uses the actual H/L recurrent states of the 14K-parameter native mini \`PureSudokuTRM\`, extracts a donor-minus-recipient \`z_H\` difference after depth 2 and projects it onto **the exact kernel of the full 11-vocabulary LM head across all 16 cells**. It compares no edit, full protected-dark state displacement and a rank-eight physical output direction of the Attention-TRM \`o_proj\` matrix followed by head-kernel projection.

The worst immediate full-softmax protection error was \`2.384185791015625e-7\` (901) and \`1.7881393432617188e-7\` (902), satisfying the registered ≤1e-6 engineering threshold. Current readout protection is exact to float32 numerical precision.

| Root | Depth | No-edit donor token agreement | Full-dark edit donor agreement | Physical-SVD8 donor agreement | Donor whole-board agreement |
|---:|---:|---:|---:|---:|---:|
| 901 | 4 | 24.2% | 23.4% | 25.0% | 0% all arms |
| 901 | 8 | **25.8%** | **25.8%** | **25.8%** | 0% all arms |
| 901 | 12 | **25.8%** | **25.8%** | **25.8%** | 0% all arms |
| 902 | 4 | 28.9% | 28.9% | 27.3% | 0% all arms |
| 902 | 8 | **30.5%** | **30.5%** | **30.5%** | 0% all arms |
| 902 | 12 | **30.5%** | **30.5%** | **30.5%** | 0% all arms |

The dark state source difference is nonzero (mean physical norm ≈3.02 for 901 and 2.74 for 902). The rank-eight weight-SVD edit also has nonzero dose. Nevertheless **neither edit increases donor token agreement** at unseen depths 8 or 12. This is consistent with an uninformative/flat learned recurrent model or downstream reconstruction of a poor solution. The script has *no full task-Jacobian-matched sham* and does not train a sparse BPD decomposition, so any apparent depth-specific edit differences are strictly **state-level physical SVD engineering diagnostics**.

## What CGC-02's failure changes next

CGC-02 already showed that a development-task top-rank subspace at depth 4 can fail after more loops. DT02-A shows a more basic admission issue: a mini TRM must learn an independently correct task and maintain task utility under closed-loop rollout before measuring explanatory gain from BPD steering.

1. **Raise native task competence first.** Use more updates, meaningful curricular difficulty, verified 4×4 Sudoku board exactness, and measured ARC program-family generalization, with heldout groups and compute/parameter budget frozen. The 14K-parameter, 64-update toy here is an engineering baseline.
2. **Use host outcome on storyworlds.** Train/evaluate on terminal-ending success and exact host reachability, comparing a scripted teacher, a typed planner, and model-rolled-out action sequences. Single-turn action accuracy of 95% is insufficient.
3. **Only after competence**, fit actual BPD V/U on physical shared matrices, match full normalized readout at relevant H commits and L boundaries, calibrate independent roots and benchmark signed *full-task*-Jacobian-and-dose shams at depths 8/12. Test physical block gauge invariance and finite nonlinear closure across depth.
4. **Keep official ARC-1 and the pretrained 7M Sudoku checkpoint independent.** Loading those requires explicit host gold, SHA-verified checkpoints and sufficient GPU/CPU budget. Any official ARC benchmark claim needs official source provenance and instance-level independent exact outputs.

**Decision:** engineering PASS, actual task-competence gates FAIL, long-horizon dark-fiber causal steering NOT ADMITTED, independent root-level science NOT RUN. The two roots 901/902 were excluded engineering seeds; no protected ARG-03-N cases, previously registered scientific roots, or real deployment actions were accessed.
