# AMM-10: dissection of exposed causal strategy memory

The [analysis plan](ANALYSIS_PLAN.md) was committed before the AMM-10
ablation scores were computed. This reuses **previously published**
AMM-09R1 TRAINING and DEVELOPMENT roots exclusively for explanatory
analysis. The study has no new prospective scientific claim and no
confirmation runner.

## Questions

The original 18-feature linear strategy-memory policy receives six equal-dose
rank-at-most-two component-mask alternatives. Seven of its features are
candidate-invariant, while two raw delayed readouts are shifted versions of
existing candidate shift features. The [standalone analyzer](analysis.py)
checks the exact **nine-dimensional effective action ranking** on every
exposed training case, including the effects of clipping.

Nine fixed feature-set ablations are retrained using **eight-fold
root-held-out cross-fitting**, with each of the 64 exposed training model
roots held out once. They share the original ridge λ=10, feature scale,
six masks and outcome definition. Six source-safe geometric heuristic rules
and pooled/family-pooled fixed-mask controls are evaluated. The
root-hindsight fixed mask and per-case test-label oracle are **audit-only
upper bounds**.

The original development archive deliberately has **no heldout feature
vectors**, so this study analyzes the frozen original selected-action bits
and counterfactual fixed-mask outcomes only. Development feature ablations,
new policy predictions, or fresh model training are outside scope.

## Reproduce from immutable archived GitHub Actions artifacts

Python 3.12 with NumPy 2.2.6. Download the original artifacts:
- [Training run 44294944939](https://github.com/MoralityLabAI/BPD/actions/runs/44294944939), \`amm09r1-training-ubuntu-22.04\` → \`training.json\`.
- [Development run 44295607653](https://github.com/MoralityLabAI/BPD/actions/runs/44295607653), \`amm09r1-development-source-locked\` → \`development.json\`.

\`\`\`bash
PYTHONPATH=mechinterp/bpd/experiments/amm10_exposed_dissection \
 python -m unittest test_analysis -v

python mechinterp/bpd/experiments/amm10_exposed_dissection/analysis.py \
 --training /path/to/training.json \
 --development /path/to/development.json \
 --out /tmp/amm10_exposed_results.json
\`\`\`

The analyzer verifies original raw SHA-256, exact root/phase/family/order
memberships, six permitted actions, training feature hashes, heldout
case outcome reconstruction, and original AMM09R1 policy totals. It
discloses all refits and analytic rules; root-bootstrap intervals are
**descriptive** because all model-root populations are selection-exposed.
The dedicated [workflow](../../../../.github/workflows/bpd-amm10-exposed-dissection.yml)
downloads the original registered archives and reconstructs the locked
\`results/ANALYSIS.json\` without running a single trained model.

Scientific boundary: 64 training model roots (128 donor cases/root)
and 64 already-scored development roots (512 donor cases/root).
The original AMM09R1 +8pp meaningful effect, pooled-repeatability sham
and corrected EvalMath planning gate belong to the original
**prospective** experiment. AMM10 introduces no independent shams
or new scientific admission verdict.

All 128 AMM09R1 confirmation roots and earlier protected cohorts
remain sealed. Results remain restricted to synthetic trained 6D
rank-three linear recurrence; any chosen simpler geometric policy
requires fresh model-root prospective validation.
