# INK-BPD-02: native PyTorch routing continuation pilot

**Frozen source/engineering registration, 21 November 2026.** Development-only, synthetic learned router. It is a real PyTorch nn.Module with recurrent reused weights and a physical activation hook. It does not use Inkling 2 Mini weights, Goodfire's unpublished activations, pretrained Huginn, or any confirmation cohort.

## Source model and task

Two content bits and a query for which slot to read form a fixed-size independent synthetic task. `TinyCrossLoopRouter` has a writer projection invoked at **every recurrent pass**, a query-dependent two-slot softmax read, a state-feedback projection, a shared transition core, and a final binary head. Train source data using seed 2701 and 260 AdamW updates on independently generated batches. The source model is evaluated for source validation once. The checkpoint digest is a SHA-256 of the sorted physical tensor weights; no checkpoint is claimed pretrained.

## Frozen source-selected physical intervention

At source seed **102704** (96 calibration tasks, never reused as development roots), apply candidate two-coordinate **orthogonal rotations** with delivered vector L2 dose **0.16** to the second writer output immediately before the reader at pass index 1 (the second pass). The source-only ranking uses the **mean absolute gate-weight displacement** and a physical admission threshold of 98% for candidate pairs. Select the most gate-moving coordinate pair; select the least gate-moving **disjoint** eligible pair as matched sham. This selects on gate movement alone, before the new development task labels are scored. The sham is a weak-gate null by construction, so its effect does not establish specificity against all possible alternative coordinate pairs.

The patched physical writer vector retains its L2 norm and hence its RMS. Both arm patches change its physical coordinates by the same delivered L2 distance. At every actual forward call, the altered writer state propagates through the real shared core and subsequent recurrent steps. The runner audits delivered dose and norm for every jointly admitted case.

**Five arms:** clean, target full, disjoint sham full, selector-only (patched selector with original writer content), content-only (patched content with clean selector). Selector/content ablations operate at the same hook pass, and later layers continue normally. They are proposed intervention-graph controls rather than an established natural direct/indirect effect identification.

## Development population and scoring

Test roots **202701..202712** (12 roots x 32 cases = 384 assigned). A root is an independently drawn task batch from the generator while all roots share **one trained checkpoint**. Track all assigned and joint dose-admitted cases; never discard missing controls silently. Primary descriptive readout is root-macro final binary accuracy difference: target full minus sham full. Secondary endpoints include clean task competence, target/sham gate changes, final logit distances, corrected and corrupted clean answers, selector-only and content-only task success.

The root count is below EvalMath's 59 independent sham-root quantile calibration minimum. Model-weight replication has n=1. Report any causal difference as synthetic developmental evidence. A future preregistered native-model replication must add independent trained checkpoints, an actual matched-sham variation floor and a sealed held-out task family.

## Reproduction

```bash
# Use a CPU-only PyTorch install. Do not require GPU or real pretrained weights.
python -m unittest discover -s mechinterp/bpd/experiments/inkling_routing -p 'test_native_model.py' -v
python mechinterp/bpd/experiments/inkling_routing/run_native_dev.py --out /tmp/ink_bpd02.json
python mechinterp/bpd/experiments/inkling_routing/run_native_dev.py --out /tmp/ink_bpd02.json --check
```

The CI artifact preserves per-root admissions, all five conditions, source selection, source-validation accuracy and checkpoint tensor SHA. The original INK-BPD-01 source-only results and all BPD target cohorts are untouched. Next step is native hooks into an actual accessible pretrained looped model, with source-only calibration, complete physical sham admission and source-locked continuation interfaces.
