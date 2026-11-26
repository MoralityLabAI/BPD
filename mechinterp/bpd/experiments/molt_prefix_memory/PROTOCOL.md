# MOLT-BPD-01: causal selection of prefix memory components

Development proposal, 2026-11-26. Source: https://arxiv.org/abs/2611.34923.
Companion research branches: DataGoblin, HybridTRMLDT, LoopedTransformers, Compactification-Theory
at `research/molt-goblin-01-20261126`. This proposal leaves all NC09, AMM09, AMM09R1 and ARG-03 seals intact.

## Hypothesis and physical implementation

For frozen checkpoint theta and a strict prefix j<i, decompose the projected value space into
a complete basis (or an explicitly reconstructed BPD factorization) with component c:

    V_j = sum_c V_{j,c} + V_{j,residual}
    V'_j = sum_c g_{i,j,c} V_{j,c} + V_{j,residual}
    z_i = sum_{j<i} softmax(q_i K_j / sqrt(d)) V'_j

Start with value interventions to isolate memory content. Compare a distinct key-routing intervention
later; it changes the attention denominator and cannot be treated as equal to value gating. Use
full-head reconstruction parity at g=1, preserve bias and undecomposed residual; report
reconstruction error and actual write norm. Select gates from training roots only, aligned across
loop visits by physical component correspondence, with a frozen prefix-local input contract.
Gate functions must not depend on the previous same-position state, future tokens, changing
pass index, or mutable global state if finite stabilization is claimed. A static basis may be
trained, but freeze it within each evaluated forward graph. Compare exact no-op and
component sign/permutation invariance as appropriate for the native factorization.

Mechanistic prediction: a source-valid donor improves delayed verified answer recovery on
corrupted recipient trajectories more than a full signed future-output-Jacobian and
write-norm matched sham, without excess damage to clean answers. A probe correlation,
future-logit capture alone or reduced damage is insufficient. Proposed minimum meaningful
root-macro causal recovery-minus-corruption gain is 5 percentage points; development
calibration may force redesign before registration. Freeze actual physical dose, residual
radius, norm tolerance, component budget, sham quantile and instrument resolution before
confirmation; do not treat the 5pp proposal as a registered threshold.

## Arms and gates

1. Native prefix KV intact, full reconstruction parity, residual-loop matched block control.
2. BPD-selected component value gate, random equal-sparsity gate, equal actual displacement
   sham, signed future-output-Jacobian matched sham, source-shuffled donor, no-memory.
3. Frozen component labels versus training-only alignment tracking; same evaluation mask and
   budget. Compare normal task training, task+targeted steering, task+sham steering.
   Compute actual parameter gradients and ordinary evaluation with steering disabled.
4. Fit on development roots only. Disclose selection count and every failed arm. Require task
   competence above exact constant/heuristic baseline before causal interpretation.

Use the EvalMath mechanistic-power skill and preflight on independent trained development roots
and paired shams; cluster case variants and repeated visits within roots. FSI-01 for correctness
pairs and root-level inference, multiplicity adjustment over the frozen comparison family.
Eight roots are variance discovery only; no confirmatory run until prospective power and
separate release. Use recovery, corruption, exact verified closure, retained path utility,
full-head parity, state displacement, cache version, GPU memory and elapsed time.

## RAM-bank and training feedback

Store versioned frozen calibration component maps, donor IDs and measured intervention receipts in
RAM. Fixed thread count and 2/4/8/16 GiB total RSS caps. Measure useful verified solutions per hour,
bank hit/miss, bytes, stale hits and recomputation. Never detach a live training activation and
credit it as equivalent gradient reuse. Fresh-cache and stale-cache arms are distinct. DataGoblin
can learn an array selector on public receipts (rank, source checkpoint, dose, lower/upper effect,
cost, qualification); the independent membrane and operator retain authority over hard updates
and execution. No actual BPD implementation, model training or causal evidence in this commit.
