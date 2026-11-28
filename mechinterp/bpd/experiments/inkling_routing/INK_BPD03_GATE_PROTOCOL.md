# INK-BPD-03: native gate-inversion competence control

**Registered 21 November 2026 before any INK-BPD-03 gate-intervention development readout.** This is a **post-INK-BPD-02 engineering follow-up**, motivated by the disclosed 384/384 clean/target/sham ceiling in INK-BPD-02. Consequently the INK-BPD-03 run is developmental and cannot convert the previously viewed population into a prospective confirmation result.

## Frozen contrast

Use the same source-trained `TinyCrossLoopRouter` weights (seed 2701, 260 AdamW steps, 16D, 3 physical shared passes). Generate new synthetic task roots seeded **303701 through 303712** (12 roots, 32 independent cases each).

For each root, run three native continuations without any patch to writer tensors:

1. **Clean**: regular learned softmax gate at each recurrent pass.
2. **One-pass gate inversion**: at pass index 1 only, replace the native two-slot softmax gate `(beta2,beta3)` with `(beta3,beta2)`; remaining shared passes proceed normally.
3. **All-pass gate inversion**: apply the same gate reversal at all three passes, recomputing every intermediate state and final logits.

The intervention is **directly on the router gate after softmax**, so it is a deliberately strong positive control. It is *not* a matched-physical-write test and cannot be compared as if it had the same 0.16 L2 dose as INK-BPD-02. The one-pass versus all-pass comparison tests the BPD transient-versus-sustaining *instrument sensitivity*, with the clean task performance and final classification readout.

## Development readouts

Primary: mean root-level final task accuracy for clean, one-pass inversion and all-pass inversion. Also report the fraction of originally correct clean predictions corrupted, and native final-logit displacement L2. Preserve the full denominator of 384 cases even if a control fails. Root IDs have independent task-generator seeds but share one source-trained checkpoint, giving **one model-weight sample**.

This demonstrates only whether the synthetic training objective actually depends on the selector and whether later passes can reconstruct correct answers after a transient disruption. It does not identify Goodfire's Inkling outlier, prove a BPD factor is causally necessary, or measure real-world model behavior.

Any subsequent physical writer dose sweep must have its own source-only registration, complete target/sham control admission, and new development task roots. Protect the original INK-BPD-02 receipts and the separate sealed BPD/LoopTransformers populations.

## Command

```bash
python mechinterp/bpd/experiments/inkling_routing/run_gate_control.py --out /tmp/inkbpd03_gate_dev.json
python mechinterp/bpd/experiments/inkling_routing/run_gate_control.py --out /tmp/inkbpd03_gate_dev.json --check
```
