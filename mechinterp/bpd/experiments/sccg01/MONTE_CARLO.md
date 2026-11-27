# SCCG Monte Carlo backend (secondary, opt-in)

Default exact reference: rebuild. The optional atlas backend compiles the
union of the predeclared sampled supports and updates exact kernels using
SCCG-06. Paired mode checks nullities and exact lifted bases against rebuild
for every distinct sampled support at every epoch. Duplicate draws are
counted in the Monte Carlo tally; both backends deduplicate within epochs.
This is offline-roster structural Monte Carlo, not a streaming sampler,
a native checkpoint experiment, or a replacement for the finite oracle.

From the repository root:

    python mechinterp/bpd/experiments/sccg01/monte_carlo_atlas.py
    python mechinterp/bpd/experiments/sccg01/monte_carlo_atlas.py --mode atlas
    python mechinterp/bpd/experiments/sccg01/monte_carlo_atlas.py --mode paired --output fixed.json
    python mechinterp/bpd/experiments/sccg01/monte_carlo_atlas.py --mode paired --quotient-drift --output drift.json

The deliberate drift challenge changes one fixed-row column during epochs
5..8 and restores it at epoch 9. Any changed quotient triggers a complete
recompilation for that roster. No approximate reuse is allowed here.
The default backend remains rebuild. Existing SCCG-06 exhaustive workflow
is unchanged. The new workflow separately exercises both paired regimes.

Local constructed development run: seeds 51000..51007, 12 epochs, 256
uniform draws with replacement per epoch; 24,576 draws and 21,679 distinct
support-epoch checks in each regime. The paired arms share all samples.
Fixed: 12 positive draws in both arms, 21,679 versus 7,635 modular rank
screens, 0.202590243 versus 0.077072947 seconds for rebuild versus atlas.
Drift: 11 positives in both arms, 21,679 versus 22,905 screens, 16 quotient
rebuilds, 0.208240120 versus 0.222995598 seconds. No mismatches.
Timings are single local CPU observations with alternating root arm order,
including compilation but excluding the common fixture and external parity
verification; no significance, population or hardware speedup claim.
Locally executed Monte Carlo source Git blob f42d934c7f7dafb4e75e660937c50b52dce5cc3b.
Local imported dependency was the relevant unchanged subset of dynamic_atlas.py.
Hosted replay uses the complete repository module.
