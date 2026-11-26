# Goodfire looped-transformer review and BPD update

**Reviewed 26 November 2026.** Repository baseline:
[`6d3ca23a`](https://github.com/MoralityLabAI/BPD/commit/6d3ca23a817095f38789751004594777f54cf7d8).
The code accompanying this review is
[INK-BPD-06](../experiments/inkling_component_flow/README.md).
Its evidence class is source instrumentation, checked with constructed
PyTorch fixtures. Pretrained-model outcomes remain pending.

## Current source picture

Goodfire's [research index](https://www.goodfire.com/research/), checked on
26 November, lists **Interpreting Inkling 2 Mini**, dated **20 November 2026**,
as its latest study directly about looped transformers. The existing BPD
Inkling campaign already cites that article.

| Primary source | Finding used in this update |
|---|---|
| [Interpreting Inkling 2 Mini](https://www.goodfire.com/research/interpreting-inkling-2-mini/), 20 November | Signed write contributions reconstruct reader inputs. Alignment concentrates at block boundaries. Layer 18 writes coordinate 2027 positively in the first two loops and negatively in the last. Goodfire proposes a routing role. Tuned and coda lenses decode intermediate states better than the direct logit lens; coda evaluation uses the full context. |
| [Discovering Neuron Communities](https://www.goodfire.com/research/neuron-communities), 17 November | Co-attribution nominates neuron groups. Removal and retention interventions assess their contributions on held-out prompts. |
| [Neurons as an Interpretable Basis](https://www.goodfire.com/research/neurons-as-an-interpretable-basis), 10 November | On Qwen3.6-35B-A3B, filtering 92.2% of routed-neuron activations preserves much of ordinary benchmark performance. Generated reasoning can deteriorate; subsequent SFT and RL repair it. Shared experts remain available. |
| [Shaping and Monitoring Models Through Their Internal Representations](https://www.goodfire.com/research/mechanistic-shaping), 5 November | Internal concepts can enter training penalties. Directions can be refreshed as representations move, and monitorability needs reassessment after training. |

For the architecture and model-specific execution contract, use
[Thinking Machines' release](https://thinkingmachines.ai/news/inkling-2-mini/)
and the upstream code pinned by
[INK-BPD-05](../experiments/inkling_routing/INK_BPD05_PRETRAINED_PROTOCOL.md).
The sparsity comparison needed for BPD would vary loop count within a
controlled model family and hold the ablation budget fixed. The publications
above leave that causal comparison open.

## What BPD already contains

| Existing work | Implemented scope | Evidence available |
|---|---|---|
| [INK-BPD-01](../experiments/inkling_routing/INKLING_BPD_IMPLICATIONS_20261121.md) | Routing/content algebra, signed projection shares, physical dose controls | Mathematical fixtures |
| [INK-BPD-02](../experiments/inkling_routing/RESULTS_INKBPD02_DEV_20261121.md) | A writer patch in a trained synthetic recurrent router | Selected and sham edits both retain 384/384 correct final answers on one checkpoint |
| [INK-BPD-03](../experiments/inkling_routing/RESULTS_INKBPD03_GATE_DEV_20261121.md) | A direct gate control with native continuation | Single-pass inversion produces 92 intermediate errors that later recover; all-pass inversion leaves 197 final errors |
| [INK-BPD-04](../experiments/inkling_routing/INK_BPD04_TANGENT_PROTOCOL.md) | Reader derivatives with idealized normalization | Algebra, finite differences and synthetic native parity |
| [INK-BPD-05](../experiments/inkling_routing/INK_BPD05_PRETRAINED_PROTOCOL.md) | An offline pretrained read-edge hook with native epsilon, four message types, source checks and dose admission | Hook/interface tests; full checkpoint execution pending |
| [BPD core](../components.py) | Shared factors, execution-indexed masks and residual weights | Native `TiedComponentLinear` and its existing tests |

INK-BPD-06 fills the interface between a physical component write and a
named reader. It accepts separately observed states, validates their
reconstruction, then computes the effect of each supplied write at that
reader. Existing Inkling source files and registered recipes retain their
published bytes.

The reader inventory needs explicit indices. Physical layer 13 in the
article's one-based numbering corresponds to loop reader index 6. The
existing INK-BPD-05 registered readout uses index 11, physical layer 18.
Both have a place in an inventory; their identities must survive reporting.

## What the current BPD results change

The recent steering work lives on separate development branches. This
review uses the exact published snapshots below.

**RP03**, commit
[`402ef776`](https://github.com/MoralityLabAI/BPD/blob/402ef7764e11bc7736873d66357b8dc24b996e94/mechinterp/bpd/experiments/training_steering_stability/RESULTS.md),
compares four training arms across eight independent model roots. Its fixed
causal projector scores 3,248/4,096 ordinary delay-31 answers, against
3,243/4,096 for ordinary task training. The effect is +0.1221 percentage
points. All three Holm-adjusted p-values equal 1. The final fixed/current
edit cosine averages 0.99609, while the fixed edit still corrupts 192 of
810 initially correct unaffected answers.

The archived EvalMath input was rerun for this review using EvalMath commit
`958b3fc797cdc8fda927eff153d056b5e65be861`. The result matches the archived
output exactly: `REDESIGN_OR_PILOT_ONLY`. The
[recheck receipt](../experiments/inkling_component_flow/evidence/rp03_preflight_recheck.json)
records both input hashes and the planner's source hash. It adds zero
development outcomes.

**RP04**, commit
[`aa4b55d8`](https://github.com/MoralityLabAI/BPD/blob/aa4b55d8c968084c1414846441cb539356ba7995/mechinterp/bpd/experiments/component_selectivity/RESULTS.md),
examines those exposed checkpoints with a fresh development draw. All
state directions in the following table have L2 dose 0.75.

| Final B intervention | Targeted accuracy | Unaffected correct answers corrupted |
|---|---:|---:|
| Donor displacement in the fixed span | 22.36% | 22.45% |
| True single-slot hybrid displacement in the fixed span | 31.05% | 16.68% |
| Hybrid displacement isolating selected reads | 10.16% | 5.85% |
| Full-space hybrid displacement | 39.01% | 9.69% |

The selected-read isolation treatment delivers 58.23% of the live-span
selected-read amplitude at that same state dose. The desired hybrid gate
write has a 0.77109 relative residual outside the chosen U-row span.
RP04 retains `DEVELOPMENT_ONLY / REDESIGN_OR_PILOT_ONLY`; the independent
sample consists of eight exposed model roots.

These results put donor specificity and delivered response near the front
of the next design. A stable projector can still omit much of the desired
change. Constraining other reads also changes how much useful signal reaches
the selected receiver. A reader audit makes that cost visible before a
training sweep.

RP03/RP04 use recurrence through sequence time, including distractor inputs.
Inkling loops its core in depth. INK-BPD-06 carries an explicit recurrence
kind so these executions keep their separate meanings.

## A role defined by its recipient

The following derivation is the BPD extension proposed here. For one clean
native linear invocation, the repository's convention is

\[
W_{\rm target}=(VU)^\top+\Delta W,
\qquad a=xV,
\qquad w_c=a_cU_c.
\]

The observed output must satisfy

\[
y_{\rm observed}=\sum_c w_c+x\Delta W^\top+b.
\]

Each term keeps its identity. A completed block can also contain other
writes, supplied explicitly as a remainder. Its independently captured
state provides a second reconstruction check. The first implementation
accepts clean target mode and rejects active component schedules or
interventions. Extraction leaves the wrapper's statistics unchanged.

For a reader query \(q\), normalization scale \(g\), width \(d\) and
epsilon \(\epsilon\), define \(u=q\odot g\) and

\[
\rho(v)=\sqrt{\|v\|^2+d\epsilon},
\qquad s(v)=\frac{u^\top v}{\rho(v)},
\qquad \beta_3=\sigma(s(v_3)-s(v_2)).
\]

Differentiating with respect to the raw block state gives

\[
J_3=\nabla_{v_3}s(v_3)
=\frac{u}{\rho_3}-\frac{(u^\top v_3)v_3}{\rho_3^3}.
\]

The quantity

\[
R_{r,t,c}=J_{r,t}^\top w_{t,c}
\]

measures a local role: the change in reader \(r\)'s score when the
coefficient of component \(c\)'s observed write increases at execution
\(t\). The corresponding probability derivative is
\(\beta_3(1-\beta_3)R_{r,t,c}\). Signed activation enters through
\(w_{t,c}\), so identical factor weights can have opposite effects on two
executions.

This coefficient intervention holds the residual write fixed. Updating
the factors and recomputing their exact residual instead preserves the
clean target operator by construction. The distinction defines which
physical operation the reported derivative describes.

A term's fixed-state score contribution is a different quantity,
\(u^\top w_c/\rho_3\). Deleting a physical write also changes the
normalizer. INK-BPD-06 reports both quantities and recomputes the finite
reader after an edit. The gap between a derivative prediction and that
finite result measures local linearization error.

The radial derivative provides a useful check:

\[
J_3^\top v_3=\frac{(u^\top v_3)d\epsilon}{\rho_3^3}.
\]

It vanishes in the idealized zero-epsilon limit. For positive epsilon, the
formula also handles a zero block state. A projection share divides by
the observed state's squared norm and therefore needs its own validity
flag near zero. Negative shares and values above one remain intact.

### Gauge and causal scope

Replacing \(V_c\) by \(kV_c\) and \(U_c\) by \(U_c/k\), for any nonzero
\(k\), preserves the physical write and its reader effect. General mixing
of component coordinates preserves the summed operator but can redistribute
individual entries. Component labels stay local to their factorization.

An edit at a completed block boundary fixes the other recorded writes.
Changing an earlier source writer can alter every intervening computation.
The instrument names the boundary treatment, while a source-writer claim
requires its own hook and full native continuation. Message keep/gain gates,
later layers and task outcomes lie beyond the two-block reader calculation.

## Training proposal informed by the review

The proposed target is a useful response at a particular execution and
recipient. Training can then preserve a memory read on one phase and allow
a clearing operation on another. A possible auxiliary objective measures
the deviation of a selected response vector from a task-conditioned target,
with a separate penalty for unwanted receiver changes:

\[
\mathcal L=\mathcal L_{task}
+\lambda\|R_{selected}-R^*_{phase,task}\|^2
+\mu\|\Delta R_{unaffected}\|^2.
\]

This is a design hypothesis. The instrument deliberately detaches its
observations, so a differentiable training implementation still needs to
be built and checked. Grouping components by the recipients they affect is
also a candidate for the existing neuron-community work. Group membership
would be selected on training data and tested through physical interventions.

The proposed training comparison and its prerequisites are recorded in
[PROTOCOL.md](../experiments/inkling_component_flow/PROTOCOL.md).
Receiver-conditioned single-slot donors address the specific contamination
observed in RP04. Matched state dose remains part of the design; selected
reader amplitude and physical write amplitude are additional measurements.

For sparsity experiments, record the ordinary answer score together with
completion behavior. Runtime and memory require direct measurement for
each backend. A component budget alone supplies neither quantity. The
reader instrument processes one selected site at a time, allowing CPU
analysis of captured vectors while the model performs its forward pass
on the chosen device.

## Delivery and remaining empirical work

INK-BPD-06 adds native factor accounting, named reader inventories and
finite boundary-edit diagnostics. Its tests exercise real
`TiedComponentLinear` instances and compare reader derivatives against
autograd and finite differences. An independent empirical effect estimate
for the new method awaits development data.

The first empirical integration should capture a known competent model's
clean writer and block states, establish observed-output closure, then
evaluate matched boundary edits through full native continuation. The
existing INK-BPD-05 runner supplies the starting point for a locally
provisioned pretrained Inkling checkpoint. A new prospective adapter is
needed to connect its captures to these factor records and evaluate the
expanded reader inventory.
