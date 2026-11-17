# SAU-TRACE01: decodability, retained KV state, and native causal access

**Source audit and prospective translation, 2026-11-17.** This note is a source-grounded study proposal. No BPD model has been evaluated and no ARG-03-N held-out examples have been accessed.

Primary source: [SauersML/retained-reply-state](https://github.com/SauersML/retained-reply-state). Especially [hidden_choice.py](https://github.com/SauersML/retained-reply-state/blob/main/hidden_choice.py), [probe.py](https://github.com/SauersML/retained-reply-state/blob/main/probe.py), [published probes](https://github.com/SauersML/retained-reply-state/blob/main/results/probes.json), and [Qwen3-1.7B recall statistics](https://github.com/SauersML/retained-reply-state/blob/main/results/q17A/q17A_stats.txt). The implementation here is protocol text only; no upstream source is copied.

## Mechanism under test

An item (one of 50 animals or countries) is placed in the first turn's thinking. The visible final reply is always `I understand.`. The next turn asks which item was chosen. The repository implements four comparisons:

- **Stripped:** the visible reply's keys and values are recomputed without access to the hidden thinking; every test turn therefore has the same visible content.
- **Retained:** the reply's original KV tensors are retained from the first turn even though the thinking tokens' own KV entries are removed and rotary positions of the reply's keys are shifted to match their positions in the stripped text.
- **Neutral:** the same retained cache, with a subsequent question requesting *any* item. This tests generic priming.
- **Visible:** the thinking remains in context. This is a task-access positive control.

The repository also records reply-token residual vectors at each layer and trains regularized linear probes with context-held-out five-fold validation and shuffled-label controls.

### Published observations and limits

- **Linear decodability:** `results/probes.json` gives Qwen3-1.7B layer-20 accuracy **0.926**, compared with a shuffled-label mean of 0.019, for predicting a 50-way chosen item from a reply's residual. It also reports Qwen3-0.6B layer-28 accuracy 0.973, Qwen3-4B layer-36 accuracy 0.866, and Qwen3-8B layer-32 accuracy 0.861. These are probe readouts; architecture and dataset coverage vary.
- **Native use:** `results/q17A/q17A_stats.txt` reports 598- and 599-run Qwen3-1.7B cohorts. The retained-cache arm changes the selected item's target log score relative to stripped by **-0.0128** and **-0.0259** respectively, with an upper-tail randomization p near one. The visible arm changes that target score by roughly **+32.8** and **+32.4**. These numbers do not support improved native recall from merely retaining reply KV under the tested harness.
- **Causal edits:** the source includes layer/head deletion, scaling and counterfactual controls; e.g. in one 599-run selected-head edit the reported top-1 rate moves from 0.0117 to 0.0317 while remaining close to a 1/50 reference. A change in target discrimination by itself does not demonstrate useful free recall. Candidate heads were selected in the source's discovery work and require independent verification for any new claim.

The probe and retained-cache measurements use different readout objects and scripts. Their gap must be described as an **empirical decodability-vs-access distinction**, conditional on exact experimental provenance, rather than a theorem that reply residuals can never be accessed downstream.

## Relation to BPD observable quotients

In a recurrent model, distinguish the visible response `q(h_t)`, a trace function `C^trace_{t,k}`, a current-state readout `C^now_{t,k}`, the recorded intervention target `h_t`, and the continuation's behavior. Observational decoding may use arbitrary learned maps from `h_t`, while the native continuation is restricted to the model's own state transition and readout. The existence of an accurate probe `r(h_t)` does not imply that the *native* map `g(F^k(h_t))` uses the same component.

A useful source-to-BPD hypothesis is that an item can be linearly recoverable from selected residuals but absent from the native accessible subspace or suppressed by an attention routing bottleneck. A competing explanation is cache erasure or positional surgery, so state at the precise native retrieval site and attention/cross-time routing must be audited.

**Prospective experimental port (SAU-TRACE01):**

1. Construct fresh, strictly disjoint 50-way symbolic latent choices with identical visible final messages and identical future prompts. Preserve the original Recurrent/BPD case keys and all frozen splits. Run stripped, retained, neutral, visible conditions on fixed recurrent checkpoints.
2. At each physical loop/pass, harvest candidate answer-position states *after* passing the [SAU-OBS01 observation-parity checks](https://github.com/MoralityLabAI/LoopedTransformers/tree/research/sauersml-observer-invariance-20261117/mechinterp/observer_footprint_cpu_v1). A passive recorder is an experimental factor until its compiled numerical parity has been proven.
3. Fit regularized linear probes on train/validation families only, report 50-way top1, rank, conditional mutual information estimate and cluster bootstrap, and score **native** exact-answer token logits on untouched test families. Include early-prompt/control lexical leakage probes, prompt-form changes, and shuffled-label tests.
4. Intervene at the exact recurrent pass with donors chosen from same visible prompt, different hidden choice; test state replacement, BPD component-block masks and matched-rank/norm sham edits. Use bidirectional native continuations and preserve the preselected output metric.
5. Compare native current-state access and trace reconstruction at each depth with the existing `LOOPSCOPE` causal Gramian readouts. Test whether a BPD block that carries high linear information also has causal value for future item recovery.
6. Keep item families, model/checkpoint seeds, and prompts as clustering units. Test-fold selection is prohibited. Report positive controls and possible nulls, and avoid extrapolating a 50-way probe accuracy to a proof of endogenous retrieval.

**Interpretive gate:** evidence for a BPD causal-access block requires a held-out native recovery effect and exclusion of visible-text, response-token, observer, positional-cache, and uniform-prior explanations. If decoding remains high while native access stays at chance, report a measured readout-access gap rather than declare a dark-fiber mechanism proved.

### Licensing

Source methods and published results are attributed. All proposed code must be independently authored or covered by a compatible source license; no source from retained-reply-state was copied.
