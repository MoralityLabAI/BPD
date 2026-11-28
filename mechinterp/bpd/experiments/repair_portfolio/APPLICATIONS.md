# Applications for Behavioral Parameter Decomposition

## Repair after quantization

BPD exposes a physical shared weight matrix through learned read and write factors and an input-dependent component router. Quantizing those factors can change the computation at every recurrent pass. A compact set of component gate corrections could restore task behavior while retaining the compressed factors. Each proposed correction can be scored through the model's complete continuation on a labeled calibration set.

BPD-RP-01 implements this application on the repository's TinyCrossLoopRouter. It compares a single greedy search, independent continuations from several starting codes, and a beam that reallocates its candidate slots after each round. Their calibration-call budgets match. The deployment recomputes learned importance from the message reached at each pass. Four global gate offsets are the editable policy; the learned factors remain quantized and frozen.

The measured outcomes are recovered answers and newly corrupted answers on held-out draws, with calibration cross-entropy and per-pass physical displacement as diagnostics. A claim that learned BPD factors improve repair would additionally require a representation control such as a fixed SVD basis and a matched direct-weight repair family. The current gym isolates search allocation within one trained BPD representation.

## Inference-time repair with delayed evidence

A recurrent solver can reach an uncertain state before a later constraint identifies the useful path. BPD offers a way to intervene on a small set of shared components, carry several resulting states forward, and revise which branch receives more work when additional evidence arrives. This application would use a verifier that is available during deployment, such as a constraint check or a partial program execution. An offline calibration objective can train the selector, while the terminal answer remains hidden during deployment search.

A separate proposed gym, **BPD-RP-02: Delayed Pointer Repair**, would expose a source model to a sequence of writes into two addressable memory slots. Distractor writes and a later query would make the initially plausible memory choice ambiguous. The model would process the sequence through recurrent state; an intervention would act on learned shared components at the ambiguous step, then each candidate would continue its own state. A late checksum or address-consistency constraint would provide the permitted selection signal. Complete held-out answers would be scored after selection.

The root would include an independently trained model and a freshly generated address/value encoding. Cases sharing that model and encoding would stay in the same cluster. Development work must first demonstrate source competence, a physical fault that produces recoverable errors, and a valid repair witness available only to the audit. The fault dose and the comparison family would then be fixed before assigning fresh roots. One-pass damage that later recurrence repairs on its own gives a distinct control for endogenous recovery.

The core comparison would retain equal candidate-continuation calls across a greedy policy, independent branches and a dynamic beam. A matched intervention control would compare actual delivered displacement at the intervention site, then report subsequent state divergence separately. Recovery and corruption would remain paired within each root. The proposed gym requires its own preflight and has not been run here.

## Transfer between checkpoints

Shared components can provide a small editing interface when a new checkpoint changes how it routes or sustains information. Several plausible component alignments could be carried into a fixed target-calibration budget, and the target model's behavior could decide which alignment survives. This could support adapting a repair from one checkpoint to another without searching every weight independently.

The evaluation unit would be an independently trained source/target pair. Each method would receive the same target labels and query budget. A target-only repair would measure the value of transfer, while simultaneous factor/router permutations would expose changes caused only by component naming. Sign-reversed edits and controls with matched parameter distortion would add causal discrimination. State displacement would still need its own measurement because equal parameter norms do not imply equal effects in a recurrent model.

## Fault localization across recurrent depth

BPD's component-by-pass interventions can distinguish a transient disturbance that later computation repairs from a component whose repeated contribution sustains the answer. This can guide a repair toward the pass where the error first becomes persistent. An engineering use would be to localize a deployment regression or decide which portion of a recurrent calculation needs higher precision.

A suitable gym would contain several independently trained recurrent solvers and a disclosed physical fault family. The audit would compare damage from a single-pass intervention with the same intervention repeated across depth, using complete solver outputs and per-pass state measurements. A repair would have to improve terminal task outcomes under a fixed budget. Intermediate logit movement alone would remain a diagnostic.

## What the current source supports

The BPD core uses one learned V/U pair for each wrapped physical matrix and supports a pass-shared adversary. Its trainer computes learned importance from the clean reference trajectory before masked reconstruction. RP-01 adds a deployment hook that recomputes importance on each reached pre-mix input. This conditioning change receives an explicit fidelity check.

The current pilot provides an engineering fixture for quantization repair. The other applications above are proposed experiments. Their outcomes require independent empirical tests, and exact scalar LS recurrence certificates supply no certificate for these trained multidimensional models.

## Primary source anchors

- [BPD components](https://github.com/MoralityLabAI/BPD/blob/6d3ca23a817095f38789751004594777f54cf7d8/mechinterp/bpd/components.py)
- [BPD trainer](https://github.com/MoralityLabAI/BPD/blob/6d3ca23a817095f38789751004594777f54cf7d8/mechinterp/bpd/trainer.py)
- [Component intervention analysis](https://github.com/MoralityLabAI/BPD/blob/6d3ca23a817095f38789751004594777f54cf7d8/mechinterp/bpd/analysis.py)
- [TinyCrossLoopRouter fixture](https://github.com/MoralityLabAI/BPD/blob/6d3ca23a817095f38789751004594777f54cf7d8/mechinterp/bpd/experiments/inkling_routing/native_model.py)

Source files and hashes are pinned in evidence/upstream_sources.json. The experiment's measured results are reported separately in RESULTS.md.
