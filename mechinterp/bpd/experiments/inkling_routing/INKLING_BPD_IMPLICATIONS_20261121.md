# INK-BPD-01: routing, content and loop-clock mechanisms in Inkling-style models

**Research design / source-only mathematical instrument, 21 November 2026.** This is a BPD interpretation and preregistration proposal, not an executed Inkling causal study. No weights, private traces, HF secrets or model activations are required. Original NC-13, NEC-03Q and LOOPSCOPE-03 claims and source/confirmation cohorts remain unchanged.

## Evidence and today's integration points

* [Thinking Machines, *Introducing Inkling 2 Mini*, 20 Nov 2026](https://thinkingmachines.ai/news/inkling-2-mini/). Inkling 2 Mini uses 24 stored layers in four six-layer blocks. Only the two middle blocks are looped, with three passes, and each core layer's attention-residual input can mix previous-layer block outputs with a prior-loop message. The variable-loop research sibling supports 1 to 5 trained passes and up to 10 tested passes, with reported no-CoT serial-reasoning gains. The report does not establish that larger loop count necessarily worsens CoT monitorability.
* [Goodfire, *Interpreting Inkling 2 Mini*, 20 Nov 2026](https://www.goodfire.com/research/interpreting-inkling-2-mini/). It decomposes block-reader inputs into contributions from individual previous-layer/loop writes, observes writer-reader alignment concentrated at block boundaries, isolates an unusually large coordinate (2027, d=2304), and compares logit, tuned and coda lenses. The normalized absolute activation 32.4 represents about 46% of squared RMS-normalized norm; the authors **hypothesize** a cross-loop routing role. This is observational attribution and evidence of a candidate instrument, not a surgical causal necessity result.
* [Kimi Team, *Attention Residuals*, arXiv:2603.15031](https://arxiv.org/abs/2603.15031). Input-dependent mixtures across previous depth blocks motivate an explicit router. Inkling's cross-loop gate adds an interaction beyond a simple sum of residual writers.
* [Wu, Hu & Zhang, *Shared Weights, Selected Computations*, arXiv:2609.39892](https://arxiv.org/abs/2609.39892). A frozen recurrent block can execute different learned graph transitions when its entering state is steered. The learned steering map does not create capabilities outside the backbone by fiat.
* [Wang & Reid, *Looped Transformers under the Jacobian Lens*, arXiv:2609.01924](https://arxiv.org/abs/2609.01924). Ouro and Huginn differ in how workspace content persists across recurrences, motivating explicit intervention-window tests.
* [Monea et al., *Shared Memory in Looped Transformers*, arXiv:2610.02383](https://arxiv.org/abs/2610.02383). Shared early-loop KV memory can change gradient flow and later-loop utilization, a separate confound for putative recurrent component persistence.

**Today, 21 Nov:** [LoopedTransformers NEC-03Q merge `d4ef3fa`](https://github.com/MoralityLabAI/LoopedTransformers/commit/d4ef3fa1ba68ca5e493a9d5447d97a1d1d7c0d98) recorded **+0.016276 percentage points** source SAE-versus-sham difference-in-differences on an eight-world development ITT, well under +2 points; exact exploratory sign-flip p=1.0, original six causal roots audited. [NEC-03P `5fab770`](https://github.com/MoralityLabAI/LoopedTransformers/commit/5fab770dd676cfc1a83331e1200a0d2e39a13994) recruited 69 paired source-eligible roots from 88, though their causal sham tail is unmeasured. [BPD NC-13 report](../../nc13/NEXT_STEP_REVIEW.md) found every original D endpoint ineligible from unmatched delivered displacements. [BPD LOOPSCOPE-03](../../../loopscope03/README.md) already records three-ledger `V_t/C_t/R_t` and path-independence tests.

## Mathematical transfer into BPD

For two previous-loop block outputs `v_2,v_3`, an idealized Inkling-style reader with weight `u` computes

\[
\hat v_i=\frac{v_i}{\operatorname{RMS}(v_i)},\qquad
s_i=\frac{u^\top\hat v_i}{\sqrt d},\quad
\beta_3=\sigma(s_3-s_2),\quad m=\beta_2\hat v_2+\beta_3\hat v_3.
\]

The layer reads `x'=kappa * x + gamma * m`, with elementwise learned keep/gain coefficients. A content writer's path has a direct effect on `m` and an indirect effect through the softmax selector. Factorizing one physical weight `W` into BPD components `V,U` must therefore distinguish **what a component writes** from **which previous-loop block receives attention**. A large coordinate may gate a read without directly carrying a semantic task bit.

For the chosen, idealized graph with block 2 frozen and block 3 altered from `v_3` to `v_3'`, the message change has an *exact telescoping decomposition*:

\[
\Delta m = (\beta'_3-\beta_3)(\hat v_3-\hat v_2)
+\beta_3(\hat v'_3-\hat v_3)
+(\beta'_3-\beta_3)(\hat v'_3-\hat v_3).
\]

The terms are respectively a **selector-only path**, a **content-only path**, and their interaction. This identity is exact for the *specified patched graph*, before any downstream re-execution, gate recalculation, or nonlinear continuation. It is a candidate source-specific mediation decomposition, not a general identification theorem for a learned network.

Goodfire's observable write attribution uses `x'=sum_w c_w` and projective shares `s_w=<c_w,x'>/||x'||^2`. These signed shares sum to 1 when the decomposition includes every contribution, including residual/bias terms. Signed contributions can cancel, and a large share is not a causal effect. `routing_math.py` validates the identity and the two-channel message split.

## Instrument choice and controls

**R1, message-routing necessity:** instrument source write coordinates in the last layer of each recurrent block and their subsequent receiver sublayers. Track observed `s3-s2`, `beta3`, `kappa`, `gamma`, and the final native task objective. A single coordinate's normalized magnitude, writer-reader cosine and projective share are exploratory rankers. Preselect the candidate on **disjoint source roots**, including the observed coordinate 2027 for the specific Inkling checkpoint only. Verify tensor layouts and actual hook sites before patching.

**R2, delivered-dose matching:** for all candidate, sham and donor interventions, record physical block-output L2 displacement, RMS, attention gate movement, native prelude output, task accuracy and future-logit margin. `rotate_pair_at_fixed_dose` supplies a *synthetic* exact L2/RMS-matched coordinate rotation control. Real-model matching must use native physical activations, compare the post-hook state, and audit every comparator's admissibility. A matched activation norm by itself does not imply matched displacement or matched gate movement. If eight required shams cannot be admitted, declare the causal endpoint ineligible, as in NC-13.

**R3, selector vs content:** at a fixed previous-loop state, replay (a) full patch, (b) changed selector/gates with original normalized content, (c) patched content with original selector, and (d) a matched-norm, matched-displacement sham. Measure selector-only, content-only and interaction contributions before and after **actual native continuation**. Intervening on a reader and on a writer are distinct physical manipulations with different information access. Change one variable at a time while retaining a frozen matched control family.

**R4, clock versus message:** the last loop's coordinator coordinate may have a different sign simply because no future loop is scheduled. Test a continuation-aware stop-clock indicator and variable recurrence depth as separate factors. Use BPD's `state_only` / `state_plus_clock` gates and a matched pass-count control. Avoid inferring a future-loop flag from an observed post-hoc sign flip alone.

**R5, readout-lens vs behavioral necessity:** evaluate logit lens, calibrated tuned lens, and the **whole-context** coda lens on the same model, depth and sources. Lens capture belongs to observational evidence. Only native continuation at the same physical dose can certify downstream effect. Tuned-lens fitting must use independent development roots. Carry source hashes and all failed lens/readout arms in the archive.

**R6, monitorability:** re-run no-CoT tasks, CoT length controls, and a tool-only versus tool+text monitor under fixed false-positive calibration. Record native task success and whether the candidate intervention changes monitor detection. Goodfire's model-specific coordination observation alone cannot imply more covert computation or decreased monitorability.

## Falsifiable protocol and statistical admission

The primary *future* endpoint should be a native paired task metric on independently trained model/world roots, e.g. net task-success difference between a physically matched selected-router patch and a registered sham. Maintain both errors corrected and originally correct answers corrupted; require a source-specific effect over calibrated matched-sham and measurement floors. Secondary outcomes are gate weight movement, cross-loop message mediation, BPD projector fidelity, coda-lens capture, per-pass schedule, pathwise conditional refill and monitor FPR.

Register the compared set before accessing fresh target outcomes. Calibrate the 95th-percentile/95%-confidence sham tolerance bound from **at least 59 independent sham roots**, assuming independent source roots. Set `delta_mech = max(delta_scientific, matched_sham_floor, measurement_resolution)`; use EvalMath FSI-01 and a complete Holm family. A source-only 6/8 or 69/88 candidate pool does not create measured 59-world causal sham effects. Freeze a fresh model checkpoint family and benchmark test roots prior to post-selection. Retain aborted and unmatched-dose cases in the denominator and artifact list.

### Proposed first execution gate

Start with independent, public synthetic/cyclic-copy development worlds and an exposed small looped checkpoint supporting intermediate hooks. Limit initial work to a small CPU/GPU smoke whose receipt must demonstrate **actual hook semantics, complete dose-control admission and exact native replay**, then run a powered development cohort. Proceed to Inkling after activation-access permission, hook compatibility, precision, time budget and model-weight provenance are independently verified. The public 20GB deployed Inkling Mini weight footprint exceeds a standalone 16GB GPU, so plan explicit CPU offload or a verified remote trace route if needed. No paid server or new GPU training is authorized by this note.

## Evidence classes

* **Reported external observations:** TML model and Goodfire architecture/readout studies.
* **Exact source-only identities:** two-block routing-logit decomposition, message three-term split, signed projective-share unit sum, L2/RMS-preserving synthetic rotation.
* **Negative internal evidence:** NEC-03Q's matched SAE source/sham result, NC-13 D admission failures. These concern different models and treatment definitions.
* **Untested hypotheses:** coordinate-2027 physical necessity, selector-versus-content mediation, loop-clock encoding, BPD cross-model transport to Inkling, change in real agent monitorability.

Never translate a source-only identity into native-mechanism confirmation.
