# BPD-EBC-01: expert support, physical components and recurrent utility

**2026-11-27. Source-only mathematical instrument and proposed development protocol.** Extends [INK-BPD-04](INK_BPD04_TANGENT_PROTOCOL.md) and [INK-BPD-06](../inkling_component_flow/PROTOCOL.md), [EBC-01](https://github.com/MoralityLabAI/Compactification-Theory/pull/313), and [INK-CHANT-01](https://github.com/MoralityLabAI/LoopedTransformers/pull/417). No native pretrained weights, trained roots, or confirmation data were opened. Existing seals remain intact.

## 1. Physical BPD decomposition

The physical router matrix is W=sum_j u_j v_j^T + Delta W. For a fixed native router input h, component j writes the entire logit vector c_j=u_j(v_j^T h). Include residual and bias in closure; reciprocal rescaling of factors leaves each write unchanged. The same physical factors are reused across pass visits. A router-only factor gain intervention changes logits; a physical upstream write can change logits and content simultaneously. Distinguish these interventions, and capture the actual normalized router input with complete graph-slot identity. In a nonlinear expert MLP, linear projection components cannot simply be added as independent post-activation outputs; intervene through the actual nonlinear computation.

With native sigmoid scores s_e, selection ranks s_e+b_e, and unbiased mixture probabilities p_e=s_e/Z for the selected top-k, define v_e as the expert output at that same input. Let y=g sum p_e v_e. The derivative for scalar gain alpha with logit write c, expert-output derivative d_e and adjoint lambda on y is

    dU/dalpha = (g/Z) sum_{e in top-k} s_e(1-s_e)c_e lambda^T(v_e-vbar)
               + g sum_{e in top-k} p_e lambda^T d_e

where vbar=sum p_e v_e. Derive by differentiating the quotient and applying the chain rule. Use only within a strict fixed-membership region or a fixed support of exactly k. With a shared factor applied at multiple visits, sum the local terms with the true downstream adjoint at each physical invocation; for upstream interventions use the full state Jacobian and avoid double counting. The local derivative is signed, and a predicted utility gain is not evidence of a verified task repair. A task loss can substitute for utility with its appropriate sign. Equal expert outputs imply zero selector-only derivative even if logits change dramatically.

## 2. Robust allowed support under declared component interventions

For independent gain changes alpha_j in [a_j,b_j], fixed native state and physical component logit writes c_{j,e}, compute

    L_e = z_e + sum_j min(a_j c_{j,e}, b_j c_{j,e})
    H_e = z_e + sum_j max(a_j c_{j,e}, b_j c_{j,e})

Native ranking belongs to [sigmoid(L_e)+bias_e, sigmoid(H_e)+bias_e]. Let A be the original full-support top-k and S the candidate allowed support. If A subset S and every lower bound for an e in A strictly exceeds every upper bound for a rival in S minus A, the masked perturbed router selects precisely A for all gains in the box. Proof: sigmoid is monotone and each affine logit contribution is bounded independently. This is a sufficient real-arithmetic bound; correlations can make it conservative. The optional slack parameter is a caller-supplied per-score error allowance, not verified interval floating arithmetic. Ties require a separate explicit native convention. When S=A, membership is forced; this says nothing about fidelity to an unrestricted perturbed router. Choose S=all experts to certify full-support route invariance. Even with identical membership the mixture can shift, and recurrent state perturbations require separate full-state tube bounds. Native quantization and index parity are mandatory.

## 3. Finite path split

At a specified boundary and intervention graph, take independently supplied normalized routing vectors p0,p1 over original IDs and content vectors V0,V1. Then the output difference divided by g equals

    (p1-p0)V0 + p0(V1-V0) + (p1-p0)(V1-V0).

These are selector, content and interaction terms. The reference constructs all four mixtures, including experts that receive nonzero weights only in one arm. For true native intervention outcomes, implement and audit each counterfactual hook, run full continuation and score the actual verifier. Do not assume teacher-forced expert outputs are reachable under the patched recurrent state. General TV=sum abs(p1-p0)/2 remains valid when shared scores drift; EBC-01's simplified common-mass formula applies only at the same scores. Local output error for fixed content is bounded by g*diameter(V0)*TV.

## 4. Candidate selection and new development gym

Use independent train roots with full-support quantized expert traces and actual downstream adjoints, source-hashed factors and residual closure. Train candidate support from complete expert coalitions, optionally weight each visit by nonnegative train-only predicted harm (e.g. clipped absolute task-adjoint intervention effect). Test the incremental benefit of this weighting; exact full-route coverage remains a different objective from task regret. A teacher trace is off-policy after a route miss. Freeze factors, candidate count, support and phase policy before dev evaluation.

Equal logical support K, identical quantization, matched physical resident capacity and exact fetch policy across arms:

- A: unrestricted native router (quality/traffic reference; distinct logical capacity).
- B: published DeepMath support (K=96).
- C: random matched support.
- D: train-only frequency ranking.
- E: train-only complete-route hyperedge coverage.
- F: BPD-weighted coalition coverage.
- G: matched train-only shuffled BPD weights.

Repeat for a separately defined K=48 task support and phase-indexed masks only after baseline parity. Use real native selection, capture unbiased scores, bias, original IDs, factor writes, residual, recurrent pass, expert outputs and measured cache transfers. The InkyNano runtime owns exact packed fetch and GPU memory accounting. Each arm must finish through the actual native loop and verifier. In a separate causal intervention study, compare targeted component gain against norm-matched randomized logit writes and a phase-permuted control, with measured native pre-router state displacement for physical upstream patches. Router-logit dose and physical hidden-state dose are separate interventions. Include zero hook full-head parity, source closure, finite route/content factorial, equal support capacity, and recoveries/corruptions by original correctness. Do not combine causal comparisons across intervention types without a registered contract.

## 5. Resolution and provenance

Independent unit: source task-generator root for a fixed checkpoint; independent trained model root for cross-model claims. New units/cases: 0/0. Proposed primary comparison F-E (incremental value of BPD weighting), and F-G (shuffled control) in a prespecified multiplicity family. Task verifier accuracy at matched support is primary; route coverage, TV and measured bytes/latency are diagnostics. Proposed meaningful floor +5 percentage points, to be frozen with appropriate task and independent development-sham calibration. For the separate physical intervention contrast, choose and freeze actual dose and its matched-sham distribution before test. Dose, sham floor, root effects and statistical resolution/power pending; no preflight readiness claim. Actual inferential comparisons: zero. Selection exposure: published masks, source theory, existing exposed BPD results. Read EvalMath mechanistic-resolution skill, run preflight.py --pilot pilot.json --strict with independent development-root effects and shams, and apply FSI-01 before confirmation release. Do not release existing sealed roots.

## 6. Local executed tests

Run from repository root:
    python -m unittest discover -s mechinterp/bpd/experiments/inkling_routing -p test_expert_compactification.py -v

Five unit tests passed locally in Python, 1,960 box corners across 245 certified synthetic boxes, 400 finite-difference utility derivatives including content, and 300 finite factorial closure checks. Additional checks cover rank ties, missing support, invalid inputs, and the identical-content zero effect. This is synthetic algebraic instrumentation; real-arithmetic theorem and floating fixture checks are separate evidence classes. Tested Git blob hashes: expert_compactification.py fe734b3c91fd91695c1668428e3c28bd0fff4f9b; test_expert_compactification.py 0f3925ef5b2a2debf2a2757cd8d6f189605894b5. No full repository test, native hook, GPU job, or training in this turn.
