# BPD-SF01 — Authorized Secret Transport Through Recurrent Dark Fibers

**Prospective source-only protocol, 2026-11-17.** No trained model, neural BPD factor, GPU result, or sealed-test evaluation is claimed here. This branch introduces mathematical audit utilities and analytical positive controls.

## Source motivation

Daniel Rodríguez's [Spectral Faithfulness](https://github.com/idolum-ai/spectral-faithfulness) studies origin attribution, cross-user contamination and private information escaping into public poker table talk. Its "spectral" language is a metaphor; it does not examine hidden-layer spectra. The reported provider rates depend on a detector, task and interaction setting. The paper explicitly leaves a lexical-domain explanation versus a boundary-control failure unresolved.

Source anchors:
- [paper source](https://github.com/idolum-ai/spectral-faithfulness/blob/main/paper/main.typ)
- [source-tag instrument](https://github.com/idolum-ai/spectral-faithfulness/blob/main/experiments/phase2_spectral/source_tag_confusion.py)
- [poker analysis](https://github.com/idolum-ai/spectral-faithfulness/blob/main/experiments/phase4_poker/analysis_v2.py)
- [BPD-CQ-03 trained recurrent intervention](https://github.com/MoralityLabAI/BPD/pull/20) (registered seed gate 6/8)
- [BPD-CQ-04 adaptive-mask pilot](https://github.com/MoralityLabAI/BPD/pull/25) (only 67/1024 donor pairs eligible)
- Local `papers/shared_weights_dark_fibers/CAUSAL_QUOTIENT_ARG03.md` and `mechinterp/bpd/theory/OPTIMAL_BASIS_AND_MASKS.md`.

These behavioral findings motivate an independent BPD research question. The external model rankings are not teacher labels, priors or evaluation data.

## Prospective hypotheses

**H1: Readout-invisible retention.** A randomized private variable is carried by a learned recurrent state along a direction invisible to the current true readout, yet affects a future authorized answer.

**H2: Recipient-conditioned routing.** An authorization switch allows the same private information to influence the authorized answer while preserving a matched unauthorized output boundary.

**H3: BPD mediation.** A development-selected, gauge-robust block/subspace intervention changes the subsequent authorized use or unauthorized disclosure of the secret more than matched sham interventions. Distinguish provenance binding from simple reconstruction of a secret.

## Variables and observer-relative information

Draw balanced randomized secret `S`, public context `X`, and receiver/authorization `A`. Keep `S` independent of `X` within source families. `H_t` is the *complete* recurrent state including all buffers and future input caches. `q(H_t)` is the actual readout, including norm/head and all token sites relevant to the task. `B` is a publicly observable action and `T` public speech. `Y_A` denotes authorized output.

Define unauthorized total and incremental disclosure in bits:

\[
L_U=I(S;(B,T)\mid X,A=U),\quad
L_{\mathrm{talk}}=I(S;T\mid X,A=U,B).
\]

The exact chain rule gives

\[
I(S;(B,T)\mid X,A=U)
=I(S;B\mid X,A=U)+I(S;T\mid X,A=U,B).
\]

A poker bet can legitimately convey hand information; the additional conditional term asks how much table talk contributes *given* the publicly observed action. Always report total and incremental information with the complete public history in `X`, using the actual chronological action/transcript protocol. For multiple users carrying independent `S_1,S_2`, measure wrongful source transfer by `I(S_{\mathrm{other}};Y_A\mid X,S_{\mathrm{authorized}},A)` as well as correctness of authorized retrieval.

**Crucial scope:** An activation probe, a current-logit observer, and a transcript recipient have different information sets. Continuous states can decode a secret at arbitrarily small Euclidean displacement; a small logit change alone is never a secrecy guarantee. Finite-sample plug-in mutual information is biased, especially on natural-language outputs. Use exact finite output channels when possible, predeclared answer alphabets and cross-fitted decoders otherwise, independent-root intervals and source-independent controls.

For an exactly specified finite output distribution, convexity of KL yields

\[
I(S;Y\mid X=x,A=a)\leq
\mathbb E_{s,s'\mid x,a}\mathrm{KL}\big(p(Y\mid s,x,a)\,\Vert\,p(Y\mid s',x,a)\big).
\]

The bound may be infinite if channel supports differ. The information audit returns bits; BPD's KL objectives typically use nats. A one-token KL at a shared prompt prefix does not bound leakage across an entire generated transcript.

## Recurrent-fiber causal test

For matched reachable states with the same `X,A` and different `S`, require the *complete* current readout equality before any edit:

\[
\|q(h_s)-q(h_{s'})\|\leq\epsilon_{\rm read},\qquad
D^{a}_{t,k}=d(q(F^k_{X,a}(h_s)),q(F^k_{X,a}(h_{s'}))).
\]

Test delays `k=1,2,4,8` after calibration. Report delayed authorized `D^A`, unauthorized `D^U`, exact task outcomes, tokenized generations, and immediate readout conservation separately. Define a local restricted observability Gramian

\[
G^a_{t,K}=P_{\ker Dq}^{\top}
\Big(\sum_{k=1}^{K}J_{k,a}^{\top}J_{k,a}\Big)P_{\ker Dq},
\quad J_{k,a}=D(q\circ F_{X,a}^k).
\]

Its first-order eigen-directions are candidate causal directions; held-out, finite-radius matched-fiber transfers are the evidential test.

In shared-weight BPD, compare native `u_cv_c^\top` writes, dark-projected writes, and complete factor blocks. Singleton masks change under general invertible latent gauge transformations. Preserve full physical `W` under a `GL(C)` reparameterization and require block-level effect stability or explicitly report the gauge dependence. Do not equate an attractive probe axis with a canonical secret module.

### Exact positive control only

An analytic tied linear transition on `h=(s,carry,out)` with readout `q(h)=out` is

\[
F_a(h)=(0,h_1,(a+(1-a)\lambda)h_2).
\]

It has two potential rank-one writes and is run *twice* without changing parameters. From `h_0=(s,0,0)` the initial and first-pass outputs are both zero; the second-pass readout is `(a+(1-a)\lambda)s`. With `a=1` it discloses to the authorized role. With `a=0,\lambda=0` it suppresses the public readout. Positive `lambda` is an intentionally leaky control. This is algebraic routing built into the model, not learned mechanism recovery. See `information.py` and `test_information.py`.

## Phased prospective campaign

1. Verify exact finite-support conditional information, action/talk chain rule, missing-support KL, source routing, and tied two-pass hidden-to-visible access. Treat these as source-only sanity tests.
2. Train new independent recurrent teachers with synthetic balanced secrets, separate public objective, authorized secret output, overlapping-topic interleaved users, and a strategic action/speech option. Include both safe-router and pressure-leak conditions. Keep diagnostic source families disjoint from training.
3. Freeze the physical teachers; train actual `BPDModel/TiedComponentLinear` factors `(V,U)` on development only. Select candidate blocks before donor pair sampling. Log per-pass importance, factor geometry and hard masks independently.
4. Run same-source, opposite-secret recipient/donor transfer at a frozen pass with exact current readout preservation. Controls: clean/full-state, same-secret/self-donor, source-scrambled, visible-only, native versus dark-projected BPD write, unselected factors, random dark of equal norm, manifold calibration and task-sensitivity-matched sham. Report all attempted/eligible donors.
5. Calibrate the 95th-percentile null using at least **59 independent source-root shams** for the distribution-free 95% confidence order-statistic bound. Freeze code, all thresholds, random seeds, source families, depth and intervention selection *before* a new sealed confirmation source. Suggested starting confirmation: 32 independent roots with 256 new paired trials each. Use root-level uncertainty and multiplicity correction. A 64-trial aggregate cannot resolve the targeted mechanism.
6. Only after the trained linear teacher passes, extend to a separately registered open-weight 7M looped transformer with true layer/pass/token activation hooks, full rollout emissions, source-independent data and prompt-only/output-filtering baselines.

**Protected exclusions:** Keep ARG-03-N's 168-case sealed test untouched. No previously registered CQ-03/CQ-04, B1–B4 or Dark Fibers results are amended. This is an additive new experiment.

## Reporting and scientific admission

Freeze the protocol SHA before new training/data. Record by root and task family: total/eligible donors, authorized utility, public action and incremental speech bits, source contamination, all intermediate readout errors, component/full-block gauge behavior, observer-budget probe performance, signed causal outcomes, null and random directions, independent calibrated sham tolerance, and test code provenance. Prespecified source-disjoint replication, retained utility, exact current-readout equality and a gauge-robust causal effect above the sham tolerance are all necessary for a mechanistic success claim. Negative outcomes, selective eligibility and underpowered outcomes remain reportable.