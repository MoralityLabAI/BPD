# BPD-REV-01: evidence revision as a delayed causal-observability test

**Source-only bridge, 18 November 2026.** Upstream [Beyond Bayes BB-CQ-05 #24](https://github.com/patrickdugan/beyond-bayes/pull/24), merged at `abc89f73d223f00305eb0416a34fe5aee602494d`. This complements BPD's [Causal Observable Masks theorem](../causal_observable_masks/THEORY.md) and its learned feedback / nonlinear-rank experiments. It changes no original NC, SF, PD or CQ experimental thresholds or archived outputs.

## New causal gym

The original ECA-04 three-regime Bayesian task has 12 match bits. A correction set `S` of size `j` is chosen independently, hidden from the encoder until after it has compressed old evidence, and later revealed together with new accepted report bits. A compressed representation that preserves the **current regime posterior** may merge two old histories that produce different **future posteriors** after the same schedule is revealed. Under exact finite conditions and `1<=j<n`, all `2^n` old histories are distinguishable by some legal continuation. For `n=12`, the self-contained exact-state minimum is 12 bits; if `S` arrives before encoding, a count sufficient statistic updates exactly. The external archive may communicate the old corrected-slot match total in `ceil(log2(j+1))` bits while itself keeping full source records.

The included independent source-only finite witness enumerates all 16 histories for n=4, j=2, using exact fractions. It returns the shortest one-disclosure distinguisher between present-posterior collisions, and shows all 16 are separable despite only five current-count labels. Its finite combinatorial signature is an analogy to BPD's stacked observability signatures; it establishes no native model hidden-state quotient.

## Structural and neural predictions

For a native recurrent transition `h_{t+1}=F_{e_t}(h_t)` with event stream `e`, and a chosen posterior readout `q(h)`, present-readout equivalence is `q(h)=q(h')`. **Continuation closure** demands equality of `q(F_{e_k}(...F_{e_1}(h)))` for every allowed future evidence revision sequence. The linearized, local necessary condition for an event `e` is `Dq(F_e(h)) DF_e(h) v=0` for every `v` in the candidate tangent kernel of the present readout. Rank/gauge choices that satisfy only `Dq(h)v=0` can break after the schedule token. The existing BPD Gramian criterion remains the exact linear contract when `F_e` is linear and the event schedule fixed.

A next neural **BPD-REV-02** (separate preregistration required) should:
- Use a source-frozen recurrent model and actual native recurrent states at the moment *before* late schedule revelation; hold current posterior output matched between donor and receiver.
- Compare donor-specific post-revision **Brier risk** to the correct reduced-information Bayesian posterior for each admitted message set. Decompose approximation error and irreducible information loss by the conditional expectation identity. Preserve the full-history strong Bayes control and an early-schedule count control.
- Apply physical interventions with matched input source, mask rank, L2/activation dose, full-future-logit Jacobian response, random/gauge directions, and a no-edit arm. Keep a donor-specific task-grounded endpoint alongside mechanistic diagnostics.
- Use independent issuer roots, separate independent *model* roots when claiming cross-training transfer, and run the EvalMath 59-root sham calibration or a properly justified stronger alternative. Freeze effect size, selected comparison family, model weights, and protected cohort before scoring.

A positive observational probe alone shows decoding. A learned native change that improves future posterior correctness above equal-dose, matched shams would support causal mediation in that specific model and event family. Previous BPD source-gating and equal-dose failed gates remain intact. The numerical source-only witness here is **not** such a positive outcome.

## Source-only reproduction

```sh
python -m unittest discover -s mechinterp/bpd/theory/revision_predictive_quotient -p 'test_*.py' -v
```

The exact ECA-04 twelve-bit and Brier curve remain owned by Beyond Bayes; this BPD fixture only audits a separate small-domain source-readable collision using no neural checkpoint.
