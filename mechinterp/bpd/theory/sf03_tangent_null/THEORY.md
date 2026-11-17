# BPD-SF03: matched tangent response and the curvature remainder

Let a recurrent model have differentiable transition `F:X→X` and scalar readout `q:X→ℝ`. At a fixed post-pass state `h` define the next visible map `G=q∘F`. Let `K=ker Dq(h)` be the current readout-dark tangent subspace. For the SF02 synthetic teacher, `q(h)=h_0` and `K={v:v_0=0}` exactly.

Suppose `δ,η∈K` satisfy both `||δ||=||η||=ρ` and `DG(h)δ=DG(h)η`. The two edits are equally sized and have the same *signed* first-order future-visible effect. They may differ in their higher-order influence, route identity and whether they remain on a reachable state manifold.

**Elementary curvature proposition.** Assume `G` is twice differentiable on the two segments `[h,h+δ]` and `[h,h+η]`, with Hessian operator norm bounded by `M` along both. Then

\[
\left|G(h+\delta)-G(h+\eta)\right|
\leq M\rho^2.
\]

**Proof.** Taylor's theorem with integral remainder yields `G(h+v)=G(h)+DG(h)v+R_v` with `|R_v|≤\frac12 M||v||²`. The constant and first-order terms cancel between `δ` and `η`, while the two remainder magnitudes sum to at most `Mρ²`. QED.

This is an exact mathematical bound conditional on a valid Hessian bound. SF03 does **not** establish a uniform empirical `M` on the trained teachers; a sampled Jacobian is a local instrument. The observed recovery gap is also a thresholded 0/1 quantity, so even small floating output differences may change recovery rates near the ±0.10 acceptance boundary.

## Random matched controls

Let `g=P_K ∇G(h)`. For `g≠0`, write `u=g/||g||`, `α=u·δ`, and `r=\sqrt{ρ²−α²}`. Choose randomized `z∈K∩u^\perp` with `||z||=r`. Then

\[
\eta_{\rm signed}=\alpha u+z,\quad
||\eta_{\rm signed}||=ρ,\quad DG(h)\eta_{\rm signed}=DG(h)\delta.
\]

Flipping the sign of `α` with a fair random sign preserves only the **absolute** future Jacobian effect:

\[
\eta_{\rm unsigned}=\sigma\alpha u+z.
\]

The signed control uses the BPD-selected write's gradient coefficient, which is a *privileged* local response quantity. It asks whether the selected BPD factors carry additional finite-radius causal information *after matching their first-order effect*. A successful BPD transfer relative to ordinary equal-norm random directions may vanish under this stronger control.

The first/second pass source gradient profile measures a related question. If `|∂q(F(h_0))/∂S|` is small while `|∂q(F²(h_0))/∂S|` is large in an authorized role, sensitivity emerges only after recurrence. Gradient sensitivity relative to a fixed input is different from mutual information under a secret distribution and from causal attribution to a canonical BPD component.

## Gauge and statistical scope

For `\hat W=VU` and any invertible component-coordinate change `R`, the physical weight is invariant under `V'=VR, U'=R^{-1}U`. Singleton masks are generally changed by this reparameterization. The signed-gradient sham is defined in physical hidden-state coordinates after constructing the candidate BPD edit; it is invariant to arbitrary notation for that fixed physical edit. The **candidate edit itself** can change under a factor gauge.

SF03 calibrates root-to-root variation of one sham-versus-shams statistic, not a universal physical-output curvature. It uses 59 independently trained teacher roots for a one-sided order-statistic tolerance statement. An actual language-model leakage interpretation would additionally require a public transcript channel and a recipient-aware observer.
