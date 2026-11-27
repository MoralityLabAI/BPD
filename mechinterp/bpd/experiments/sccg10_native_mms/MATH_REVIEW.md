# SCCG-10 / MMS-TRM mathematical review: changing mechanisms, admission, and the value of refreshing

2026-11-27. Workstream A. This review covers the SCCG-10 conditional bounds
(THEORY.md, admission.py) and the native ComponentTRM. Each proposition gives
its scope, its proof or proof sketch, whether it is sharp, and a counterexample
where the natural stronger claim fails. Numbers marked *measured* come from
`results/engineering/root_29201/{drift,chronology}`: fresh engineering root
29201, TRAIN-fit panel of 24 edges, AdamW at lr 1e-3, checkpoints after
0..64 committed updates. They describe that panel and those checkpoints only.
Lean companions are in MathCan `MathCan/MMSTRMEconomics.lean` (commit
e0347e3) and the earlier `MathCan/SCCG10Admission.lean` (f0e6154).

## 0. Setting

X = (θ, optimizer moments, RNG states, data cursor). A committed update is the
map X → X′ produced by one clipped AdamW step on a selected minibatch. Fix a
*panel*: one semantic single-slot edit (receiver r, hybrid h, slot s), one
nuisance draw, four queries, and delays D = (d_1 < … < d_k). An *action space*
is a physical frame for interventions u:

* carried state: u ∈ R^64 is added to the post-write joint state [z_H, z_L]
  before the first distractor. Metric G = I.
* native factor write: u = c U_S is added to `up_out` at the first suffix low
  visit, with c ∈ R^|S| and U the shared factor rows. Metric
  G_S = U_S U_Sᵀ (physical write Gram).

Responses are signed relative logits m_{d,q,v} = y_{q,v} − y_{q,ref_q} computed
by the *full* recurrent suffix. The target response is
f_θ(u) = m_{d_k, s, new}(u) − m_{d_k, s, new}(0). The protected response
g_θ(u) ∈ R^p collects all rows with q ≠ s at every delay (p = 27 for three
delays). The linearizations are a_θ = ∇f_θ(0) and C_θ = Dg_θ(0). They are
computed exactly by `native.state_jacobian` / `native.write_jacobian`, with
parity to CURATE-02 `observe_edge` (rtol 1e-5) and finite differences.

## 1. What a reuse decision needs (and what it does not)

**Prop. 1 (direction-restricted admission).** Let u be a fixed candidate with
‖u‖_G ≤ e. Put α_u = |(a′ − a)ᵀu|, γ_u = ‖(C′ − C)u‖ and the remainders
R_f(u) = |f′(u) − a′ᵀu| and R_p(u) = ‖g′(u) − C′u‖. Then
f′(u) ≥ aᵀu − α_u − R_f(u) and ‖g′(u)‖ ≤ ‖Cu‖ + γ_u + R_p(u).
*Proof:* triangle inequality (Lean: `stale_target_lower`,
`stale_protected_linf`, `near_null_persistence`). SCCG-10's α, γ, R_f, R_p are
suprema of these quantities over the dose ball, so the SCCG-10 lower bound
is implied by Prop. 1 and is weaker for any particular u.

**Corollary 1 (verification is the certificate on the panel).** f′(u) and g′(u)
are the quantities a native finite continuation computes. One native
evaluation at the current checkpoint therefore decides admission of u on that
panel *exactly*, up to float32 rounding (§6). No derivative-drift bound can
improve on it, and its cost is one suffix continuation per delay.
*Measured:* verification costs 0.05–0.10 CPU-s per edge for three delays, while
exact observation costs 0.32–0.53 s (state) and 0.25–0.40 s (write). So
verifying a cached candidate is 4–10× cheaper than re-observing it.

**Prop. 2 (asymmetry: verifying a positive is local, certifying a negative is
uniform).** The SCCG-10 *upper* bound (no headroom in the span) needs
uniform α, γ, R_f, R_p over the whole dose ball. No finite set of native
evaluations implies a uniform bound. *Counterexample:* for any finite
evaluation set S in the ball there are smooth responses f, f̃ that agree on
S and differ elsewhere by an arbitrary amount (a bump supported off S).
Lean: `sample_residual_not_uniform`. *Consequence for the ladder:* a failed
certificate or failed verification sends the entry to refresh or ordinary
fallback. It never labels the span infeasible, and sampled finite errors are
never used as a uniform remainder (drift.py labels them
`SAMPLED_REMAINDER_NOT_UNIFORM`).

**Measured scope of SCCG-10 on the native TRM.** For the k = 4 and k = 8 state
supports we used the exact operator norms of a′ − a and C′ − C for each
measured checkpoint pair. These are the true uniform α, γ for that pair on
that panel, because the full live derivative was measured. SCCG-10 returned
REFRESH_OR_ORDINARY for 24/24 edges at every checkpoint, including
checkpoint 0. At checkpoint 0 the sampled R_p alone exceeds the room
η − ‖Cu‖, because SCCG-01 places u on the linear budget boundary. After one
update the mean ‖C′ − C‖₂ is 25.4, so γe ≈ 19 against η = 0.25. After 64
updates γe ≈ 32. Uniform certification is out of reach by two orders of
magnitude. Yet the cached k = 8 directions still deliver 60–110% of the live
target gain (§7). The certificate is sound and uninformative here.

## 2. Low-rank changes, exact quotients, approximate quotient residuals

**Prop. 3a (exact invariant quotient for readout-only updates).** Write the
protected rows at delay d as C_d = E_d W_o J_d, where W_o is the output layer,
E_d is the fixed margin-contrast matrix, and J_d = ∂z_H(T_d)/∂u is the
suffix Jacobian. If an update class changes only W_o (and the output bias),
then J_d is constant and ker J_d ⊆ ker C′_d for every member of the class.
The quotient R^64 / ker J_d is an exact invariant (SCCG-06 with L chosen to
annihilate im E_d ΔW_o), and C′ − C = E_d ΔW_o J_d has rank ≤ min(4, rank J_d).
*Proof:* direct from the factorization.

**Prop. 3b (no exact invariant for block updates; counterexample).** An update
that touches the shared factors (V, U), `down`, the norm scale or the input
map changes every per-step Jacobian D_t = ∂z_{t+1}/∂z_t. For generic
updates, no nonzero vector stays in ker C over a whole update class
(exact persistence of u ∈ ker C needs (C′ − C)u = 0, a codimension-p
condition). *Measured:* one committed update moves C by 44.5% (relative
Frobenius) and a by 48.8%. The protected row space rotates with maximal
principal sine 0.68 after one update and 0.95 after 32. Exact SCCG-06
structural reuse is therefore not available for native training. Near-null
numerical columns are not rational dependencies.

**Prop. 3c (approximate low rank of the drift; measured).** The *drift* is
far from full rank even though it is large. 95% of the energy of C′ − C
lies in its top 3.8 (after 1 update) to 5.2 (after 64) singular directions,
out of 27. The SCCG-06 split C_t = C_0 + U_t V_t therefore holds with a
small residual for rank 4–6 and an update-dependent U_t. The residual is an
*approximate quotient residual*: it must enter γ explicitly, and U_t must
be re-estimated, so this does not certify reuse. It does suggest that a
rank-r drift sketch is a cheap screen. That is unresolved problem 1.

## 3. Signed chronological transport

**Prop. 3d (chronological product bound; sharp).** Let J = D_T⋯D_1 and
J′ = D′_T⋯D′_1 with ‖D_t‖ ≤ M_t and ‖D′_t − D_t‖ ≤ ε_t. Then
‖J′ − J‖ ≤ ∏(M_t + ε_t) − ∏ M_t. *Proof:* telescope
J′ − J = Σ_t D′_T⋯D′_{t+1}(D′_t − D_t)D_{t−1}⋯D_1 and bound each term, or
induct on T as in the Lean proof `chrono_prod_bound` (scalar factors). The
bound is attained for aligned nonnegative scalars (`chrono_prod_sharp`).

**Counterexample to its usefulness (measured).** On the delay-31 suffix the
exact per-step Jacobians have spectral norms up to 7.6 (geometric mean
1.8–3.7 per step). After 8 updates the per-step drift averages 0.7–2.0.
The bound evaluates to 3.7·10^10 – 5.9·10^20, while the actual end-to-end
drift ‖J′ − J‖₂ is 0.30–11.4. The products themselves stay bounded,
‖J_t‖ ≤ 11.7 for t ≤ 32. Per-step expansions are not aligned: RMSNorm
projections and sign cancellations keep the product near its clean scale.
Operator-norm transport is sound and off by 10–20 orders of magnitude. Only
*signed* product information is usable: the actual product, or a
directional product J′u computed by one native continuation. That
continuation is again a verification (Cor. 1).

**Prop. 3e (local reader certificates do not certify recurrent outputs;
counterexample).** SCCG-09's affine envelope certifies normalized
fixed-query reader scores at one step. Take the 2-D linear recurrence
z_{t+1} = R z_t with R the rotation by 90°, reader r = e_1ᵀ and edit
u = e_2. Then r u = 0, so the local protected response is zero, but
r R u = 1 one step later. A local RMS certificate covers the delay
at which it is evaluated. Extending it to later outputs needs either the
chronological bound (vacuous here) or a native continuation.

## 4. Singular subspaces and near-null directions

**Prop. 4a (persistence without a gap).** For any stored direction u with
‖Cu‖ ≤ τ‖u‖, ‖C′u‖ ≤ (τ + ‖C′ − C‖)‖u‖. Sharper and computable:
‖C′u‖ ≤ ‖Cu‖ + γ_u. Weyl: |σ_i(C′) − σ_i(C)| ≤ ‖C′ − C‖. Lean:
`near_null_persistence` (per row). *Measured:* Weyl holds at every
checkpoint and edge (`weyl_ok`). This is a sanity check on the instrument,
not a finding.

**Prop. 4b (kernel angle).** Let C have full row rank p with smallest
singular value σ_p. Every unit x ∈ ker C′ satisfies
‖P_{row C} x‖ ≤ ‖C′ − C‖ / σ_p(C). *Proof:* Cx = (C − C′)x, and C restricted
to its row space has smallest singular value σ_p. The bound is attained by
2×2 diagonal examples. *Measured:* σ_p(C) ≈ 1·10^−3 against σ_1 ≈ 38
(condition number ≈ 4·10^4), so the bound is about 1.3·10^4 after one update
and vacuous (it holds trivially). The row space of the 27×64 protected map
contains near-null directions with no spectral gap. Identifying a *cached
near-null subspace* is ill-posed here, even though individual cached
directions keep part of their value. Cancellation circuits must be
verified as directions, not as subspaces.

## 5. Physical metric and factor-frame changes

**Prop. 5 (metric equivalence).** If (1 − ε)G ⪯ G′ ⪯ (1 + ε)G, then
‖u‖_{G′} ≤ √(1+ε)‖u‖_G. A cached candidate with ‖u‖_G ≤ e/√(1+ε) is
dose-feasible at the new checkpoint, and α, γ stated in the G-norm convert as
α_{G′} ≤ α_G/√(1 − ε). *Measured* (write frame, generalized eigenvalues of
(U₀U₀ᵀ, U_kU_kᵀ)): within [0.989, 1.012] after one update and
[0.931, 1.088] after 64. The physical write metric is stable to about 9%
over 64 updates, so dose rescaling is cheap and exact.

**Prop. 6 (factor gauge).** (V, U) ↦ (VS, S⁻¹U) leaves W = VU and every
physical response invariant. Component coordinates transform as c ↦ cS.
Sparse component supports are invariant exactly under monomial S
(permutation times nonzero diagonal). *Counterexample* for any non-monomial
S: with two components and S the 45° rotation, the support {1} spans U_1
before and (U_1 − U_2)/√2 after. Atlas entries keyed on component supports
must carry the factor-frame digest. State-space entries are gauge-free.
Training changes U genuinely rather than by a gauge. *Measured:* keeping
the cached coefficients and using the current frame ("factor transport")
gave the same target gain as reusing the cached physical write to three
decimals at every checkpoint, so frame drift is not the bottleneck.

## 6. Curvature, finite dose, precision

**Prop. 7 (finite-dose remainder).** |f(u) − aᵀu| ≤ ½ sup_{‖v‖≤e} ‖∇²f(v)‖ e².
For a T-step composition the Hessian bound inherits the chronological
product (Prop. 3d), so no cheap uniform R_f exists. *Counterexample to
"linear null implies safe" (measured):* the zero-collateral full-state
direction (C u = 0, dose 0.75) produced finite protected changes of 0.67–0.88
on the 29201 panel. In the 28101 smoke test it reached 5.3. The k = 4 / k = 8
sparse supports stay at the budget (0.25–0.30) with finite errors below 10^−3.
Write-space directions are nearly linear: target finite error ≤ 6·10^−4.

**Prop. 8 (storage and arithmetic precision).** Storing C in fp16 adds
‖ΔC‖₂ ≤ ‖ΔC‖_F ≤ 2^−11‖C‖_F (normal range), which enters γ additively. The
native ComponentTRM computes in float32 (TiedComponentLinear casts inputs),
so a float64 recomputation is not available as a reference. Verification
thresholds (gain ratio 0.7, protection slack 25%) exceed the float32
rounding of a 36-step suffix by about five orders of magnitude.
*Not established:* a rigorous rounding bound for the native float32 suffix.
The atlas stores float64 copies of float32-computed quantities.

## 7. When do old cancellation circuits remain useful?

Combine Props. 1 and 4a: a cached direction u stays useful at the new
checkpoint iff its *measured* live outcome satisfies
f′(u) ≥ ρ·f(u)|_source and ‖g′(u)‖ ≤ max(η, ‖g(u)‖_source)(1 + slack). The
ladder tests this at L4. Exact persistence of a cancellation (u ∈ ker C′_S)
holds only under an invariant split (Prop. 3a or SCCG-06) and is non-generic
for native training (Prop. 3b). *Measured* (state k = 8, mean over 24 edges):

| updates since cache | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cached / live target gain | 1.10 | 1.01 | 0.74 | 0.60 | 0.81 | 0.84 | 0.80 |
| cached protected change (budget 0.25) | 0.33 | 0.46 | 0.45 | 0.52 | 0.56 | 0.84 | 0.54 |
| live protected change | 0.26 | 0.27 | 0.26 | 0.28 | 0.25 | 0.30 | 0.26 |

Target value persists. Protection degrades within 2 updates, so a cached
direction is a good *proposal* for the target term but not for protection.
Write-space k = 8 directions keep 0.98 of the live gain after one update
and about 0.45 after 4 or more, with protection near budget (0.24–0.36).

## 8. The value of refreshing, and why steering value is not learning value

**Prop. 9 (value of information of a refresh; sharp).** Let K be a candidate
set with realized one-window gains r_b, fresh scores s*_b and stale scores
s_b. If |s_b − s*_b| ≤ δ and |s*_b − r_b| ≤ ε for all b, then picking by
stale scores loses at most 2δ + 2ε against picking by fresh scores. Lean:
`voi_refresh_le`, with equality witness `voi_refresh_sharp`. For top-k
selection by sums the bound is 2k(δ + ε). For the curriculum score
s_b = −⟨g_cal, Δθ_b⟩, δ ≤ ‖g_cal − g′_cal‖ · max_b‖Δθ_b‖ + max_b|⟨g′_cal, Δθ_b − Δθ′_b⟩|.
The AdamW step satisfies ‖Δθ_b‖ ≤ lr·√P·c(β₁, β₂, t) (at t = 1,
‖Δθ‖ = lr√P exactly up to decay).

**Prop. 10 (steering gain does not translate to learning value without further
assumptions; counterexamples).**
(a) *Large steering headroom, zero learning value.* Let θ = (θ₁, θ₂), let the
validation loss depend on θ₁ only, and let every candidate batch have the
same θ₁-gradient while mechanistic scores differ through θ₂. The SCCG-10
upper bound can be arbitrarily large while every mechanism-based curriculum
choice has exactly zero validation value. Lean:
`steering_gain_not_learning_value` (a gain can coexist with worse loss).
(b) *No steering headroom, positive learning value.* If a = 0 on the span,
the transport term is constant and the mechanistic objective reduces to
w_ordinary·CE. Curation by it is then ordinary curation and can help.
Headroom is not necessary.
(c) *Positive one-step value, negative multi-window value.* Lean:
`myopic_not_two_step`: action A gains 2 then 0, action B gains 1 then 3.
One-step greedy picks A.

**Prop. 11 (sufficient conditions for translation).** Let V be the validation
objective and Φ the calibration objective. Suppose that at each window
(i) ∇V = κ∇Φ + ξ with ‖ξ‖ ≤ ρ‖∇Φ‖ (alignment), (ii) V is β-smooth, and
(iii) ‖Δθ_b‖ ≤ s. Then the one-window V-improvement of candidate b is at least
κ·(−⟨∇Φ, Δθ_b⟩) − ρ‖∇Φ‖s − βs²/2. The V-value of a refresh is at most
κ·VOI_Φ + 2ρ‖∇Φ‖s + βs². Summing over windows also needs separability: the
choice at window t must not change the opportunity set at t+1, which (c)
violates. None of (i)–(iii) is established for the native TRM. This is why
the trainer measures value directly as the realized drop of ordinary
TRAIN-feedback CE per charged CPU-second, and the primary endpoint is
measured on fresh roots.

## 9. Economics of refreshing versus reinvesting

**Prop. 14 (ratio rule).** With stationary, additive per-window values v_a and
costs c_a, no split of a CPU budget B beats spending it all on the action
with the largest v_a/c_a. Lean: `ratio_rule`. An audit or refresh with cost
c and value bound VOI is worth paying only if VOI ≥ c·max_a v_a/c_a. Lean:
`refuse_audit`. With non-stationary, diminishing returns the rule is only
greedy-per-block. That is what `economics.RateController` implements: EMA
of measured ordinary-CE drop per CPU-second for the ordinary and the curated
action, deterministic exploration every fifth block, and audits skipped when
no candidate has positive predicted value.

**Measured cost ratios** (engineering, CPU-s): ordinary window 0.05–0.1;
curated window with analytic previews 0.6–1.0; full mechanism refresh per
edge 1.2–2.0 (observation 0.17 + support search 1.0–1.7 + verification 0.05).
With 12 feedback edges, a cold refresh costs 15–25 s, about 200–400
ordinary windows. A curated step must therefore be worth about 10× an
ordinary step to break even, and a full refresh must be amortized over many
windows. The atlas makes that amortization possible: 75% of edges reuse
after verification in steady state. The ratio rule then says reinvestment
wins unless a curated step is worth more than its cost ratio. Section
`RESULTS.md` reports what happened.

## 10. Proof obligations: status

| Obligation | Status |
|---|---|
| Prop. 1 / Cor. 1 direction-restricted admission | proved (triangle); Lean `stale_target_lower`, `stale_protected_linf` |
| Prop. 2 no finite sample gives a uniform bound | proved; Lean `sample_residual_not_uniform` |
| Prop. 3a readout-only quotient | proved (factorization) |
| Prop. 3b generic non-persistence | argued (codimension); measured counterexample |
| Prop. 3d chronological bound | proved; Lean scalar case `chrono_prod_bound`, sharpness `chrono_prod_sharp`; matrix case by telescoping (not in Lean) |
| Prop. 3e local ≠ recurrent | explicit counterexample |
| Prop. 4a/4b near-null persistence, kernel angle | proved; Lean 4a per row |
| Prop. 5 metric equivalence | proved |
| Prop. 6 gauge invariance of supports | proved with counterexample |
| Prop. 7 remainder | standard Taylor; measured counterexample |
| Prop. 8 precision | fp16 storage bound proved; native float32 rounding bound **open** |
| Prop. 9 VOI of refresh | proved; Lean `voi_refresh_le`, `voi_refresh_sharp` |
| Prop. 10 steering ≠ learning | counterexamples; Lean (a), (c) |
| Prop. 11 translation conditions | proved under (i)–(iii); hypotheses **not established** for the TRM |
| Prop. 14 ratio rule, audit refusal | proved; Lean `ratio_rule`, `refuse_audit` |

## 11. Ranked unresolved problems for the next frontier review

1. **Certified low-rank drift screens.** Drift energy is 95% in 4–6
   directions. Is there a cheap, *sound* estimator of the residual
   ‖(I − P_r)(C′ − C)‖ from r directional derivatives plus a certified
   remainder? Randomized sketches give probabilistic bounds only. A
   deterministic bound needs structure (Prop. 3a) that block updates lack.
2. **Rounding certificates for float32 native continuations.** We need an
   interval or compensated evaluation of the suffix so that native
   verification (Cor. 1) is a theorem rather than a float32 measurement.
3. **Signed transport bounds.** Operator-norm transport is vacuous by
   10^10–10^20 (§3). Can a cone or Lyapunov-weighted norm in which the trained
   recurrence is non-expansive be found and certified, giving transport
   bounds that reflect the measured bounded products?
4. **Translation from calibration to validation.** Prop. 11's alignment
   (κ, ρ) is unmeasured. Estimating it online, and deciding when a
   mechanistic calibration objective is aligned with ordinary held-out CE,
   would turn the controller's empirical rates into bounds.
5. **Multi-window value of curricula.** Prop. 10(c) rules out one-step
   sufficiency. What sufficient conditions (e.g., an exchangeability of
   minibatch effects) make one-step scores rank-consistent with W-window
   value, and how large must W be?
6. **Behavioral headroom versus dose.** Clean target margins are −4 to −7
   logits while dose-0.75 SCCG directions move them by 0.05–0.35. A
   dose-aware selective-control theory that predicts the dose at which
   targeted flips beat matched random flips would make the behavioral gate
   prospective instead of empirical.
7. **Exact sparse support maintenance under approximate drift.** SCCG-06's
   inverted index needs exact fixed D. Extending it to an ε-perturbed D with
   certified rank decisions is open.
