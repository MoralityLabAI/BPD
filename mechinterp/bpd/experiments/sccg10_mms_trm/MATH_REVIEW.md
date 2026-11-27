# SCCG-10 / MMS-TRM mathematical review: changing mechanisms between committed updates

Scope: the 5,380-parameter ComponentTRM (width 32, rank 29, one low visit
per token, 64 carried coordinates) of CURATE-02 (`af43e48`), the SCCG-10
conditional admission reference (`sccg10_native_mms/admission.py`, PR 175)
and the instrument in this directory. Every statement is labelled:

* **[proved]**: complete proof here or in Lean (`MathCan/SCCG10Admission.lean`,
  MathCan `f0e6154`; kernel build and `#print axioms` recorded in section 9);
* **[standard]**: a cited textbook result applied as stated;
* **[measured]**: an observation from the receipts in `results/`, not a bound;
* **[open]**: a proof obligation not discharged.

## 0. Setting and notation

The training state is `X = (θ, optimizer moments, RNG, cursor)`. At checkpoint
θ and a fixed TRAIN panel item, the native margin map
`m_θ : U → R^R` sends an intervention `u` in an action space `U` to the
signed relative-logit rows `y[q,v] − y[q, labels_r[q]]` over the four queries
and delays (2, 7, 31) (R = 36). The target row `f` is (d=31, edited slot,
hybrid value); the protected rows `g` (27 of them) are every row with a query
other than the edited slot. Two action spaces are kept separate:

* carried state: `u = δ ∈ R^64` added to the post-write joint state
  `[z_H, z_L]` entering the first suffix token, metric `G = I`;
* native factor write: `u = α ∈ R^29`, pre-activation write `αU` at the low
  visit of the first suffix token, metric `G = UUᵀ` (U the live
  `TiedComponentLinear` factor), so `‖α‖_G = ‖αU‖_2`.

`a(θ) = ∇_u f(0)`, `C(θ) = ∇_u g(0)`. A cached observer is `(a₀, C₀, G₀, U₀)`
from θ₀; the live one is `(a₁, C₁, G₁, U₁)` at θ₁ after k committed updates.

## 1. What SCCG-10 actually guarantees

SCCG-10 (THEORY.md) admits a cached candidate `u`, `‖u‖_G ≤ e`, when

    ‖C₀u‖ ≤ η − E_p,   a₀ᵀu − E_f ≥ floor,   E_f = eα + R_f,  E_p = eγ + R_p.

**Proof obligations (all must hold at θ₁, for the declared response):**

* P1 (target drift, uniform over the ball): `|(a₁−a₀)ᵀu| ≤ α‖u‖_G` for all u,
  equivalently `‖a₁ − a₀‖_{G⁻¹} ≤ α` (dual norm). [proved, standard duality]
* P2 (protected drift): `‖(C₁−C₀)G^{-1/2}‖_2 ≤ γ`.
* P3 (uniform finite remainder over the whole dose ball):
  `sup_{‖u‖_G ≤ e} |f₁(u) − f₁(0) − a₁ᵀu| ≤ R_f`, and the same for g with R_p,
  including floating-point evaluation error of the response being certified.
* P4 (common frame): `‖·‖_G` is the same norm at both checkpoints. If
  `G₁ ≠ G₀`, either transport the dose (section 2.4) or certify in `G₀` and
  add the distortion factor.

Under P1–P4 the admission inequalities are [proved] (Lean
`stale_target_lower`, `stale_protected_linf` for the ℓ∞ form,
`squared_l2_functional_bound` with integer Cauchy–Schwarz for the ℓ₂ form;
the real-valued ℓ₂ statement is the same argument and is not separately
formalized). The SCCG-10 upper bound

    V* ≤ min_λ sqrt((e² + λ(η+E_p)²) · a₀ᵀ(G + λC₀ᵀC₀)⁻¹a₀) + E_f

is a bound on attainable *steering gain in the declared span, metric,
checkpoint, response and dose*. It says nothing about training value
(section 6).

**Key structural observation [proved].** P1–P3 are needed only to certify
*unqueried* candidates or to bound the value of refreshing. For a finite,
explicit candidate list they can be replaced by cheaper exact operations:

* a single forward-mode JVP along `u` returns `a₁ᵀu` and `C₁u` exactly (up to
  float32 evaluation), eliminating P1/P2 for that `u`;
* one native finite continuation at `u` returns `f₁(u) − f₁(0)` and
  `g₁(u) − g₁(0)` for the declared queries, eliminating P1–P3 for that `u`.

So for candidate *verification* the uniform premises are unnecessary; they are
only needed for (i) certifying candidates that are never evaluated, and
(ii) the refresh-value upper bound of section 5. This is why the admission
hierarchy places explicit native verification (tier 4) before refresh.
The measured cost ratio (section 8) decides when tier 4 beats refresh.

## 2. Reliable, inexpensive bounds on change between committed updates

### 2.1 Parameter displacement under AdamW: an a-priori, gradient-free bound [proved]

PyTorch AdamW with `β₁, β₂, ε ≥ 0`, decoupled decay `wd`, fresh moments,
per coordinate:
`m_t = (1−β₁)Σ_{k≤t} β₁^{t−k} g_k`, `v_t = (1−β₂)Σ β₂^{t−k} g_k²`,
`θ ← θ(1 − lr·wd) − lr · m̂_t/(√v̂_t + ε)`, with `m̂ = m/(1−β₁^t)`, `v̂ = v/(1−β₂^t)`.

Cauchy–Schwarz on `Σ (β₁/√β₂)^{t−k} · √β₂^{t−k} g_k` gives, with
`ρ = β₁²/β₂ < 1` and `S_t = Σ_{j<t} ρ^j`,

    |m̂_t| / (√v̂_t + ε) ≤ κ_t := (1−β₁) sqrt(S_t (1−β₂^t)) / ((1−β₁^t) sqrt(1−β₂)).

If `v_t = 0` then `m_t = 0` and the step is zero. Gradient clipping rescales
each `g_k` and does not affect the bound. Hence for one committed update

    ‖θ' − θ‖₂ ≤ lr (κ_t √d_t + wd ‖θ‖₂),

with `d_t` the number of coordinates that have received a nonzero gradient.
For `β = (0.9, 0.999)`: `κ₁ = 1`, `κ₂ = 1.0013`, `κ₈₀ ≈ 2.02`,
`κ_∞ = (1−β₁)/sqrt((1−β₂)(1−ρ)) ≈ 7.27`. CURATE-02 curation (fresh AdamW,
lr 1e-3, ≤ 80 steps) therefore moves θ by at most `≈ 0.148` per committed
update regardless of the data. [measured] the first committed update moved
θ by `0.07335 = 10⁻³ √5380` on all four drift roots, the `κ₁ = 1`
sign-step case.

This bound is cheap and rigorous, but it controls parameters, not mechanisms.

### 2.2 From parameter displacement to observer drift [open]

`‖a(θ₁) − a(θ₀)‖ ≤ sup_{ξ∈[θ₀,θ₁]} ‖∂_θ a(ξ)‖ · ‖θ₁ − θ₀‖`, where `∂_θ a` is
the mixed second derivative `∂²f/∂θ∂u` of the full 32-step recurrent suffix.
An a-priori bound needs (i) a lower bound on every RMSNorm input RMS along
the segment (the RMSNorm Jacobian norm scales like `‖γ‖_∞ / rms(z)` and is
unbounded as `rms(z) → 0`), (ii) SiLU first/second-derivative bounds
(`|silu'| ≤ 1.0999`, `|silu''| ≤ 0.5`), (iii) products over the suffix
length. Layer-product Lipschitz constants of this type are valid but
typically vacuous at delay 31. No such bound is claimed here.
[measured] one committed update changes the live carried-state target
derivative by a relative `α/‖a₀‖` reported in section 8. Drift of this size
after a single sign-like step is strong evidence that any useful a-priori
`L_a` will be loose.

### 2.3 Sketches and sampled residuals [proved / standard]

A zero residual on sampled directions is not a uniform bound (Lean
`sample_residual_not_uniform`: two maps agreeing on any finite sample set can
differ elsewhere). Randomized sketches do give *probabilistic* bounds: for a
Gaussian test matrix with r columns, Halko–Martinsson–Tropp (2011, Lemma 4.1)
give `‖ΔC‖₂ ≤ 10 sqrt(2/π) max_i ‖ΔC ω_i‖` with probability at least `1 − 10^{-r}`.
Each `ΔC ω_i` costs one live JVP (or VJP). Such statements are labelled
`PROBABILISTIC`, never `CERTIFIED`, and still require live evaluations.

### 2.4 Physical metric and factor-frame change, a priori [proved]

`U` is a block of θ, so `‖ΔU‖₂ ≤ ‖ΔU‖_F ≤ ‖Δθ‖₂` (section 2.1 bounds it).
Then

* `‖G₁ − G₀‖₂ ≤ 2‖U₀‖₂‖ΔU‖₂ + ‖ΔU‖₂²`;
* metric distortion: `‖α‖²_{G₁} ≤ (1 + ‖G₁−G₀‖₂/λ_min(G₀)) ‖α‖²_{G₀}`;
* frame residual: for a cached physical write `w = αU₀`,
  `dist(w, row U₁) ≤ ‖αΔU‖ ≤ ‖w‖ ‖ΔU‖₂ / σ_min(U₀)`, and component-identity
  transport `w' = αU₁` satisfies the same bound `‖w' − w‖ ≤ ‖w‖‖ΔU‖₂/σ_min(U₀)`.

Proof of the frame residual: `w = αU₁ − αΔU` with `αU₁ ∈ row U₁`, and
`‖α‖ ≤ ‖w‖/σ_min(U₀)` because U₀ has full row rank. These quantities need
only `σ_min(U₀)`, `λ_min(G₀)` (stored at insertion) and the optimizer bound, so
the atlas can screen frame transport without any forward pass. The receipts
report the measured residuals beside the a-priori bound.

### 2.5 Singular subspaces and near-null directions [proved]

For the protected map at numerical rank r, the null space `N₀` of `C₀` moves
by `sin Θ(N₀, N₁) ≤ ‖ΔC‖₂ / σ_r(C₁)`: for a unit `n ∈ N₀`,
`σ_r(C₁) dist(n, N₁) ≤ ‖C₁n‖ = ‖ΔC n‖ ≤ ‖ΔC‖`. Row spaces obey Wedin's
`sin Θ ≤ ‖ΔC‖ / (σ_r(C₀) − σ_{r+1}(C₁))` when the gap is positive [standard].
Near-null directions are by definition those with tiny `σ_r`, so they carry
no gap protection. Numerically null columns are not exact rational
dependencies, and tolerance-based membership is never claimed as exact.

### 2.6 Exact invariant quotients versus approximate residuals [proved]

An exact quotient is a subspace K with `C(θ)K = 0` and `a(θ)ᵀK = 0` for
*every* θ in the region, by architecture (for example a coordinate the suffix
provably never reads). Such K transports across checkpoints at zero cost.
The RMSNorm scale invariance `RMSNorm(cz) = RMSNorm(z)` gives a θ- and
state-dependent invariant direction of the normalized block input, not a
fixed subspace of the carried state; it does not yield a cross-checkpoint
quotient. In the native TRM no fixed exact quotient of the 64-dim carried
state is known. All measured null spaces are approximate, and their
residuals are logged (`rank1_*`, `nullspace_sin_*`).

### 2.7 Signed chronological Jacobian transport [proved]

The write-space Jacobian is the chain `J_w = J_{post-block} · D_RMS · W_down ·
diag(silu'(p))` with p the low-visit pre-activation. `silu'(x) = σ(x)(1 +
x(1 − σ(x)))` is negative for `x < −1.2785`, so transporting a signed
state-space observer to the write site (or across checkpoints whose
pre-activations cross −1.2785) can flip row signs. Signed transport across
sites is valid only with live local factors `D_RMS`, `silu'(p)`; the
downstream factor `J_{post-block}` is the expensive part, and its drift is
exactly the drift of section 2.2.

### 2.8 Curvature and finite dose [proved / open]

If `‖∇²f₁‖ ≤ M` on the dose ball then `R_f ≤ ½Me²` [standard Taylor].
No rigorous M for the native suffix is available (same RMSNorm obstruction
as 2.2) [open]. [measured] the dose-grid remainders on the live full-span
direction (0.25/0.75/1.5/3) give an empirical scaling exponent; a fitted
exponent is a description, never a bound.

### 2.9 Precision and compressed storage [proved]

Round-to-nearest storage at unit roundoff u with no overflow satisfies
`‖fl(C) − C‖₂ ≤ ‖fl(C) − C‖_F ≤ u‖C‖_F + η sqrt(numel)`, η half the smallest
subnormal spacing (`atlas.quantize`, tested in `test_mms.py`). For float16
`u = 2⁻¹¹`. Storage error enters P2 additively:
`γ ← γ + (u‖C₀‖_F + η sqrt(numel)) · ‖G^{-1/2}‖₂`, and P1 likewise with `a₀`.
Compared with the measured one-update drift (section 8), float16 storage is
negligible: the binding uncertainty is mechanism drift, not storage.
float32 evaluation of the native verifier differs from reverse-mode rows by
≈1e-5 [measured, `row_parity_max_abs`, forward-mode test].

## 3. Local fixed-query RMS certificates do not certify recurrent outputs [proved]

Counterexample. State `h = (h₁, h₂)`; local reader `r(h) = h₁`; the suffix
map swaps coordinates each step, terminal output `y = h₁` after an odd number
of steps. The perturbation `δ = (0, s)` has local reader change 0 (any local
RMS certificate passes with zero margin used) and terminal change `s`, which is
unbounded. More generally the terminal deviation of a locally certified
perturbation is controlled by `‖P_out Φ_suffix (I − P_local)‖`, the suffix
Jacobian restricted to the local reader's kernel. Translating SCCG-09's local
reader envelope into full-task protection therefore requires a bound on that
restricted suffix map over the dose ball. That is SCCG-02's tube premise, which
has not been established for the native TRM. Results stay scoped: the drift
instrument certifies nothing about outputs beyond the declared 36 rows, and
those only by explicit native evaluation.

## 4. When do old cancellation circuits remain useful? [proved criterion, measured outcome]

Let a cached circuit be a support S (columns `B_S`) with a cached zero- or
low-collateral direction `u = B_S c`. Write `α_S = ‖(a₁−a₀)ᵀB_S‖_{G_S⁻¹}`,
`γ_S = ‖(C₁−C₀)B_S G_S^{-1/2}‖₂`, `G_S = B_SᵀB_S`. The circuit remains
admissible at θ₁ iff (sufficient, under P3 on the support)

    ‖C₀u‖ + eγ_S + R_p ≤ η   and   a₀ᵀu − eα_S − R_f ≥ floor.

Because `C₀u ≈ 0` lies in a near-null direction, section 2.5 gives no gap
protection: live collateral is `‖ΔC u‖ ≤ γ_S e`, linear in drift. `γ_S ≤ γ`,
and for small supports it can be much smaller. `γ_S` and `α_S` are measured
exactly with |S| JVPs (the *support-restricted refresh* tier), which is
cheaper than a full refresh only when `|S| · c_JVP < c_full`. [measured]
on this model `c_JVP ≈ 0.3 s` versus a full observer at 0.63 s (state) or
1.72 s (write) per edge, so restricted refresh beats a full observer only for
`|S| ≤ 2` (state) or `|S| ≤ 5` (write). Counting the 0.6–1.1 s support search
that a full refresh also pays, k = 4 restricted refresh is cheaper end to end
(section 8).

## 5. Upper bound on the benefit of refreshing an observer [proved, conditional]

Over a declared span with live feasible set
`F₁ = {u : ‖u‖_G ≤ e, ‖g₁(u) − g₁(0)‖ ≤ η}`, let `L` be the *verified* live
gain of the cached candidate (exact by tier 4) and U any valid upper bound
on `sup_{F₁} f₁(u) − f₁(0)`. Then the steering-gain benefit of refreshing is at
most `U − L` (Lean `refresh_value_le_gap`). With uniform premises P1–P4,
SCCG-10's dual bound supplies U *without* refreshing, so

    refresh benefit ≤ min_λ sqrt((e²+λ(η+E_p)²) a₀ᵀ(G+λC₀ᵀC₀)⁻¹a₀) + E_f − L.

Refreshing is unjustified in steering terms when this is below the value
of the refresh cost. Without uniform α, γ, R (the native case, section 2.2)
no a-priori refresh-value bound exists. The posterior value `V* − L` is
observable only after paying for the refresh, and the receipts measure it
(`live_*` versus `cached_*`). This bounds steering gain, not learning value.

## 6. From steering gain to learning value: conditions and counterexamples

**Counterexample 1 [proved, Lean `steering_gain_not_learning_value`].** A
certified-positive steering gain coexists with a strictly worse ordinary
validation loss after the corresponding update (`G(θ)=θ`, `V(θ)=θ²`, step
0→1).

**Counterexample 2 [proved].** SCCG quantities are derivatives at `u = 0`;
ordinary accuracy depends on clean margins. Adding a constant correct-class
logit offset changes accuracy arbitrarily and leaves `a, C` and every SCCG
bound unchanged. So no SCCG gain bound implies an accuracy bound without
margin information.

**Counterexample 3 [measured, CURATE-02].** The mechanistic calibration
score predicted realized one-step mechanistic-loss gains with Spearman 0.915,
yet the mechanistic arm did not beat equal-CPU reinvestment
(49.15% vs 49.44%). Predictive accuracy of the scorer is not learning value.

**Sufficient conditions for a positive translation [proved under stated
assumptions].** Let ordinary validation loss V be L-smooth on the step
region and the mechanistic calibration gradient satisfy
`‖∇L_mech − c∇V‖ ≤ ε‖∇V‖` for some c > 0 (alignment). For candidate steps
`Δθ_i` with `‖Δθ_i‖ ≤ D` (section 2.1 gives D a priori),

    |−∇Vᵀ Δθ_i − (−∇L_mechᵀ Δθ_i)/c| ≤ (ε/c)‖∇V‖ D,
    V(θ+Δθ_i) − V(θ) ≤ −∇VᵀΔθ_i + ½ L D².

Hence the mechanistic ranking certifies the validation ranking whenever its
score gaps exceed `2(ε/c)‖∇V‖D` (Lean `ranking_certificate`,
`ranking_topset_threshold`), and the selected step decreases V when its
validation-direction gain exceeds `½LD²`. Every quantity involving ∇V uses
validation data. Estimating it on TRAIN tune keeps custody, but the alignment
ε is then itself an estimate. Multi-window value further needs (A4)
bounded interaction between successive selections, which AdamW moment
coupling violates generically. Without alignment there is no positive
translation (counterexample 1 is the misaligned case).

**Fixed-CPU break-even [proved, first-order additivity].** If an ordinary
window costs `c_o` and yields expected improvement `v_o`, and an audited
window costs `c_o + c_a` and yields `v_c`, the audited policy wins at fixed
CPU iff `v_c / v_o > 1 + c_a / c_o`. CURATE-02 measured `c_o ≈ 100/2188 ≈
0.046 s` and `c_o + c_a ≈ 100/80 ≈ 1.25 s`, so audits had to be about 27× more
valuable per window than ordinary windows. That they tied (49.15% vs 49.44%)
says the measured per-window advantage was about 27×. The limiting
cost is the audit, not the mechanism signal. An atlas helps only if it
removes a large share of `c_a` without lowering `v_c` (section 8 decomposes
`c_a`).

## 7. Admission scopes used in code

| Scope | Meaning | Where |
|---|---|---|
| `CERTIFIED_CONDITIONAL_IN_DECLARED_SPAN` | SCCG-10 inequalities with premises of provenance `uniform` | `atlas.decide` tier 2/3; exact fixtures only |
| `VERIFIED_FINITE_ON_DECLARED_QUERIES` | candidate evaluated by native continuation on the declared rows | tier 4; drift `cached_*`/`live_*` outcomes |
| `HEURISTIC_MEASURED_INPUTS` | SCCG-10 arithmetic run on measured drift and sampled remainders | drift `admission_k4` |
| `HEURISTIC_PROPOSAL` | atlas reuse awaiting the trainer's committed-update acceptance check | economics `D/E/F` |

A failed certificate or screen means reuse has not been justified. It never
labels the control problem infeasible.

## 8. Empirical anchors [measured]

Pooled over drift roots 28000, 28001 (historical, replay only), 29200 and
29201 (fresh), 24 panel edges each. Source: `results/engineering/summary.json`.

| updates | ‖Δθ‖ | AdamW a-priori bound | state α/‖a₀‖ | state γ/‖C₀‖ | target cos | null-space sin (rtol 1e-3) | ΔC rank (90% energy) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.073 | 0.073 | 0.47 | 0.33 | 0.92 | 0.59 | 3.3 |
| 2 | 0.114 | 0.147 | 0.90 | 0.53 | 0.84 | 0.79 | 3.3 |
| 4 | 0.167 | 0.294 | 1.08 | 0.64 | 0.74 | 0.89 | 3.6 |
| 8 | 0.231 | 0.59 | 1.29 | 0.76 | 0.67 | 0.93 | 3.7 |
| 64 | 0.488 | 6.38 | 3.09 | 1.06 | 0.45 | 0.98 | 3.7 |

* A single committed sign-like AdamW step changes the full-suffix target
  derivative by about half its norm and rotates the protected near-null space
  by about 36°. Cross-checkpoint reuse of a signed observer as if current is
  not justified on this model at any age ≥ 1.
* The change is low-rank: 3–4 directions carry 90% of `‖ΔC‖_F²` (27 rows).
  That is a measured regularity, not a structural guarantee (section 2.3).
* The section 2.1 bound is exact at step 1 and 13× loose at step 64. The
  section 2.4 frame bound (with measured ‖ΔU‖) is valid and about 10× loose:
  measured physical-write frame residual 0.4% at step 1 and 2.3% at step 64,
  against bounds of 4.3–6.4% and 28–33%.
* Remainder-versus-dose log-log slopes are 1.74–2.06 on every root and
  action space, consistent with a quadratic remainder (section 2.8).
* Section 4 criterion in action (state space, k = 4, budget 0.25). A cached
  direction keeps 92% → 48% of the live target gain over 1 → 64 updates, while
  its collateral grows 0.28 → 0.52. The same cached *support* re-solved with
  exact live JVPs (support-restricted refresh) holds collateral at 0.25 with
  gain 0.109 → 0.085, against 0.12–0.13 for a fresh search. Old cancellation
  supports stay useful, but their coefficients must be refreshed.
* Full-span cached proposals keep target gain (0.28–0.33 vs live 0.30) but
  collateral grows 0.76 → 1.91, against 0.63–0.75 live. Finite collateral of
  the live full-span proposal already exceeds its linear budget of 0.25
  because of the quadratic remainder at dose 0.75.
* Factor-write action space: gains 0.04–0.07 margin units. Cached transport
  keeps 94% → 36% of the live gain. The heuristic SCCG-10 arithmetic on k = 4
  returns `NO_HEADROOM_IN_DECLARED_SPAN` for 64–77 of 96 cases (floor 0.1).
* The heuristic SCCG-10 admission (measured drift, sampled remainder) never
  returned `CONDITIONAL_REUSE_CANDIDATE` in 1,344 cached-k4 cases across both
  action spaces.
* Per-edge CPU (2 threads): full observer 0.63 s (state) / 1.72 s (write),
  support search 1.13 / 0.57 s, native verification of one candidate 0.13 s,
  support-restricted refresh with forward-mode AD 1.27 s (k = 4) / 2.5 s
  (k = 8). Verification is about 13× cheaper than refresh plus search.
  Forward-mode JVPs cost about 20 plain forwards each in this implementation.
  Restricted k = 4 refresh (1.27 s) therefore undercuts full refresh plus search
  (1.76 s state, 2.29 s write) only modestly, and costs about 10× a verification.
* Privileged full-hybrid transport: matched dose 0.75 flips 5 of 78 eligible
  targets with 10 of 674 protected corruptions; natural dose flips 27 of 78
  with 42 of 674. The CURATE-02 behavioral selective-control failure is
  reproduced in kind, and causal auxiliary training stays closed.

Curriculum economics (fresh engineering roots; RESULTS.md has full tables):

* One audited window costs about 0.65 CPU-s with transactional previews:
  16 previews 0.41 s (63%), calibration gradient about 0.14 s (20%), acceptance
  check about 0.1 s. A plain window costs about 0.06 s. A cross-checkpoint
  calibration atlas can remove only the 20% share.
* The batched analytic AdamW preview (`batched_preview.py`) matched the
  transactional preview to within 2.6e-6 (gains of 0.1–0.35), with identical
  top-4 sets in every benchmark window. It reproduced the transactional
  fixed-update checkpoint SHAs of B and C on both roots (80 windows) at
  2.2–3.0× lower total arm CPU. This exact economy attacks the dominant cost.
* Alignment, not scoring, is the limiting factor. With training delays 1–3
  after warmup, even the best single candidates have positive predicted
  delay-15 feedback gain in only 17–75% of windows, and realized gains are
  positive less often (curvature). Curated arms act mainly as a gate that
  rejects updates harmful to long-delay memory (section 6, misaligned case).

## 9. Lean formalization (MathCan `f0e6154`, `MathCan/SCCG10Admission.lean`)

`lake build`: `Build completed successfully (134 jobs)`, rerun independently by
the parent session. `#print axioms`:

| Theorem | Axioms |
|---|---|
| `stale_target_lower`, `stale_protected_linf`, `refresh_value_le_gap`, `ranking_certificate`, `ranking_topset`, `ranking_topset_threshold` | propext, Quot.sound |
| `steering_gain_not_learning_value` | none |
| `sample_residual_not_uniform`, `cauchy_schwarz`, `squared_l2_functional_bound` | propext, Classical.choice, Quot.sound |

`lake env leanchecker MathCan.SCCG10Admission` exited 0. Statements are over
`Int` (exactly scaled rationals), with vectors as `Fin n → Int`. Not
formalized: real ℓ₂ norms, the real Cauchy–Schwarz dual upper bound, the
AdamW bound of 2.1, anything about native transformer outputs.

## 10. Ranked unresolved problems for the next frontier review

1. **A-priori mixed-derivative bound** `sup ‖∂²f/∂θ∂u‖` along an AdamW step
   for the native recurrent suffix (needs certified RMSNorm input-RMS lower
   bounds). Without it, neither the refresh-value bound nor any cross-checkpoint
   certificate is available for native outputs.
2. **Uniform finite remainder R over the dose ball** for the native suffix
   (interval or Lipschitz propagation through 32 steps with RMSNorm/SiLU);
   currently only per-candidate verification is sound.
3. **Restricted suffix tube bound** `‖P_out Φ_suffix (I − P_local)‖`, needed to
   lift SCCG-09 local reader certificates to full-task protection (section 3).
4. **Alignment ε between mechanistic and validation gradients** with a
   TRAIN-tune estimator that has a valid confidence statement under the
   custody rules; the only route to a learning-value claim (section 6).
5. **Cheaper exact candidate scoring**: closed-form AdamW previews from
   per-candidate gradients (vmap) to cut `c_a` without changing selections;
   requires a float32 rounding analysis to certify identical top-k sets
   (Lean `ranking_topset_threshold` supplies the gap condition).
6. **Probabilistic certificates** (HMT-type sketches) for `γ_S` with explicit
   failure probability, and how to price their failure in the ledger.
7. **Multi-window additivity**: bound the interaction term between successive
   curated windows under AdamW moment coupling.
8. **Exact structural quotients** of the carried state, if any, from the
   tied-factor architecture.
