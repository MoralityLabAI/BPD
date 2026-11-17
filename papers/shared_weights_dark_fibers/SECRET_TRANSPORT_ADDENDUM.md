# Secrecy as an observer-conditioned dark-fiber problem

**BPD paper addendum, 17 November 2026.** Additive research note for the shared-weights Dark Fibers program. Historical v014/published receipts are unchanged. The new SF01/SF02 results and ongoing SF03 controls are synthetic.

## 1. Information boundaries and recurrent observability

Inspired by [Rodríguez, Spectral Faithfulness](https://github.com/idolum-ai/spectral-faithfulness), which empirically distinguishes source-tag confusion, multi-user interference, and secret-bearing poker talk. Its "spectral" metaphor does not imply a measured eigenbasis. Our BPD research gives a separate mechanical meaning to the possible hidden-to-visible pathways.

Let `S_A,S_B` be randomized private inputs, `X` the public context, `R` the recipient authorization, `h_t` the complete recurrent state, and `q(h_t)` the actual visible output. A legitimate task may require

\[
I(S_A;Y_A\mid X,R=A)>0
\]

while bounding the information a public recipient obtains,

\[
I(S_A;Y_U\mid X,R=U)\ll H(S_A).
\]

These information quantities need an **explicit distribution and observer**. Under fixed fully observed input and deterministic recurrence, unconditional hidden-state entropy can be vacuous. Continuous deterministic outputs can create singular information quantities; report finite response alphabets, declared quantization or noisy observers rather than silently interpreting logit differences as bits of leakage.

An observable quotient `q∘F=T∘q` fails on a matched fiber when two distinct reachable states with the same current true readout yield different future readouts. In a secret-conditioned task, this failure can be **desirable** for an authorized recipient when future access to the secret is necessary. For an unauthorized recipient, the same delayed dependence can signal a boundary failure. The output observer, authorization and complete carried state determine the claim.

## 2. BPD-SF01: exact finite mathematical controls

[Merged SF01, PR #28](https://github.com/MoralityLabAI/BPD/pull/28) introduced conditional information ledgers and a two-pass tied linear positive control. Its chain rule separates ordinary public action information from the *incremental* information contributed by table talk:

\[
I(S;(B,T)\mid X)=I(S;B\mid X)+I(S;T\mid X,B).
\]

An output can reveal information already exposed by action; the conditional speech term prevents counting the same exposure twice. The synthetic XOR example shows that `I(S;K)=I(S;C)=0` while `I(S;(K,C))=1` bit, warning against interpreting zero singleton probe scores as lack of a coalition-level mechanism.

## 3. BPD-SF02: learned two-pass causal transport

[Merged SF02, PR #34](https://github.com/MoralityLabAI/BPD/pull/34) trains a shared 12D nonlinear recurrent transition to produce public context at its first readout and source/role-conditioned output at its second readout. It freezes the physical weights, then trains a **full-rank**, separately initialized 12-component BPD parameter factorization. Six components are selected on independent development source pairs and intervened at the first hidden step with exact current-readout preservation.

The [hosted locked eight-root receipt](https://github.com/MoralityLabAI/BPD/actions/runs/44117355061) reports **1,633/1,738 authorized donor-readout recoveries** within ±0.10, and all eight roots pass the exploratory differential recovery gate relative to 255 norm-matched random-dark directions. Complete dark donor patches recover all eligible heldout donor readouts in each root.

**Registered engineering status: FAIL.** Physical factor reconstruction MSE on root 1021 is 0.0002616, above the preregistered 0.0001 bound; only 7/8 roots pass all engineering tests. Selected partial-factor interventions can also cause public output changes comparable to random off-manifold dark edits. Neither result establishes an actual model secret leak. The full physical factor block remains gauge-invariant under invertible component reparameterizations; singleton interventions do not.

These outcomes show that *reconstructed physical factors can mediate source-specific delayed output changes in a trained synthetic system*. They do not establish recovery of unique secret modules or performance on natural-language tasks.

## 4. BPD-SF03: first-order matched sham

A secret-carrying direction may appear superior to a random same-norm perturbation merely because it aligns with the future-output Jacobian. [SF03](https://github.com/MoralityLabAI/BPD/pull/37) prospectively matches the physical dose and the **signed** downstream tangent response before comparing effects. For `G=q∘F` and current-dark directions `δ,η` with equal norm `ρ` and equal `DG(h)` action, Taylor's theorem gives

\[
|G(h+\delta)-G(h+\eta)|\le M\rho^2
\]

provided the Hessian of `G` is bounded by `M` along both edit paths. This is an assumption-conditional theoretical bound. Neural tests report finite nonlinear outcomes without assuming a verified global Hessian bound.

The prospective SF03 calibration uses 59 independent teacher roots to bound the upper tail of a sham-versus-shams statistic, then freezes a threshold before 32 fresh-root confirmation tests. Its results are pending in this note. Earlier excluded seeds 901 and 902 yielded tangent-sham recovery near the selected-factor recovery, motivating the more stringent study; those two seeds are engineering pilots, not confirmation.

## 5. Conditions for a stronger language-model claim

A future native small transformer experiment should freeze the model and source families before reading the confirmation data. It must capture all relevant current readout positions, heldout role/secret pairs, private/public transcript and action channels, and actual post-normalization output probabilities. Include source-swap, component-gauge and signed-gradient-matched nulls, full-block and full-dark positive controls, and exact post-intervention readout preservation. Report distinct authorized utility and unauthorized disclosure, with observer and quantization conventions made explicit.

The key question remains: does the selected BPD block transmit a secret to an authorized future output *with causal specificity beyond first-order output sensitivity*, and does it avoid sending that information to unauthorized recipients under the true generation protocol?
