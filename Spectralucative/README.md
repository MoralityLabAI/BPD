# Spectralucative

Spectralucative applies spectral proofs to BPD's component masks and recurrent
state. It records the matrices needed to inspect loop transport and evaluates
complete Boolean intervention cubes so that response energy can be accounted
for exactly in the underlying finite mathematics.

The first study runs the repository's native miniature TRM and real
`BPDModel` wrappers on CPU. Its four models start from random initialization.
One warmup transition precedes the four measured transitions.
The four-token fixture measures implementation behavior; it supplies zero
trained-model or task-performance results. The protocol is in
[`PROTOCOL.json`](PROTOCOL.json).

The [first completed run](RESULTS.md) passed 53 tests and independent replay.
Its complete matrices, intervention responses, and execution history are
saved under [`results/`](results/).

## What this adds to BPD

BPD already exposes one physical factor pair per shared matrix and
pass-indexed masks through `PassMaskSchedule`. DT04 measures subspace closure,
and DT05 follows JVPs through several recurrent steps. Spectralucative extends
those instruments with signed matrices and finite-horizon error accounting.

| BPD quantity | Spectralucative instrument | Useful interpretation |
|---|---|---|
| Signed state Jacobian projected into a chosen subspace | Trace powers and a determinant expansion with a stated truncation bound | Describes feedback within that frozen local operator. |
| State-dependent Jacobians across actual passes | Ordered transport products with leakage and a norm envelope | Measures how far a reduced model drifts from the full linearized trajectory. |
| Full-head logit response over every mask of four components | Walsh coefficients, Parseval, and residual energy after dropping higher degrees | Measures how much of this intervention response needs interactions among several components. |
| BPD factors of a square linear recurrence and its retained weight residual | Exact linear-layer intertwining witness | Identifies the part of linear closure error contributed by the retained residual. |

The existing `compute_transition_operator` pools RMS attribution magnitudes.
It describes an attribution graph. Signed state Jacobians are recorded here
to study perturbation propagation. Similarly, the archived DT05 singular
values cannot recover the matrices' traces; this study saves the operators
themselves.

## Trace series at a frozen local operator

For a reduced real matrix `A`, the instrument checks

\[
S_K(z,A)=\sum_{k=1}^{K}\frac{z^k\operatorname{tr}(A^k)}{k},
\qquad q=|z|\|A\|_\infty<1.
\]

The [Lean theorem](https://github.com/patrickdugan/OntologicalProofs/blob/520006b4611f549ad9f0501dfee250f4052ce7e4/Neoplatonic/LTR/TraceDeterminant.lean)
proves that the exponential of the full series equals `det(I-zA)^-1`.
For dimension `d`, the omitted logarithm tail is bounded by

\[
|S_\infty-S_K|
\le\frac{d\,q^{K+1}}{(K+1)(1-q)}.
\]

This follows from `|trace((zA)^k)| <= d*q^k` and a geometric tail.
The software reports floating-point agreement separately from that analytic
bound. It retains the raw `z=1` eligibility result and records any damping
chosen for a convergent diagnostic. A damped calculation describes its
specified scaled operator.

## Transport through changing passes

Let `B_t` be the full-state Jacobian, `A_t` its proposed reduced update, and
`L_t` the map from reduced to full state. Define

\[
E_t=B_tL_t-L_{t+1}A_t,
\qquad
D_t=\Phi_B(t,0)L_0-L_t\Phi_A(t,0).
\]

The exact recurrence

\[
D_{t+1}=B_tD_t+E_t\Phi_A(t,0)
\]

gives a norm envelope

\[
u_0=0,\qquad
u_{t+1}=\|B_t\|_\infty u_t+
\|E_t\|_\infty\|\Phi_A(t,0)\|_\infty.
\]

It follows by expanding `D_(t+1)` and inserting
`B_t L_t Phi_A(t,0)`. The software independently reconstructs the telescoping
sum. Products retain their chronological order, and every changing
inclusion map remains explicit. This is the algebraic extension of the
[fixed-operator Lean result](https://github.com/patrickdugan/OntologicalProofs/blob/520006b4611f549ad9f0501dfee250f4052ce7e4/Neoplatonic/LTR/ApproximateIntertwining.lean).
Its implementation is checked numerically in this repository.

For orthonormal columns `Q_t`, a useful reduced update is
`A_t = Q_(t+1)^T B_t Q_t`; the closure defect records the part of `B_t Q_t`
outside that next subspace. A nonlinear model also needs a finite-displacement
remainder. Native receipts retain that measurement so the tangent estimate
can be compared with the delivered perturbation.

There is also a direct factor-level application to a square recurrent weight.
In column-state convention,
`W_hat = U^T V^T`, `L = V^T`, and `K = V^T U^T` satisfy
`K L = L W_hat`. For `W = W_hat + DeltaW`, the closure defect is
`K L - L W = -V^T DeltaW`. For a rectangular physical matrix, a Jacobian `R`
from its output back to its next input is needed to form `V^T R U^T`.
The native study uses the full recurrent-state Jacobian, which includes that
return path and the other state updates.

## Complete component-mask cubes

Each Boolean bit keeps or removes one BPD component at a declared invocation.
The invocation is the first one in the first measured transition, at the first
token after the prefix. This study uses zero ablation with sphere rescaling
disabled.
The residual weight remains active in every condition. Thus the all-ones
mask has a target-parity check, while the all-zeros mask still contains the
retained residual and bias.

The response `f(m)` is the final logit-delta vector over all four token
positions and all eleven head rows. Using that vector preserves the signs
of the response. Squared logit drift is recorded separately, since squaring
an additive response itself creates degree-two terms.

On all `2^C` masks,

\[
\widehat f(s)=2^{-C}\sum_m f(m)(-1)^{\langle s,m\rangle},
\qquad
2^{-C}\sum_m \|f(m)\|_2^2=\sum_s\|\widehat f(s)\|_2^2.
\]

If `f_<=r` keeps coefficients of degree at most `r`, then

\[
2^{-C}\sum_m\|f(m)-f_{\le r}(m)\|_2^2
=\sum_{|s|>r}\|\widehat f(s)\|_2^2.
\]

The [Walsh Lean module](https://github.com/patrickdugan/OntologicalProofs/blob/520006b4611f549ad9f0501dfee250f4052ce7e4/Neoplatonic/LTR/WalshFourier.lean)
proves scalar Parseval and general residual energy under the uniform cube
measure. Applying it to each output coordinate and summing gives the vector
identities above. Spectralucative checks the degree truncation with an independent parity-matrix
calculation. A partially sampled or duplicated mask table is rejected.
Delivered intervention doses are recorded because different masks can change
the physical displacement.

## Evidence and reproducibility

The pinned EvalMath preflight was rerun on the original eight-root DT05
development input. Its strict exit status is `2`, and its preserved verdict
is `REDESIGN_OR_PILOT_ONLY`. The source, input and output hashes appear in
[`evidence/provenance.json`](evidence/provenance.json). That result concerns
the historical transported-versus-static task comparison. The present packet
tests instruments on separately named, fully exposed engineering fixtures.

All 16 mask outcomes and all four recurrent transitions are kept for every
initialization. Native receipts distinguish the number of initialized models
from repeated measurements within each model. They include source hashes,
matrices, physical-dose measurements and numerical tolerances for independent
replay. The existing frozen source and archived evidence checks run in CI.

### Run the numerical checks

From the repository root:

```bash
python -m pip install numpy
python -m unittest discover -s Spectralucative/tests -p 'test_math.py' -v
```

### Run the native CPU study

```bash
python -m pip install -r Spectralucative/requirements.txt
python -m Spectralucative.native --out runs/spectralucative --roots 4 --steps 4 --rank 3 --components 4 --cpu-threads 2
python -m Spectralucative.verify --run runs/spectralucative --out runs/spectralucative/verification.json
```

The GitHub Actions workflow runs the independent receipt verifier after the
native study and uploads its receipts. It also reruns the historical preflight
and checks BPD's frozen source and evidence. The standalone historical replay
is available through `python -m Spectralucative.verify --historical-only`.
