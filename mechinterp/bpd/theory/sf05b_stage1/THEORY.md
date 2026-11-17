# SF05B Stage 1: a global public-readout bound for a predicted correction

**Source-independent mathematical analysis, November 18, 2026.** The trained model and oracle-free policy are specified in [PROTOCOL.md](PROTOCOL.md). This theorem is an analytical bound conditional on the model's physical weights; it is separate from the posttraining OOD measurements.

Let \`h∈R^d\`, an event/query \`x\`, \`A∈R^{d×m}\`, \`B∈R^{p×m}\`, \`b∈R^m\`, and \`C∈R^{m×d}\`. The weight-tied recurrent transition is

\[
F(h,x)=h+\tanh(hA+xB+b)C,\qquad q(h)=h_0.
\]

The public future scalar for query \`x_U\` is \`G_U(h)=q(F(h,x_U))\`. Write \`c_0=C e_0∈R^m\`. The map \`tanh\` is 1-Lipschitz componentwise and as a map on Euclidean vector spaces. Consequently, for any hidden states \`z,z'\`,

\[
|G_U(z)-G_U(z')|
\le |z_0-z'_0|+\|A\|_{\mathrm{op}}\|c_0\|_2\,\|z-z'\|_2.
\]

**Proof.** The residual skip contributes exactly \`z_0-z'_0\`. Apply Cauchy–Schwarz to the remaining scalar product with \`c_0\`, the Euclidean 1-Lipschitz property of \`tanh\`, then the spectral operator norm bound \`\|(z-z')A\|_2≤\|z-z'\|_2\|A\|_{\rm op}\`. QED.

If two interventions \`d,d'\` both conserve the actual current readout (\`d_0=d'_0=0\`), the skip term cancels and gives the sharper, **global** inequality

\[
|G_U(h+d)-G_U(h+d')|
\le L_{\rm dark}\|d-d'\|_2,\quad
L_{\rm dark}=\|A\|_{\rm op}\|C_{:,0}\|_2.
\]

Thus if an oracle-generated repair \`d_*\` yields \`|G_U(h+d_*)-G_U(h)|≤\epsilon_*\`, while an oracle-free learned policy outputs \`\hat d\` with zero current readout, its public residual obeys

\[
|G_U(h+\hat d)-G_U(h)|
\le\epsilon_*+L_{\rm dark}\|\hat d-d_*\|_2.
\]

This bound holds over the **complete finite edit**, with no local Taylor or Hessian approximation and no reachable-manifold assumption. It is conservative and may be loose. Its value is that it offers a checkable global upper certificate once \`A,C\` are frozen, and cleanly separates predictor approximation error from the oracle level-set tolerance.

To certify a statistical public-error guarantee, one must additionally bound \`\|\hat d-d_*\|\` for new root and history distributions with independent tests. An average regression MSE does not alone certify a worst-case per-example privacy threshold. The oracle output itself uses privileged \`G_U\` queries at training time, while policy inference sees only \`[h,d,x_U,q(h)]\` and has no oracle. The theorem concerns one specified synthetic scalar query channel, not a complete generated text transcript or data-exfiltration security.

## Gauge and source identity

For any invertible physical component-basis change \`R\`, \`V' = VR\`, \`U'=R^{-1}U\` leaves \`VU\` and therefore the full donor physical write unchanged. An output-SVD projector of \`VU\` is similarly gauge invariant up to numerical issues at tied singular values. The policy sees the **physical** proposed edit, so its output is unchanged by pure BPD-factor gauge transformations when the physical write is identical. The teacher true \`C\` and its rank-24 projection remain privileged controls; they are not evidence that a canonical individual BPD component was discovered.

The Stage 1 source-family split tests history-length and overwrite-template generalization with the *same* three source identifiers. A heldout failure must be reported separately from a policy failure. Scientific calibration over 59 independent roots and subsequent confirmation over 32 roots remains unexecuted.
