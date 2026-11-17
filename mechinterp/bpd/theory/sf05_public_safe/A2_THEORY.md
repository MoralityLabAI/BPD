# SF05-A2: a finite public-level-set Newton correction

**Post-Stage-A mathematical note.** The failed Stage A tangent projection and its immutable receipts are recorded in [RESULTS_STAGE_A.md](RESULTS_STAGE_A.md). The new numerical correction is governed by [A2_PROTOCOL.md](A2_PROTOCOL.md) and [A2_LOCK.json](A2_LOCK.json). All A2 evidence comes from excluded teacher roots.

## Tangent plane and nonlinear level set

Fix a prequery hidden state \`h\` and an unauthorized role-U query \`x_U\`. Write \`q(h)=h_0\`, \`G_U(v)=q(F(v,x_U))\`, and \`H(v)=G_U(h+v)-G_U(h)\`. The **finite** constraint set is

\[
\mathcal M_{h,U}=\{v:v_0=0,\ H(v)=0\}.
\]

The Stage A linear intersection computes the tangent plane \`ker Dq(h)∩ker DG_U(h)\`. For a nonlinear recurrent transition, this tangent plane is only a first-order approximation to \`\mathcal M_{h,U}\` and has no global preservation guarantee. In the excluded-root pilot, the exact full physical donor edit lay near \`\mathcal M_{h,U}\` even though its tangent-plane projection moved it far away in terms of the actual public output. Euclidean proximity is the relevant geometric optimization criterion; the teacher's public output and authorized task loss depend nonlinearly on direction.

## One-step residual theorem

Suppose the projected gradient \`g(v)=P_0∇G_U(h+v)\` is nonzero. The minimum-Euclidean-norm **linearized correction** to \`H(v)\` in the current-readout-dark space is

\[
s=-\frac{H(v)}{\|g(v)\|^2}g(v),\quad s_0=0,
\]

because \`DH(v)s=g(v)^\top s=-H(v)\` and Cauchy–Schwarz identifies \`s\` as the minimum-norm solution to that one scalar equation.

If the gradient of \`G_U\` is \`L\`-Lipschitz along \`[h+v,h+v+s]\`, Taylor's theorem bounds the actual corrected residual:

\[
|H(v+s)|\le \frac{L}{2}\|s\|^2.
\]

This is a local bound conditional on an unmeasured \`L\`, not a certified global guarantee. The exact SF05 A2 implementation uses 8 capped Newton steps with a 4-option backtracking line search, retaining only updates that **reduce the measured** residual. If the projected gradient vanishes while the residual is nonzero, the solver records a failure. Cases that remain outside the tolerance after 8 steps remain recorded as nonconverged.

## Implicit-function intuition

At a point \`v_*\` satisfying \`H(v_*)=0\` and \`g(v_*)\ne0\`, the implicit function theorem locally describes a smooth codimension-one public-output level set inside the 19-dimensional current-readout-dark space. The level set has tangent space \`ker g(v_*)^\top\`; its tangent subspace varies with \`v_*\`. The *initial* public-Jacobian kernel at \`v=0\` therefore need not approximate the finite set accurately at the full donor intervention radius.

A corrected point \`v\` on the level set can retain authorized influence \`G_A(h+v)-G_A(h)\`, which is a separately queried channel. However, the solver accesses both \`G_U\` and its gradient at candidate hidden states, a privileged oracle unavailable to a normal external observer. The construction demonstrates mathematical feasibility of corrections in the synthetic teacher and is **not** a learned privacy gate, source-provenance proof or deployable leakage-prevention mechanism.

## Gauge and baseline controls

The raw physical \`W_hat=VU\` donor write and its output-SVD rank truncations are invariant under \`V→VR,U→R^{-1}U\`; their finite public corrections depend only on the resulting physical edit. The SF04 14-singleton coalition retains arbitrary factor gauge dependence. Apply the **identical** solver and cap policy to each candidate and to equal-norm random-dark and signed-authorized-gradient-matched *initial* shams. Corrections can alter the latter's authorized tangent, so label that matching as **pre-correction** in every report.

Primary accounting is unconditional over all preregistered eligible authorized examples; reporting only converged cases would create selection bias. Report convergence, public residual after correction, added correction dose, and full-sham distribution separately. This Stage A2 work cannot establish independent statistical significance or a new security property.
