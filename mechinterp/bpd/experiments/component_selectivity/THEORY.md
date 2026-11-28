# RP04: component reads, physical writes and memory carry

RP04 examines how a proposed state edit travels through the native BPD factors
and the GRU recurrence. A selected factor span fixes part of the read response.
Other components can respond to the same edit because their read vectors share
directions in hidden space. The resulting writes reach several nonlinear gates,
while the hidden state also enters a carry path directly.

This note specifies the algebra behind those measurements and the conditions
under which they support a future training objective. RP04 uses exposed RP03
checkpoints and remains **DEVELOPMENT_ONLY**. The saved warmup and four final
arms within a root are repeated measurements of that root.

The executed operator is defined in
[`TiedComponentLinear`](../../components.py). The GRU equations and the
hidden-state intervention follow
[`ComponentStateMemory`](../training_steering_stability/model.py).
The [RP03 protocol](../training_steering_stability/PROTOCOL.md) supplies the
original component groups and their selection history.

## 1. Row-vector convention and native BPD execution

Let a hidden state be a row vector \(h\in\mathbb R^{1\times d}\). The saved
factors have shapes

\[
V\in\mathbb R^{d\times C},\qquad
U\in\mathbb R^{C\times 3d},\qquad W=VU.
\]

Here \(d=C=32\). Native component execution computes

\[
a=hV,\qquad w=aU=hW.
\]

The vector \(a\) contains component reads, and \(w\) contains the reset,
update and candidate preactivations. PyTorch's stored reconstructed linear
weight has shape \(3d\times d\) and equals \(W^\mathsf T\).
Every component mask is open in this model. The residual mask is zero, so
the saved initialization buffer `W_target` contributes zero to the executed
write. These checkpoints train V and U directly; their factorization belongs
to this particular learned parameterization.

For selected component indices \(G\), let \(M_G\) be their diagonal selector.
Define the physical group operators

\[
W_G=VM_GU=V_GU_G,\qquad
W_{\bar G}=V(I-M_G)U,\qquad W=W_G+W_{\bar G}.
\]

The state edit \(\delta\) changes every read according to
\(\Delta a=\delta V\). Component \(j\) contributes the physical write

\[
\Delta w_j=(\delta v_j)u_j,
\qquad
\Delta w_G=\delta W_G,
\qquad
\Delta w_{\bar G}=\delta W_{\bar G}.
\]

In the final expression, the subscript `outside` is clearer in machine-readable
receipts: `write_outside = delta @ W_outside`.

## 2. What projection onto selected V columns preserves

Let \(P_G\) be the orthogonal projector onto \(\operatorname{col}(V_G)\).
For a proposed memory displacement \(d_*\), the raw span edit is

\[
q_{\rm span}=d_*P_G.
\]

Since \(P_GV_G=V_G\), it obeys

\[
q_{\rm span}V_G=d_*V_G.
\]

It is the minimum Euclidean-norm hidden-state change satisfying those selected
read equations. The complementary part \(d_*(I-P_G)\) lies in the null space
of the selected reads.

Outside reads take values \(d_*P_GV_{\bar G}\), which can be nonzero. The
induced write therefore has the exact decomposition

\[
q_{\rm span}W
=d_*W_G+d_*P_GW_{\bar G}.
\]

This identity separates the requested selected-component response from the
additional response through outside components. Their sum can help or harm
the requested memory change. Behavioral controls determine that sign.

The identity uses the projector from the **current** V columns. RP04's frozen
warmup projector has a different role: at a later checkpoint,
\(P_G^{(0)}V_G^{(t)}=V_G^{(t)}\) is a measurable approximation. RP04 records
projector drift and current-factor reads for each delivered edit.
A direct selected-read residual is a possible extension.

Receiver, donor and hybrid states are \(h_R,h_D,h_H\), measured immediately
after the four writes. The hybrid changes the selected semantic slot while
preserving the receiver's other values. Two useful proposed displacements are

\[
d_D=h_D-h_R,\qquad d_H=h_H-h_R.
\]

The actual data generator shares order, noise and suffix inputs across this
triplet. Hence \(d_D-d_H=h_D-h_H\) isolates the state difference associated
with the donor's other memory values, conditional on this sampled triplet.
The nonlinear recurrence makes this a joint context-dependent difference.
An additive effect for each individual slot would require further evidence.

## 3. A dual edit that isolates selected reads

The desired read change for a component-isolating edit is

\[
b=d_*VM_G.
\]

Solve the constrained state problem

\[
\min_q\|q\|_2\quad\text{subject to}\quad qV=b.
\]

When V is square and invertible, the unique answer is

\[
q_{\rm dual}=d_*VM_GV^{-1}.
\]

For a general V, the minimum-norm candidate is \(bV^+\), with feasibility
condition \(bV^+V=b\). Evaluate its actual residual. A rank-deficient factor
matrix can make the requested read vector infeasible. Its least-squares
approximation retains a measured residual and a corresponding scope label.

A feasible dual has the exact responses

\[
q_{\rm dual}V_G=d_*V_G,\qquad
q_{\rm dual}V_{\bar G}=0,\qquad
q_{\rm dual}W=d_*W_G.
\]

The linear map \(VM_GV^{-1}\) is an oblique projector in hidden coordinates.
It can have a large Euclidean operator norm. Float64 solves should record the
rank threshold, factor conditioning, read residual and physical write
residual. Casts back to the model dtype require another delivered-edit check.
Zero proposed edits and failed solves stay in the outcome inventory.

### Fixed physical dose changes the read amplitude

Every valid standardized state edit has dose \(\kappa=0.75\):

\[
\delta(q)=\kappa\frac{q}{\|q\|_2}.
\]

The raw dual satisfies all selected-read constraints of the span edit, plus
outside-read constraints. Consequently,

\[
q_{\rm dual}=q_{\rm span}+n,\quad nV_G=0,\quad
\langle q_{\rm span},n\rangle=0,
\quad \|q_{\rm dual}\|_2\geq\|q_{\rm span}\|_2.
\]

For nonzero feasible edits, define amplification

\[
A=\frac{\|q_{\rm dual}\|_2}{\|q_{\rm span}\|_2}\geq1.
\]

At equal state dose, the dual delivers \(1/A\) times the span edit's
selected-read change. Its selected physical write has the same scaling.
The registered comparison of hybrid dual with current hybrid span therefore
tests the complete isolation policy, including the cost of supplying that
read response through hidden state. Report A and delivered selected-write
norms beside answer accuracy.

An optional future diagnostic can match selected-read amplitude by applying
a common multiplier to both raw edits. That comparison has different physical
state doses. Its purpose and dose contract would need their own declaration.

## 4. Physical leakage, cancellation and gauge choices

For any delivered state edit, set

\[
g=\delta W_G,\qquad o=\delta W_{\bar G}.
\]

A useful outside-group write fraction is

\[
L_{\rm outside}
=\frac{\|o\|_2^2}{\|g\|_2^2+\|o\|_2^2}.
\]

Its denominator measures the separate group write magnitudes. Interference
appears in the identity

\[
\|g+o\|_2^2=\|g\|_2^2+\|o\|_2^2+2\langle g,o\rangle.
\]

Record the cosine between g and o and the cancellation statistic

\[
K=1-\frac{\|g+o\|_2}{\|g\|_2+\|o\|_2}.
\]

Positive K measures cancellation in preactivation space. Undefined ratios
receive an explicit null value and a count. A large outside fraction describes
how the edit reaches the recurrent operator; its value alone determines
neither semantic spillover nor task harm.

The factorization has the exact reparameterization freedom

\[
V'=VR,\qquad U'=R^{-1}U
\]

for any invertible \(R\), while W remains unchanged. Different metrics have
different invariance guarantees:

| Quantity | Preserved transformations |
|---|---|
| Full operator W and ordinary recurrence | Every invertible R |
| Group operator, selected span and feasible dual edit | R commutes with the group selector, including arbitrary invertible mixing inside each block |
| Individual physical write \((\delta v_j)u_j\) | Reciprocal component scaling and sign changes; joint index relabeling permutes the inventory |
| Raw component-read L2 magnitudes | Depend on the selected factor gauge |

Mixing factors across the group boundary changes the physical hypothesis
encoded by G while leaving ordinary model behavior unchanged. Component roles
are therefore claims about the specified factorization and intervention
policy. The saved factor bytes and selected indices belong in every receipt.
Per-component write magnitudes are useful diagnostics under that gauge;
group-summed physical writes support the broader within-block invariance.

A raw factor-coordinate penalty can shrink under reciprocal rescaling without
changing any model write. Physical operators or behavioral outcomes avoid
that particular degree of freedom. Numerical condition numbers still depend
on the representation supplied to the solver, so report them as solver
diagnostics.

## 5. How BPD writes interact with GRU carry

Hold the next token's encoded input \((i_r,i_z,i_n)\) fixed. Split the
recurrent write into \((w_r,w_z,w_n)\) and define

\[
r=\sigma(i_r+w_r),\qquad z=\sigma(i_z+w_z),\qquad
n=\tanh(i_n+r\odot w_n).
\]

The next-state function, with state and write exposed separately, is

\[
F(h,w)=(1-z)\odot n+z\odot h.
\]

A physical state edit supplies both \(h'=h+\delta\) and
\(w'=w+\delta W\). Let \(r',z',n'\) denote the gates induced by w'. Then

\[
F(h+\delta,w')-F(h,w)
=z'\odot\delta
+(z'-z)\odot(h-n)
+(1-z')\odot(n'-n).
\]

This finite identity separates direct memory carry, update-gate movement and
candidate movement. It remains exact at the full delivered dose.

The four pathway conditions are

\[
H_{00}=F(h,w),\quad H_{10}=F(h+\delta,w),\quad
H_{01}=F(h,w'),\quad H_{11}=F(h+\delta,w').
\]

Their one-step carry effect and interaction satisfy

\[
H_{10}-H_{00}=z\odot\delta,
\qquad
H_{11}-H_{10}-H_{01}+H_{00}=(z'-z)\odot\delta.
\]

Continuing these states through the same native suffix gives a behavioral
pathway diagnostic. RP04 saves route-specific answer outcomes and terminal
displacement norms. A future terminal interaction calculation would retain
all four terminal-state vectors. Each pathway inherits the
same parent state edit and its induced write. The write-only pathway's
preactivation dose is \(\|\delta W\|_2\); its units and dimensionality
differ from the state dose of 0.75.

### Prospective per-component local response

For component write \(\Delta w_j\), split its reset, update and candidate
coordinates. At the clean state, its differential gate response is

\[
\dot r_j=r\odot(1-r)\odot\Delta w_{r,j},\qquad
\dot z_j=z\odot(1-z)\odot\Delta w_{z,j},
\]

\[
\dot n_j=(1-n^2)\odot
\bigl(\dot r_j\odot w_n+r\odot\Delta w_{n,j}\bigr),
\qquad
e_j=(h-n)\odot\dot z_j+(1-z)\odot\dot n_j.
\]

The linearized full response is \(z\odot\delta+\sum_j e_j\). This differential
diagnostic is derived here for later work. Comparing it
with the finite next-state difference measures local nonlinearity at the
chosen dose. A large physical write can have a small local response when
gates saturate. The exact finite pathway experiment supplies the behavioral
check for such a differential observation.

### Implementation boundary

The RP03 model uses `TiedComponentLinear` in `component` mode. In the current
implementation, `_apply_intervention` is called by its `target` branch.
Pathway controls therefore require an explicit analysis wrapper or output
hook around native component execution. Their zero-edit, clean, and full-edit
corners should reproduce the corresponding saved-model continuations.
Changing the module to target mode would introduce `W_target` into the
experiment and change the operator under study.

## 6. State alignment and operator alignment answer different questions

For a true hybrid displacement d and a raw candidate q, useful additional
measurements are the state residual \(\|q-d\|_2\), induced-write residual
\(\|(q-d)W\|_2\), and their normalized forms when the respective reference
norm is nonzero. These paired residuals extend the registered RP04 metrics. At fixed
dose, the full-hybrid reference is \(\delta_H=\kappa d/\|d\|_2\), so
comparisons of standardized edits should use that standardized reference.
The natural hybrid state retains its separately declared displacement.

W weights directions by their effect on recurrent gates. The carry path
transmits state directions directly through z. Consequently, an edit's
induced-write alignment leaves a further question about its carried memory.
The finite next-state and terminal-answer diagnostics jointly address both
routes.

The output factors impose a second geometric constraint. Let \(Q_{U_G}\)
be the orthogonal projector in \(\mathbb R^{3d}\) onto the row span of
\(U_G\). For a desired physical write t, every selected-component write
satisfies the lower bound

\[
\|cU_G-t\|_2\geq\|t(I-Q_{U_G})\|_2
\quad\text{for every read vector }c.
\]

The ratio \(\|t(I-Q_{U_G})\|_2/\|t\|_2\) measures a limit imposed by U's
selected write span. It complements the V-span state residual. A state edit
must also supply c through V at the allowed physical dose, so a small
U-span residual leaves an actuation-cost question. These spans and their
group operators preserve the within-block gauge invariance stated above.

## 7. How a later training loss can use these findings

RP04 trains zero parameters. A future experiment can choose a physically
grounded BPD objective after observing which failure occurs, then freeze its
recipe before assigning new model roots.

A behavioral role objective can score every query address q for a proposed
edit of semantic slot s:

\[
\mathcal L_{\rm role}
=\mathbb E\left[
\operatorname{CE}(f_\theta(x_R^{(s)};\delta_s),y_H^{(s)})
+\frac{\lambda_{\rm keep}}{S-1}
\sum_{q\ne s}\operatorname{CE}
(f_\theta(x_R^{(q)};\delta_s),y_R^{(q)})\right].
\]

This objective asks the model to deliver the requested value and preserve the
other addressed values. Ordinary task loss remains part of training. All
comparison arms should receive the same ordinary receiver, donor and hybrid
examples, with declared gradient and compute budgets.

If the diagnostic supports component isolation, a candidate additional term
is a penalty on the **physical outside-group write**,
\(\|\delta_sW_{\bar G}\|_2^2\), together with an explicit selected-response
target. Pairing these terms matters because shrinking every response also
reduces an isolated leakage penalty.

For example, take a frozen teacher snapshot, a dose-matched hybrid reference
\(\delta_H\), and its physical write
\(t=\operatorname{stopgrad}(\delta_H\bar W)\). A candidate write objective is

\[
\mathcal L_{\rm write}
=\frac{
\|\delta_sW_G-t\|_2^2
+\lambda_{\rm outside}\|\delta_sW_{\bar G}\|_2^2
}{\operatorname{stopgrad}(\|t\|_2^2)+\epsilon}.
\]

It has a defined physical target and preserves reciprocal factor-scale
invariance. Teacher choice, detach boundaries and the treatment of zero
reference writes require prospective specification. A state or next-step
teacher term can address a carry failure revealed by the pathway test.
The U-span residual supplies a useful feasibility diagnostic before fixing
the teacher target or the loss weight.

These objectives encode a stronger component-role hypothesis than ordinary
task training. Their usefulness must be measured through fresh root-level
task outcomes and selective intervention outcomes. In particular, a component
group shared by two slots can represent joint information; any subsequent
constraint that demands separate groups changes the model hypothesis.

The registered RP04 contrasts compare the complete frozen edit policies.
Inference stays clustered by the eight exposed model roots. Native factor
readouts, alternate checkpoints and the gate/carry pathways provide
descriptive evidence for designing that subsequent training experiment.
