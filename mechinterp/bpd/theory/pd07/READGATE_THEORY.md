# Factor-read gate identifiability and perturbation margin

**BPD-PD-07, analytic companion note (2026-11-17).** The [prospective protocol](PROTOCOL.md) is fixed before the registered science run. This note explains the sufficiency and limits of using the learned factor readout as the input to its own gate. [ReadGateSeparation.lean](ReadGateSeparation.lean) proves the exact-magnitude finite subset rule; the real-norm perturbation bound here is a paper proof.

## Planted-basis case

The observed input is \(x=aQ^\top\), with orthogonal \(Q\in\mathbb R^{4\times4}\) and latent \(a\) supported on at most three coordinates. If learned read columns satisfy \(V_j=c_j Q_{\pi(j)}\) for nonzero scales \(c_j\) and one permutation \(\pi\), normalization yields

\[
z_j=\frac{xV_j}{\|V_j\|_2}
   =\operatorname{sgn}(c_j)\,a_{\pi(j)}.
\]

The gate is \(h_j=1[|z_j|\ge\tau_j]\). Active latent magnitudes lie in \([.75,1.25]\), while inactive magnitudes equal zero. For every \(\tau_j\in(0,.75)\), \(h_j\) equals the latent activity indicator for every possible support, including the withheld pairs and triple. This follows from coordinatewise structure, so novel combinations do not directly change routing. It is a stronger inductive bias than a generic raw-input MLP that can memorize observed support mixtures.

This is a planted-basis statement. It does not establish that reconstruction and binary sparsity training find that basis, or that an arbitrary learned dictionary has a unique component correspondence. Read-column sign and scale do not affect the gate because of absolute normalized coefficients; a global permutation only changes labels.

## Robust sufficient margin under imperfect read factors

Let \(u_j=V_j/\|V_j\|_2\), and align its sign to the correct source column \(q_{\pi(j)}\). Suppose
\[
\|u_j-s_jq_{\pi(j)}\|_2\le\delta_j.
\]
Cauchy–Schwarz gives
\[
\left|\langle x,u_j\rangle-s_j a_{\pi(j)}\right|
\le\delta_j\|x\|_2
=\delta_j\|a\|_2
\le1.25\sqrt3\,\delta_j,
\]
since at most three latent components are active. The gate executes exactly on every support if
\[
1.25\sqrt3\,\delta_j
<\min(\tau_j,\ .75-\tau_j)
\quad\text{for all }j.
\]
At initial threshold \(\tau_j=.5\), a sufficient condition is \(\delta_j<0.11547\). In absolute cosine \(c_j=|\langle u_j,q_{\pi(j)}\rangle|\), the condition becomes approximately \(c_j>0.99333\), since \(\delta_j=\sqrt{2(1-c_j)}\) after aligning signs.

The bound is sufficient, with exact orthogonality of the generating Q and source amplitudes bounded as registered. The preregistered mean-cosine acceptance threshold of .90 is weaker than this worst-case guarantee. The first-run receipt must report complete-board support accuracy, per-component cosines and learned thresholds. Near-perfect average read alignment may coexist with poor gate execution when one axis or threshold has inadequate margin.

## Interpretation of the controls

The read_coupled arm permits gate-feature derivatives to update V, alongside physical factor reconstruction gradients. The read_detached arm has identical forward masks and suppresses the gate-feature V derivative while retaining physical-factor gradients. The raw_independent arm uses a general shared MLP on observed signed x and has no explicit factorwise read channel.

The raw_count_supervised arm receives true support cardinality as an auxiliary training loss, a form of privileged information absent from the unlabelled primary and comparator. The read_planted_frozen control probes planted-basis sufficiency; oracle_support receives exact source labels. The mathematical theorem and learned-optimizer experiment remain separate evidence.
