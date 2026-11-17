# SF05B Stage 2: exact source-blind conditional means under source-aware queries

**Pre-model source-only mathematical audit, November 18, 2026.** Query-role stratification alters the conditional blind law from Stage 0, and recipient/donor blind means need separate treatment. The source generator computes both by a direct 24-configuration enumeration, rather than assuming equality.

## Source histories and conditional role law

Fix a heldout record with length L∈{5,6} and abstract three-source overwrite template t. Denote its final position by m=L−1 and the abstract source at m by o. Write \(j_o<m\) for o's latest **earlier** write. Let a,b be the other two abstract source IDs, and \(\ell_a,\ell_b\) their respective latest-write positions.

Independently permute all three abstract sources onto observable A/B/C names. The recipient final write belongs to s; the donor changes only that final tag to uniformly chosen distinct t. The shared query role r is s or t, each with probability 1/2. All numeric values, source history except that tag, public scalar and query marker agree.

An observer who sees only ordered numeric values, L and the common query role, while source-name labels are hidden, has:

- Probability 1/2 that r is the *recipient's original final source* s. Then recipient target is v_m, while donor target is v_{j_o}, because the donor reassigns the last write away from s.
- Probability 1/4 that r is the first other abstract source a and 1/4 that r is b. Then recipient target is v_{\ell_a} or v_{\ell_b}, while donor target is v_m, because the donor adds the last write to the queried source.

Therefore the **unconditional** source-blind squared-error Bayes estimators differ:

\[
\begin{aligned}
\mathbb E[Y_r\mid v,L,r]&=\frac12 v_m+\frac14 v_{\ell_a}+\frac14 v_{\ell_b},\\
\mathbb E[Y_d\mid v,L,r]&=\frac12 v_{j_o}+\frac12 v_m.
\end{aligned}
\]

They share the same visible conditioning variables, but answer **different counterfactual tasks**. The source-only generator must report both and should never silently copy the recipient blind mean to the donor.

## High-contrast conditional law

The high-contrast stratum admits an authorized pair precisely when \(|Y_r-Y_d|\ge\Delta\) with fixed \(\Delta=0.30\), determined using the exact source-label oracle before any trained model is consulted. Define indicator weights:

\[
w_o=\mathbf1\{|v_m-v_{j_o}|\ge\Delta\},\quad
w_a=\mathbf1\{|v_m-v_{\ell_a}|\ge\Delta\},\quad
w_b=\mathbf1\{|v_m-v_{\ell_b}|\ge\Delta\}.
\]

Every accepted pair has \(Z=2w_o+w_a+w_b>0\). Conditioning on the event and the observed role changes the 1/2,1/4,1/4 probabilities proportionally to \(2w_o,w_a,w_b\). Hence the exact **high-contrast** source-blind conditional means are

\[
\boxed{
\begin{aligned}
\mathbb E[Y_r\mid v,L,r,\mathrm{admit}]
&=\frac{2w_o v_m+w_a v_{\ell_a}+w_b v_{\ell_b}}{Z},\\
\mathbb E[Y_d\mid v,L,r,\mathrm{admit}]
&=\frac{2w_o v_{j_o}+(w_a+w_b)v_m}{Z}.
\end{aligned}}
\]

The rule selecting the first n accepted records from a deterministic stream of 4n iid candidate histories preserves the individual accepted-record law conditional on admission, while the number rejected remains part of the experiment's evidence.

For the independent public U-query stratum, both exact targets and both source-blind conditional means equal the same public scalar P.

## Exhaustive certificate and limitations

There are six uniformly likely source-name permutations, two donor-switch possibilities and two equally likely role choices, giving 24 equiprobable hidden configurations before conditioning. Restrict them by the observed role and optional exact label-gap admission, compute exact recipient/donor last-write outcomes and average. The implemented enumerator follows this finite definition. An independent test checks both closed-form expressions on every generated record, including the recipient/donor distinction.

These conditional means minimize **squared error** under the declared source-aware population. Their MAEs remain descriptive baselines. The hidden-source observer conditions on ordered numeric values, history length and the observed shared query role while withholding all source-tag identity. The public cohort has a separate target law. This mathematical source audit has no model weights, future readout or security outcome.
