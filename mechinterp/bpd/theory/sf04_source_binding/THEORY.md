# BPD-SF04: exact source-blind risk and recurrent observation boundary

## Proposition 1: an ordered value bag is insufficient for source-binding

For a fixed authorized query role `R∈{A,B}`, let two independent source labels `U_1,U_2` be uniform over `{A,B}`. The associated scalar values `V_1,V_2` are independent of those labels. The target is the last value written by the queried source, with zero if the source never appears.

Conditioning on `V_1=v_1,V_2=v_2,R` while **withholding source labels** gives the conditional distribution:

\[
Y^* \sim
\begin{cases}
v_2&\Pr=\tfrac12\quad(U_2=R),\\
v_1&\Pr=\tfrac14\quad(U_2\ne R,U_1=R),\\
0&\Pr=\tfrac14\quad(U_2\ne R,U_1\ne R).
\end{cases}
\]

Therefore the unique squared-error Bayes predictor (apart from null sets) is

\[
f_{\rm blind}(v_1,v_2,R)=\mathbb E[Y^*\mid v_1,v_2,R]
=\tfrac12 v_2+\tfrac14 v_1.
\]

The conditional irreducible squared error is

\[
\operatorname{Var}(Y^*\mid v_1,v_2,R)
=\tfrac14v_2^2+\tfrac3{16}v_1^2-\tfrac14v_1v_2.
\]

If `V_1,V_2\overset{\rm iid}{\sim}Uniform[-a,a]` then their independent zero means and variance `a^2/3` yield the blind minimum expected authorized-role MSE

\[
\mathbb E\,\operatorname{Var}(Y^*\mid V_1,V_2,R)
=\frac{7a^2}{48}.
\]

For equiprobable third role `U`, the public target is the observed `P` and has zero conditional variance; the all-role minimum MSE is `7a^2/72`. At SF04's `a=0.8` this is **0.062222... MSE**. A learner that sees source labels can in principle achieve zero Bayes MSE for noiseless source records. This is an exact discrete/continuous distribution calculation, independent of a trained model.

**Metric clarification:** `f_{\rm blind}` minimizes **squared error**. The SF04 code also reports its **absolute error** as a descriptive comparison to the teacher's MAE. That MAE is not generally the minimum achievable source-blind MAE. Model-versus-blind MAE ratios should be interpreted as comparisons with this fixed conditional-mean baseline, not as an unconditional bound on every source-blind L1 estimator.

## Proposition 2: tag-only counterfactual tests provenance

For the recipient/donor pair the first event, both values, event chronology, public scalar and query role are identical; the second event's source label is toggled `U_2\mapsto \neg U_2`. The donor target is recomputed using the same last-write-wins rule. Thus any change in ideal target depends solely on the changed association between a source and a value. This is a controlled provenance-binding challenge with no change to lexical topic or numerical payload.

The model may still exploit fixed source-tag positions, so passing this synthetic challenge is an intervention-specific functional property. A more demanding real-language experiment would randomize user names, reorder histories, insert distractors, and use recipient-conditioned transcript evaluations under independent source families.

## Distinguish state information, observable readout and authorized use

The true observable readout at each recurrent pass is `q(h)=h_0`. Intermediate training constrains `q(h_1),q(h_2)` to near zero. At the pre-query second state, source-identity changes can alter the hidden state while the visible readout differs only slightly. An intervention `\delta\in\ker Dq` preserves that actual scalar readout exactly. Its influence on a later query outcome is tested by `q(F(h_2+\delta,x_q))`, and compared with exact dark donor-state swaps, full physical factor reconstructions, random norm-matched dark perturbations and signed-gradient-matched shams.

For shared physical write matrix `C≈VU`, any invertible basis change `R` gives `(VR)(R^{-1}U)=VU`. Consequently the complete factor block has a gauge-invariant physical effect, while a 14/20 singleton-selected coalition generally does not. A strong donor recovery for such a coalition is only a property of its particular trained gauge and selection rule.

This source-binding gym tests output behavior on synthetic private scalars. The unauthorized output may move under a partial off-manifold intervention even when clean source-tag swaps preserve it. Such a shift is a **diagnostic sensitivity**, not an information-theoretic secrecy proof or confirmed disclosure to a recipient.
