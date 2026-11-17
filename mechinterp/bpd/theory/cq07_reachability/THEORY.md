# A conditional reachability no-go for fixed rank-two BPD writes

Let `W` denote the *frozen and learned* full BPD reconstructed 16x12 physical operator. In the CQ-06 first-visit source family, a legal initial cue is `(x,y,0,-1)Q` and the hidden state starts at zero. Consequently

\[
  p(x,y) = b - Q_{3,:}W_{\mathrm{cue}} + x\,Q_{0,:}W_{\mathrm{cue}} + y\,Q_{1,:}W_{\mathrm{cue}} = p_0 + (x,y)A.
\]

The rows of `A` are the first-visit input-reachable preactivation directions, a subspace `S` of dimension at most two. Choose a fixed two-component BPD write block after canonicalizing the complete physical weight product. Its two output write rows are `B=U_{0:2,:}` and its allowed preactivation displacement lies in `T=span(B)`, dimension at most two.

**Conditional proposition.** Suppose `rank([A;B])=4`. Then `S∩T={0}`. Since every real tanh coordinate is strictly monotone and injective, any nonzero preactivation change in `T` sends `tanh(p(x,y)+δ)` to a state that cannot equal `tanh(p(x',y'))` for any real first-cue coefficients `(x',y')`.

**Proof.** If a nonzero `δ` belonged to both subspaces, `δ=aA=bB` for row coefficient vectors `a,b ∈ R²`. Thus `(a,-b)[A;B]=0`. The four independent rows force `a=b=0`, contradicting `δ≠0`. Equality of the two tanh states would imply equality of their preactivations by injectivity, yielding the same impossible intersection. ∎

The legal signed-magnitude restrictions on `(x,y)` are stricter, so the proposition continues to hold. There is a numerical check for the full-row-rank premise: the minimum singular value of the 4x12 stacked matrix. It is reported for each newly trained root, with the fixed threshold 1e-6. This is an exact mathematical implication conditional on each *real-valued* matrix meeting the rank premise; finite-precision singular values are numerical evidence for that premise in the eight engineering models.

**Consequences for causal identification.** Tanh ensures a point is in its coordinatewise state box. The image of legal inputs is a much thinner set. A head-preserving rank-two patch that stays inside the box can therefore remain outside the legally reachable state manifold. This is independent of whether it changes the future label and independent of how well the future controller responds.

The clean-source comparator uses a state obtained by applying the learned first transition to a different valid cue. Matching to the large full-donor physical dose in the signed-parity task selects the distant sign-flipped branch of the first-parity fibre. Such a branch already carries the donor x sign. This makes an equally successful clean null an **expected structural rival** to the BPD intervention rather than evidence that the BPD write uniquely mediates the delayed variable. The matcher uses the first parity, head logit and physical dose; sign equality with the target is checked only as a post-match diagnostic.

**Way forward.** For true reachable-state mediation, replace the recipient's state by an independently obtained clean donor state, and intervene on the *readout of the selected component block at the next pass* while holding that same natural state and cue fixed across arms. This changes the scientific estimand from "was the rank-two patch itself naturally reachable?" to "was a selected causal component block needed to propagate an already naturally reachable donor variable?". That requires a newly registered gym with target-independent selection, equal-cost component controls, and a source population containing nuisance variations that can match donor displacement without automatically carrying the target sign.
