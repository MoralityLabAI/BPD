# SF05B Stage 2: conditional source-blind baselines under source-aware queries

The original Stage 0 source-blind conditional mean assumes the final query role is independently uniform. Stage 2 deliberately selects the authorized role from the old or new last-write source tag. This changes the conditional distribution and requires a new Bayes baseline **before** model evaluation.

Fix a structurally heldout history of length L∈{5,6}, with abstract source overwrite template t. Let its final write be at position m=L−1 and let a,b be the two other abstract source IDs. Write \(\ell_a,\ell_b\) for their respective latest-write positions in t. Independent uniformly random source-name permutations hide the mapping between abstract source IDs and visible A/B/C identities. The donor changes only the final source label s to a uniformly chosen distinct label u. The query role r is chosen as s or u with probability 1/2, then held fixed across recipient and donor.

Condition on all numeric values \(v_0,\ldots,v_m\), history length, and the observed query role \(r\), **without seeing any source-name tags**. Exactly half of the equally likely hidden role choices match the original final source, with the other half matching the replacement source; the other abstract source ID is symmetric. The recipient and donor squared-error Bayes conditional means therefore agree:

\[
b_{\rm uncond}(v,L,r)
=\mathbb E[Y_{\rm recipient}\mid v,L,r]
=\mathbb E[Y_{\rm donor}\mid v,L,r]
=\frac12v_m+\frac14v_{\ell_a}+\frac14v_{\ell_b}.
\]

For the **high-contrast** population, admission requires \(|Y_{\rm recipient}-Y_{\rm donor}|\geq\Delta\) with \(\Delta=0.30\), determined solely by exact source-oracle labels and **before** any trained model call. Set \(w_k=\mathbf1\{|v_m-v_{\ell_k}|\geq\Delta\}\), k∈{a,b}. Every accepted example has \(w_a+w_b\ge1\). By symmetry of donor-role selection under swapping s↔u, both blind conditional means now become

\[
b_{\rm contrast}(v,L,r)
=\frac12 v_m+\frac12\frac{w_a v_{\ell_a}+w_bv_{\ell_b}}{w_a+w_b}.
\]

When the public role U is requested, both target values and both source-blind conditional means equal the known public scalar P.

**Proof by finite exhaustive enumeration.** There are six uniformly probable source-name permutations, two equally likely distinct donor-source choices and two equally likely query-role choices. For each configuration, evaluate both recipient and donor exact last-write-wins values, retain configurations compatible with the observed role, and additionally enforce contrast admission when requested. Every remaining configuration has equal conditional weight. The corresponding arithmetic means give the formulas above. The generator computes this direct 24-configuration enumeration, and independent tests compare it against the closed forms for each length and source permutation.

These source-blind baselines minimize squared error under their **own conditioned populations**. Absolute error compared with the blind conditional mean remains a descriptive quantity. The 4n-candidate high-contrast screen is a distinct data-generating population and must carry the attempted, qualified and retained counts into any subsequent model assay. The generator never selects cases based on neural-model success or an intervention outcome, and the public cohort has its own independent seed and denominator.
