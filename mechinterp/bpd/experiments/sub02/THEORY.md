# SUB-02 theory: finite-dose causal observability

## Why global future-output Jacobians

The SUB-01 original experiment retained high-variance state subspaces through PCA. It found a useful *behavioral* projector, but hidden-state perturbations along discarded fibers continued to alter downstream logits. A causal rank constraint should use the dynamics of the terminal readout instead of the covariance of states.

For a fixed trained recurrent model and input x, write

- \`h1(x)=F_x(0)\`;
- \`L_x(h)=R(F_x^3(h))\`;
- \`j_x=∇_h L_x(h1(x))\` in original physical state coordinates;
- \`C_obs=E_x[j_x j_x^T]\`, the source-only future-output observability Gramian.

The rank-r orthogonal projector \`P_obs,r\` spans the r largest eigenvectors of C_obs. The state-PCA comparator \`P_pca,r\` instead spans top eigenvectors of \`E_{x,t}[h_t h_t^T]\`, an entirely different quadratic form.

## Exact first-order optimality at matched physical dose

Fix rank r and let k=d-r. Let z be isotropic standard Gaussian in R^d. Define the rank-matched physical perturbation

\`δ_P = α (I-P)z / ||(I-P)z||\`,

for any rank-r orthogonal projector P. Then \`||δ_P||=α\`, \`Pδ_P=0\` and the perturbation is uniformly distributed on the radius-α sphere in ker(P). Its conditional covariance is

\`E_z[δ_P δ_P^T | x] = α²/k (I-P)\`.

Taylor expanding the terminal readout at clean h1 gives

\`L_x(h1+δ_P)-L_x(h1) = j_x^T δ_P + O(α²)\`.

Therefore

\`E_{x,z}[(j_x^Tδ_P)²] = α²/k Tr[(I-P)C_obs]\`.

For fixed r and α the Ky Fan variational principle proves that the top-r eigenspace of C_obs minimizes the expected squared *linearized final-logit* response. The result is independent of the spectral basis orientation inside any degenerate eigenspace, though the rank-r minimizer can be nonunique at a tied r/r+1 boundary.

This is why SUB-02's primary direction can be preregistered without reading any finite-dose heldout results. The same theorem predicts nothing about an exact nonlinear finite-dose inequality when α is large, and nothing about the entire projected hidden-state transition. These are measured rather than inferred.

## Dynamics and observational limits

The finite-dose response measured at rank r is

\`S_r(P) = sqrt(E_{x,z}[(L_x(h1+δ_P)-L_x(h1))²])\`.

Because the original source executes all three post-impulse recurrent steps, S_r measures a genuine source-model intervention. A lower S_r implies weaker terminal-output sensitivity for the registered impulse law. It does not imply a closed quotient transition \`q∘F=G∘q\`, which requires the stronger projected next-state fiber-constancy condition throughout the reachable/counterfactual domain.

A low source-vs-projected-student KL can coexist with a high S_r. Conversely an observability projector could yield excellent S_r while the autonomous projected student loses clean accuracy due to removing state needed to reach the source trajectories. The experiment measures both.

## Root and sham resolution

Model seed is the independent root. The same raw Gaussian vector is used under PCA and observable Gramian; projection and norm reset impose the registered physical dose. Eight independent random rank-r sham projectors provide an engineering reference. They cannot by themselves establish a distribution-free 95/95 tail bound, which needs at least 59 independent calibration roots. The observed mean and root bootstrap interval at eight roots remain engineering evidence.

An exact rank-r finite-sample Gramian minimization theorem is an algebraic witness for the *linearized discovery objective*. The empirical scientific question is whether that source-only optimality transfers to finite-dose heldout causal response on fresh trained recurrent model roots.
