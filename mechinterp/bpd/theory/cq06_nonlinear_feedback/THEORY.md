# CQ-06: hidden information, nonlinear causal writes, and gauge fixing

## Exact one-pass ceiling

Let x,y,z have independent balanced signs and magnitudes in [.65,1.20]. First-step input is an orthogonal rotation of (x,y,0,-1), second-step input is the same rotation of (0,0,z,+1). Targets are L1=sgn(xy) and L2=sgn(xz). Conditional on first-step information I=(x,y), z is an independent balanced sign. For every deterministic classifier f observing only I,

\[ \Pr[f(I)=L_2\mid I]=\tfrac12. \]

The confirmation task samples 128 matched first cues, each with two opposite z signs, so any one-step behavioral predictor attains exactly 128/256. A static mask can still use the **physical model's** late z at visit two, so its prediction accuracy can exceed this ceiling. An exact parity solver using both visits has a 100% structural ceiling. Donors flipping x and y preserve L1 while reversing L2, requiring delayed causal content beyond the visible first-step target.

## Recurrent BPD and donor patch

A learned nonlinear recurrent state update is

\[ s_{t+1}=\tanh(([s_t,u_t]V)U+b),\qquad q(s)=Cs. \]

Here V (16x4), U (4x12) arise from a second training phase on the separately learned physical rank-four shared operator. A mask m on a visit inserts componentwise multiplication between V and U. The first mask is all ones and the second is a hard learned function of reached s1 and newly revealed u1. A training-only exhaustive oracle selects the sparsest mask that preserves full-model logits and state RMS; the independent labels are scored after training.

For a donor and recipient differing in first-step x,y while sharing late z, components 0 and 1 are fixed by descending singular-value order. The preactivation patch is

\[ \Delta_{pre}=\bigl(([0,u_d]-[0,u_r])V\odot[1,1,0,0]\bigr)U. \]

The physically measured state patch is \(\delta=\tanh(a_r+\Delta_{pre})-s_{1,r}\), where \(a_r=[0,u_r]VU+b\). Its dark projection is \(P\delta\), with \(P=I-C^\top(CC^\top)^{-1}C\). Restore the original state norm and match a Gaussian random-dark patch and an unrelated **same-first-parity clean-state** dark difference pair-by-pair. Report out-of-range state coordinates beyond [-1,1] after patching, since equal Euclidean displacement does not guarantee manifold equivalence. The full donor intermediate-state replacement has a larger physical budget.

## Gauge invariance and calibration

For invertible R in GL(4), (VR)(R^-1 U)=VU, but with a diagonal hard mask D, typically V R D R^-1 U differs from V D U. Canonical SVD of the *entire* physical matrix fixes rank-one terms up to compensating sign when all nonzero singular values are distinct. Under near degeneracy, canonical components can rotate. Compare raw sheared masks with re-canonicalized execution and live second-stage gate recomputation.

The independent statistical root is a newly rotated and trained model world. In 59 independent sham-calibration worlds, the maximum absolute difference between two same-dose matched-manifold random controls bounds the population 95th sham quantile with at least 95% confidence under the sample-maximum rule: \(1-.95^{59}\approx .9515\). These same development worlds estimate pilot effect variance. The *effective mechanism floor* is the maximum of .15, the sham tolerance bound and the 1/128 score resolution. Two registered donor comparisons receive one-sided Bonferroni alpha .025 each and a root-cluster bootstrap on 26 separately withheld worlds. The narrow task family remains a limitation on generalization.
