# Observable-risk sparse BPD masks

## Exact finite-horizon risk

Use BPD-CQ-01's reached-state linearization. Let a be the actual component
activation, U its write map, D_m a diagonal binary keep mask, e=U(I-D_m)a.
For a common linear continuation T and readout C, the difference of
full versus masked observed trajectories at visit h is C T^h e.
For nonnegative weights gamma_h,

    G_H = sum_h gamma_h (T^h)^T C^T C T^h
    R_H(m) = e^T G_H e
           = a^T (I-D_m) U^T G_H U (I-D_m) a.

This follows by expanding the sum of squared observed differences.
Zero risk is exactly the intersection-of-kernels condition from CQ-01
if all included weights are strictly positive. For restricted reached
inputs the statement applies pointwise or in expectation on that law.
The observable component Gramian U^T G_H U is PSD but its entries can
have either sign. Physical risk uses G=I instead. A horizon-dependent
metric changes component interactions, not the full-write identity.

Define v_i=a_i [sqrt(gamma_h) C T^h u_i]_h and z=1-m.
Then R_H=||sum_i z_i v_i||^2 = z^T K z, K_ij=v_i^T v_j.
Selecting a fixed number of components is a binary quadratic problem.
The marginal decrease from adding i to a kept set S is

    Delta_i(S)=2 <v_i, sum_{j notin S} v_j> - ||v_i||^2.

Thus the observable diagonal score ||v_i||^2 ignores cross-terms.
If all v_i are pairwise orthogonal, top-k observable diagonal is
exactly optimal. Otherwise the marginal decrease can be negative
or increase as S grows, so generic monotone-submodular greedy guarantees
do not follow from PSD. A cost-aware policy may instead minimize
R_H(m)+lambda sum_i c_i m_i; resource costs must include observer
evaluation and routing overhead when claiming compute savings.

## Exact witnesses

1. Physical/observable separation: U=I_2, a=(2,1), C=(0,1),
   H=0, k=1. Physical top-1 keeps coordinate 1, with physical error 1
   and observable error 1. Observable optimum keeps coordinate 2,
   with physical error 4 and observable error 0. With T swapping axes
   and H=1, G_H=I, so physical and observable risks coincide.
2. Cancellation: U=I_3, a=(6,5,-6), C=(1,1,1), H=0, k=1.
   Top observable diagonal keeps coordinate 1 (lexicographic tie),
   giving squared error 1; keeping coordinate 2 cancels both omitted
   components, giving 0. Diagonal scores alone cannot solve this mask
   problem, even in a perfectly orthogonal physical dictionary.
3. Nonmonotonicity: U=I_2, a=(1,-1), C=(1,1). Keeping zero components
   has observable error 0, either singleton error 1, both error 0.
   The gain of adding 1 to the empty mask is -1; adding it after
   keeping 2 gives +1. Risk reduction is neither monotone nor
   submodular in general. Signed interference is the reason.

## Orthogonal coordinate control

Let Q be orthogonal, b=Qa and U'=UQ^T. Then U'b=Ua exactly.
Consequently the complete output is invariant at every linear visit.
The individual masked writes U'D_m b need not equal UD_m a.
For the same diagonal mask D, equality for every a holds if
U(Q^T D Q-D)=0; when U is injective this is equivalent to commuting
with Q. Orthogonal coordinate controls test sparsity of an observed
write; a rotated coefficient need not be a physically realizable
single SwiGLU activation or independently executed learned BPD gate.
This is consistent with CQ-01's relative basis analysis.

## Nonlinear extension and decision boundary

At a fixed reached state x, define F_H(x) as the concatenation of
weighted readouts of the actual nonlinear continuation, and J_H its
Jacobian. If J_H is L-Lipschitz in operator norm on the segment from
x to x-e (and the native gate regime is differentiable there), then

    F_H(x)-F_H(x-e) = J_H e + r, ||r|| <= (L/2)||e||^2.
    | ||F_H(x)-F_H(x-e)||^2 - ||J_H e||^2 |
        <= 2||J_H e|| b + b^2, b=(L/2)||e||^2.

This is a sufficient local bound; an actual recurrent gate may switch,
so a finite edit and re-evaluation of its feedback remain necessary.
For a finite candidate set and approximate risk uniformly within
epsilon of true risk, the minimizer of estimated risk plus identical
exact cost has true excess objective at most 2 epsilon: compare its
estimated objective to the true minimizer, then apply the error bound
at both masks. No uniform approximation bound has been established
for the Sudoku data.

For discrete actions with uncertain effects, BPD-OUT06 separately
computes expected class-conditional utility from a calibrated posterior
and permits abstention. The Gramian objective can supply candidate
writes or an observable-error budget; it cannot replace the posterior,
finite effect calibration, action cost, or the existing regret bound.
