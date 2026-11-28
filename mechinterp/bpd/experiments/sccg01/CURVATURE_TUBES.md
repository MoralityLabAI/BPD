# SCCG-02: curvature tubes and exact finite RMS reader control
Date: 2026-11-27. Additive theory and engineering fixtures; no new model roots.
Companion: THEORY.md and SPARSE_COMPLEXITY.md. INK-BPD-06 (#161) supplies
the two-block reader convention. The calculus uses classical differentiation,
composition and spherical-cap optimization. Literature priority is open.
All analytic bounds below concern real arithmetic; floating reference
outputs require interval/outward-rounding machinery for machine certificates.

## 1. Sharp RMSNorm curvature
Let N(x)=sqrt(d)x/rho, rho=sqrt(||x||^2+d eps), eps>=0, rho>0.
For u,v, direct differentiation gives
D N[u] = sqrt(d)(u/rho - x<x,u>/rho^3)
D^2 N[u,v] = sqrt(d)(-(u<x,v>+v<x,u>+x<u,v>)/rho^3
                       +3 x<x,u><x,v>/rho^5).
The first derivative has norm <=sqrt(d)/rho.

Set s=||x||^2/rho^2, and for a unit u let z be its squared cosine with x.
At x=0 the Hessian is zero. For x!=0, resolving the output into radial
and transverse components gives
rho^4/d * ||D^2 N[u,u]||^2
  = s*((1+(2-3s)z)^2+4z(1-z))
  = s + 2s(4-3s)z - 3s^2(4-3s)z^2.
The maximum over 0<=z<=1 is 9s(1-s)^2 for s<=1/3 and 4/3
for s>=1/3, with vertex z=1/(3s) in the latter range.
For d>=2 every angle is feasible. For d=1 the angle is restricted to
z=1, so these still give an upper bound. Symmetric bilinear polarization
extends the diagonal supremum to the bilinear operator norm: for unit u,v
the sum of squared norms of u+v and u-v equals four.
Thus ||D^2 N|| <= 2sqrt(d)/(sqrt(3)rho^2). This factor is sharp for
d>=2 when s>=1/3 and also at s=1/3 for d=1, eps>0.
The d=1, eps=0 case is locally constant on either half-line.
For eps>0, substitute rho^2=d eps/(1-s).
For s<=1/3 maximize 3sqrt(s)(1-s)^2 at s=1/5, attaining
48/(25sqrt(5) sqrt(d) eps). On s>=1/3 the bound is at most
4/(3sqrt(3) sqrt(d) eps), which is smaller.
The uniform positive-epsilon constant is therefore attained by a radial
direction in every d>=1 at ||x||^2=d eps/4.

For learned elementwise scale gamma multiply both bounds by max |gamma_i|.
Equality is not promised after arbitrary scaling. On a closed ball centered
at x0 of radius R, rho_min=sqrt(max(0,||x0||-R)^2+d eps).
The minimum of local and uniform positive-epsilon curvature bounds applies
throughout the ball. Reject an epsilon-zero tube touching the origin.

## 2. Inkling two-block reader with one block fixed
Let n=N_gamma(x), n2=N_gamma(x2) constant, q a fixed query and
s=q^T(n-n2)/sqrt(d), beta=sigmoid(s), message=n2+beta(n-n2).
Let l_N,k_N be certified first and second derivative bounds on the ball.
Then l_s=||q|| l_N/sqrt(d), k_s=||q|| k_N/sqrt(d).
The sigmoid bounds are |sigmoid'|<=1/4 and |sigmoid''|<=1/(6sqrt(3)).
Hence l_beta=l_s/4 and k_beta=k_s/4+l_s^2/(6sqrt(3)).
With D>=sup||n-n2|| (use 2sqrt(d)||gamma||_inf, or tighten by the clean
difference plus l_N R), the product rule yields
l_message <= l_N + l_beta D
k_message <= k_N + 2 l_beta l_N + k_beta D.
For several scalar protected gaps, their joint Euclidean Hessian bound
is <= sqrt(sum_i k_s_i^2). These are local fixed-query bounds, not a full
attention or learned keep/gain gate certificate. If query, other block,
or gate moves, include those dependencies with the product/chain rules.
RMS score uses q^T n/sqrt(d), consistent with INK-BPD-06's raw rho.

## 3. Recurrence tubes: trajectory error versus transported prediction
For a clean trajectory x[t+1]=F_t(x[t]), consider the edited recurrence
x'[t+1]=F_t(x'[t])+B_t u_t, fixed additive post-transition injection.
Assume ||u_t||<=e_t. Suppose on the closed ball about x[t] with radius R_t
we have ||DF_t||<=G_t, ||D^2F_t||<=K_t and ||DF_t(x[t])||<=j_t.
Let b_t>=||B_t||. With r_0=E_0=0 compute
r_(t+1) = min(G_t r_t, j_t r_t + K_t r_t^2/2) + b_t e_t
E_(t+1) = j_t E_t + K_t r_t^2/2.
Require r_t<=R_t for every evaluated transition; otherwise refuse the bound.
Induction with the mean-value and Taylor remainder inequalities proves
||x'[t]-x[t]||<=r_t and ||x'[t]-x[t]-z_t||<=E_t,
where z_(t+1)=DF_t(x[t]) z_t+B_t u_t.
For a scalar/vector readout, append its smooth map as a final transition
without injection. This yields separate certified upper bounds on desired
response linearization error and collateral error. It uses local clean
Jacobians for transporting errors, which can be sharper than products of
uniform Lipschitz constants. It does not use favorable cancellation
between independently bounded errors. All constants need certified
derivation over each tube. A sampled maximum is not a certificate.
For tied parameter updates the additive B_t u_t model is generally
insufficient. Augment the state with the shared parameter/control held
constant and certify the joint transition, or include mixed and direct
control Hessian blocks. Endogenous routing is covered only when its
actual smooth maps are part of the certified F_t. Hard top-k needs
region/branch stability or a separate nonsmooth analysis.

Useful composition rule for differentiable maps f,g on admitted domains:
L_(g∘f)<=L_g L_f, H_(g∘f)<=H_g L_f^2+L_g H_f.
Domains of intermediate maps must be checked. Affine maps have zero
Hessian. Coordinate sigmoid and tanh have Hessian operator bounds
1/(6sqrt(3)) and 4/(3sqrt(3)), respectively, with L<=1/4 and <=1.
No architecture-wide numerical constants are asserted here.

## 4. Exact finite fixed-radius boundary steering
There is an alternative to generic local nonlinear correction for a
particular Inkling reader intervention. Fix x, the second block, the
reader queries and scale. Let S be the physical span of allowed writes.
Let C be the matrix of effective protected queries (each row q⊙gamma);
a the corresponding target effective query. Impose ||x+delta||=||x||,
C delta=0, delta in S and ||delta||<=e. The raw RMS denominator is then
constant and the protected gaps (and mathematical sigmoid probabilities)
are exactly unchanged. Protected messages/content, later nonlinear
continuations and task outputs are outside this guarantee.

Let T=S intersect ker C, p=P_T x, b=P_T a, R=||p||. Every admissible
delta is p'-p with p' in T, ||p'||=R, ||p'-p||<=e.
The target score change is b^T(p'-p)/rho. For R=0 or b=0 the zero
edit attains the optimum. For dim T=1, the only choices are p'=p or
p'=-p, the latter feasible iff e>=2R.
For dim T>=2 and R>0, let c=p/R and
alpha=max(-1,1-e^2/(2R^2)). Feasible directions w=p'/R satisfy
||w||=1 and c^T w>=alpha.
If b/||b|| is feasible it attains the global maximum. Otherwise
decompose b into b_parallel c + b_perp. The cap optimum has
w=alpha c+sqrt(1-alpha^2) b_perp/||b_perp||.
If b_perp=0 choose any perpendicular unit vector (the boundary case
can occur only for negative b_parallel, aside from zero budget).
Proof: at fixed t=c^T w the best inner product is
b_parallel*t+||b_perp||sqrt(1-t^2). Its unconstrained maximum occurs
at t=b_parallel/||b||, so clipping to [alpha,1] gives the displayed
solution. The objective is linear in the normalized state because the
radius is fixed.

This is an exact real-arithmetic global optimum for the declared
fixed-radius *boundary* action space. The constraint can be much
stronger than generic fixed-L2 steering. In particular, a one-dimensional
admissible nullspace permits only a full sign flip or zero. The ordinary
linearized optimum can therefore predict positive gain on a direction
that has no nonzero finite fixed-radius edit at small dose.
This does not contradict the local correction theorem: the derivative
of the additional radius equality can be singular or the tangent space
can have dimension zero after including it.

The result turns Goodfire-style reader accounting into an explicit
finite design test. Enumerate small joint supports, quotient redundant
physical writes, and solve each spherical cap. An exposed native
checkpoint experiment must next verify actual boundary-hook parity,
RMS radius, protected gaps, full continuation, and matched physical
controls. Source-writer causality requires its own hook. This note does
not authorize use of protected cohorts.

## 5. Evidence and next mathematical obstacle
The reference solver uses floating SVD and reports numerical rank and
residuals. A near-singular constraint needs an explicit rank policy;
neither tiny residuals nor the analytic proof certify arbitrary floating
inputs. Tests check constructed RMS/gate Hessians with autograd, exact
rational bounds, nonlinear tube inequalities and small finite cap optima.
No trained network curvature envelope, Lean compilation, measured
training benefit or claimed research priority. Next: interval-verified
bounds through real attention, gating and tied blocks; then compare
local control, fixed-radius finite control, and unrestricted finite
oracle on fresh prospective development roots with task-level controls.
