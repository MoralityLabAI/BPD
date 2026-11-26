# SCCG-01: selective control with a collateral budget
Date: 2026-11-27. Evidence: mathematical derivations and constructed engineering fixtures.
Parent: INK-BPD-06, BPD #161, commit 32acfed8c678f8b9402c50994460daa0ea7f40ca.
This note proposes a synthesis for BPD; the linear algebra, convex duality and
implicit function argument are classical tools, and mathematical priority is
unestablished. See Boyd and Vandenberghe, Convex Optimization,
https://web.stanford.edu/~boyd/cvxbook/ and the constrained-optimization
implicit function theorem in https://ocw.mit.edu/courses/15-084j-nonlinear-programming-spring-2004/resources/lec6_constr_opt/.
The related general two-trust-region problem is discussed by Yang and Burer,
https://doi.org/10.1137/130945880. Our objective is linear and constraints convex.

## 1. Control geometry and time indexing
Let h[t+1] = F_t(h[t]) + B_t u_t to first order, with J_t = D F_t
evaluated on the clean trajectory. B_t injects into the POST-transition state.
Tangent control response at T is H = [J_(T-1)...J_1 B_0, ... , B_(T-1)].
With response Jacobians D_target and D_protected at T, A = D_target H and
C = D_protected H. Multiple time readouts can instead be stacked, padding
future action blocks by zero. For shared coefficients u = S z use AS, CS and
S^T M S before calculating the frontier. Do not treat a tied parameter edit
as independent per-pass boundary edits. A genuine source edit needs derivatives
through the actual source and endogenous routing.

Let a = A^T q, where q specifies a scalar desired response, and physical
dose be sqrt(u^T M u). For positive definite M, whiten by u = L^(-T) z,
M = L L^T. All statements below use whitened Euclidean coordinates.
At zero collateral budget, P is the orthogonal projector on ker C:
G_sel = A P A^T, V(e,0) = e ||P a||. A prescribed vector response d is
achievable iff d is in im(AP); then the minimum squared norm is
d^T G_sel^dagger d.

## 2. Theorem: exact positive-budget support frontier
For e>0, eta>0, write
V(e,eta) = max{ a^T u : ||u|| <= e, ||Cu|| <= eta }.
Then
V = inf_(lambda>=0) sqrt((e^2+lambda eta^2) a^T (I+lambda C^T C)^(-1) a).
Allow the infinite-lambda endpoint (a limit).

Proof. For every lambda>=0, feasible u obeys
u^T(I+lambda C^TC)u <= e^2+lambda eta^2. Cauchy-Schwarz in this metric
gives the displayed upper bound. For equality, if a=0 the result is zero.
Otherwise compactness gives a maximizer, and u=0 strictly satisfies both
quadratic inequalities. Convex KKT thus supplies multipliers mu,nu>=0
with a=(mu I+nu C^TC)u (absorbing the factor 2), complementary slackness,
and at least one nonzero multiplier. If mu>0, take lambda=nu/mu.
Stationarity makes Cauchy-Schwarz tight, and complementary slackness
makes the summed constraint tight. If mu=0 then nu>0, a is in the rowspace
of C, and Cauchy-Schwarz in the positive-semidefinite metric C^TC is
tight at u; the infinite-lambda limit gives that value. This proves equality.
The zero-dose case is V=0 in Euclidean coordinates; zero collateral uses P.

For computation normalize the constraints:
Q(alpha)=alpha I/e^2+(1-alpha)C^TC/eta^2, alpha in (0,1].
The squared upper bound a^T Q(alpha)^(-1) a is convex in alpha:
its second derivative is 2 a^T Q^-1 Q' Q^-1 Q' Q^-1 a >= 0.
The reparameterization alpha=e^2/(e^2+lambda eta^2) gives the same bound.
The endpoint alpha=0 is interpreted by a limit. A primal candidate from
Q^-1 a divided by sqrt(a^T Q^-1 a) can be scaled down to satisfy both
original constraints. An actual feasible objective and dual bound bracket
the optimum in exact arithmetic. The reference implementation reports the
floating-point gap and physical residuals; it is not interval-certified.
Numerical nullspace thresholds are reported at eta=0.

### Singular physical metrics
Let M>=0 and N=ker M intersect ker C. If a has a nonzero component in N,
the scalar problem is unbounded even at zero budgets. Otherwise the
objective and constraints descend to N^perp, where M+C^TC is positive
definite. The feasible set there is compact: u^T(M+C^TC)u <= e^2+eta^2.
For both positive budgets, Slater and KKT yield the normalized dual
inf_(alpha in [0,1]) sqrt(a^T Q(alpha)^dagger a), where
Q=alpha M/e^2+(1-alpha)C^TC/eta^2 and endpoints with a outside im Q
are assigned +infinity. For zero budgets first restrict to the corresponding
nullspace(s). A singular metric caused by redundant coefficient directions
in a physical map B should satisfy ker B subset ker A and ker C;
otherwise the instrument has attributed a behavioral effect to a zero
physical displacement. Boundary adapter uses an SVD basis of im B with
Euclidean physical displacement and reports numerical rank.

## 3. Theorem: nonlinear exact protection by a local correction
Use whitened coordinates. Let g: R^n -> R^m be the change in protected
responses, g(0)=0, C=Dg(0) full row rank. Let R=C^dagger,
kappa=||R||, and assume ||Dg(x)-Dg(y)|| <= L||x-y|| on a closed ball
of radius D containing all points used below. Let f be a scalar target
response with a=Df(0) (column convention) and Lipschitz gradient constant
L_f on that same ball. If L=0, g is linear on the ball and no correction
is needed. Assume L>0, m>0. Choose r>0 such that kappa L r <= 1/4 and
r+b<=D, where b=kappa L r^2. For any v in ker C, ||v||=r, there exists
a unique w in range(C^T) with ||w||<=b in that correction ball satisfying
g(v+w)=0. Moreover ||v+w||<=sqrt(r^2+b^2).

Proof. On the closed b-ball in range(C^T), define
T(w)=-R[g(v+w)-C(v+w)]. Taylor's integral bound gives
||T(w)||<=kappa L (r+b)^2/2 <= (25/32)b < b.
The derivative of g-C on the r+b ball has norm at most L(r+b),
so T is Lipschitz with constant <= (5/4)kappa L r <= 5/16.
Banach's contraction theorem gives its unique fixed point in this ball.
CR=I and Cw=-[g(v+w)-C(v+w)], while Cv=0, imply g(v+w)=0.
Orthogonality of ker C and range C^T gives the squared dose bound.
This proof is local; no uniqueness beyond the stated correction ball.

If Pa is nonzero, choose v=r Pa/||Pa||. The exactly protected finite
target gain is at least
r||Pa|| - b||(I-P)a|| - (L_f/2)(r^2+b^2).
Indeed a^T w >= -b||(I-P)a||, then apply Taylor to f.
Choose r with sqrt(r^2+b^2)<=e for the prescribed physical budget.
The guarantee is conditional on independently justified derivative bounds
on the entire ball. Sampling local Hessians does not certify such bounds.
If Pa=0, the first-order positive-gain conclusion is absent.
The full row rank assumption is essential: g(u)=u^2 at zero has C=0
and every direction is tangent, but only u=0 exactly preserves g.
If protection is approximate instead, a simple sufficient feasibility
condition is ||Cu|| + L||u||^2/2 <= eta, with target lower bound
a^T u - L_f||u||^2/2. These are sufficient, not necessary.

## 4. Exact sparse-selection obstruction
Let three orthonormal physical actuators have target row a=(1,1,delta)
and protected row C=(1,-1,0), for rational 0<delta<1.
Define F(S)=max_{supp u subset S, ||u||<=1, Cu=0}(a^T u)^2.
F({1})=F({2})=0, F({3})=delta^2,
F({1,2})=2, F({1,3})=F({2,3})=delta^2.
Thus F is monotone but violates diminishing returns:
the marginal of adding 2 to empty is zero; adding 2 to {1} is two.
With a two-actuator budget greedy marginal selection chooses 3 first
and cannot recover either cancelled pair. Its squared-objective ratio
is delta^2/2, arbitrarily small; its unsquared gain ratio is delta/sqrt(2).
This holds with perfect attribution, exact linear response and matched
physical dose. A local top-k or sequential greedy guarantee needs extra
assumptions or joint support search. Pair search resolves this fixture,
but can require larger groups: with C's first k columns forming the
(k-1)-dimensional chain difference constraints, every proper subset
has zero protected-null gain, whereas all k columns admit a common
nonzero coefficient vector. Add a protected-free distractor of small gain
to defeat any greedy group search restricted to batches smaller than k,
under a k-actuator budget (provided it first selects positive gain).
This is an obstruction for this particular selective-control objective;
no claim about every submodular Gramian objective.

## 5. Predictive equivalence versus control equivalence
Exact controlled recurrence F((s,b),u)=(s+b*u,b), b in {-1,+1}.
Readout y=s. Under ordinary u=0, (0,+1) and (0,-1) have identical
future observations at all depths. For one intervention |u|<=1 to
maximize y, the full state controller selects u=b and obtains y=1.
A deterministic controller seeing only s selects a common u and has
worst-case reward -|u|<=0. Any randomized state-blind controller
has worst-case expected reward -|E u|<=0. Under equal prior, average
reward is also zero. One extra bit suffices. Thus an exact passive
predictive quotient may discard information essential for choosing
interventions even with zero ordinary prediction error. This is a
finite controlled-system witness, not a measurement of a neural model.

## 6. Why this matters for BPD
RP03 #155: stable projector geometry with weak selective effect.
RP04 #156: equal state dose with reduced delivered selected-read amplitude.
BPD-NBR #147-149: local derivative ranking can fail finite mask risk.
MathCan ROC #279: a better gradient observer still loses the matched input
comparator. Goodfire Inkling (https://www.goodfire.com/research/interpreting-inkling-2-mini/)
motivates execution-indexed writes; INK-BPD-06 exposes reader-local
gap derivatives. This instrument can stack target and collateral reader
rows and calculate a boundary-control frontier before undertaking
native continuation. It cannot infer source causality from a boundary edit.

For task-level control use actual full-trajectory response Jacobians,
defined protected outcomes (logit margins are proxies for discrete answers),
independent physically matched controls and real finite-dose outcomes.
The curvature certificate requires a norm appropriate to the response map;
hardware rounding is outside the analytic real-arithmetic argument.
No trained model roots, paid compute, held-out data or historical source
locks are involved. Lean kernel verification and empirical novelty
assessment remain open.
