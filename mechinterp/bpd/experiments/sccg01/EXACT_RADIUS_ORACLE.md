# SCCG-03: exact rational-input global nonlinear reader oracle
Date: 2026-11-27. Source instrument and mathematical derivation.
Predecessor: VARYING_RADIUS.md. The tools are classical real-algebraic
root isolation and convex quadratic optimization; priority is open.

## A rational formulation
Use a rational full-column-rank basis Q of the physical write span
(not necessarily orthonormal). All quantities below are expressed in
physical coordinates, so no irrational orthonormalization is needed.
Let T=S intersect ker C, P_T its rational orthogonal projector,
p=P_T x, b=P_T a. Put c=Cx. If c is not in im(CQ), tau=1 is required.
Otherwise choose k in S, orthogonal to T, satisfying Ck=c. Compute
k by projecting any rational solution in S off T. Then every protected
edit is delta=z0+w with z0=(tau-1)k and w in T. In the fixed-only
case take z0=0 and impose tau=1. Denote rho0²=||x||²+m, m=d eps>0.
Define v=p+w, R²=tau² rho0²-m-||x-p+z0||²,
E²=e²-||z0||². Exact physical and normalization constraints become
||v||²=R², ||v-p||²<=E². For a target threshold g measured in units
of rho0 times score gain, the remaining inequality is
a^T(x+z0+v-p) >= tau(a^T x+g).
Equivalently p^T v>=H=(R²+||p||²-E²)/2 and b^T v>=K=
tau(a^T x+g)-a^T(x+z0-p). Both R²,E²,H have degree at most two,
K is affine. Enforce tau>=0, R²>=0, E²>=0. Tau=0 is impossible
because m>0. This also covers the exact fixed-only condition.

## Theorem: quartic threshold reduction
For dim T>=2, two closed halfspaces have a common nonzero recession
direction (choose perpendicular to the first normal, with sign making
the second product nonnegative; handle zero normals directly).
If their intersection contains a point of norm <=R, moving along such
a direction continuously reaches norm R. The converse is immediate.
The unique minimum-norm point in a nonempty intersection belongs to
one of four active sets: neither, first alone, second alone, both.
This follows from the normal cone of a polyhedron, including degenerate
opposing inequalities. Let P=||p||², B=||b||², D=p^T b.
The four feasibility branches are:
1. H<=0,K<=0 (v=0);
2. P>0, D H>=P K, H²<=P R² (v=Hp/P);
3. B>0, D K>=B H, K²<=B R² (v=Kb/B);
4. det=P B-D²>0 and B H²-2D H K+P K²<=det R².
The fourth candidate meets both linear equalities by inverting the
2-by-2 Gram matrix. Testing primal feasibility is enough; even a
nonoptimal candidate inside the ball proves existence. Dependent
normals need only the first three cases (or are infeasible).
These are a finite disjunction of univariate polynomial inequalities,
each degree at most four.

For dim T=0 require R²=0 and H,K<=0. For dim T=1 use any nonzero
rational basis n and v=±n sqrt(R²/||n||²). For L>=0, c rational,
c sqrt(L)>=h is equivalent to:
- c=0: h<=0;
- c>0: h<=0 or (h>=0 and c² L>=h²);
- c<0: h<=0 and h²>=c² L.
This removes square roots with their correct signs, still at degree
at most four. It captures isolated feasible solutions and the
one-dimensional sign-flip obstruction. No approximation to rank.

## Exact global decision and optimization bracket
For each branch remove constant tautologies and reject negative
constants. Isolate all real roots of the square-free least common
multiple of its nonconstant polynomials over QQ. Refine to disjoint
rational intervals. Test one rational sample in every open cell,
including unbounded tails. Signs are constant there. At each isolated
root, use exact gcd/root counts to decide vanishing of every constraint;
otherwise its sign is constant across that isolating interval. Rational
singleton roots can be evaluated directly. Thus every feasible cell
and isolated boundary is covered. A witness is a rational tau or a
polynomial plus rational isolating interval, together with the branch
index. It certifies existence via the theorem, not a decoded edit.
A false result means all cells of all branches were rejected.
SymPy's exact arithmetic and root-isolation routines are trusted
dependencies; the implementation is not a separately kernel-checked
certificate verifier.

The feasible threshold set is downward closed and includes zero.
Since a^T y/rho(y)<=||a||, the scaled gain is strictly less than
(rho0²+||a||²)/2+|a^T x|+1 by AM-GM. Bisection with exact
decision returns a rational bracket [lo,hi] with lo feasible and hi
infeasible; width shrinks by 2^-bits. Positive epsilon makes the
original physical domain compact, so the optimum exists. The method
does not claim polynomial bit complexity in input size or in the
number of supports; sparse support selection remains NP-complete.
Large rational coefficients can make isolation expensive.

## Use and evidence boundary
exact_radius.py implements this decision and bracket. Inputs are rational
numbers (no binary floats); eps_mass=d eps. Units are rho0 times score
gain; divide by rho0 to recover reader-score units. The tests include
isolated irrational roots, isolated fixed radii, dependent writes,
zero-dimensional and one-dimensional spans, and known analytic optima.
No neural checkpoint outcomes, exact hardware floating-point arithmetic,
certified full continuation, Lean proof, or claimed literature novelty.
Next steps: independent receipt verification and exact edit reconstruction,
then a preregistered small-support comparison on fresh native roots.
