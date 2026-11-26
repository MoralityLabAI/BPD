# SCCG-02B: varying-radius exact nonlinear reader reduction
Date 2026-11-27. Mathematical addendum; no trained model outcomes.
Extends CURVATURE_TUBES.md beyond the fixed-radius treatment.
Assume eps_norm>0, fixed queries, fixed other block, and the same
normalization scale throughout the treatment. All expressions are exact
real arithmetic. Use e for the physical edit budget, eps_norm for the
normalizer epsilon, and rho0=sqrt(||x||^2+d eps_norm).

## The scalar reduction theorem
Let Q be an orthonormal basis of the allowed physical edit span S,
delta=Q z, C the protected effective raw query matrix, a the target
effective raw query. Require ||z||<=e and exact protected score equality:
C(x+Qz)/rho(x+Qz)=Cx/rho0.
Write tau=rho(x+Qz)/rho0>0. Put D=CQ, c=Cx.
Then D z=(tau-1)c.

If c is outside im D, only tau=1 is possible, and the fixed-radius
theorem solves the problem exactly. Otherwise set k=D^dagger c and
let N have orthonormal columns spanning ker D. Every solution of the
linear constraints has the unique decomposition
z=z0+N w, z0=(tau-1)k.
Here z0 is orthogonal to im N. Define h=Q^T x, p=N^T h, and
h_R=h-Np, x_perp=x-Qh. The normalization constraint is equivalent to
||p+w||^2 = R(tau)^2 =
tau^2 rho0^2 - d eps_norm - ||x_perp||^2 - ||h_R+z0||^2.
The remaining dose constraint is
||w||^2 <= E(tau)^2 = e^2-||z0||^2.
These are quadratic polynomials in tau. Reject negative squared radii.
For dim ker D=0, w is empty and feasibility requires R^2=0.
Otherwise let v=p+w. We require ||v||=R, ||v-p||<=E.
This is a spherical cap of possibly unequal radius.

At each feasible tau:
- If R=0, v=0, provided ||p||<=E.
- If p=0 and R>0, the sphere is feasible iff R<=E; choose v along
  b=N^T Q^T a (any direction if b=0).
- For dimension one, test v=+R and -R against the dose constraint.
- Otherwise put P=||p||>0, alpha=(R^2+P^2-E^2)/(2RP).
  Reject alpha>1. If alpha<=-1 the whole sphere is feasible.
  In the remaining cap use the fixed-radius optimal direction formula
  in CURVATURE_TUBES.md, replacing its cap threshold by alpha.
  When b=0 any feasible direction suffices.
For a chosen optimal v, target score gain is exactly
  (a^T x + a^T Q z0 + b^T(v-p))/(tau rho0) - a^T x/rho0.
Maximize this expression over tau within
[sqrt(max(0,||x||-e)^2+d eps_norm)/rho0,
 sqrt((||x||+e)^2+d eps_norm)/rho0].
The triangle and reverse triangle inequalities imply this interval.
Conversely, each admitted tau and cap vector constructs an admissible
edit with precisely that normalization ratio. Compactness at positive
eps_norm ensures a maximum exists. Thus the scalar maximization is
globally equivalent to the original finite protected control problem.

For rational/algebraic inputs one can express feasibility and
target-threshold tests by univariate semialgebraic conditions using
the cap formulas (or real-algebraic elimination with auxiliary
nonnegative roots). Ordinary grid search alone cannot certify a
global maximum or even detect isolated feasible tau values.
A certified root-isolation/interval implementation is future work.
The fixed-radius numerical solver is already implemented.

## Exact strict separation from fixed-radius treatment
Take x=(1,1), Q=I, C=(0,1), a=(-1,0), d eps_norm=1, dose e=1.
The fixed-radius protected feasible directions lie along the first
coordinate. The only nonzero feasible radius-preserving candidate
would flip x_1, costing 2, so the optimum is zero.
The varying-radius edited state y=(1/7,5/7) has
||y-x||^2=40/49<1 and rho(y)^2=75/49=(25/49)rho0^2.
Thus tau=5/7 and y_2=tau x_2; the protected score is exactly unchanged.
The target gain is 4/(5sqrt(3))>0.
This is a strict finite geometric separation with positive epsilon,
not an empirical neural steering effect.

## Applications and limits
A radius-preserving intervention may discard attainable selective
control; varying-radius edits should be a separately declared arm
with their actual radius and dose logged. The reduction preserves
named reader routing scores (and their sigmoid probabilities);
message content, moving queries, later gates and task responses
are outside this guarantee. No guarantee for source-parameter
interventions follows without a full native trajectory model.

For Fable: build a verified univariate semialgebraic solver for this
frontier, with branch-complete feasible intervals and isolated roots,
then combine it with support circuit enumeration under physical
coefficient or write-count bounds. Compare fixed radius, variable
radius, and full nonlinear task-level continuation on fresh development
roots. Source locks and EvalMath preflight precede any empirical claim.
