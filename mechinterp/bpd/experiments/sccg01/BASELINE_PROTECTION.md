# SCCG-09: compensated finite protection with nonzero baseline scores

2026-11-27. Constructed rational development only. Additive follow-up
to SCCG-07/08. The exact Monte Carlo backend and historical results are
unchanged. This is a proposed application of classical norm bounds,
quadratic implication and univariate real algebraic geometry, not a
claim of priority.

## 1. Normalization-aware response and affine envelope

Let the physical intervention be B w, G=B^T B positive definite,
h=B^T x, c=C x, H=C B, and rho2=||x||²+m, m>0. The fixed query
matrix C includes RMS learned coordinate scaling. Scores use the
raw numerator divided by the RMS radius. Scaled score changes (rho0
times score changes) equal (c+H w)/s(w)-c, where
s(w)=sqrt(1+(2 h^T w+w^T G w)/rho2).
The dose constraint is w^T G w<=e², e>0; protection uses Euclidean
norm <=eta. The input contract requires rho2-h^T G^-1 h>0.

A zero direct protected write H w can still alter c/s. Example:
G=1, h=0, H=0, c=1, rho2=1, e=1, eta=1/4. The edit w=1 violates
the budget by normalization alone. These data are realizable with
nonzero x orthogonal to B and positive m.

Set ell=h/rho2 and b=e²/(2 rho2). The ratio s is the Euclidean norm of
an augmented affine vector divided by rho0; its supporting plane at
w=0 gives s>=1+ell^T w. The tangent bound sqrt(1+v)<=1+v/2 for v>=-1
gives s<=1+ell^T w+b. The bound uses the entire declared dose.
Require 1-e sqrt(ell^T G^-1 ell)>0 (the code verifies a conservative
rational upper bound for the root). Otherwise refuse this envelope.

For each w, the function f(s)=||H w+(1-s)c||-eta s is convex in s.
If f is nonpositive at the two affine endpoints, it is nonpositive
at the actual s. Let T=H-c ell^T. At endpoint s=1+ell^T w+beta,
beta in {0,b}, the condition becomes
||T w-beta c||<=eta (1+beta+ell^T w).
The derivative at zero of the actual scaled protected score is T.
This keeps cancellation between the direct response and changing radius.

## 2. Uniform PSD certificates

The endpoint condition is equivalent to the quadratic p_beta(w)>=0,
where p_beta=eta²(1+beta+ell^T w)²-||T w-beta c||². The right-hand
side of the unsquared inequality is nonnegative by the positivity
premise. Strict feasibility of the dose ball gives the S-lemma
(see Polik and Terlaky, 2007, https://doi.org/10.1137/S003614450444614X):
each endpoint holds throughout the ball iff some lambda>=0 makes

    [ eta² ell ell^T-T^T T+lambda G     eta²(1+beta)ell+beta T^T c ]
    [ transpose                         eta²(1+beta)²-beta²||c||²-lambda e² ]

positive semidefinite. Each endpoint has its own multiplier.
The cap is the bottom constant divided by e²; a negative cap
proves the endpoint condition impossible. Even complete endpoint
decisions only provide a sufficient test for the original nonlinear
uniform condition because the envelope may be conservative.
The implemented grid searches 65 rational points per endpoint.
SCCG-08's exact_multiplier.decide can consume the endpoint pencils
but is not called in this benchmark.

Cheap separate bound: use U(A)=sqrt(tr(G^-1 A^T A)) and L=
sqrt(ell^T G^-1 ell). Then e U(H)+(e L+b)||c||<=eta(1-e L)
suffices. Compensated bound replaces that left side with
e U(T)+b||c||. All computed square-root upper bounds use exact
integer square roots rounded upward at denominator 2^24. These
are sufficient screens, not exact tests. PSD witnesses are checked
using independent principal minors.

A cancellation fixture G=1,H=1,h=1,c=2,rho2=2,e=1/4,eta=1/16
has T=0. Separate screening refuses, compensated screening certifies.
The exact line oracle verifies uniform safety.

## 3. Complete exact line decision

Let y=c+H w and s² as above. Violation is equivalent to
V>J sqrt(s²), with V=||y||²+(||c||²-eta²)s², J=2 c^T y.
For J>=0 this is V>0 and V²>J² s².
For J<0 this is V>=0 or V²<J² s².
For a one-dimensional span the dose polynomial has degree two, V
degree at most two, J at most one and V²-J²s² at most four.
Isolate all distinct real roots of their square-free LCM. The truth
is constant on open sign cells; the two dose-boundary roots bound
the in-dose domain. Strict violation is continuous because the
normalization offset is positive. If a violation exists at an
isolated polynomial root or dose boundary, an adjacent open in-dose
interval also contains a violation. Checking rational samples of all
open in-dose cells is therefore complete, and returns a rational
violating edit when unsafe. Equality at the budget is allowed.
Unlike nonstrict feasibility in SCCG-08, this strict violation test
does not need isolated root witnesses. SymPy 1.14.0 root isolation
is a trusted dependency. No independently implemented root isolator.

## 4. Local constructed results

Fresh constructed seeds 71000..71095 (one-dimensional) and
72000..72095 (dimensions 2..4). Nonzero c, positive normalization
offset, varied budgets, rational nonorthogonal physical dictionaries.
Half the seeds use small planted compensated derivatives; the other
half use larger ones. Consequently these fixtures test the mechanism
and cannot estimate natural neural cancellation frequency.
Sixty-four exact rational sampled edits per root are a check, not
a proof of uniform safety. The exact line oracle checks every line
fixture. Fixed generator and arms were hashed before first execution.
After the first run the boundary fixture was clarified and exact equality
and nonpositive-lower-endpoint checks were added. Both receipts retained
locally; all 192 historical rows and totals match exactly.

| Regime | Separate | Compensated | PSD grid | Exact safe | Unsafe | Unresolved |
|---|---:|---:|---:|---:|---:|---:|
| Line | 21 | 40 | 46 | 51 | 45 exact | 0 |
| Multi | 6 | 23 | 34 | unavailable | 57 witnessed | 5 |

Combined admission is 80/192, 53 additional beyond separate screening.
PSD adds 17 beyond both trace screens. All 160 endpoint witnesses on
80 admitted subspaces pass principal-minor checks. No certified budget
violations among 12,288 sampled rational edits. Five line-safe cases
were not certified by the conservative pipeline; five multi-dimensional
cases are neither certified nor witnessed unsafe. Grid failure is
inconclusive; no approximate cache admission is treated as exact
structural exclusion.

Reproduce:
    python mechinterp/bpd/experiments/sccg01/baseline_benchmark.py --output sccg09_receipt.json

Local source Git blobs:
baseline_atlas.py c179dcc69bb499414754f79fd58ad7664dfeeb57
baseline_benchmark.py 9acf5e38b54f06f13255d1281f7fffab69f05489
First local benchmark source c3840e2450f4aee471241b1fcd0eb49828d3463b.
First local receipt SHA256
8fdb7b4762c129efb526a805f594f5e989f294b81b6a398eb5e8a153ff7c93a2.
Final local receipt SHA256
d0670180d71e5a156602f49e5971c276cf60799809db16fc9d7af833485f5cc9.
Local final runtime ~1.74s includes every comparator and sample replay;
no policy cost comparison.

## 5. Scope and follow-up

Fixed queries, rational inputs, exact mathematical RMS response,
no downstream continuation, measurement uncertainty, actual native
checkpoint evolution, measured RAM or training gains. The next step
is to connect the endpoint pencils to the complete multiplier solver,
tighten the envelope (e.g. radius or dose subdivisions), and use
the line oracle as a regression control. Quantify costs of reading
and refreshing the true geometry before using a budgeted native
training campaign; preserve seals and run EvalMath preflight.
