# SCCG-07: robust finite collateral certificates for stale atlases

Constructed rational development only. Follow-up to merged SCCG-06 and its
opt-in Monte Carlo backend. No change to the exact test runner or its default.

## Mathematical reduction

Assume the clean protected scores are zero, physical edit delta=B w is
restricted to a cached span B=W Z, and the actual reduced protected map
is H=C_actual B (or C_actual Z if C_actual is in coefficient coordinates).
If the old protection map annihilates B, H is its residual restricted to B.
Let G=B^T B be positive definite, h=B^T x, rho2=||x||²+m, m>0.
Use scaled collateral norm rho0 ||H w||/rho(x+B w), rho0=sqrt(rho2).
The radius constraint is w^T G w <= e² with e>0, and the protected-score
budget is eta>=0. Zero clean score is essential.

For every w in the ball the budget holds iff the quadratic
p(w)=eta² rho2+2 eta² h^T w+w^T(eta² G-rho2 H^T H)w >=0.
The normalization denominator is positive; the implementation requires
rho2-h^T G^-1 h>0. By the S-lemma (see Pólik and Terlaky, 2007,
https://doi.org/10.1137/S003614450444614X), strict feasibility of the
dose ball gives equivalence to existence of lambda>=0 with
p(w)-lambda(e²-w^T G w)>=0 globally. The latter is exactly the PSD
condition for the augmented matrix

    [ (lambda+eta²)G-rho2 H^T H    eta² h          ]
    [ eta² h^T                    eta² rho2-lambda e² ].

Any rational lambda and exact PSD witness certifies every physical
edit in the cached subspace, including finite RMS response. The bottom
diagonal bounds lambda by eta² rho2/e². This is an exact real theorem
using the classical S-lemma, not a priority claim for a new optimization
theorem. The code tests 65 equally spaced rational multipliers in
this interval. For nonorthogonal x that search is sufficient only:
no witness is inconclusive, not evidence of an infeasible intervention
or an unsafe whole subspace. A separately checked violating rational
edit proves the subspace is unsafe for uniform reuse.

When h=0, the denominator is rho2+w^T G w; along every fixed ray the
squared normalized protected effect increases with dose. At full dose
the exact condition is

    eta²(rho2+e²)G - rho2 e² H^T H >=_PSD 0.

The lambda upper endpoint gives the same condition. This special case
is completely decided by exact rational PSD without a grid.

Cheap sufficient norm bound: trace(G^-1 H^T H) upper bounds the largest
generalized eigenvalue and rho2-h^T G^-1 h lower bounds the denominator
squared. Hence rho2 e² trace(G^-1 H^T H) <=
eta² (rho2-h^T G^-1 h) is sufficient. Compare before searching the
more expensive multipliers. A production policy should account for
computing the actual H, metric, clean response, and certificate cost.

## Local constructed benchmark

96 seeds 61000..61095 in each of two paired regimes: clean state
orthogonal to the cached span, and varying clean editable coordinates
(which can include zero). Dimensions 1..4, nonorthogonal physical
dictionary, 1..3 residual rows, positive dose, varied budgets.
Every PSD witness is independently checked by all principal minors.
The orthogonal decisions are independently compared to the exact
matrix condition. 64 rational in-dose edits per fixture check for
violating witnesses; sampling is an engineering test, not the proof.

| Regime | Trace certified | PSD certified | Extra over trace | Breach witnessed | Unresolved |
|---|---:|---:|---:|---:|---:|
| Orthogonal | 35 | 41 | 6 | 50 | 0 (55 exactly unsafe) |
| Varying clean coordinates | 29 | 36 | 7 | 58 | 2 |

All 77 PSD certificates passed independent principal-minor replay.
All 12,288 sampled edits agree with the guarantees. Orthogonal unsafe
cases without a sampled breach remain exactly unsafe by the theorem.
For varying coordinates the two unresolved cases are neither certified
safe nor proven unsafe. Combined coverage is 77/192, versus 64/192 for
the trace bound. These are paired constructed fixtures, not independent
neural roots or evidence of population-level gains.

The program refuses singular physical metrics and invalid dimensions.
The gate currently checks rational data, with zero clean protected
scores and fixed reader queries. It does not bound floating measurement
error, downstream nonlinear continuation, or moving queries. For
nonzero protected baseline the score difference contains a varying
normalization term; use a separate valid certificate. For general
recurrent task response, append SCCG-02 curvature error bounds with
their domain and measurement premises. Never use stale nullspace
membership as an exact structural exclusion after quotient drift.
Refresh or fall back to exact testing on certificate failure.

## Reproduction

    python mechinterp/bpd/experiments/sccg01/robust_atlas.py --output robust_receipt.json

Local executed source Git blob 17309e6cd0d8e9032811c116f90aa46d314cd9cb.
Source SHA256 5d3e263dccffec74aae6a7491bf29fa828d400baba5c6646470d710ee51adf39.
Receipt SHA256 ad33ef96a6542a6cb1b5d570df97de56b76effa952b6680af25a06c6b1006ab9.
No elapsed-time or memory-speedup claim for this certificate.
Next: complete univariate multiplier feasibility (including algebraic
singleton witnesses), propagate bounded measurement error, and measure
actual cache refresh/reinvestment cost on fresh native training roots.
