# NC-06: covariance conditioning, gauge transport and the task-fidelity floor

## Weighted gradient covariance is a change of measure

On a clean, trained four-pass recurrent trajectory let a_t(x) be the prephysical-write activation, z_t(x) the full-rank BPD read factors, and g_t(x)=grad_{z_t} s(x) for a model-only final-score functional s(x). The discovered rank-r projector is constructed from the passwise weighted gradient second moment

C_t(lambda) = E[w_t(x) g_t(x)g_t(x)^T] / E[w_t(x)],

where w_t(x)=exp(-lambda ||a_t(x)||² / s_t) and s_t=E||a_t(x)||² is measured on the discovery source only. This is the Fisher-like uncentered score second moment under the tilted discovery measure dP_t^lambda=w_t dP/Ew_t.

The top-r eigenspace maximizes tr(P C_t) over orthogonal rank-r projectors P. For lambda=0 it reduces to NC-05's source-registered unweighted gradient covariance; lambda=1 is a fixed source-observable conditioning rule. That variational identity is an algebraic property of covariance. It does not imply that retaining the top-r physical write minimizes terminal held-out true-label NLL, since the future nonlinear recurrence can amplify discarded directions and the model-score functional differs from external labels.

## Score sign invariance

For any source-dependent sign epsilon(x) in {−1,+1} detached from the gradient graph, g_t changes to epsilon(x)g_t. Each outer product is invariant. Thus the gradient second moment and its eigenspace do not gain target-label information from changing the score sign. The positive co-attribution graph may still be sign-sensitive. NC-06 uses model final-logit signs in discovery solely to reproduce NC-05's physical-derivative definition.

## Gauge covariance and its boundary

For an **orthonormal** factor rotation R, the BPD read and write factors transform V'=VR and U'=R^T U, preserving physical W=VU. Read activations and gradients transform z'=zR and g'=R^T g, while physical prewrite activation norms are invariant. Therefore C'_t=R^T C_t R, and, where the rank-r eigengap is nonzero, P'_t=R^T P_t R. The physical retained write map V'P'_tU'=V P_t U is identical in exact arithmetic. Its intervention must transport to the same future task outputs under fully closed-loop execution. This is the direct NC-03/NC-05 gauge test.

The same statement requires a declared physical metric under general nonorthogonal GL(d) coordinate changes; Euclidean top eigenvectors alone are not invariant under arbitrary invertible rescaling. NC-06 tests only orthonormal gauge changes, a deliberately restricted scope.

## Exact ESS-matched shuffle placebo

For fixed per-pass positive observed weights w_1,...,w_N and a uniformly sampled independent permutation pi, the placebo computes C_t^pi = sum_n w_pi(n) g_ng_n^T / sum_n w_n. Its empirical weight multiset and effective sample size (sum w)^2 / sum w² are exactly preserved, and

E_pi[C_t^pi | g,w] = (1/N) sum_n g_ng_n^T.

The permutation expectation equals the **uniform empirical covariance**, regardless of the weight distribution. For one realized placebo permutation C_t^pi is still a random matrix and its eigenspace can fluctuate. As PD-16 showed in its sparse-linear source, identical ESS is not sufficient for informative factor recovery. NC-06 tests the same association in a learned recurrence with a genuinely external terminal task.

## Absolute terminal fidelity and NC-05 rank frontier

NC-05's rank-four full-gain developmental arm met mean kept-minus-clean true-label NLL +0.021219 nats, adjusted upper +0.045398; its rank-two arm exceeded the +0.05 ceiling. A rank-three result must satisfy an absolute fidelity criterion and an improvement against same-rank controls; comparing only relative sham sensitivity can promote a severely damaging intervention. At all four passes, closed-loop physical updates must remain numerically matched between native and orthogonal factor coordinates and the final external-label outcome must be held out from discovery.

The model's original updates can have nonconservative curl, as LOOPSCOPE-08's trained 44K recurrence demonstrated. An observed gradient-covariance projector is a **causal intervention candidate**, not a proof that the recurrent update performs gradient descent. The actual retained physical write and future true-label loss determine this experiment's practical effect.

## Inference unit

Eight independently trained model roots form the development sampling units. A 2,000-resample whole-root bootstrap summarizes variability in the fixed training family; its small-sample interval is an engineering decision aid. At least 59 genuinely independent sham-calibration roots and a new power preflight are needed before any protected 32-root confirmation, consistent with the program's EvalMath variation-floor requirements. Repeated passes, prompts and shams inside the same model are nested observations and never added to the number of independent roots.
