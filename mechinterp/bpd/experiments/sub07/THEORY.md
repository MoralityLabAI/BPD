# BPD-SUB-07: finite-depth information certificates and clock-indexed causal quotients

## 1. A compositional task with provable future-token necessity

Define the state space H_oracle=F₂² and the four bijections X,S,C,R by

X(a,b)=(a⊕1,b), S(a,b)=(b,a), C(a,b)=(a,b⊕a), R(a,b)=(a⊕b,b).

Start at (0,1), execute an ordered six-token sequence s∈{X,S,C,R}⁶ from left to right, and return final second state bit. The task's label is a deterministic finite computation. Source code enumerates all 4096 ordered sequences, so it has an exact finite population of labels, with 2291 positive labels and 1805 negatives.

For any restricted statistic T(s), the Bayes-optimal accuracy under the uniform law over all 4096 sequences is

\`B(T)=1/4096 Σ_v max(#{s:T(s)=v,y(s)=0}, #{s:T(s)=v,y(s)=1})\`.

The oracle enumerator returns exact integer numerators for first two-token prefixes B=2399/4096, first four-token prefixes B=2735/4096, last token B=2291/4096, and unordered four-token count histograms B=2335/4096. Every predictor measurable with respect to T, even with unlimited parameters and perfect training, has accuracy at most B(T) on the full uniform population. This is an information-theoretic ceiling, independent of a neural architecture or a fitted source head. A finite holdout subset can fluctuate around the population bound, and an individual source-root's actual prefix classifier is a separate descriptive measurement.

The conditional entropy \`H(Y|T)=Σ_v P(T=v)H₂(P(Y=1|T=v))\` is 0.96949 bits for the two-token prefix and 0.87109 for the first-four-token prefix. Future tokens retain substantial label information. Noncommuting order matters: two six-token sequences with identical token-count histograms, (X,X,S,S,R,C) and (X,X,S,S,C,R), have different final labels.

The exact oracle witnesses **data/information depth necessity** under the full ordered-sequence input model. It cannot prove that the trained GRU uses all six recurrent passes in the same way: a different architecture can read all six tokens at once. That model-mechanism question requires observed source competence, counterfactual interventions and fair controls.

## 2. Physical recurrence with changing external inputs

A tied six-pass GRU is a driven dynamical system

\`h_(t+1)=F_θ(h_t,x_(t+1))\`, with constant θ and six *distinct* input tokens. The output logit is \`R_θ(h₆)\`. The Jacobian for a null-fiber intervention after pass t is \`D_t(x)=D_h F_θ(h_t,x_(t+1))\`. The next-token argument is part of the causal mechanism and must match between clean and intervened source rollouts.

For rank-r orthogonal projectors P_t on the physical hidden state, a quotient transition \`G_(t,x):im(P_t)→im(P_(t+1))\` factors the source on a specified domain precisely when \`P_(t+1)F_θ\` is constant on each P_t fiber for that t and next-token x_(t+1). The local necessary differential condition is

\`P_(t+1)D_t(x)(I-P_t)=0\`.

For any finite δ∈ker(P_t) and an admissible convex fiber segment,

\`P_(t+1)[F_θ(h+δ,x)-F_θ(h,x)] = ∫₀¹ P_(t+1) D_h F_θ(h+sδ,x)(I-P_t)δ ds\`.

A uniform bound ε on the cross-block operator norm over the segment gives the finite-distance guarantee \`||P_(t+1)ΔF||≤ε||δ||\`. The SUB-07 source projector optimizer minimizes an empirical **linearized** approximation of this condition on 64 predefined discovery input sequences and all five recurrent transitions. Its heldout test measures actual finite-dose nonlinear causal responses through GRUCell dynamics.

## 3. Matching temporal source-fitting exposure

Let \`C_t=E_x[g_tg_tᵀ]\`, \`g_t=∂R(h6)/∂h_t\`, computed on all 512 discovery sequences. The static and time-indexed source projectors are fitted by the frozen QR/Adam procedures in PROTOCOL.md.

Each of seven wrong-clock controls permutes the full list of discovery transition-Jacobian matrices and the six passwise future-output covariance matrices **before** training its own six-projector bundle. It receives the same initial orthonormal static basis, 120 steps, source examples, optimizer and loss normalizations as the correct-clock candidate. The independent fitted wrong-clock bundle is evaluated in its native fitted order under the real input sequence and physical model. The strongest wrong-clock family member and static baseline form one conservative, source-root-level primary null statistic:

\`d_i=min(C_i(P_static),C_i(P_wrong,0),…,C_i(P_wrong,6))-C_i(P_aligned)\`.

Each C_i is the finite-dose root-level pooled next-state RMS. Seven wrong-clock fits from one source are *dependent* repeated measurements within that trained root. Root bootstrap and EvalMath planning count one independent trained source model, even though the training sequence split is shared across roots. A future model-distribution or task-distribution generalization claim would require those distributions to be specified prospectively.

## 4. Gauge covariance and limits of identification

Changing orthonormal coordinates by Q transports states and projectors as z=Qh and P'_t=QP_tQᵀ. The physically transformed GRU is \`F'_θ(z,x)=QF_θ(Qᵀz,x)\`, so correctly transported impulses δ' = Qδ obey exact algebraic parity on clean and intervened trajectories, subject only to floating-point error. This is the source-only gauge positive control.

A uniformly nonzero cross-block on heldout interventions is an obstruction to exact quotient closure on that counterfactual domain. A small empirical RMS is a protocol-relative estimate of approximate closure under the specified L2 physical intervention distribution. The GRU does not represent a biological synapse or a physical neural implant, and the result offers no evidence about phenomenal consciousness or personal identity transfer.

The preregistered +0.010 RMS effect, clean model fidelity gates and strict EvalMath sham-tail boundary are engineering/statistical decisions separate from the finite exact oracle theorem. A failure of source competence is a model-training result and must be preserved before any causal interpretation.
