# PD-17 mathematical revision: source tilt, exact hyperplanes and observable fibres

## 1. What PD-15 estimated

Let the observed 16D linear source be X=Q A, with orthogonal Q, and a sparse latent support S with K=|S|. The PD-15 Gaussian objective after fixed observed weight exp(-lambda ||X||^2) is an ordinary Gaussian near-zero expectation under the exponentially tilted probability distribution P_lambda(dA)=exp(-lambda ||A||²)P(dA)/Z_lambda.

For the original clustered source, nonzero positive magnitudes are independent Uniform[0.75,1.25], conditional on S, so phi(lambda)=E[exp(-lambda B²)]. Conditional on K=k, E[weight | K=k]=phi(lambda)^k and

P_lambda(K=k)=P(K=k) phi(lambda)^k / sum_l P(K=l) phi(lambda)^l.

The registered source has P(K=k)=1/4. At lambda=1, phi=0.37530734, and its tilted cardinality profile is approximately (0.6373,0.2392,0.0898,0.0337), versus (0.25,0.25,0.25,0.25) in the original law. For the same support distribution, the source expected effective sample-size ratio is E[w]^2/E[w²]=0.48430. The weighted optimizer therefore sees more low-cardinality support, even though every training vector remains in the pool.

This statement identifies a **change of measure** under the original source model, not a general feature of pretrained recurrent-model activations. It also shifts active-amplitude distributions: the conditional nonzero B density acquires a factor exp(-lambda B²). Treat any causal attribution solely to changing K as a testable hypothesis.

## 2. A paired radial-camouflage law

Keep precisely the original clustered support S and hidden Q. Generate positive B_i iid Uniform[.9,1.1], independent signs and R independent Uniform[1.0,1.1]. For i in S define A_i=R sign_i B_i / sqrt(sum_{j in S} B_j²), and set A_i=0 outside S.

Then ||X||²=||A||²=R² exactly, making K and ||X||² independent. The weighting distribution is now P_lambda(K=k)=P(K=k)=1/4 for every lambda. The signed nonzero amplitudes remain continuously distributed on their support, with lower bound

min |A_i| >= min(R)*min(B)/(sqrt(4)*max(B)) = 0.9/2.2 = 0.4090909.

Thus the original physical read gate 0.375 correctly admits every exact planted-Q component in both laws.

The original and camouflaged laws have **the same exact support distribution and the same ideal zero-mass optimization problem**. Their Gaussian-smoothed optimization landscapes can differ. This makes PD-17 a source-distribution test of the norm-weighted heuristic and a falsifier of a stronger identifiability claim.

## 3. Exact hyperplane-mass theorem via a determinant matching

For any nonzero latent read vector m with support T={i:m_i != 0}, define z(m)=P[m^T A=0]. Conditional on S, a nontrivial linear form of continuously sampled active B_i (and independent signs) vanishes on a measure-zero set, including the radial-camouflage normalization because a positive denominator never changes zero locations. Therefore

z(m)=P[S intersect T = empty].

Let M be any invertible d×d matrix of read columns m_j, with d=16. Because det M != 0, at least one permutation pi has the nonzero determinant monomial product_j M_{pi(j),j}. In particular pi(j) lies in support T_j of m_j for every column.

Write e_i for the canonical latent coordinate read. Then

z(e_{pi(j)})-z(m_j)
 = P[ pi(j) not in S, and S intersect (T_j without pi(j)) nonempty ] >= 0.

For any column with |T_j| >= 2, choose i' in T_j different from pi(j). The event S={i'} has probability (1/4)*(1/16)=1/64 under **both** registered laws and is inside the event on the right. Hence

z(e_{pi(j)})-z(m_j) >= 1/64.

Summing over the matched columns, the normalized average exact-zero mass of any invertible basis with at least one genuinely mixed read is **at least 1/(64d)=1/1024** below the canonical coordinate basis. Coordinate sign changes and permutations preserve this exact objective. This includes invertible nonorthogonal read bases, although the registered fit constrains read columns orthogonally.

The ideal exact-zero population optimum is thus uniquely the canonical latent-axis family up to rescaling and permutation. This proof requires neither norm/cardinality dependence nor recurrent dynamics. It relies on a positive probability of each singleton source, continuously varying active amplitudes, and the stated full-basis condition. The gap is discontinuous at a column with an arbitrarily small extra nonzero coefficient, so it is **not** a uniform finite-sigma basin or convergence theorem.

The companion exact_source_math.py enumerates the actual PD-12 clustered source with rational probabilities and checks the 120 coordinate-pair hyperplane mass inequalities. This source-only audit is separate from the neural experiment.

## 4. An observable alternative: projective-ray concentration

The singleton fraction is 1/4. For a singleton sample, X=+/-A Qe_i; after normalizing X to its direction, the observed ray is exactly +/-Qe_i independent of radius. In N=8192 observed pass vectors, each of 16 coordinate rays appears in expectation N/(4*16)=128 times.

A label-free empirical frequency of nearly parallel observed directions can therefore expose the axes directly. PD-17's fixed angular control uses the first 1024 training vectors as ray candidates and the full 8192 as an index, counts absolute-cosine neighbors, applies a fixed nonmaximum separation rule, and polar-orthogonalizes 16 selected rays. This estimates the same signed-permutation-gauge latent basis by a **different sufficient observable** that survives the proposed cardinality-independent radius.

The angular method is a source-specific information comparator with an extra O(1024×8192×16) similarity budget. Its success would demonstrate that losing the norm/cardinality correlation need not remove all information about Q. It does not solve sparse-source identification when singleton rays are absent and does not automatically translate to nonlinear recurrent features.

## 5. Bridge to causal quotient and recent trained-model evidence

The ideal sparse-basis theorem is a statement about samples and latent coordinates; physical BPD interventions and future task effects are distinct. In the identity teacher, VU=I admits arbitrary factor rotations for full-mask reconstruction while component-specific masks depend on basis, a gauge ambiguity. The correct observable is the task-conditioned **physical intervention operator** and its downstream effect.

Compactification CGC-01 formalizes the protected tangent K=ker DP and task-conditioned rank rank(DT|K), with SVD-tail and finite-radius bounds. BPD NC-03 verifies gauge-covariant recurrent projector transport on constrained synthetic recurrences. NC-04's matched-sham gain without adequate absolute keep-only fidelity shows why a projector can be selective while insufficient. LOOPSCOPE-08's shared four-pass energy with large curl and weak steering gains is an even closer warning that a favorable potential need not be the computational direction that causes future task improvement. ROC-05B and LS-SAE12 reinforce the need for dose-matched, source-legal, heldout behavior.

A future recurrent BPD experiment should declare one hidden-state edit site and remaining-loop horizon, require full-mask clean closure at every pass, compare the task derivative of a gauge-covariant projector to same-radius/source-edit and random-projector shams, and use independent model-root variation floors from EvalMath. It must be registered separately from PD-17 and train genuinely recurrent nonlinear models rather than scoring independent identity passes.

## 6. Claim classes

Theorem: the support-matching exact-zero gap for both specified source laws.

Population calculation: exponential cardinality/amplitude tilting under the original source and exact radial independence under camouflage.

Engineering diagnostic: label-free ray feasibility, source hash parity, and oracle mask admissibility.

Prospective experiment: paired 32-root comparison of the norm weighting advantage across two laws, under frozen training and exact mechanism gates.

Trained recurrent neural mechanism identification: untested by PD-17.
