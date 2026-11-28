# Reach-Transport Behavioral Parameter Decomposition

RT-BPD-02 mathematical extension, 23 November 2026

A shared weight matrix acts on a different input distribution each time it is called. A physical component can write the same vector at two occurrences and have different effects when a later computation reads that vector. This paper separates compression at a registered occurrence from the causal effect of a finite edit at that occurrence. The companion [registration](EXTENSION_PROTOCOL.md) fixes the synthetic and archive comparisons.

The starting point is the [Nanbeige Q4 synthesis](https://github.com/MoralityLabAI/LoopedTransformers/blob/c074e354e8bcd4d26c02104fadd289dfdad8a207/reports/nanbeige_bpd_gym/NANBEIGE_RESEARCH_SYNTHESIS_20261123.md). Its physical-closure, pass, matched-dose, and routing checks motivate which quantities to preserve. In particular, an all-one full-head check qualifies a write decomposition, local reader geometry identifies a possible path, and later causal efficacy requires an intervention on a paired source contrast.

## 1. Occurrences and the reached second moment

Fix a parameter occurrence $p$ with a linear physical write $Wh$, input coordinate space $X$, output space $Y$, and a distribution $\mu_p$ over reached $h\in X$. A precision backend, token position, recurrence pass, and module site are part of the occurrence definition. Let $\Sigma_p=\mathbb E_{\mu_p}[hh^\top]$ and let $A_p$ map writes into fixed measured coordinates. With finite second moments, linearity of expectation gives

$$
 \mathcal E_p(K)=\mathbb E_{\mu_p}\big[\|A_p(W-K)h\|_2^2\big]
 =\operatorname{tr}\big[A_p(W-K)\Sigma_p(W-K)^\top A_p^\top\big].
$$

The moment is uncentered. Centered covariance omits the error made on the mean and corresponds to a different fitting problem unless a separate offset is restored. The paired contrast moment $\mathbb E[(h_B-h_A)(h_B-h_A)^\top]$ answers the separate question of how closely the component preserves donor–recipient writes. For equally weighted roots with two worlds, use half the sum of both world terms within a root and then average across roots. Changes in recurrence, input prompts, quantization, or prior edits can change the law of $h$ while the parameter identity stays fixed.

This is an expectation over a fixed law. If replacing $W$ changes the reaches, the objective describes a frozen activation replay. The edited rollout requires a new law or a bound on how far it moves.

## 2. Exact weighted low-rank approximation

**Theorem 1 (weighted Eckart–Young lift).** Let $W:X\to Y$, $\Sigma\succeq0$, $S=\Sigma^{1/2}$, and $A:Y\to Z$. For any nonnegative integer $r$, set $M=AWS$. If $M_r$ is obtained by keeping at most its $r$ largest singular modes, then

$$
 K_* = A^+M_r S^+,\qquad
 \min_{\operatorname{rank}K\le r} \|A(W-K)S\|_F^2
 = \sum_{i>r}\sigma_i(M)^2.
$$

The statements include zero supports, zero rank, and a budget exceeding the target rank. Any feasible $AKS$ has rank at most $r$, so ordinary Eckart–Young–Mirsky gives the lower bound. The nonzero left and right singular vectors used in a truncation lie in $\operatorname{im}A$ and $\operatorname{im}S$. Thus $AA^+M_rS^+S=M_r$, and the lift attains the bound with rank at most $r$. The same support argument applies to a choice within a tied positive singular block.

For a fixed optimal image $M_r$, the minimum-Frobenius-norm lift is the displayed pseudoinverse formula. The map $K\mapsto AKS$ constrains the block with columns in $\operatorname{im}A^\top$ and rows in $\operatorname{im}S$. Every other block is Frobenius-orthogonal to that block, and setting those blocks to zero attains minimum norm. The rank-constrained optimizers consist of all lifts of optimal images that satisfy the rank budget. This proof gives minimum norm for the chosen image individually. A different tied truncation can have a different lift norm. If $\sigma_r>\sigma_{r+1}$ at a positive cutoff, the projected optimizer is unique. Singular metrics can still leave physical actions undetermined. For example, with $W=I$ and $\Sigma=\operatorname{diag}(1,0)$, every rank-one map sending $e_1$ to $e_1$ fits observed inputs exactly.

One can replace $S$ by any thin factor $C$ with $CC^\top=\Sigma$ and use $C^+$ in the lift. If weighted sample rows are $D_i=\sqrt{w_i}\,h_i^\top$ with normalized positive total, a thin SVD of $D$ supplies the factor and its supported directions. A fixed reader on an empirical train support then requires a target with one column per retained sample mode, rather than a full dense input moment.

**Numerical policy.** The code retains moment eigenvalues greater than $\tau\lambda_{\max}$ and reader singular values greater than $\tau\sigma_{\max}$. It constructs the target and lift on exactly those retained supports. It reports both the effective objective and the full loss for the supplied reader and symmetrized PSD moment. The latter evaluates the same estimate, including the modes omitted in fitting. Small negative eigenvalues within the PSD validation tolerance are clipped. For an illustrative failure, $A=\operatorname{diag}(1,10^{-9})$, $W=\operatorname{diag}(2,10^{10})$, and $\Sigma=I$ put the second singular value of $AWS$ at 10. Dropping that reader direction only during lifting can spend the entire rank-one budget there. Consistent retained support spends it on the first direction, for effective loss zero and full loss 100. A small full loss, a stable subspace, and a well-conditioned lift require additional spectral information. At a tie the selected basis can change under arbitrarily small perturbations.

**Coordinate covariance.** If $h'=Gh$ and $y'=Hy$ for invertible $G,H$, use $W'=HWG^{-1}$, $K'=HKG^{-1}$, $\Sigma'=G\Sigma G^\top$ and $A'=AH^{-1}$ to keep the measured output fixed. The measured residual for every sample and its squared expectation are invariant, and rank is preserved. This bijects exact feasible and minimizing solution sets. A coordinate Euclidean minimum-norm lift is equivariant under orthogonal changes in the corresponding frames, when the projected optimum is unique or an equivariant tie choice is supplied. General invertible maps change Euclidean lengths, the off-support minimum-norm extension and potentially numerical ranks. With $W=I$, $\Sigma=e_1e_1^\top$, $A=I$, $H=I$ and $G=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$, the new minimum-norm map is $e_1e_1^\top$. Transporting the old lift instead gives $e_1(1,-1)$, with identical actions on the measured support. Physical-dose claims fix a metric on $Y$ and transport that metric when coordinates change.

## 3. Disagreement between two occurrences

**Theorem 2 (exact paired identity).** Embed both occurrences in the same physical frames. For $b$ minus $a$, bars denote arithmetic means and $\Delta$ denotes a difference. Expanding the bilinear map yields

$$
 K_bh_b-K_ah_a=\Delta K\,\bar h + \bar K\,\Delta h.
$$

When the readout changes, $\Delta(AKh)=\Delta A\,\overline{Kh}+\bar A(\Delta K\,\bar h+\bar K\,\Delta h)$. These operations can be evaluated from low-rank map actions on the two reached inputs. The corresponding energy includes the signed cross term $2\langle\Delta K\bar h,\bar K\Delta h\rangle$ and, with varying readers, two more cross terms. A large operator term can cancel a large state term. For scalar $K_a=1$, $K_b=2$, $h_a=2$, $h_b=1$, both writes equal 2 and the terms are $1.5$ and $-1.5$.

With exact tied weights in a shared native frame, $\Delta K=0$ for a truly fixed component. A separate fit at each pass can have nonzero $\Delta K$ even when the parent weight is tied. Across precisions both the recorded reached states and effective operators may change. An equality for this decomposition gives a paired descriptive accounting. A manipulation of each term under a common baseline is an additional causal design.

## 4. Future readers, delayed visibility, and direction

**Theorem 3 (fixed linear future reader).** Inject a write $\delta y$ through $B$ into a dynamical state with tangent transitions $J_1,\ldots,J_L$. Fix local readout maps $C_\ell$ and nonnegative weights $w_\ell$, including any output metric factors. The stack with blocks

$$
 A_\ell=\sqrt{w_\ell}\,C_\ell J_\ell\cdots J_1 B
$$

maps the physical write to the weighted measured trajectory. At horizon zero the transition product is the identity. The squared norm of the stack equals the sum of squared readout errors over those horizons. Thus Theorem 1 solves the rank problem for this stack and a fixed reach law. The identity holds for finite edits of a linear dynamical system and for first derivatives along a frozen differentiable trajectory. If the true readout $f$ has an $L$-Lipschitz Jacobian along the edit segment, the linearization remainder is at most $L\|\delta y\|^2/2$ by integrating the derivative. A fixed stack shared across roots is an explicit approximation when the native Jacobian depends on the root. A general root-dependent $A(h)$ induces a quadratic form involving the expectation of $(hh^\top)\otimes(A(h)^\top A(h))$, which need not factor into two separate metrics.

**Theorem 4 (delayed-visible rank within the local kernel).** On a write domain $D$, let $C$ be the present reader and $O$ the transported future stack. The number of independent future-visible directions among currently dark writes is

$$
 \operatorname{rank}\big(O|_{D\cap\ker C}\big)
 = \operatorname{rank}\big([C;O]|_D\big)-\operatorname{rank}(C|_D).
$$

Project the image of the stacked map onto its current-reader block. Its range is $\operatorname{im}(C|_D)$ and its kernel consists of images $(0,Od)$ of currently dark writes. Rank–nullity proves the identity. A cache slot read only after a transition is an example. A nonzero delayed rank certifies existence of a linear direction. It does not bound its gain or guarantee that a reached donor contrast excites it. A small singular value may be unusable at a stated dose and noise floor. Numerical ranks depend on these scales and the chosen metrics.

For a reference readout displacement $v$ and observed displacement $u$, use signed gain $u^\top v/\|v\|^2$, readout length $\|u\|$, and cosine $u^\top v/(\|u\|\|v\|)$ where defined. Equal physical $\|\delta y\|$ constrains intervention size. It allows both positive and negative projections of equal magnitude, as from writes $e_1$ and $-e_1$ under reader $e_1^\top$. Distinct source worlds supply the reference displacement. Valid comparisons track zero writes and zero reference readouts explicitly.

## 5. Cross-pass and cross-precision transport

**Theorem 5 (sufficient common-target bound).** For one common target and candidate in aligned frames, let $P_j=A_j^\top A_j$. If $\Sigma_q\preceq M\Sigma_p$ and $P_q\preceq cP_p$, then $\mathcal E_q(K)\le Mc\,\mathcal E_p(K)$. Write $D=W-K$. First use $\Sigma_q\preceq M\Sigma_p$ in $\operatorname{tr}(P_q D\Sigma_qD^\top)$. Then use $P_q\preceq cP_p$ on the positive semidefinite matrix $D\Sigma_pD^\top$. The two trace inequalities give the bound.

There is a finite constant dominating one PSD matrix by another exactly when the target support is contained in the source support. On the source support the least constant is the top eigenvalue after whitening the target by the source. Any target direction visible in the source kernel gives a witness against finite domination. This also shows why a thin empirical CAL moment may give a useful within-fold fit yet provide no finite guarantee on a new root. As a simple witness, $\Sigma_p=e_1e_1^\top$, $\Sigma_q=e_2e_2^\top$, $W=I$, and $K=e_1e_1^\top$ yield errors 0 and 1. An analogous witness takes equal state moments and orthogonal reader supports.

For a changed target, the Hilbert-space triangle inequality gives
$\sqrt{\mathcal E_q(W_q,K_p)}\le\sqrt{Mc\,\mathcal E_p(W_p,K_p)}+\|A_q(W_q-W_p)\Sigma_q^{1/2}\|_F$ under the same metric conditions. For $K_q$ in place of $K_p$, add the corresponding norm of $K_q-K_p$. Approximate domination with a residual term is possible when its action is bounded separately. To transfer a directional behavioral effect, additionally require a common outcome direction, signed gain on reached edits, and a bound on the linearization remainder at the actual matched dose. None follows from a small Frobenius reconstruction error alone.

## 6. Signed paths and interactions

**Theorem 6 (signed interference on a linear unrolled DAG).** Order scalar nodes in causal time and place edge derivatives in strictly lower triangular $T$ with entries $T_{ji}$ for $i\to j$. A perturbation at source $s$ reaches target $t$ with coefficient $[(I-T)^{-1}]_{ts}=\sum_{k=0}^{n-1}(T^k)_{ts}$. Each power enumerates products on paths of its length. Replacing $T$ by $|T|$ gives the sum of the absolute path products, an upper bound on the absolute signed sum. Vector paths use ordered products of linear blocks before scalar projection. Parallel paths of products $+1$ and $-1$ give a zero result despite two individually nonzero paths.

For nonlinear finite edits, the mixed intervention contrast $f(1,1)-f(1,0)-f(0,1)+f(0,0)$ can be nonzero. For $f(u,v)=uv$ at the zero baseline it equals 1 while both single effects and the first-order derivatives vanish. Corresponding single and joint interventions at physical matched dose are needed to distinguish cancellation, redundancy, and a genuine joint effect. Path accounting with a fixed graph gives a derivative prediction and may miss a routing change when a finite edit alters the graph's effective coefficients.

## 7. What can be tested

The synthetic gym gives exact planted channels, independent paired roots, and a delayed reader whose fixed map can be verified by explicit continuation. Its stress panel changes the reached distribution and tests where a fixed compression loses its advantage. The learned T1/T2 atom banks are evaluated separately on exposed Nanbeige CAL archives. For those archives the target is a frozen fitted rank-eight sum and closure against the original BF16 affine module is audited separately. A fixed K-reader response is a linear functional of an archived physical write in one occurrence. For native causal confirmation, future work needs exact all-one full-head parity, aligned per-root donor and recipient states and output directions, native single and joint interventions at matched physical doses, and fresh roots with independently calibrated shams. The processed archive has no authority to release confirmation data.

A hypothesis left open is that fewer calibrated physical factors can produce equal or better causal rescue on unseen roots than Euclidean SVD or the existing learned banks at equal rank and measured dose. In particular, cache delay, uncertainty in the reference direction, and signed downstream cancellation can reverse an apparent advantage measured by a local reader. The fixed-metric theorems specify what can transfer and expose failure cases. The developmental measurements test specific instances under the registered support and readers.
