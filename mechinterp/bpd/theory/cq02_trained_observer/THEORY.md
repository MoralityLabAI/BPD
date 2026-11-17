# BPD-CQ-02: identifiability, exact-kernel visibility and causal transport

For frozen BPD read/write factors V in R^(d x c), U in R^(c x d), let the native component write for input x and binary mask m be w_m(x)=((x V) odot m) U. This is exactly the `TiedComponentLinear` component contribution under zero residual mask and no sphere correction. A trained BPD gate selects m for every pass, but the resulting factors need not minimize an idealized expected-L0 objective globally.

For an observer C in R^(r x d) and externally specified state transition T in R^(d x d), a perturbation delta applied after component execution has visible effect at delay j:

    delta_y_j = C T^j delta.

For perturbations constrained to the trained write span delta = U^T a, the finite-horizon component observability matrix is

    O_k = stack(C U^T, C T U^T, ..., C T^k U^T).

The component-space dark directions at horizon k form ker(O_k). Their dimension is c-rank(O_k). The observable rank can grow across recurrences because kernels intersect, but never decreases with k. The Gramian G_k=O_k^T O_k gives the same nullspace, and therefore a precise finite-horizon *linear* BPD-CQ-01 witness. This is a statement about input-to-output response to controlled additive writes, independent of dictionary reconstruction MSE.

The registered synthetic source has d=4,c=2, C=e2^T and T swapping e1 and e2. Thus C T=e1^T, and the two-pass observed component matrix equals the 2x2 slice of learned U on its first two output channels. In any cohort where that slice is full rank, rank(O_0)=1 and rank(O_1)=2. The two-mask-space observer therefore gains one observable component coordinate at delay 1. The frozen identity-transition control has rank(O_1)=rank(O_0). These rank facts hold for this designed observer/transition and provide a necessary dimensional condition for distinguishing component perturbations over two time points.

The trained first-component native donor write is Delta=(x_d V)_c U_c-(x_r V)_c U_c. The immediate/delayed sensitivity ratios are |C Delta|/||Delta|| and |CT Delta|/||Delta||. The projected-fiber control removes the instantaneous visible part and rescales to the original norm. Random controls sample uniformly from the 3D kernel of C, independently per pair at matched norm. A full donor write swap is a deliberately easy positive functional control: in the two-component source the donor future value is itself defined by the component write. Its successful transfer is not proof of a uniquely localized semantic variable. The random-fiber and identity controls test whether the operator/routing math is implemented correctly.

A nonzero singular value of O_1 identifies an observable direction in a **chosen trained component basis**. It does not identify a canonical individual component if the expected-L0 optimum family contains nonmonomial transformations; BPD-CQ-01's commutant and block-mask algebra remains the correct intrinsic discussion. Likewise, the synthetic channel swap is not a learned loop operator or native computation path. Generalizing requires the actual per-visit Jacobian of a trained recurrent network, matched-fiber state pairs and independent heldout causal responses with sufficient seed-level power.
