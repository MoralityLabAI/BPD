# BPD-PD-13 post-hoc audit: the Gaussian sparsity surrogate reverses the basis ordering

**Status:** exploratory analysis conducted after the original source-locked scientific FAIL. This supplements RESULTS_20261118.md and leaves the first registered cohort, protocol, scorer, original artifact and verdict unchanged.

## An explicit off-axis counterexample

The registered 16D clustered source selects supports of sizes 1,2,3,4 uniformly. For size at least two, it chooses one of the four anchored contiguous blocks with probability 0.85, and another permitted support otherwise. Each nonzero latent has an independent Rademacher sign and a uniform positive amplitude.

On any anchored pair, for example (0,1), enumerate the registered support law. The probabilities for the activation states (0,0), (1,0), (0,1), (1,1) are respectively

0.770379024670, 0.033527225330, 0.033527225330, 0.162566524670.

Let A,B be independent active magnitudes on the registered interval, with independent random signs. For Gaussian near-zero mass exp(-z²/(2 sigma²)), put:

- g = E exp(-A²/(2 sigma²))
- h = E exp(-A²/(4 sigma²))
- c = E[exp(-(A+B)²/(4 sigma²)) + exp(-(A-B)²/(4 sigma²))].

This remix is defined in the planted latent-coordinate frame using the known Q only for a post-hoc diagnostic. Its physical read basis is Q times a fixed four-pair Givens matrix. It is an existence witness for the objective landscape, not a label-free algorithm that recovers those directions from x.

Rotating the pair (A,B) by 45 degrees gives the exact expected *two-read score-sum* difference

    D = (p10+p01) (2h-1-g) + p11 (c-2g).

The first term captures the loss when exactly one latent is active. The second captures near-cancellation when both latents coactivate and their signs oppose. Four fixed disjoint 45-degree rotations at (0,1), (4,5), (8,9), (12,13) change the dimension-averaged objective by D/4.

Using 256-node Gauss-Legendre quadrature of the uniform amplitudes, with the *original frozen smoothing widths*, gives:

| Margin / amplitudes | sigma | Planted basis mass | Four-pair remix mass | Remix advantage |
| --- | ---: | ---: | ---: | ---: |
| Strong [0.75,1.25] | 0.15 | 0.84375003 | 0.85569739 | **+0.01194736** |
| Weak [0.25,1.25] | 0.05 | 0.84375001 | 0.83378472 | **-0.00996528** |

Holding the amplitude interval fixed, the sign crossover for this explicit remix is sigma approximately 0.0688879 (strong) and 0.1330386 (weak). They are quadrature estimates, rather than interval-certified proofs involving exponentials.

**This establishes an objective mismatch for the strong-margin population.** A mixed orthogonal basis is a concrete feasible point with better expected smoothed sparsity than the true Q dictionary. The ideal exact-zero hyperplane objective favors the planted coordinate basis in this source because singleton supports give every genuinely mixed direction a strict cost and continuous amplitudes give exact accidental cancellations probability zero.

## Independent source replay

The original full cohort artifact is at https://github.com/MoralityLabAI/BPD/actions/runs/44140294730/artifacts/14353309264 (JSON SHA256 04bbda829276662e1fb37edbbd6cf5c49a2b8321cd55a712a2cd79c09b88ad20).

The audit script imports the unchanged PD-12 support generator and PD-13 fixed seed roster, reconstructs all 32 2048-board pools, and confirms SHA256 agreement with the original receipt for each pool and rotation. The same fixed four-pair remix then scores:

- Strong: 16/16 pools higher than Q, average +0.01167025; range [+0.01001544,+0.01275586].
- Weak: 0/16 pools higher than Q, average -0.01006207; range [-0.01096193,-0.00919001].

The learned searches have an additional optimization limitation: mean final priority2048 Gaussian mass is 0.828111 in strong and 0.674852 in weak, both below the planted baseline near 0.84375. The post-hoc basis inversion is therefore an identifiable *objective* issue, while failure to reach even the planted score diagnoses a separate *search* issue. These observations do not allocate a causal fraction of the registered failures to either one.

## Consequences for prospective work

A future registered experiment should check objective consistency before optimizer comparisons, using a fixed suite of off-axis transformations and fresh source seeds. Smoothing width, restart policy and sweep count can then be crossed without changing the PD-13 first-run decision. The strong source's explicit inversion suggests that a better pair ordering alone will not guarantee identification under the current Gaussian objective.

Reproduce with posthoc_zero_mass_audit.py from the repository root. The script performs only deterministic source replay and analytic quadrature. It does not train, access held-out families for optimization, or alter the source-locked scientific receipt.
