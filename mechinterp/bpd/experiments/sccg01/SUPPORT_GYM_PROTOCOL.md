# SCCG-04: constructed support-selection pilot
Status: registered engineering/development experiment. No native weights
or new trained neural backbone, no reserved cohort access, no paid compute.

## Exact orthogonal-source family
State x=3 e_side, normalization mass m=7, rho0=4, physical edit dose<=3.
Eight orthonormal editable coordinate directions, choose up to 3
(supports of size 3 include smaller supports). One or two protected
effective raw query rows C; their baseline scores are c/4. Target
baseline raw numerator a0. Let t=rho0/rho(x+delta)>0, z=t delta,
r=4t. Then ||z||²+r²=16, r>=16/5 (upper dose 3), protected constraints
4 C_S z+c r=4c. Target scaled score gain is a_S^T z+a0 r/4-a0.
This converts the finite nonlinear task exactly into an affine sphere
with a single linear cap and target inequality. The upper bound r<=4
follows from the sphere. The clean augmented point is (0,0,0,4).
The min-norm affine center is its rowspace projection; residual sphere
lies in a nullspace of dimension>=2. SCCG-03's two-halfspace theorem
decides each target threshold by rational arithmetic without root
isolation. Float cap optimizer uses a separate batched pseudoinverse.
This restricted family makes a strong analytical baseline available;
the learning study does not assert that general nonlinear BPD is easy.

## Frozen pilot
Train seeds 11000..11095; development seeds 21000..21047, disjoint.
Mode seed modulo 4, one/two protected rows and random/planted dependent
three-column groups. Generator is in support_gym.py. Thresholds 2,3,4
in units rho0 times target score gain. No tuning based on development
labels. All 56 size-three supports labeled exactly. A shared 3-pass
tied recurrent ranker is trained on the 96 training roots, three
initializations 41,43,47, 160 Adam updates, batches of 8 complete roots.
No development-based early stopping or arm selection. This is a small
recurrent proposal ranker, not the native TRM architecture.
Each arm scores all 56 supports, then verifies at most 4 in order,
stopping on first exact success. Baselines: random, target magnitude,
zero-collateral tangent, analytic finite cap, exhaustive labels.
Primary descriptive endpoints: top-1 and hit by 4, total actually
replayed exact checks, solvable-root ceiling. Report all seeds.
Training-label cost and all-candidate scoring work are explicit;
query savings versus exhaustive labels do not imply wall-time,
FLOP, general neural, or training-amortization gains.
Exact-vs-float mismatch beyond 1e-8 is a hard engineering error;
near-boundary rows counted separately. No inferential gate, no
promotion. Same development roots across model seeds.
Store JSON including root IDs and per-root successes/checks.
The exact arithmetic is trusted Python Fraction; learned proposals
never certify success. No claim of novelty for affine spherical caps.
