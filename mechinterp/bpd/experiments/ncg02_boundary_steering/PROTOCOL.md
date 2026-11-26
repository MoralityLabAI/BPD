# NCG-02: native BPD training on boundary charges

## Scope

This development pilot asks whether causal steering during training improves
the ordinary answers of BPD's recurrent model on a finite charge task.
The graph is fixed. Each independent root contains a fresh model initialization,
a new input encoding and its own training noise stream. Test cases within a
root are repeated observations. The pilot measures transfer across sensor noise
and a longer delay on the same 64 semantic assignments.

We use two disjoint triangles over F_2. Each triangle receives a cycle bit c
and two vertex syndromes b1 and b2. The unique target edges are
(c, c XOR b1, c XOR b1 XOR b2). Thus a solver can recover the exact answer
from the inputs. This solver defines the structural ceiling.

These are classical bit assignments that index a finite physical basis in the
toy system. The experiment makes no claim about quantum amplitudes or gravity.

## Native source and architecture

The unmodified BPD files in `vendor` come from commit
`402ef7764e11bc7736873d66357b8dc24b996e94`. `PROVENANCE.json`
records their hashes. The runner calls RP03 `ComponentStateMemory`, `projectors`
and `state_patch`, along with the repository's `TiedComponentLinear`.

There are 16 hidden coordinates and 16 components in the trainable V/U pair.
All components execute at every token, with the residual delta gate at zero.
The factor pair permits full hidden rank. It exposes component directions for
interventions; a parameter-compression study would need a separate rank sweep.
The model receives two tokens containing the triangle inputs followed by
NOOP tokens. Six output logits predict the edge bits. Every arm has the same
architecture, training batches, update count and forward/backward calls.

## Causal edit and selection

Donor and receiver inputs differ in one cycle bit. The desired edit flips all
edges in that triangle and preserves the other triangle. Each arm receives both
receiver and donor as ordinary supervised examples.

After 400 common warmup updates, each triangle selects one of four disjoint
groups of four components. Ninety-six independent training-stream pairs per
triangle supply the selection data. Candidate donor-target loss determines the
choice; lower index resolves a tie. The selected V-column projector is frozen.

The patch occurs after both input tokens, before the first NOOP. RP03 projects
the detached donor-minus-receiver hidden-state difference through the frozen
projector, then scales its L2 length to 0.75. The receiver remains differentiable.
A fixed root-seeded signed permutation rotates the sham patch and preserves
that length. Norms, state radius and optimizer displacement are logged.

## Training comparison

| Arm | Update |
|---|---|
| A_task | Task gradient |
| B_causal | Task gradient plus the causal auxiliary gradient |
| C_rotated | Task gradient plus the rotated auxiliary gradient |

The auxiliary gradient has L2 norm 0.25 times the task gradient before clipping
and Adam. A computes both auxiliary gradients as well. Every branch performs
80 updates, each containing 32 paired examples, at learning rate 0.005.
Each branch starts from the common warmup weights with a fresh Adam optimizer;
its first and second moments start at zero.
Warmup and branch training use two NOOP tokens. Gradient clipping has radius 1.

## Endpoints and controls

The primary endpoint is ordinary exact accuracy on all six edges after eight
NOOP tokens, with steering disabled. The primary contrast is B minus A. B minus
C shares the same two-comparison family. A five-percentage-point gain is the
scientific reference floor. Constraint satisfaction and bit accuracy are
reported separately. An output can satisfy the syndromes while choosing the
wrong cycle bits.

Each arm receives 512 fresh ordinary cases at delay 2 and another 512 at delay 8.
An additional 256 donor pairs measure the causal edit and corruption of edges in
the untouched triangle. All raw ordinary predictions and labels are retained.
Recovery and corruption compare B against A on identical ordinary cases.

Engineering roots are 962600 and 962601. Development roots are 962610 through
962617. Each phase writes its source hashes and recipe before the first run.
Output directories must be new. The engineering result is retained unchanged.
No checkpoint selection, dose search or hyperparameter sweep is allowed in this
recipe. A larger prospective study needs a separate protocol.

The official EvalMath preflight runs after the engineering roots, then again
on the development roots. Its input uses root-level B-A effects and C-A sham
effects. Eight roots cannot calibrate the requested 95th-percentile population
sham bound with 95% confidence. The output remains `DEVELOPMENT_ONLY` throughout.
There is no generated or evaluated confirmation population.

## Verification

The tests enumerate every semantic assignment for the structural solver,
compare native component execution with the dense VU expression, check patch
norm preservation, finite-difference the normalized patch gradient and verify
that the causal gradient changes the optimizer update. A's update must equal
plain supervised training. The source snapshot preserves the MIT license.
