# NCG-02: boundary charges with native BPD steering

Eight fresh development roots produced a mean **-1.37 percentage-point** change
in ordinary long-delay exact accuracy from causal steering during training.
The descriptive root-bootstrap 95% interval is **[-3.03, +0.29] percentage points**.
EvalMath returns `REDESIGN_OR_PILOT_ONLY`.

The executable uses the native RP03 BPD model at commit
[`402ef7764e11bc7736873d66357b8dc24b996e94`](https://github.com/MoralityLabAI/BPD/tree/402ef7764e11bc7736873d66357b8dc24b996e94/mechinterp/bpd/experiments/training_steering_stability).
The pinned source snapshot is included with its license. The model has 1,558
trainable parameters. Each root has its own input encoding and training stream.

## Task

For each of two disjoint binary triangles, the inputs provide a cycle bit c and vertex
syndromes b1 and b2. The target edge bits are

`(c, c XOR b1, c XOR b1 XOR b2)`.

Across both triangles there are 64 semantic assignments. The model reads two
input tokens and predicts all six edges after a delay. Training sees two NOOP
tokens. The main development score uses eight. An exact specialized solver
scores 100% on this finite task.

A donor changes one cycle bit, so the desired edit flips one triangle's edges
and preserves the other triangle. Training-only pairs select a group of four
native BPD components for each triangle. The corresponding frozen projector
defines a hidden-state patch of L2 length 0.75. A signed permutation supplies
the matched rotated control. Every arm sees the same ordinary receiver/donor
labels and executes the same forward and gradient calls. Auxiliary gradients
have norm 0.25 times the task gradient before Adam and clipping.

The full recipe is in [PROTOCOL.md](PROTOCOL.md).

Every branch restarts Adam's moment estimates at zero from the common warmup
weights. The V/U pair has 16 components for a 16-wide hidden state and permits
full rank. This parameterization exposes intervention directions. Compression
of model parameters would require a separate experiment.

## Development results

| Arm | Exact answer, delay 2 | Exact answer, delay 8 | Constraints satisfied, delay 8 | Edge accuracy, delay 8 |
|---|---:|---:|---:|---:|
| A: task only | 96.729% | 36.792% | 38.062% | 80.151% |
| B: causal steering | 96.289% | 35.425% | 37.036% | 80.160% |
| C: rotated steering | 96.753% | 35.156% | 36.597% | 79.537% |

The B-C difference is +0.269 percentage points, with descriptive bootstrap
95% interval [-2.76, +3.76]. The two planned contrasts yield Holm-adjusted
two-sided sign-flip p values of 0.375 for B-A and 0.922 for B-C. These calculations
treat each trained root as one unit and assume signed exchangeability for
the permutation reference.

Across 4,096 paired long-delay cases, B recovers 142 complete answers that A
misses and corrupts 198 answers that A gets right. At the individual edge level,
868 bits are recovered and 866 are corrupted. The primary score requires all
six bits to agree, which captures the joint state that the task asks for.

On the separate intervention diagnostic, the causal-trained model gets 36.898%
of edited bits right, compared with 31.999% in A. Desired full-state recovery
under the patch is 3.271% in B and 2.100% in A. Ordinary long-delay performance
remains the primary outcome.

Two of eight roots choose the same component group for both triangles. The
candidate group loss and chosen indices are preserved per root. This overlap
is a source-selection diagnostic; the pilot does not establish semantic
selectivity of the selected subspaces.

## Evidence and limits

This is a development study on a fixed pair of triangles. Training and scoring
share the same finite semantic assignment domain while using independent
sensor noise. Claims about unseen graph structures require another experiment.
There is no generated confirmation population.

The network must retain the syndromes together with the cycle bits because it
receives them only in the initial tokens. Its semantic input spans 64 states,
which requires six classical bits for lossless discrimination. NCG-01's shorter
cycle coordinates condition on a known graph and known charges. A future
decoder can retain those charges externally and ask the model to store the
cycle bits. The present pilot measures delayed charge memory and causal edits
on disjoint triangles. NCG-01 supplies the exact shared-boundary gluing checks.

The prespecified scientific floor is +5 percentage points. Eight root samples
leave the 95th-percentile sham bound uncalibrated at 95% confidence. EvalMath
also records the absence of a sealed confirmation population. Its hypothetical
power estimate cannot authorize a scientific confirmation label.

The two engineering roots and their negative effects remain under
`results/engineering`. The eight development roots use the same source recipe
without a hyperparameter search. Each phase saves its source hashes before
running. The later package initializer change is documented in [PACKAGING.md](PACKAGING.md).

## Verification

Five tests passed. They cover the exact solver, native dense VU parity,
physical patch norms, the normalized patch's finite-difference derivative,
and the training update. A matches plain supervised training exactly; adding
the causal gradient changes the parameters.

An independent NumPy implementation of the GRU equations replayed **184,320
prediction bits with zero disagreements** across engineering and development.
It also checked 184,320 label bits, 30 paired initialization hashes and 800
matched training batches. The 14 historical source-hash checks passed.
This replay covers inference from saved checkpoints. Optimizer trajectories
are represented by the recorded source, batches, update diagnostics and
the dedicated update test.

Patch norms lay between 0.749999821 and 0.750000179. The measured auxiliary
gradient ratios in B stayed between 0.249999970 and 0.250000030. The recorded
runtime was 5.83 seconds for the two engineering roots and 21.08 seconds for
the eight development roots on the available CPU environment.

## Run

### Post hoc geometry check

An independent review compared the warmup projectors with the corresponding
factor spans after training. Mean `Tr(P_warmup P_final)/4` was 0.997964 in A,
0.997716 in B and 0.997700 in C. The smallest squared principal cosine across
all final selected groups was 0.989052. These measurements place the remaining
design work on behavioral selectivity and memory of the joint state.

The companion source-pair analysis reuses the pairs that selected the component
groups. Its own-slot loss advantage at warmup follows from that selection rule.
`diagnostics/posthoc_review.json` labels this exposure explicitly. The replay
script verifies historical source hashes and records every checkpoint hash:

```bash
python diagnostics/posthoc_review.py --root . --output /tmp/ncg02-posthoc-review.json
```

### Main runner

Requirements: Python 3.12, PyTorch 2.5.1+cpu and NumPy 2.3.5 were used.

```bash
python -m unittest -v test_pilot
python pilot.py --phase engineering --out results/new_engineering
python vendor/evalmath_preflight.py --pilot results/new_engineering/preflight_input.json --strict
python pilot.py --phase development --out results/new_development
python audit.py
```

The strict preflight returns status 2 for the recorded pilot-only decision.
`audit.py` audits the committed `results/engineering` and `results/development`
directories. A replay into alternate directories preserves the original files;
comparison of a new trajectory requires pointing an audit at those directories.

## Files

- `pilot.py`: native model runner and the fixed task recipe.
- `test_pilot.py`: mathematical, gradient and update checks.
- `audit.py`: independent inference replay and descriptive root analysis.
- `PROVENANCE.json`: source commits, file hashes and runtime versions.
- `results/`: raw predictions, all terminal checkpoints, training traces and preflight receipts.
- `ARTIFACT_HASHES.json`: checksums for retained result files.

The next task can use the exact cycle-coordinate encoder from NCG-01 to
separate memory of the two cycle bits from the known syndrome decoder.
That would make lawful reconstruction a property of the decoder, while a
trained recurrence remains responsible for retaining the cycle information.
