# RLT-EXIT-04: support-aware stopping on fresh 40/52-blank Sudoku roots

**Date:** 2026-11-21. **Status:** development evidence, real 7M pretrained
Attention-TRM execution with frozen source-only learned stop model.
Original [GitHub Actions run #44914654368](https://github.com/MoralityLabAI/BPD/actions/runs/44914654368)
and independent cross-replayed [run #44915182662](https://github.com/MoralityLabAI/BPD/actions/runs/44915182662)
both succeeded. Full run including the independent offline audit is retained
in artifact bpd-rlt-exit-support-guard-new-roots #14712000093.

## Registration and source custody

The original 50-blank controller was trained and calibrated in merged BPD
PR #140, using TRAIN/CAL/DEV source roots distinct from this experiment.
Controller model and threshold were loaded from frozen report SHA-256
dda473da08434484f4d67e8f0e6fa77b4296d9fa1489a810f529c699dc94a7e1.
The pretrained 7M Attention-TRM checkpoint SHA-256 remains
932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3.

Merged BPD PR #141 recorded an incorrect learned early stop at 52 blanks.
After seeing that failure, this new support guard was designed using a
public input property. It allows the originally frozen logits-only learned
stop threshold when Sudoku has **at most 50 blanks**; otherwise it runs the
fixed two-step model. The rule is posthoc with respect to the original
difficulty-failure finding and was frozen before collecting the following
fresh development roots.

- Fresh easy cohort: 24 uniquely solvable 40-blank puzzles, seeds
  20261411 + 101*i.
- Fresh hard cohort: 24 uniquely solvable 52-blank puzzles, seeds
  20261412 + 101*i.
- Input question, both independently checked candidate boards, six
  logits-derived features, full checkpoint/threshold/source hashes,
  per-policy decision, measured native CPU time, and physical visit
  count are saved for every root.
- No pretrained weights, learned logistic coefficients, or threshold
  were updated. All prior TRAIN/CAL/DEV roots were excluded by digest;
  the previous transfer cohort's registered seeds were excluded.
  No protected BPD confirmation/sealed data were opened.

## Final measured and independently audited outcomes

The first run and the cross-audited replication produced identical
counts of correct boards, stops and incorrect exits. CPU times varied
slightly with runner scheduling.

| Cohort/policy | Correct boards | Recurrent outer steps | Native time (audit run) | Wrong early exits |
| --- | ---: | ---: | ---: | ---: |
| 40-blank, fixed 2 | 24/24 | 48 | 11.886519 s | 0 |
| 40-blank, exact verifier exit | 24/24 | 24 | 5.930708 s | 0 |
| 40-blank, original learned exit | 24/24 | 25 | 6.314108 s | 0 |
| 40-blank, **support-guarded learned exit** | **24/24** | **25** | **6.151625 s** | **0** |
| 40-blank, rate-matched random | 24/24 | 39 | 9.807077 s | 0 |
| 52-blank, fixed 2 | 22/24 | 48 | 11.943368 s | 0 |
| 52-blank, exact verifier exit | 22/24 | 41 | 10.159272 s | 0 |
| 52-blank, original learned exit | 22/24 | 42 | 10.495693 s | 0 |
| 52-blank, **support-guarded learned exit** | **22/24** | **48** | **11.774915 s** | **0** |
| 52-blank, rate-matched random | 17/24 | 39 | 9.637505 s | 7 |

For 40 blanks, the guard conserved 23/48 recurrent outer steps (47.92%)
while retaining 24/24 exact boards. Measured native CPU step time was
48.25% below fixed-two under this runner sample. One first-step
valid board fell just below the original learned classifier threshold,
so the gate executed an unnecessary second step.

For 52 blanks, the guard necessarily uses the full two-step budget.
It spent six more recurrent steps than the unguarded learned policy,
which happened to make **no incorrect stops on this new 24-root cohort**.
Consequently this new sample demonstrates the guard's adherence to
its frozen support rule and cost, without observing an accuracy gain
from the guard. A distinct older 52-blank cohort did contain one
incorrect learned stop. Preserve both observations; their difference
is finite-sample heterogeneity.

## Verification and audit scope

[Independent audit code](audit_exit_support_guard.py) replays the entire
written Sudoku input and candidate board, root lineage, correctness
at each pass, known public difficulty guard, frozen threshold, and
six-feature original trained classifier score. It also recomputes
the deterministic random-control decisions and every aggregate
verifier/cost count, rejecting changed scores and candidates even if
someone has resealed the input file digests.

The independent audit passed 48/48 roots in the reproducible GitHub
Actions run. Saved report SHA-256:
b5f8d716db1b9b1b069513793fa80afd5d1a787718c50a30ed32a80821444364.
It does **not** reexecute the 7M pretrained neural weights from saved
activations. The original hosted CPU run executed genuine native TRM
recurrence; the replay verifies its archived outputs and policy decisions.

The scientific claim remains bounded to this posthoc-designed,
development-only support rule. Zero observed false stops across
48 new roots cannot certify a future-zero risk. A prospective
source-root sample with a prespecified acceptable false-stop margin,
paired compute/accuracy inference and cross-task verifiers is required
to claim controlled generalization.

The TRM's recurrent H/L states are distinct from Inkling's
Attention Residual cross-loop readers. This experiment does not claim
an Inkling feature or Goodfire coordinate-2027 mechanism.
