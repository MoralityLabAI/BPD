# BPD RLT-EXIT-01: online verifier-gated native pretrained TRM recurrence

**First original result:** development-only, real SHA-pinned pretrained recurrent attention model,
November 21, 2026. [GitHub Actions #44903527858](https://github.com/MoralityLabAI/BPD/actions/runs/44903527858),
[raw artifact #14706404454](https://github.com/MoralityLabAI/BPD/actions/runs/44903527858/artifacts/14706404454).

- Model: gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att, pretrained
  checkpoint SHA-256 932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3.
- Source: native_trm_exit.py SHA-256
  2fd754ce38506787fdf724952ae8c64c1eedb78412b6c6746bd2f5fa9f0a7634.
- Raw original JSON SHA-256
  c621861c5fc0b950d23cd30ae4de5ea677d07b893e53ecbfbd0b73b84259f948.
  All per-root candidate grids, runtime records, source IDs and model settings
  remain in the original Actions artifact.
- Fresh **development** cohort: 24 separate, uniquely solvable 9x9 Sudoku
  questions with 50 blanks, seed 20261123 + 101*i for each root.
  Prior 40-blank and 50-blank cohorts used seeds 20261121 and 20261122.
- Fixed policy: run two real outer recurrent steps using continued z_H / z_L
  states, then grade. Adaptive policy: run one outer step, run an exact Sudoku
  clue/row/column/box verifier, and stop immediately when complete.
  Otherwise run the second outer step and check again.
- Match controls: identical checkpoint, same 24 tasks, same generated states
  from each initialization and first-step candidate string, alternating
  policy order across roots to moderate execution-order timing effects.
  The verifier only sees the candidate and original question, never the
  planted solution. Independent receipt review recomputed Sudoku legality
  and per-root depth policy behavior.

## Original measurements

| Metric | Fixed 2 | Online verifier exit |
| --- | ---: | ---: |
| Cases solved completely | 23/24 | 23/24 |
| First outer step solved | 15/24 | 15/24 |
| Additional roots solved on second step | 8 | 8 |
| Subsequent regressions on earlier correct grids | 0 | 0 (stopped) |
| Total outer steps | 48 | 33 |
| Mean outer steps per task | 2.000 | 1.375 |
| Native model forward seconds, total | 10.133934 | 6.974778 |
| End-to-end policy seconds, total | 10.147272 | 6.983677 |
| Exact verifier seconds, total | 0.003163 | 0.002071 |

Result: **31.25% fewer outer recurrent steps**, 31.174% lower measured
native forward seconds, 31.177% lower policy-level elapsed seconds.
These are sums over sequential model calls on a hosted GitHub CPU runner,
not amortized parallel serving, GPU inference throughput or watt-hours.
Two outer steps each involve three H-cycles and seven L_level calls per
H-cycle (six L followed by one H), for 42 versus 33/48 × 42 total
L_level block applications under this model and cohort.

Exactly 15 roots halted after the first verifier-valid Sudoku grid.
Eight more were successfully completed after two; one remained unsolved
at the permitted depth. The fixed model also solved 23, with zero
observed late-depth regressions among its first-step correct roots.

## Claim boundary

The trained recurrent model is **TRM**, not the native 26B Inkling 2 Mini
Attention Residual architecture. This experiment demonstrates that
task-verifier feedback can save real inference compute with preserved
Sudoku accuracy on a small, exposed development cohort. The model is
not learning an exit gate here. Its exact Sudoku checker is task-specific,
and a general language-model replacement would need externally
trustworthy verification or a calibrated learned stop controller.

One run on 24 roots is not a prospectively powered scientific confirmation.
No additional protected BPD NC09/AMM cohorts were accessed; all historical
gates and verdicts remain unchanged. No model weights were committed.

### Next scientific experiment

Fit a tiny exit-policy predictor on disjoint development roots using
the first-step logit margins, state-change norms, and future verified
outcomes. Compare learned routing to exact-verifier stop and fixed
depth under the same compute budget, with source-root bootstrap and
sham-calibrated minimum effect in EvalMath. Preserve the exact verifier
as an upper-quality source of labels, avoid exposing held-out outcomes
to controller selection, and test the stopping logic across other
reasoning task families. Native Inkling cross-loop readers still require
a separate instrumented checkpoint and memory-suitable runner.
