# Dory Q4 benchmark SAE tower pilot

The [Oakhaven SAE pilot](../dory_story_sae/README.md) found that an extra
recurrent pass attenuated an identical first-pass residual edit. The benchmark
pilot extends that comparison to INTELLECT-3 math and logic, GSM8K, MATH-500,
and RLT rule revision. Inside the two-pass decode, the action-position SAE
edit shifts the answer margin more at the final endpoint than at the first-pass
endpoint on all five fixed-choice panels. Reconstruction error rose slightly
on the sampled original-format response trajectories.

## Matched loop endpoints

| Suite | Edit after pass 1, read after 1 | Identical edit, read after 2 | Edit after pass 2 | Matched sham after pass 2 |
| --- | ---: | ---: | ---: | ---: |
| INTELLECT-3 math | +2.07 | +0.04 | +1.02 | −0.03 |
| INTELLECT-3 logic | +2.70 | +0.02 | +3.37 | −0.02 |
| GSM8K | +3.61 | +0.04 | +3.60 | −0.11 |
| MATH-500 | −0.40 | +0.03 | +2.02 | +0.10 |
| RLT rule revision | +0.26 | −0.01 | +0.86 | +0.04 |

Each entry is the mean shift in foil-position minus correct-position logits
over both answer orders of three development roots. The correct answer and
foil switch positions between donor and recipient. A positive shift favors
the recipient foil. At the last endpoint the action-trained edit made the
foil the full-vocabulary top token in 18 of 21 clean-correct cases across the
five suites. The complete receipts also retain recovery and corruption counts.

The full-arm panel used an outcome-independent hash order within each suite.
All 240 first-pass writes were byte-identical across loop budgets. The first-pass
residuals matched on all 60 development prompts at each of the three sampled
sites. All 120 same-value writes and 60 repeated selected treatments reproduced
their full-head fingerprints. The pass-2 intervention uses the donor direction
from the pass-2 residuals and the same physical dose as the earlier edit.

## Calibration, controls, and scope

- **Cohorts.** Thirty-two roots fit the SAE towers, eight each from INTELLECT-3
  math, INTELLECT-3 logic, GSM8K train, and RLT. Six disjoint development roots
  per suite support the primary comparison. MATH-500 is readout-only. The
  INTELLECT-3 math and MATH-500 panels use integer answers. The logic panel uses
  Boolean expressions, bracket completion, and word sorting. RLT uses the
  verified final node after a public successor edit.
- **Tower.** Dory 1.0 4B Q4_K_M uses shared dictionaries across passes at
  zero-based residual sites 25/65, 45/85, and 55/95. Top-8 activation and rank-8
  projected donor edits use separate prompt-only and prompt-plus-action fits.
  Matched controls include PCA, Haar, direct donor, donor-orthogonal sham,
  same-label cross-root donor, and same-value writes. The supplementary full-arm
  roster records a zero-shot Oakhaven dictionary control. Nontrivial edits use
  a chord length of 5% of the recipient radius and preserve that radius.
- **Primary gate.** The registered endpoint is the first-pass edit read after
  two passes, across 30 independent source problems. The bounded action-SAE
  minus sham mean was +0.00325, against a 0.05 mechanistic floor. The largest
  observed absolute sham was 0.01726. The 95/95 sham tail remains uncalibrated
  and the readout unsealed. EvalMath returned `REDESIGN_OR_PILOT_ONLY`.
- **Original prompts.** Twenty separate greedy activation replays matched the
  generated prefixes and next tokens, two roots per suite and loop budget.
  For the ten two-pass traces, relative reconstruction error at the sampled
  second-pass endpoint averaged 0.921 with the prompt-only tower and 0.941
  with the action-position tower. These were early generated prefixes of up to
  16 tokens. GSM8K solved one of two sampled problems at each loop budget. All
  sampled open INTELLECT-3 math and MATH-500 responses hit the output cap.

The intervention measures causal routing among supplied answer candidates.
The original-format activation study measures observational transfer to early
generation. Unrestricted benchmark-solving accuracy remains unevaluated at
scale. The checkpoint, source data, verifier, and reader hashes
appear in the [registration](receipts/registration.json) and
[release manifest](receipts/manifest.json).

## Offline audit

Run from the BPD repository root with Python 3.10 or later.

```sh
python mechinterp/dark_fiber_gyms/dory_benchmark_sae/verify_receipts.py
```

The verifier checks 960 intervention receipts, 120 clean selector receipts,
20 original-response grades and linked activation replays, and the root-level
power input. It keeps all reported arms and the complete
[DataGoblin decision log](receipts/dory_bench_sae_decisions.jsonl).
The release manifest also supports exact source-bundle binding via
`--source-bundle dory_bench_sae_bundle.zip`. The local experiment archive
retains the native orchestration source. Full prompts, activations, and model
weights stay local.
