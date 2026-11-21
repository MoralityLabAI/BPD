# INK-BPD-03: gate intervention and recurrent recovery in a trained synthetic native model

**21 November 2026. Development experiment with post-result intermediate-state diagnosis.** This study extends the matched physical-writer pilot [INK-BPD-02](RESULTS_INKBPD02_DEV_20261121.md) with a separately registered, **direct post-softmax gate inversion** control. It uses new task roots and the same 946-parameter, three-loop source-trained PyTorch checkpoint. It does not run pretrained Inkling Mini.

## Frozen contrast and evidence

The [INK-BPD-03 protocol](INK_BPD03_GATE_PROTOCOL.md) was committed as `404c304` before any gate-inversion outcomes on the new seeds. Clean, single-pass and all-pass conditions share a freshly sampled 12-root x 32-case development cohort (root IDs 303701..303712). All three execute native recurrent continuations through the final classification head.

- Original scored gate-inversion experiment: [Actions run 44912164009](https://github.com/MoralityLabAI/BPD/actions/runs/44912164009), original archived [artifact 14709694862](https://github.com/MoralityLabAI/BPD/actions/runs/44912164009/artifacts/14709694862). It reported 100%, 100%, 48.6979% terminal accuracies. The attached JSON SHA-256 was `d7914e02ae7b1d5fbe1b521662e63b04379c270e352bdd5c63050da38a5bdf9f`.
- Post-result diagnostic code adds three intermediate readouts and per-case repair accounting on **the same already exposed development roots**. [Actions run 44912329601](https://github.com/MoralityLabAI/BPD/actions/runs/44912329601), [artifact 14710785220](https://github.com/MoralityLabAI/BPD/actions/runs/44912329601/artifacts/14710785220). ZIP SHA-256 `d276e3ebb0cfa702457b24055bbcdf1717ef03257f115f76080794dc928107a4`. The new `inkbpd03_gate_dev.json` SHA-256 is `c01b4291c61d7c6d32ab636cfed6e47e543fa48318edb201e34e872b3668a557`. The retained `inkbpd02_native_development.json` remained byte-for-byte identical to its original receipt (SHA-256 `e71850d4620341d1bfc6cfddbff6d275c1ec9c3bcd0c536ea34aa80591ff7d87`).
- All native tensor-hook tests, source-only attribution tests and deterministic reruns passed on hosted CPU. The all-pass gate treatment has **no matched writer L2 dose** and belongs to a separate causal contrast.

## Native per-pass task outcomes

| Direct gate condition | After pass 0 | After pass 1 | Final after pass 2 |
|---|---:|---:|---:|
| Clean | 384/384 | 384/384 | 384/384 |
| Invert only at pass 1 | 384/384 | 292/384 | 384/384 |
| Invert at all 3 passes | 187/384 | 187/384 | 187/384 |

With the one-pass intervention, **92 cases** were originally correct at pass 1 under the clean policy, became incorrect under the inverted gate, and were restored by the next native recurrent pass. All 92 were recovered at terminal readout. The all-pass inversion left **197/384** final answers incorrect (51.3021 percentage points lower accuracy than clean) on these synthetic inputs.

Mean final-logit L2 displacement from the clean model was **0.943908** after a one-pass inversion and **6.297662** after all-pass inversion. Both are actual native final logits following full model continuation.

| Development root | All-pass correct / 32 | Gate-induced intermediate errors, repaired by final |
|---|---:|---:|
| 303701 | 20 | 7 |
| 303702 | 18 | 7 |
| 303703 | 16 | 9 |
| 303704 | 14 | 9 |
| 303705 | 17 | 8 |
| 303706 | 16 | 6 |
| 303707 | 17 | 3 |
| 303708 | 13 | 9 |
| 303709 | 13 | 6 |
| 303710 | 11 | 11 |
| 303711 | 16 | 9 |
| 303712 | 16 | 8 |
| **Total** | **187/384** | **92/92** |

## BPD implication and inference limits

The source-only BPD taxonomy predicts that **transient importance** can be invisible at the terminal output when later passes rebuild an intermediate computation. Here, the final decision under a one-pass gate inversion is unchanged on all 384 cases even though 92 intermediate predictions were wrong and subsequently corrected. Applying the same gate inversion on every recurrent pass removes the recovery opportunity and produces a large terminal task deficit. This is a concrete positive-control witness **for the trained synthetic model and intervention**.

The gate inversion is direct and deliberately strong. It is not equivalent to changing the large-coordinate candidate observed by Goodfire in Inkling, and it is not physically dose-matched against the 0.16 writer patch from INK-BPD-02. The result establishes neither natural mediation nor generalization to other checkpoints. A single trained checkpoint is shared across the twelve independently drawn task roots, and both studies used development data.

For a pretrained BPD/Inkling experiment, register physical hooks at the actual last-block writer and next-pass reader; test clean competence, route-gate movement and delivered-dose sham admission before interpreting a model factor. Compare single-pass versus all-pass ablation alongside correct-answer corruption/recovery and full native continuation. Under EvalMath, retain the independent checkpoint and task-root populations separately, and calibrate the matched-sham null before opening protected confirmation outcomes.

## Reproduce

```sh
python -m unittest discover -s mechinterp/bpd/experiments/inkling_routing -p 'test_native_model.py' -v
python mechinterp/bpd/experiments/inkling_routing/run_gate_control.py --out /tmp/inkbpd03.json
python mechinterp/bpd/experiments/inkling_routing/run_gate_control.py --out /tmp/inkbpd03.json --check
```

The intermediate error-and-repair counts were added as a **post-result** readout. They strengthen the developmental instrument diagnosis and are not a newly registered confirmatory endpoint.
