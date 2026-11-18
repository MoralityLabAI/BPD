# BPD-NC-08: allocate physical BPD rank differently at each recurrent pass

**Prospectively registered November 18, 2026.** [PROTOCOL.md](PROTOCOL.md) was committed FIRST at `27e09f396392f9ba0c8a169b84aea3712ed4c6cc`, before implementation or any training of the eight new engineering roots 800–807.

NC-07's actual four-pass trained-model experiment selected ranks 3,4,3,4,6,3,6,3 under a shared rank at every pass. Its original engineering compression gate failed: mean rank 4.0, and only 7/8 models had the required eight physically dose-matched local shams. The next controlled question is whether a smaller *sum of pass-specific retained ranks* can support terminal task behavior, especially when some passes need more of the shared physical write than others.

NC-08 freezes **four source-only total-rank budgets: 8, 10, 12, 14**. For each budget it enumerates passwise ranks from {2,3,4,5,6} and chooses the tuple minimizing the source-only squared omitted first-order model-score change. A separate independent 512-input stream then validates the actual four-pass closed-loop model's clean-to-retained predictive KL and physical write fraction. The smallest passing budget is selected, without external true-label access.

The [experiment](experiment.py) evaluates the selected projector through real BPDModel/TiedComponentLinear override-read interventions on every nonlinear recurrent pass. Its final labels are taken only from a further independent 1,024-example source; fixed 3-3-3-3 (total 12), fixed 4-4-4-4 (total 16), and predeclared DP budget-12 comparisons are scored on the same heldout stream. Each root receives eight independently seeded local pass-specific-rank projector shams from a frozen 2,048-proposal search, admitted on a clean-discovery physical retained-write RMS tolerance of 15%. Gauge-equivalent factor bases must reproduce the same physical retained operator and closed-loop logits.

The prospective eight-root **engineering-development** gate requires all roots fully admitted, mean retained total rank <=12.5 with at least five roots using B<=12, upper bootstrap true-task kept-minus-clean NLL <=.05 nats, selected-vs-fixed-rank4 heldout NLL cost upper <=.02 nats, and a positive source-dose-matched sham advantage >=.02 nats. The bootstrap resamples only whole independently trained model roots (2,000 replicates; fixed RNG 808008).

[THEORY.md](THEORY.md) sets out the multiple-choice knapsack, the omitted score-derivative surrogate, its nonlinear limitations, and the distinct final task criterion. [PasswiseBudget.lean](PasswiseBudget.lean) kernel-certifies the fixed rank-count arithmetic. [Source-only excluded-799 tests](test_experiment.py) and [synthetic registered-decision tests](test_aggregate.py) establish initial engineering admission before releasing development root 800.

The entire first eight-root run will retain either ENGINEERING_POSITIVE or ENGINEERING_NULL_OR_INELIGIBLE without revising the protocol. Future 59-root sham-calibration, EvalMath power review and 32-root scientific confirmation remain independently sealed and ungenerated.


## Frozen first-run developmental outcome

[Original source-locked Actions run #44265232267](https://github.com/MoralityLabAI/BPD/actions/runs/44265232267) completed every engineering/source/physical evidence gate and archived all eight independently trained model roots. **Original development verdict: ENGINEERING_NULL_OR_INELIGIBLE**, from one failed prospective task-cost bound versus flat rank four.

All **8/8** model roots were fully admitted, including eight physical same-tuple RMS-matched shams each from the 2048-proposal source-only pool. The selector chose total physical write rank budgets **8,8,10,10,12,12,12,14** (mean **10.75**), saving **32.8%** of the 16-slot flat-rank-four reference. Absolute task NLL cost upper root-bootstrap was **+0.036017** (cap +0.05) and the same-rank sham margin lower bootstrap was **+0.051208** (required >0), both PASS.

The registered **selected-minus-flat-rank-four task NLL 95% upper bound was +0.024799**, above the fixed **+0.020** cap, so the original overall development result is a FAIL. The model-only selector's source agreement was insufficiently protective on one root: root 800 selected budget 10 and incurred **+0.051944** true-label NLL relative to flat rank four. The unweighted source-DP **budget-12** projector outperformed the equal-total-dimension flat rank-three comparator by **0.013068 nats mean** across the same eight roots, a prespecified **descriptive** contrast without a new significance claim.

[Original full results](RESULTS_20261118.md), [frozen eight-root CSV](PER_ROOT_20261118.csv), [independent receipt auditor](test_published_results.py), and [original immutable nine-file artifact #14409767606](https://github.com/MoralityLabAI/BPD/actions/runs/44265232267/artifacts/14409767606) retain the first-run outcome. Original ZIP SHA256 is 533b150c6021093c1bf6d91d1152cce690eb3b09d357d641ec9de350aa9ca061. No new 59-root independent sham-calibration or 32-root confirmation sets were opened.
