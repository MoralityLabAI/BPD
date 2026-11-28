# ER-11/12: contamination, calibration and joint-rate sufficient state

Reproduce from this directory with NumPy 2.3.5:
```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python robust.py > robust.json
OPENBLAS_NUM_THREADS=1 python hierarchical.py > hierarchical.json
```

64 new roots 296..359, same 4060 candidate family and unequal known weights as ER-08. 64 probes, budget 16. Each observation is independently replaced with probability e by uniform noise on the 31-symbol measurement alphabet (-6..24 in offset coordinates). Rates 0,.1,.3 reuse roots, so regimes are not independent samples. Robust adaptive and random assume e=.1. Clean and matched likelihood arms replay robust-adaptive actions and observations; matched is an oracle inference comparator, not an independently optimized scheduler. Predictive entropy maximization equals MI here because the conditional entropy is constant across graph candidates and actions. No out-of-alphabet corruption tested.

| true e | robust adaptive MAP 4/8/16 | robust random MAP 4/8/16 | clean replay MAP 4/8/16 | matched replay MAP 4/8/16 |
|---|---|---|---|---|
|0|32/64/64|13/56/63|32/64/64|32/64/64|
|.1|23/58/64|12/53/63|22/32/16|23/58/64|
|.3|11/36/64|6/35/60|5/5/0|11/39/64|

At probe 16 clean likelihood empty counts 0/48/64 by regime. Every positive-contamination posterior preserves true support. At e=.3, probe 8, robust adaptive 95% credible-set coverage 53/64; matched 62/64; robust random 59/64. Misspecified robust adaptive mean truth NLL 3.108 bits vs 2.761 for matched replay and 2.587 for robust random. Thus adaptive did not beat random on all interim metrics, and a positive likelihood is not calibrated uncertainty. Credible sets include ties. Clean NLL on surviving finite cases is not a valid unconditional score; zero truth mass has infinite NLL.

## ER-12: exact joint uncertainty state

Uniform prior over six rate grid values .01,.05,.1,.2,.3,.5 and graph support. Replay ER-11 robust-adaptive histories, not a newly scheduled arm. Conditional likelihood has three levels L0=(1-e)*.7+e/31, L1=(1-e)*.15+e/31, L2=e/31. Given global t, two uint8 counts (exact matches and distance-one matches) per graph determine joint likelihood L0^c0 L1^c1 L2^(t-c0-c1). With a known nonuniform prior multiply by that prior. No general neural distribution assumption. Uint8 budget <=255; decoding uses floating-point weights. Persistent payload 8120 bytes vs 194880 bytes float64 joint table, excluding grid, lookup, priors, global step and temporary allocations. 3072 replay checks passed, max absolute difference 2.220446049250313e-15. No runtime/RSS win claimed.

| true e | hierarchical MAP 4/8/16 | 95% coverage 4/8/16 | mean inferred e at 16 |
|---|---|---|---|
|0|32/64/64|64/64/64|.0400|
|.1|23/59/64|63/63/64|.1218|
|.3|11/38/64|60/62/64|.3062|

At e=.3, probe 8 NLL 2.777 bits; the hierarchical filter restores empirical coverage relative to the fixed rate, but exact true rate is not an inference guarantee. The zero-contamination regime is outside the rate grid and the mean rate does not vanish. Exploratory root-level results, no preregistration, shared graph family, no real neural intervention claim.
