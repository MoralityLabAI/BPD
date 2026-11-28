# ER-40: causal oracle-supervised cache admission

## Design

Reuse ER-38 access traces (roots 78,79 at widths128/256) for training and ER-39 traces (80,81 at both widths) for development. Generate fresh trained worlds 82,83 at each width with ER-25/32 generator and reconstruct twelve cache traces using er39.py. Each trace uses twelve probes and one of three noise regimes (.1,1,3 times RMS); three episodes per model. Root identity is (width,seed); training, development and test root sets are disjoint. Each split has four trained models. All fresh traces pass independent exact eager MAP and energy comparison, spanning 491520 pairs; 4/12 true injected faults recovered. 223422 exact candidate-probe evaluations over the test traces; 130258 unique-in-batch requests, 8480 distinct keys summed across episodes. Fixed search traces; no learned intervention selection or runtime kernel change.

Oracle labels: each request begins a retention interval up to its next occurrence. Label 1 iff that interval survives the ER-39 optimal optional-admission greedy schedule and ends in a hit. For every trace assert sum(labels)=request count minus offline oracle misses. This supervises one optimal schedule, which need not be unique. Causal request features use only current batch and pre-batch history: width, normalized component index, gap-ranked probe position, log elapsed batches, log batch size, log key count, log time since prior access, previous reuse gap, exponential moving average of gaps, component and probe counts, key rate, probe share, within-batch first-use position, cold-key flag and log capacity/width. Features are generated chronologically without future trace length, next access, hidden pair, oracle labels or test performance. Probe ranking is available before search. Training standardization and logistic regression (C=1, max_iter=200) use training only. Model has 16 inputs, 17 fitted coefficients including intercept (plus scaling statistics and causal history arrays). One shuffled-label model fitted with seed261144, same architecture and replay. The primary capacity is512. The 128-entry analysis is a secondary sensitivity on the same already evaluated test traces; refit labels/model and choose decay using training/development only, no new independent evaluation roots.

Replay: evaluate hits against batch-start cache. For every requested entry refresh base score s_g (model logit, shuffled logit, log prior frequency, negative log previous gap with cold penalty, or zero). Rank the union of resident and just-requested entries by s_g - alpha*log(1+age), ties by latest batch then lowest key index. Keep top K, optionally bypass a newly computed entry. Age is number of batches since last request. Tune alpha independently for each arm on the twelve development traces using grid {0,.25,.5,1,2,4}; select minimum total misses, ties lowest alpha. Fixed global FIFO and LRU use ER-39 snapshot and insertion semantics. Recency-only priority replay is a separate control, with batch-key tie breaking (not identical to first-use LRU). All trace-level misses must be >= offline oracle. Test labels are computed for scoring the predictor, never used to rank admissions. No prospective preregistration, actual compiled online policy, end-to-end timing, or process RSS measured. Model parameter count does not count history storage or policy-selection overhead.

## Primary capacity 512

Training 181592 requests, oracle-positive fraction .902628; development154281 requests, test130258. Development AUC .84352, test AUC .82629, shuffled test .64047 (one shuffle, not a chance-level assumption), learned test logloss .19782. Decay selected learned .5, shuffled .25, freq .25, gap4, recency0. All policy differences descriptive across only four independent test models; individual requests/episodes correlated.

|width|FIFO|LRU|oracle|learned|shuffled|frequency|gap|recency|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|128|5630|4469|3787|4338|4442|4511|4536|4469|
|256|9635|6462|5000|6309|6483|6424|6794|6467|
|total|15265|10931|8787|10647|10925|10935|11330|10936|

Development totals FIFO37533, LRU29921, oracle16373, learned24704, frequency25854, shuffled29705. On test learned reduces misses30.25% vs FIFO,2.60% vs LRU; closes 284/2144=13.25% of remaining LRU-to-oracle gap. Learned vs LRU wins/ties/losses 3/4/5 episodes, 2/0/2 model roots. Root sums (LRU/learned): 128_82 2109/1963, 128_83 2360/2375, 256_82 2641/2840, 256_83 3821/3469. Learned has5302 nonempty miss batches vs LRU5281, oracle4459; 488 newly requested absent entries bypassed. Aggregate miss improvement does not establish speed. Retain LRU as control; no promotion.

## Secondary capacity128

Same splits, new capacity-specific labels and fit. Training-positive .60036; development AUC .81029, test .77282, shuffled test .55109. Decay learned .25, shuffled .5, freq0, gap4, recency0.

|width|FIFO|LRU|oracle|learned|shuffled|frequency|gap|recency|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|128|26888|26189|12754|19296|26051|20808|25550|25502|
|256|53727|49404|25153|40184|49289|41259|48523|48835|
|total|80615|75593|37907|59480|75340|62067|74073|74337|

Learned saves16113 misses (21.32%) vs LRU and2587 (4.17%) vs frequency, closes42.75% of LRU-to-oracle gap. Beats LRU on11/12 episodes and4/4 model roots. Root LRU/learned totals128_82 10595/8089,128_83 15594/11207,256_82 22081/17013,256_83 27323/23171. However nonempty miss batches7901 learned vs7654 LRU and7525 frequency; learned bypasses31901 newly requested entries. A more fragmented miss workload may erase compute savings. Candidate for a frozen tight-budget replication, not evidence of a runtime win. 128-entry capacity selected after primary result, so treat as exploratory sensitivity.

## Reproduce

From experiment directory (NumPy2.3.5, sklearn1.8.0, PyTorch2.10.0+cpu in this environment):
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Generate or reuse ER-39 cal and test traces as described in ER39_RESULTS.md.
for d in 128 256; do
  python er32.py prepare er40_worlds --width "$d" --roots 82 83
done
python er39.py trace er40_worlds/128_*.npz > test128.jsonl
python er39.py trace er40_worlds/256_82_*.npz > test256_82.jsonl
python er39.py trace er40_worlds/256_83_*.npz > test256_83.jsonl
cat test128.jsonl test256_82.jsonl test256_83.jsonl > er40_test_traces.jsonl
python er40.py --train er39_cal_traces.jsonl --dev er39_test_traces.jsonl --test er40_test_traces.jsonl > er40_results.json
python er40.py --cap 128 --train er39_cal_traces.jsonl --dev er39_test_traces.jsonl --test er40_test_traces.jsonl > er40_cap128.json
```
Source SHA256 1005bcf13abba11a6beb50312caefb49985537422f415f785012761c13905537; Git blob fb933a9cdb8b8c7903886778f2bf614281461869 verified against local source. Test traces SHA256 f021fb47d0e25d2e68477f149f74556df319d226785cdeed048ffbd54236a009; capacity512 receipt 17d0d6655c50fd43e7f7fb42f5f195f948cbcf361f0977b13dec989a5be2091b; capacity128 receipt dd6094138fafd70933c0ed7c4eefc24b61df9d815ad6d45c3e2da53fe3c8b315.

## Next gate

Freeze the 128-entry model and decay; compile a causal feature/priority kernel with bounded history storage and compare against LRU plus frequency admission at matched total bytes. Charge all policy overhead, count nonempty miss batches, measure actual prefix kernel time and end-to-end RSS. Use more fresh roots and register root-level improvement criteria; do not use these spent test traces for confirmation. Learned native BPD factors, machine-certified numerical bounds and hard OS memory enforcement remain pending.
