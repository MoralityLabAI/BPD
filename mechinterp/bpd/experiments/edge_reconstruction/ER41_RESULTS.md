# ER-41: compiled frozen admission, including policy-state memory

## Design

Use the unmodified capacity-128 ER-40 logistic fit and chosen age decay .25 (frequency decay 0). The compact er41_frozen_model.json preserves its coefficients, standardization, intercept and decay settings; source receipt SHA256 dd6094138fafd70933c0ed7c4eefc24b61df9d815ad6d45c3e2da53fe3c8b315. No fit, hyperparameter selection or adaptation on ER-41 test episodes. Fold the standardization into the 16 coefficients and intercept at load time. Four fresh synthetic teacher/student trained models from ER-25/32: widths 128 and 256, roots 84 and 85 at each width; twelve episodes, three per model at .1, 1, 3 times calibration RMS noise. This is a fresh-roster exploratory confirmation, not a preregistered campaign.

All policies run inside the same compiled full reconstruction search: ER-35 streamed screen, minimum-upper-bound exact incumbent, stable candidate order and probe order, batch64, one-probe pruning, deduplicated per-batch prefix requests and exact output evaluation. Each entry holds z3 and all second-component projections. LRU uses the ER-38 linked-list policy, touching batch hits before inserting misses. Frequency and learned policies score the resident/request union and optionally bypass misses. Frequency ranks by count (equivalent to the ER-40 stored prior log-count ranking with alpha=0), latest batch then ascending key. Learned reproduces all sixteen ER-40 causal features with pre-batch history, then ages retained logits with .25*log(1+age). Stable sorts enforce tie rules. All features and admission/eviction work are inside the timed kernel. Cache state resets each episode.

Memory budget is the persistent cache/policy-state accounting for learned128. In bytes, with C components, D hidden width, P probes and K entries, the common term is 8[K(D+C+1)+CP]. Charge LRU 8(2K+3) additional bytes for list metadata/scalar accounting; frequency 8(2CP) for last request and count; learned 8(5CP+C+2P+17) for last/count/gap/EMA/logit, component/probe counts, probe ranks and folded coefficients/intercept. Match this nominal persistent byte ceiling by enlarging simpler controls as far as integer capacity permits. This accounting excludes common input arrays, transient sorting/batch buffers, Python/JIT objects, most local scalars and overhead; it is not a hard memory limit or matching of whole-process RSS. The LRU scalar accounting is conservative. All arms receive the same frozen coefficient arguments even if unused. Compare same-capacity controls as secondary contrasts.

|width|learned budget bytes|LRU128 bytes|freq128 bytes|LRU157 bytes|freq146 bytes|
|---:|---:|---:|---:|---:|---:|
|128|338248|277528|300032|337616|337040|
|256|675144|551960|599040|671440|672912|

One BLAS thread; shared Numba disk-cached compiled kernel, warm all modes before timing; fresh process per width/mode. Two repetitions per episode, median within episode. Modes sequential at each width, reverse width order except the width256 combined command timed out during learned and that configuration was rerun from scratch before the remaining controls. All included rows complete. Timings exclude training, model load, warmup and independent audit; include screen, history, ranking and prefix compute. Peak RSS is process high-water after all episodes, not isolated policy allocation. Timing order is not randomized and only two independent roots per width.

## Verification

60 mode-episode records (120 timed calls), all match independent eager MAP index and energy within EPS=1e-10. Independent eager audit spans 491520 exact candidate pairs across twelve episodes; max score discrepancy 2.3592239273284576e-15. Every policy uses identical incumbent, exact candidate-probe count and batch-local dedup on every episode. Total candidate-probe work per policy 276767 (53461 at width128,223306 at width256). All evaluated floor assertions pass. Four of twelve injected faults recovered; MAP correctness is distinct from latent recovery. Float64 checks, not certified outward rounding.

An independent trace reconstruction using ER-39 plus original Python ER-40 feature extraction and policy replay reproduces misses, nonempty miss batches and bypass (for optional admission) on all 60 records, along with exact work and dedup. Frozen learned model uses folded logits in replay; no test labels enter scoring. The 128-entry offline oracle on these traces has 73107 misses. The online policy does not have that future information.

## Measurements

|width|mode|K|misses|nonempty miss batches|bypasses|median solve ms|median finish ms|median paired budget-LRU/mode|wins vs budget LRU /6|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|128|lru128|128|17569|2213|0|85.958|59.856|.973|2|
|128|freq128|128|14735|2086|10560|114.638|79.640|.760|0|
|128|learned|128|13617|2244|6413|103.895|78.544|.878|1|
|128|lru_budget|157|14395|2133|0|86.618|57.951|1|ref|
|128|freq_budget|146|13206|2054|8922|123.855|82.513|.717|0|
|256|lru128|128|117829|8290|0|937.645|741.850|.955|2|
|256|freq128|128|93885|8281|85539|908.721|746.637|.962|3|
|256|learned|128|92272|8416|58200|845.186|685.412|.995|3|
|256|lru_budget|157|101594|8264|0|854.621|685.346|1|ref|
|256|freq_budget|146|88370|8250|79479|1068.931|860.359|.826|1|

Aggregate misses: LRU128 135398, frequency128 108620, learned128 105889, budget-LRU157 115989, budget-frequency146 101576. At equal entry capacity, learned saves 21.8% vs LRU and 2.5% vs frequency; it beats both on all four model-root aggregate miss comparisons (episodes 11/12 and 7/12 respectively). Learned closes about 47.4% of the same-capacity LRU-to-oracle miss gap. At the nominal equal-byte budget, learned saves 8.7% vs LRU (9/12 episodes, all four root totals) but incurs 4.25% more misses than frequency (only 2/12 episode wins, zero root wins). Nonempty miss batches for learned 10660, budget LRU 10397, budget frequency 10304.

Runtime: learned vs same-capacity LRU median paired ratio .937 (5/12 wins), vs same-capacity frequency 1.103 (9/12 wins); vs budget LRU .950 (4/12 wins), vs budget frequency 1.195 (11/12 wins). Root-level median paired budget-LRU/learned ratios .914/.835 at width128 and 1.00016/.991 at width256. Runtime advantage over frequency is specific to these implementations and workload; rank sorts and miss batch distributions differ. Learned fails incremental runtime promotion against LRU. The timing comparison is descriptive and based on only four models.

Final process peak RSS MiB (LRU128/freq128/learned/LRU budget/freq budget): width128 188.984/188.852/189.035/188.848/188.867; width256 202.926/202.844/203.180/202.898/202.949. Post-warmup high-water baselines approximately 183.4 MiB throughout. No meaningful process-level memory win; persistent policy-state budget is the intended control.

## Reproduction

From experiment directory, use fresh output directories:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for d in 128 256; do
  python er32.py prepare worlds --width "$d" --roots 84 85
done
python er41.py --model er41_frozen_model.json --mode learned --reps 1 worlds/128_84_0.npz
for d in 128 256; do
  modes='lru128 freq128 learned lru_budget freq_budget'
  [ "$d" = 256 ] && modes='freq_budget lru_budget learned freq128 lru128'
  for m in $modes; do
    python er41.py --model er41_frozen_model.json --mode "$m" worlds/${d}_*.npz > bench${d}_${m}.jsonl
  done
done
cat bench*.jsonl > er41_bench.jsonl
python er41.py --model er41_frozen_model.json --mode audit worlds/128_*.npz > audit128.jsonl
python er41.py --model er41_frozen_model.json --mode audit worlds/256_84_*.npz > audit256_84.jsonl
python er41.py --model er41_frozen_model.json --mode audit worlds/256_85_*.npz > audit256_85.jsonl
cat audit128.jsonl audit256_84.jsonl audit256_85.jsonl > er41_audit.jsonl
python er39.py trace worlds/128_*.npz > trace128.jsonl
python er39.py trace worlds/256_84_*.npz > trace256_84.jsonl
python er39.py trace worlds/256_85_*.npz > trace256_85.jsonl
cat trace128.jsonl trace256_84.jsonl trace256_85.jsonl > er41_traces.jsonl
python - <<'PY'
import json
from er40 import features,replay
from er39 import online,oracle
from er41 import model
w,b=model('er41_frozen_model.json')
rows=list(map(json.loads,open('er41_bench.jsonl')))
tests=list(map(json.loads,open('er41_traces.jsonl')))
audit={r['path']:r for r in map(json.loads,open('er41_audit.jsonl'))}
assert len(rows)==60 and len(tests)==len(audit)==12
results=[]
for r in tests:
    name=r['path']; ref=audit[name]
    bb={x['mode']:x for x in rows if x['path']==name}
    assert len(bb)==5 and len({x['incumbent'] for x in bb.values()})==1
    X,k,s=features(r,128); item=(r,X,k,s,None)
    for mode,rec in bb.items():
        assert rec['best']==ref['best'] and abs(rec['score']-ref['score'])<1e-10
        assert rec['payload_bytes']<=rec['budget_bytes']
        if mode.startswith('lru'):
            miss,profile=online(r['batches'],rec['capacity'],r['p'],'lru')
            expected=dict(misses=miss,nonempty_miss_batches=sum(x>0 for x in profile))
        else:
            expected=replay(item,X@w+b if mode=='learned' else X[:,5],.25 if mode=='learned' else 0.,rec['capacity'])
        assert all(rec[key]==expected[key] for key in expected)
        assert rec['pairprobes']==r['work'] and rec['dedup']==r['dedup']
        results.append(dict(path=name,mode=mode,expected=expected))
print(len(results),sum(oracle(r['batches'],128)[0] for r in tests))
json.dump(results,open('er41_replay_audit.json','w'),indent=2)
PY
```

Executed source SHA256 6beb856cb41c74b33368e477afca90627b9263c724a68508f0f9dc8bbc5743dc; Git blob 7b5251b9860edee34f47b6ce87836d397b92e011. Frozen model SHA256 3f2551fbdcbe79bbf25c7f23e9a7dc05257ec64695f5bf78b173b39f8e46578d, Git blob d86f51a991845def59ece03fa9e6b4f508605c02. Bench SHA256 6ffb5d171ae38fac6b78148ebd51339bdae724cd4226bc6f58a096646e6461fd; eager audit cc6c1990af9a58d6af717f3fdfde351e80c74c9c83f2705f89e4ab9548f3d0ac; traces 01f5936bd76d76aa28b4ba9a26504756c750ffefbe52c8e312dff7f235a90edc; replay audit d8ac9d211d2f5a0a7f47cadbfd46dd068006050342ae6bfcc23266a24bd95992.

## Next gate

Retain budget-LRU runtime baseline and budget-frequency miss baseline. Reduce learned state and selection cost before further promotion: cache-only priority metadata with compact counters/age, single top-K selection rather than three full stable sorts, or a bounded two-choice eviction approximation. Require causal replay parity for exact policy implementations; approximate policy changes must receive separate controls. Freeze fresh roots and primary equal-byte runtime endpoint prospectively, compare against efficient frequency admission, and charge full persistent state and transient allocations. Native learned BPD factors, formal numerical bounds and hard OS memory enforcement remain outstanding.
