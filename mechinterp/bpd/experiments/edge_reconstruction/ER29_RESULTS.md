# ER-29: probe ordering and partial-stage sizing

## Design

Nine fresh trained model instances, widths 32,64,128 with roots 57..59 at each width, using ER-25 world generator. Three distinct 12-probe episodes per model, Gaussian noise multipliers .1,1,3 times calibrated RMS. Compare eager exact batch64, the ER-28 gap4 incumbent, fixed index order and seeded random permutation with four-probe stages, gap2 and gap6, and observation-conditioned utility4. All partial arms use identical floors, incumbent, candidate sorting, batch64 and exact-column kernel. The gap ranking sums the upper-minus-lower cost width over all candidates, as in ER-28. Shuffle seed namespace 261142 includes width, root, episode; fixed across timing repetitions. Every method runs twice in rotated order, one BLAS thread, widths run sequentially. Timings include prefix construction, floor construction, ranking, candidate sorting, exact evaluation and pruning. Training, SVD and environment generation excluded. This is exploratory, not preregistered. No measured peak RSS.

For candidate g with current lower energy L_g=sum_p l_gp and initial exact incumbent E, replacing the contribution of probe p by its exact squared residual excludes g if the exact response lies outside [y_p-R_gp,y_p+R_gp], with R_gp=sqrt(max(0,E+EPS-L_g+l_gp)). Use a uniform pseudo-model over the admissible prediction interval [fhat_gp-e_gp,fhat_gp+e_gp] to estimate exclusion probability: one minus intersection length divided by prediction interval length. Rank probes by mean probability over the initially eligible candidates (excluding incumbent). This ranking is computed once. It is a heuristic estimate, not a calibrated probability model; true exact responses are deterministic given the model. Admissibility and MAP correctness are independent of the ranking, because actual squared residuals replace their respective lower floors before pruning. Utility4 stores two additional candidate-by-probe float64 arrays during screening and uses extra temporary arrays; no memory-efficiency claim.

## Results

27/27 distinct episodes match exact eager MAP and cost within 1e-10 for all seven methods, both repetitions. Every evaluated cost floor checked against the corresponding exact term. Work counts deterministic across repeats. True hidden recovery 10/27 (9/9 at .1 RMS, 1/9 at 1 RMS, 0/9 at 3 RMS). Exact decision agreement is not latent recovery. Episodes within a trained model are correlated.

|method|total exact candidate-probe evaluations|median paired gap4/method runtime|timing wins against gap4 /27|lower-work episodes /27|
|---|---:|---:|---:|---:|
|eager|2322432|.262|0|0|
|gap4|201772|1|reference|reference|
|fixed4|242888|.944|10|4|
|shuffle4|278096|.909|5|4|
|gap2|129314|1.039|18|27|
|gap6|285672|.946|10|0|
|utility4|195988|.959|8|18|

Gap2 reduces work 35.91% versus gap4. Gap4 reduces work 16.93% versus fixed4 and 27.45% versus shuffle4. Utility4 reduces work 2.87% versus gap4 but loses runtime on the median paired comparison. Utility4 is a negative incremental speed result, and no evidence of calibrated utility estimation. Gap2 wins on median paired ratio for 6/9 trained model roots; utility4 for 1/9. A single shuffled order per episode is only a limited null. Timing results are descriptive and hardware dependent.

|width|eager ms|gap4 ms|fixed4 ms|shuffle4 ms|gap2 ms|gap6 ms|utility4 ms|
|---:|---:|---:|---:|---:|---:|---:|---:|
|32|3.579|1.590|1.607|1.823|1.717|1.766|1.686|
|64|30.950|7.644|8.338|8.787|6.976|8.461|8.016|
|128|376.588|55.212|61.211|66.740|47.888|67.858|58.860|

Median paired gap4/gap2 ratios .926,1.053,1.121 respectively, with wins 1/9,8/9,9/9. Median paired eager/gap2 ratios 2.016,4.336,7.987. Total gap4/gap2 exact evaluations by noise: .1 RMS 23720/14846, 1 RMS 84376/53332, 3 RMS 93676/61136. All results include exact incumbent evaluation in work count. Candidate-probe computations are not FLOP counts; different stage shapes change dispatch and BLAS efficiency.

## Reproduce

From the experiment directory, with er23.py and er25.py available:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for d in 32 64 128; do
  python er29.py --width "$d" > "er29_$d.jsonl"
done
cat er29_32.jsonl er29_64.jsonl er29_128.jsonl > er29.jsonl
```
NumPy 2.3.5, PyTorch 2.10.0+cpu. Executed local source SHA256 90fd0695621cacf4825f69ba15aa61483cfeaa5779881e1b698ab700dcdb34a5; Git blob 0adfaa8b7db45b25bd7a16fce3cb8a06eaa34367. Combined receipt SHA256 300fed0ae399e608ce058f8b8c772b7521dbef2380cbbe437563197423c1b123.

## Next gate

Freeze gap4 as small-width control and gap2 as candidate for widths >=64, prospectively verify on new roots before adopting this width rule. Test one-probe stages and a fused exact-column kernel that amortizes dispatch across surviving candidates. Compare empirical survivor work against the theoretical optimal ordering for a small offline audit, include multiple shuffled nulls and a root-level confidence interval. Memory budget, learned native BPD factors, formal interval arithmetic and larger network deployment remain untested here.
