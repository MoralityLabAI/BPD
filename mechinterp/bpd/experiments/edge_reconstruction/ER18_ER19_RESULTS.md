# ER-18/19: bounded cross-pass interactions and lazy exact MAP

## Real-arithmetic bound
For phi=tanh, sup |phi'|=1 and sup |phi''|=4/(3 sqrt(3)). For scalar perturbations a,b,c, split phi(z+a+b+c)-phi(z+a+b) off the mixed difference phi(z+a+b)-phi(z+a)-phi(z+b)+phi(z). The first term is bounded by |c| sup |phi'| on the segment (or its interval enclosure). The mixed term is the double oriented integral of phi'' over [0,a] x [0,b], bounded by |ab| sup |phi''| on the rectangle's image. Hence the absolute residual is at most |c|+4|ab|/(3 sqrt(3)). Local suprema are computed on intervals: the maximum slope occurs at the point nearest zero; absolute curvature reaches its global maximum at |z|=atanh(1/sqrt(3)) and otherwise at the nearest interval endpoint to that magnitude.

For the four-pass trained recurrence, h3 follows the second ablation and h4 is final. At h3 preactivation take d=h2_i-h2_clean, a=W d, b=-M_j h2_clean, c=-M_j d where M_j is the rank-one component matrix. Apply coordinatewise envelope to obtain r3 >= |h3_ij-h3_i-h3_j+h3_clean|. At h4 preactivation take a4=W(h3_i-h3_clean), b4=W(h3_j-h3_clean) and bound unknown mixed term by |W|r3. Apply the same bound again to obtain r4. For scalar readout c_read^T O h4, E=|c_read^T O| r4 bounds the omitted joint-response error. Uses clean and single-component trajectories; no joint outputs required for bounds. Scope: zero ablation, no sphere rescaling, tanh recurrence, fixed factors; not a bound for arbitrary nonlinear BPD gates.

For common Gaussian noise and uniform graph prior, MAP minimizes squared residual cost. For approximate row fhat and certified coordinate error E, the cost lies between L=sum(max(0,|y-fhat|-E)^2) and U=sum((|y-fhat|+E)^2). Evaluate the minimum-U candidate, then successively evaluate the unresolved minimum-L candidate while its L could compete with the exact incumbent. Stop on strict separation or exhaustion. Exact incumbent tie-breaking uses lowest index. In real arithmetic valid bounds imply exact MAP, not posterior-distribution equivalence or hidden-fault recovery. Code uses float64 and EPS=1e-10 slack, not a verified outward-rounded interval implementation.

## ER-18 executed pilot
Fresh model roots 24..31, trained with ER-13 teacher/student training. NumPy float64 rollouts from trained float32 weights and float64 SVD components. 24 calibration and 48 evaluation input/readout pairs; 48 episodes/root; 12 distinct probes per episode, common Gaussian noise from calibration scale, matched observations. Exact oracle used only for environment responses, scale, and audit; random/tangent static k=16 controls retain 16 exact rows. Adaptive algorithm invokes selected joint rollouts only on the episode's probes; envelopes constructed once for all 48 probes using 13 base batch rollouts. No preregistration.

All 27,648 checks (8 x 36 x 48 x 2 bound methods) pass; local bounds <= global bounds. Mean absolute interaction error .0075838553, global envelope .0469139406, local envelope .0396459956. Largest actual/local bound ratio .780652.

| Method | hidden hit/384 | exact MAP agreement/384 |
|---|---:|---:|
| additive |328|330|
| random k16 |346|348|
| tangent k16 |378|382|
| global refinement |380|384|
| local refinement |380|384|
| full |380|384|

| Bound | Mean joint queries/episode | median | p90 | max |
|---|---:|---:|---:|---:|
| global |3.6328125|3|6|9|
| local |3.0807292|3|5|7|

Local: 1183 exact joint queries over 384 episodes. Counts are per-episode logical joint evaluations, not amortized cache rows or end-to-end speed. Uniform response tolerance policy retaining all rows with max E > tau*sigma requires median 34.5, 32.5, 30.5, 26, 21.5 rows for tau=.5,1,2,4,8 respectively (out of 36). All omitted errors within tolerance; conservative bounds yield limited static cache savings.

## ER-19 executed amortization pilot
Same roots and cases; compare eager 36 full evaluation-batch joint rollouts (no base rollouts) with lazy exact MAP using the same envelopes and reusable exact rows computed over 48 probes on first request. Lazy charges 13 base rollouts plus each unique joint row. Three paired timing repetitions per root and workload size, alternating order, median within roots then across roots. Timing includes bank creation, bounds, inference and cache lookup; excludes training and independent environment generation. All decisions equal across all repetitions.

| episodes | median cached joint rows lazy | eager median ms | lazy median ms | median paired eager/lazy |
|---:|---:|---:|---:|---:|
|1|3|.8751|4.6548|.1910|
|8|16.5|.9376|5.3609|.1734|
|48|33.5|1.2218|7.6724|.1575|

FAIL end-to-end speed: the tiny NumPy model makes rollouts cheap; the envelope loop dominates. Repeated episodes also drive unique pair count near full enumeration. Envelope implementation materializes two full candidate-by-probe bound tables, tangent scores and additive table; no RAM reduction claim. No timing-matched native or learned-factor result. Next: vectorize/stream local derivative calculations, consider exact signed interaction residuals and interval propagation, evaluate at larger hidden widths with actual expensive recurrent passes, use eager fallback when amortization predicts no gain.

## Reproduction
From experiment directory:
```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python certified_crosspass.py > certified_crosspass.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python refinement_cost.py > refinement_cost.json
```
Python with PyTorch 2.10.0+cpu, NumPy 2.3.5. SHA256 of executed sources: certified_crosspass.py 97346866f6688f98df5cf0643706b60e6db15d95b976e3b8ec792859c649a155; refinement_cost.py e9b31b1283303e7d37dda088493ccf8bae547ad4fdd2b28aa8e3a3f5ff944b42. Local raw receipts: certified_crosspass.json c6391d88f9bd17fb5b9671135fb66f5fce59a25ede9e8d26678396b5f7091146; refinement_cost.json bea63b40bae7cff9d540f2b4bc3aeda7bd95d3725ded7632e6d8dce6b71e6505.

Native learned BPD factors, larger real networks, formal Lean proof, and machine-certified arithmetic remain pending.
