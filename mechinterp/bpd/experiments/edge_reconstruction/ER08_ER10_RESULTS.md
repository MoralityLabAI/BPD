# ER-08 through ER-10: noisy reconstruction and compact posterior

## Reproduction (Python 3.13.5, NumPy 2.3.5, CPU)
From this experiment directory:
```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python noisy.py > noisy.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python packed_build.py > packed_build.json
OPENBLAS_NUM_THREADS=1 python noisy_counts.py > noisy_counts.json
```
Source SHA256 (local): noisy.py 9fbdd3bdca06f086d7474720de3a5d860b20970eec23d37356e44c200cf2620b; packed_build.py c33bc95ca9f7778c2366fa4352e03a483c0c1d5eceb77a8425c9100ca4c7db6b; noisy_counts.py da4a157d8ec70364d9640c0a6ae702202c47a8f93146f515197b13a9555ac83c.
Receipt SHA256: noisy.json cf343434ca64d9d6f79c55cf9fdd314e2b1275b34bc9d3266611c70689a658ee; packed_build.json 42329f5b8816f11031d72c4f59a6b43d2cefc79f438f6967ed882fd274e95cf7; noisy_counts.json e749318c3e46d9fda895083c0177ae61543725549b830a38e1b56c656e588b64.

## ER-08: Bayesian sensing
Uniform prior over 4060 three-edge directed supports; per-edge known random magnitudes 1..3; fresh roots 232..295, seed 261128. Each root has 64 Rademacher bilinear probes, independent additive noise per probe (-1,0,+1 with .15,.70,.15 probabilities), paired across arms. 16 distinct probes; all arms use correct noise likelihood for inference. Adaptive scores predictive entropy: I(G;Y|a,S)=H(Y|a,S)-H(noise). Explicit mutual information check at first root/first query: 2.7160732501 bits, within 1e-12 of entropy difference.

| policy | MAP hits 4/8/12/16 | mean negative log posterior truth mass at 4 probes (bits) | true support in 95% credible set at 4/8 probes |
|---|---|---:|---|
| random | 18/60/62/64 | 2.7292 | 62/64 |
| static entropy | 18/59/63/63 | 2.8144 | 64/64 |
| adaptive entropy | 32/64/64/64 | 1.8471 | 64/64 |

Denominator 64 for each count; credible sets include boundary ties. All Bayesian updates preserved true support; by probe 8 adaptive had posterior singleton on every root. On the same adaptive trajectory, a misspecified noise-free hard elimination filter retained true support only 2/64 at probe 8, and had empty support 62/64. This comparison isolates the inference assumption on matched observations, not a competing independently scheduled policy. Bayesian model knows true likelihood and edge magnitudes; no general robustness or unknown-weight claim. Fresh roots but shared support family. Probe budgets matched, wall time not measured. Exploratory, no preregistration.

## ER-09: direct packed construction
Construct two bit planes one probe at a time via three vector gathers, avoiding full dense P x H x 3 allocation. Same masks across all five n values and six paired builds each (one allocation trace pair, five untraced timing pairs). One root per size (600+n using moments.world); alternating timing order; CPU single-thread environment.

| n | candidates | raw plane bytes | dense traced peak B | direct traced peak B | median dense ms | median direct ms |
|---|---:|---:|---:|---:|---:|---:|
|6|4060|65024|2861205|91160|2.1883|3.0541|
|7|11480|183680|8084624|236889|4.7347|7.4947|
|8|27720|443520|19517584|560365|13.7558|15.8376|
|9|59640|954240|41989264|1196661|40.5534|33.0766|
|10|117480|1879680|82708624|2349645|100.9698|74.0320|

Allocation traces exclude preexisting inputs and support enumeration, and are not process RSS. Timing covers building masks only, not scheduler rollout. Both use equal outputs. Largest size traced peak ratio 35.2, ratio of median times 1.36. Fixed sparsity 3, binary-sign outcomes; this builder is not a 2-bit codec for the unequal-weight noisy experiment.

## ER-10: integer likelihood-count posterior
For uniform prior and the symmetric noise model above, a hypothesis surviving all observations has likelihood product (7/10)^c (3/20)^(t-c). Divide by the common factor (3/20)^t: posterior proportional to (14/3)^c. Set state -1 on impossible candidates and increment c only on exact matches. For fixed budget t<=127 a signed byte suffices for each candidate. Nonuniform known prior can be factored in separately. General categorical independent noise can use a count for each distinct positive likelihood value; unmodeled noise invalidates this exact update.

3072 updates (3 arms x 64 roots x 16) checked against ordinary floating-point Bayesian update, same positive support and saved true posterior mass. Max absolute posterior error 3.3306690738754696e-16. Persistent raw count state 4060 bytes vs 32480 bytes float64, plus 136-byte shared lookup table. Decoding requires temporary floating-point weights; no peak RAM or speed claim for this method.

Next: noise misspecification (full-support contamination likelihood), unknown weights, real BPD component intervention adapter, support-disjoint training; formalize sufficient state and partition identities. No actual neural ablation or Lean check completed.
