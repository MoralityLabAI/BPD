# BPD-PD-17: radial camouflage, exact-zero matroid logic and observable angular rays

**Prospectively registered November 18, 2026.** A paired 32-root, two-source study tests whether PD-15/16 norm-aligned basis discovery depends on input energy revealing active-support cardinality.

PD-15 recovered 32/32 strongly clustered 16D mechanisms with source-observable norm weights versus 0/32 unweighted. Independent [PD-16 original first run #44237142845](https://github.com/MoralityLabAI/BPD/actions/runs/44237142845) then recovered **32/32 norm-aligned versus 0/32 shuffled weights with exactly equal ESS and weight multiset**, confirming observation-to-weight association under the original generator. Those are separate preregistered synthetic-source findings.

## Mathematical change

The original clustered law has four uniformly selected support cardinalities; weighting each observed x by exp(-||x||²) exponentially tilts these to roughly 63.7% singleton, 23.9% two-active, 9.0% three-active and 3.4% four-active. The new **radial-camouflage law** keeps the exact same support masks and hidden Q, drawing continuous active amplitudes and normalizing the per-pass latent vector to an independent radius R uniform [1.0,1.1]. Thus ||x||²=R² independent of latent cardinality. All true active magnitudes remain above the original .375 read gate.

The [exact-source mathematical note](THEORY.md) proves from a determinant support matching that the canonical latent basis is strictly optimal for **exact-zero hyperplane mass** in both laws, with at least 1/1024 normalized objective gap for any genuinely mixed invertible basis. This distinguishes ideal identifiability from the finite-smoothed Gaussian objective and its optimizer.

## Study arms

| Method | Original source | Camouflaged source | Compute per source |
|---|---|---|---:|
| Norm-weighted Givens, λ=1, σ=.15 | yes | yes | 39,960 angles |
| Uniform Givens, λ=0, σ=.15 | yes | yes | 39,960 angles |
| Label-free angular-ray concentration | yes | yes | 1024×8192×16 multiplies |
| Privileged planted-Q oracle | yes | yes | zero |

All arms use the same exact physical four-pass BPDModel/TiedComponentLinear evaluator, eight seen/withheld families of 96 trajectories, and the PD12/15 full-mechanism gate. The angular comparator is **separately costed**, using repeated source-singleton rays to demonstrate observational feasibility under radial camouflage.

The frozen primary quantity is the **within-root interaction** in norm-vs-uniform complete-recovery difference between the original and camouflaged laws. Positive scientific admission requires original-law weighted replication ≥24/32, camouflaged angular feasibility ≥24/32, at least 30/32 post-fit source correlation checks, mean interaction ≥.40 with positive 10,000-draw paired-root bootstrap 95% lower endpoint, and source-oracle full recovery under both laws.

## Provenance and evidence

- [CONFIG.json](CONFIG.json): 32 frozen model roots 3607–3853, both paired support laws, registered gates
- [PROTOCOL.md](PROTOCOL.md): complete predata exact source and physical causal design
- [THEORY.md](THEORY.md): exact-zero determinant matching and probabilistic norm tilt
- [source.py](source.py): frozen original-versus-radial-camouflage training/heldout generator
- [exact_source_math.py](exact_source_math.py): exact Fraction enumeration of the PD12 clustered support law
- [angular.py](angular.py): deterministic label-free angular feasibility comparator
- [experiment.py](experiment.py): 256-arm native physical BPD runner and root-bootstrap audit
- [test_source.py](test_source.py), [test_experiment.py](test_experiment.py): excluded engineering seed 4001 only
- [SourceBudget.lean](SourceBudget.lean): kernel-certified registered singleton, sample, and candidate counts

The [excluded-source preflight #44240421795](https://github.com/MoralityLabAI/BPD/actions/runs/44240421795) passed. The source-only mathematical and engineering results are separate from the prospectively frozen 32-root scientific test. No trained nonlinear TRM mechanism recovery is implied; that requires a separate gauge-covariant, actual closed-loop intervention protocol guided by NC-04, LOOPSCOPE-08 and EvalMath variation floors.
