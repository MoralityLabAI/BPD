# BPD-CQ-08: separately registered 32-root confirmation

**Evidence class: prospective independent-model-root synthetic confirmation, under an equal-component-cardinality ablation contrast.** The original experiment and component selection were frozen before the 8 development, 59 sham, and 32 confirmation model-world roots were tested. This report is scoped to a rank-eight learned nonlinear recurrent operator with a deliberately block-separated rank-six task and rank-two nuisance carrier. Confirmation release [CONFIRM_UNLOCK.json](CONFIRM_UNLOCK.json) and predata inference [confirmation_release.py](confirmation_release.py) were committed on a separate GitHub branch before scoring the registered 32 fresh roots. The initial one-process local command hit an execution time limit before writing a result; the fixed 32-root roster was then executed as eight disjoint four-root partitions, with source hashes and original seeds unchanged. None of the part outcomes was used to change controls, thresholds, or analysis.

## Main correct-cell damage comparison

For each valid first-cue donor, the actual first hidden state was produced by the learned physical teacher and the delayed target was defined independently by `L2=sgn(xz)`. The selected two-component BPD ablation and both independent random two-component ablations kept **six of eight components active** on the *same* natural donor state. The causal numerator is the count of full-correct donor cells turned incorrect by that second-pass mask, with **all 128 donor pairs per model root**, including unmatched controls, retained in the denominator.

| Registered arm | Damaged donor cells / 4,096 | Damage rate |
|---|---:|---:|
| Calibration-selected BPD two-component ablation | **2,139** | **52.22%** |
| Random two-component ablation 1 | 916 | 22.36% |
| Random two-component ablation 2 | 879 | 21.46% |
| Full 8-component donor continuation | 0 | 0% |

The **primary selected-minus-random1 root-mean effect is +29.8584 percentage points**. **29/32** independent roots exceed the preregistered +15-point floor. Exact one-sided Binomial(32, .5) root-sign p=**1.2780e-6**, and Holm adjusted p=**1.2780e-6** across the two registered comparisons. A 20,000-resample whole-model-root bootstrap, with precommitted seed 20261118, gives the descriptive **95% interval +26.15 to +33.54 points**.

The secondary selected-minus-random2 effect is **+30.7617 points**, with **30/32** roots exceeding the floor. Exact one-sided sign p=**1.2317e-7**, Holm-adjusted p=**2.4633e-7**, and cluster bootstrap interval **+27.39 to +34.35 points**. Both preregistered statistical gates pass. The weaker-primary roots **11011 (+14.84 points), 11023 (+1.56), and 11063 (+11.72)** remain in the result. Secondary failures are 11011 (+14.06) and 11023 (+8.59). The full source-level 95th-percentile/95%-confidence random-versus-random sham bound calibrated *before this cohort* is **13.28125 points**, below the prespecified 15-point effect floor.

## Reachability, task, and control validity

- **32/32** roots passed the preregistered ≥80% natural nuisance-state matching criterion. **4,008/4,096** generated donor pairs have valid nuisance-only clean-source matches, using the prespecified first-head and ±0.25 Euclidean state-dose bins.
- First-pass donor and matched nuisance states are exact forward outputs of legal first cues, with **zero** tanh-box violations and **zero** provenance replay error.
- The unmasked learned physical teacher and full-rank-eight BPD factorization each preserve **8,192/8,192** two-pass task outcomes; full donor label correct **4,096/4,096**.
- A first-cue-only predictor on registered paired ±z tasks hits **4,096/8,192 (50%)**, as dictated by independence of the second cue.
- The random controls, losing model seeds, component choices, matched-source failures, first-head changes, state displacements, and per-arm preactivation write norms are preserved in the immutable [raw confirmation receipt](results/cq08_confirmation_independent.json.gz.b64). [CONFIRMATION_LOCK.json](results/CONFIRMATION_LOCK.json) binds its SHA-256 and all eight partition hashes to the release registration.

## Important physical-dose limitation

The 2-off arms are **exactly matched for component cardinality and execution budget**, whereas the actual preactivation ablation write norms differ. Mean selected-block norm is **19.59**, compared with **8.51** for random1 and **8.33** for random2, approximately **2.30×** and **2.35×** stronger interventions. This is a major competing explanation for the larger selected-block damage. The results support a prospective **rank/cardinality-matched component-ablation effect** in this explicitly separated synthetic source family. They do **not** establish that BPD identifies a mediator with a larger effect than physically equal-dose random directions, nor that all masked second-pass states are reachable from valid source inputs.

The selected block was chosen from 256 calibration-only donor-label-free task examples per model using the registered singleton-loss derivative criterion. First cue contains (x,y,n1,n2); second z arrives only at pass two. Natural-state nuisance controls are sourced by modifying *legal first-cue inputs*, then rerunning the physical teacher; no hidden-state projection or rescaling turns a source control into an off-manifold candidate.

## Native and source provenance

- Registration source hashes: [REGISTRATION.json](REGISTRATION.json). Original experiment code, mask selection, and donor matching have not changed.
- Predata release: [CONFIRM_UNLOCK.json](CONFIRM_UNLOCK.json), source-verified [confirmation_release.py](confirmation_release.py), and [CONFIRM_RELEASE_PROTOCOL.md](CONFIRM_RELEASE_PROTOCOL.md).
- 59-root sham: [SHAM_LOCK.json](results/SHAM_LOCK.json) and [preflight_after_59_shams.json](results/preflight_after_59_shams.json).
- The hosted native confirmation replay is a separate engineering check from the original independent CPU result. Report its status and any cross-host retraining drift explicitly in the PR and CI receipts; a successful local statistic must never be described as hosted-native-verified before that check completes.

## Scientific next step

A prospective **physical-dose-matched** BPD ablation study should match the Euclidean norm of the removed preactivation contribution for the selected and random component blocks, including distributions at each independent model root, while retaining the same original naturally reachable donor states and legal nuisance-source control. A less structured core/nuisance architecture would test whether these modes emerge without deliberate separation.
