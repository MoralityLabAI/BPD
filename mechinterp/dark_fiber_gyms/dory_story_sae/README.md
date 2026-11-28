# Dory Q4 Oakhaven SAE tower pilot

In Dory's Oakhaven storyworld, the second recurrent pass attenuated a donor edit
that strongly shifted the action after one pass. A sparse autoencoder (SAE)
trained on action-token states recovered most of the final-pass donor effect.
This is a state-space intervention pilot for Behavioral Parameter Decomposition
(BPD) on one previously observed storyworld.

## Action margin

| Edit at action-token residual | Blind play | After the cue challenge |
| --- | ---: | ---: |
| Direct donor at end of pass 1, readout after one pass | +6.29 | +6.10 |
| Identical direct donor edit, readout after two passes | +0.11 | +1.33 |
| Direct donor at end of pass 2 | +6.21 | +6.20 |
| Prompt-trained SAE span at end of pass 2 | +0.32 | +0.69 |
| Action-position-trained SAE span at end of pass 2 | +5.69 | +6.02 |
| Donor-orthogonal sham at end of pass 2 | +0.29 | +0.10 |

Entries are mean changes in the opposite-world action's logit minus the
recipient action's logit, averaged over the two directions of transfer within
each context. The dock chooses `deploy` or `hold` when the freshness of the
channel report changes. The ceremonial lamp stays green in both worlds.

The native model reused identical prompt IDs and the same six-token JSON
continuation prefix. Its two-pass clean prefill head and greedy action replayed
the archived play on all four prompts. One of the four one-pass trials would
have diverged before the action slot under free generation. The effect table
uses the fixed prefix for all trials. An observed flip is a change in the
full-vocabulary argmax from recipient to donor at that slot.

The first-pass residuals matched across loop budgets at all sampled sites and
positions. All 60 first-pass nontrivial edits also matched by source write
digest. This holds the edit fixed while the extra computation changes. The
pass-2 direct donor is computed from the pass-2 residuals at the same relative
depth. Its cosine with the corresponding pass-1 contrast in blind play is 0.93.
The large change in native effect despite that alignment makes readout horizon
a relevant variable when indexing BPD causal importance.

## Tower and controls

- Dory 1.0 4B Q4_K_M, one and two native passes. The model and compiled reader
  hashes appear in the [primary receipt](receipts/story_results.json).
- Shared across passes, each of three separate top-8 SAEs has 96 features and
  fits 16 disjoint AGN-02 fixture roots. Corresponding residual readout sites
  are absolute zero-based layers 25/65, 45/85 and 55/95. The report uses the
  end-of-pass pair 55/95 for the table and retains all swept sites in the receipts.
- Edits replace one residual vector on its original-radius sphere, with a
  chord length of 5% of the recipient radius. The comparison includes direct
  donor tangents, rank-8 SAE/PCA/Haar projections, donor-orthogonal shams, and
  same-value writes. All 44 same-value controls match the full output head.
- The refit gives prompt and action-token states equal sampling weight, with the
  same 16 calibration roots. At the last site in each pass, Oakhaven action-state
  reconstruction error fell from 97% to 43% of the original calibration-mean
  prediction error. At the pass-2 endpoint, the refit caused donor-action flips
  in 0/4 Oakhaven cases and 8/16 development fixture cases across eight roots.
  This refit was chosen after the original readouts and reused those cases.
- The CUDA graph update aborted an initial capture after 32 rows. The full
  panel was rerun with graph reuse disabled. The overlapped activation and
  full-head records matched in all 32 rows and the interrupted receipt is
  source-hashed.

The mechanistic floor was 0.05 in the registered bounded contrast
`0.5 × (tanh(SAE donor margin / 4) − tanh(sham donor margin / 4))`. Across eight
development roots the mean was −0.01746 and the largest observed absolute
sham was 0.04056. [EvalMath input](receipts/power_input.json) preserves the
root-level pairs. EvalMath returned `REDESIGN_OR_PILOT_ONLY`, with the sham tail
uncalibrated and the storyworld unsealed. The four story prompts, both passes,
and the alternate intervention arms are repeated measurements on one story root.

## Offline replay

From the repository root, run the receipt audit with Python 3.10+.

```sh
python mechinterp/dark_fiber_gyms/dory_story_sae/verify_receipts.py
```

The audit rechecks 376 per-case scores, 60 matched writes, all 10 DataGoblin
decision records, source-world custody, and the registered development contrast.
The 376 scores include the 48 development controls, 188 primary Oakhaven rows,
and 140 action-position refit rows. [Refit metrics](receipts/action_refit.json)
and [decision records](receipts/decisions.jsonl) are included.

The [manifest](receipts/manifest.json) locks the compact receipts and C++ reader
source to the local experiment bundle. With that bundle, pass
`--source-bundle dory_story_sae_bundle.zip` to check its SHA-256 and exact member
bindings. Native inference requires the pinned local checkpoint and adapted
llama.cpp CUDA runtime. Full prompts, activation dumps, fitted dictionaries,
and model weights stay local.
