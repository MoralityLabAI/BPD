# BPD-PD-08 | Two-stage factor discovery, frozen-dictionary gate transfer

**Prospectively registered November 17, 2026.** [CONFIG.json](CONFIG.json) was committed at \`6a44eecfc51a2f6c76005774b8b35e237e44bafd\` before implementing the experiment and before any PD-08 training. Changes to the model, support law, thresholds, or training schedule after this registration require a separate hypothesis and independent seeds.

## Causal question

[PD-07](../pd07/RESULTS_20261117.md) found that a read-conditioned gate succeeds on every withheld rotated support when its dictionary is supplied exactly (24/24 planted-frozen controls). Jointly training the read and gate factors recovered the full mechanism in 1/24 seeds, with mean read cosine 0.805. A simpler raw-input gate failed in 0/24 seeds. This experiment isolates the factors learned during dictionary discovery from the later binary gate training.

**Primary comparison:** Given the *same frozen learned BPD factors* per seed, will a componentwise normalized read-threshold gate generalize across support combinations more reliably than an independent binary MLP on the raw rotated input?

The first-stage dictionary is trained once per seed on the **original PD-06 count-supervised top-k surrogate**. That method reached read-axis cosine≥0.90 in all 32 historical PD-06 seeds while having poor withheld-combination gate decisions. Stage two freezes this learned V and U, resets the gating parameters, and trains only a gate using the original support-law input samples. Both primary and comparator receive **no cardinality or support labels** during stage two; each begins from the same physical factors and sees the same per-seed training batch stream.

The first stage *does* receive true count labels through its registered auxiliary cross-entropy. The experiment is therefore a **conditional two-stage optimizer test with privileged count supervision in stage one**, not an entirely unsupervised discovery claim. State that limitation in every report.

## Source and independent test units

Use exactly the PD-06 four-dimensional source generator, seeded QR rotations and six frozen evaluation families. The held-out diagonal pairs (0,2) and (1,3), triple (0,2,3) and mixed unseen-support trajectory remain excluded from factor and gate training. Observed inputs are dense \(x=aQ^\top\) with signed latent amplitudes of absolute magnitude in [0.75,1.25] and latent cardinalities k=1,2,3.

Twenty-four **fresh** seed–rotation pairs:

\`1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289\`.

Engineering smoke seed 2017 is excluded. Stage-one training sample generator is 100000+seed. Stage-two training generator is independently initialized with 500000+seed for **each** trainable gate arm, so batch sequences are matched between read and raw gates. Held-out evaluation is sourced only after all training and gate selection are complete, using generator 200000+seed+10000×family-index with 192 four-pass trajectories per family. The model seed/rotation pair is the unit of statistical inference.

## Frozen stage-one dictionary discovery

Instantiate the actual shared \`BPDModel/TiedComponentLinear\` with one physical \(W=I_4\) bias-free layer called four times and C=4 shared rank-one components. Use the PD-06 \`GateMLP\` and \`objective(...,"count_supervised")\` without changing the source or model code. Both V and U receive gradients and updates for **360 Adam steps**, LR 0.025, batch size 32, matrix-faithfulness weight 0.30 and binary gate-usage coefficient 0.05. The auxiliary true-count CE has weight 1.0. The residual \(\Delta W\) mask is set to zero; zero ablation and no sphere rescaling preserve physical component execution.

Store factor states V,U, their stage-one gradient and update norms, and the alignment against the hidden Q **for evaluation only**. Freeze V and U bit-for-bit for every second-stage arm. A single stage-one factor snapshot must be shared by all arms within each seed.

## Frozen stage-two gates and negative/positive controls

Five registered arms per seed:

1. **read_trained (primary):** for each learned normalized factor read \(z_j=xV_j/\|V_j\|_2\), execute \(h_j=1[\sigma(10(|z_j|-\tau_j))\ge0.5]\), where \(\tau_j=0.25+0.5\sigma(\theta_j)\). Initialize \(\theta_j=0\), giving \(\tau_j=0.5\). Train only the four threshold parameters over 240 Adam updates.
2. **raw_independent (paired comparator):** initialize a new shared PD-06 raw-input MLP from torch seed 700000+seed, and train its four independent assignment-logit sigmoid gates over the same 240 Adam updates and **the identical stage-two batches**. No cardinality or support labels. Both arms execute exact binary masks with sigmoid-probability straight-through gradients.
3. **read_fixed_threshold:** apply the same learned-factor read gate at \(\tau_j=0.5\) without any stage-two training, to distinguish the usefulness of factor alignment from threshold fitting.
4. **oracle_support_learned_dictionary:** execute the *true* latent support in the globally aligned learned-component order, on the stage-one **learned** and frozen dictionary. This is a privileged capacity control. Its use of Q for oracle alignment must be explicit.
5. **read_planted_frozen:** replace the stage-one dictionary by the exact planted \(V=Q,U=Q^\top\) and apply the fixed read threshold 0.5. This is a planted factor positive control and receives no learned-factors claim.

All non-planted arms share **bitwise identical** stage-one V/U factors within a seed. The model always invokes the genuine physical BPD wrapper and disables residual writes with \`delta_mask=0\`. Gate loss is binary-forward MSE + 0.05×observed binary gate fraction, with the constant 0.30 matrix-faithfulness diagnostic carried into receipt. Train read and raw gates on the same stage-two source stream. Neither stage-two arm receives latent count/support labels, nor any held-out or calibration observations.

A seed's independent raw MLP is initialized as a fresh stage-two gate. It does not inherit the count-supervised stage-one gate weights. This prevents an inherited privileged count signal from entering the primary comparator.

## Preregistered per-seed scoring and acceptance

Align learned factor columns to the true rotated axes by **one globally optimal permutation** chosen for evaluation, with mean absolute read cosine and minimum per-component cosine reported. Its alignment may not alter a learned gate's execution.

An arm passes per seed iff all conditions hold:
- mean read-axis cosine \(\ge0.90\);
- mean binary-forward MSE across all six held-out families \(\le0.03\);
- *each seen family* has exact four-pass component support \(\ge0.80\) and MSE \(\le0.04\);
- *each withheld family*, including the novel transition, has exact four-pass component support \(\ge0.70\) and MSE \(\le0.06\);
- all masks execute exactly 0 or 1, both stage-one factors have nonzero training gradients and updates, and they remain bitwise frozen in stage two.

The **primary preregistered gate** requires read_trained success in at least **18/24** seed–rotation pairs, **six or more net paired read-only successes** against raw_independent, and exact two-sided McNemar \(p\le0.05\). The scientific result is FAIL when any condition fails, even if the read gate achieves lower MSE or a good OOD effect without full mechanism recovery.

For each arm report Wilson 95% seed intervals, exact paired discordance, read-axis cosine, learned thresholds, per-family four-pass support accuracy, false-on/false-off events, reconstruction, and factor freeze attestations. The matched seed/rotation is the statistical unit; passes and trajectories are nested.

## Prior mathematical margin and hypotheses

[PD-07 READGATE_THEORY](../pd07/READGATE_THEORY.md) proves on paper that normalized read factors within distance \(\delta<0.11547\) of planted Q and threshold 0.5 suffice for exact support execution on the original source. This corresponds to **per-component absolute cosine \(\gt0.99333\)** under the registered norm bounds. The registered success threshold of *mean* cosine≥0.9 is weaker, so include minimum matched cosine and a margin-satisfaction diagnostic.

The predicted failure mode is that stage-one learned factors meet mean alignment but one or more individual read directions retain enough cross-talk to cause false-on activation for an unseen pair/triple. The read_fixed_threshold arm isolates this without gate optimization. The read_trained arm tests whether per-component threshold fitting can accommodate the imperfect basis without labels. The oracle_support_learned_dictionary control tests whether exact masks produce adequate output reconstruction with those imperfect factors.

These are explanations to test, not guaranteed outcomes. Any failure or positive result applies only to the 4D synthetic identity teacher, not a 7M recurrent Sudoku transformer. The original PD-01 through PD-07 source-locked receipts must remain unchanged.

## Source lock and run discipline

Commit CONFIG and PROTOCOL before code and all PD-08 compute. Run an excluded-seed engineering smoke only on seed 2017 to verify Q orthogonality, withheld support exclusion, stage-one factor gradients, bitwise stage-two freezing, matched batches, exact binary masks, count/support label isolation and evaluation-only alignment. Then commit an immutable \`EXECUTE_LOCK.json\` referencing the tested source commit. Run exactly one registered 24×5 cohort, archive its unmodified JSON and scientific verdict, and preserve any scientific failure without retuning.
