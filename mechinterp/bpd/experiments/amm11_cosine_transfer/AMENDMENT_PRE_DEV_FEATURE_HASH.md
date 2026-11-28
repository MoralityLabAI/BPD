# AMM-11 pre-development receipt-integrity amendment

**Registered 2026-11-18 after two TRAIN-only hosted runs, before any DEVELOPMENT model was trained or scored.** The first training-policy lock was committed at \`06ae61c534cc8eb496c43cb96591d6428f7d68bc\`. In the next hosted run [GitHub Actions #44318019737](https://github.com/MoralityLabAI/BPD/actions/runs/44318019737), all sixteen engineering tests and the complete 64-root numerical training replay passed. The new independent DEVELOPMENT stage failed **before fitting any model** at \`validate_training()\`, which correctly rejected missing \`features_sha256\` from its per-training-assay receipt.

The imported AMM-09 scorer contains \`features: features["features"].tolist()\`, but does not produce the \`features_sha256\` field, even though AMM-11's independently written source-integrity verifier requires it. This is an **engineering receipt-serialization omission**, not a failed synthetic task, a new data-driven mechanism, or an effect-size observation. Neither prospective DEVELOPMENT nor protected CONFIRMATION data were accessed. Training-only label results remain 64 independent roots, strongest pooled mask (0,2), 1,932/8,192 hits, and unchanged learned ridge weights in the original source.

## Frozen remediation before development

1. Add \`features_sha256=m.digest(features_list)\` to each physical assay receipt, hashing precisely the full 18-dimensional candidate features on its registered donor pairs. Keep the exact trained model, the six physically admissible masks, the source-defined behavioral target, the optimizer, the score threshold, all fixed policy choices and comparison family unchanged.
2. Add an excluded-engineering test that checks \`features_sha256\` is present and exactly matches the retained training feature list; it must fail on tampering.
3. Preserve the first \`TRAINING_LOCK.json\` under \`results/TRAINING_LOCK_PRE_FEATURE_HASH.json\` for historical evidence. **Retire the active \`TRAINING_LOCK.json\`** because the executable source SHA changed; do not reuse it for DEVELOPMENT.
4. Freeze corrected scorer/tests, rerun all 64 TRAIN roots, commit a **new policy and full training-bit/feature receipt source lock**. Only afterward may one independent DEVELOPMENT cohort be generated and scored.
5. Preserve the original failing Actions run and training receipt in the README/report. No PRE-FEATURE_HASH training receipt becomes a source for the prospective DEVELOPMENT population.

This is a strictly prospective **pre-development source integrity repair**. The physical target/action definitions and the prior-selected cosine competitor are fixed. This amendment does not license tuning model weights, replacing a poor root, changing the effect floor, weakening the 95/95 independent-root EvalMath gate, or accessing any protected AMM03–11 confirmation cohort. Original numeric results, source snapshots and full logs remain available for audit.
