# BASIS-01-SCALE: exact chunked scorer and per-board receipts

Date: 2026-11-17. Frozen before the first SCALE synthetic performance comparison.
Parent registration: `BASIS01_CAPTURE_PROTOCOL.md`; this is an implementation-equivalence audit with unchanged registered beta budgets `{0.01,0.03,0.10}` and unchanged `G=I` first-pass metric.

## Change

The naive `floor_and_basis.py --matrix-npz` performs an unnecessary dense solve for orthogonal dictionary bases, stores per-sample `[N,C]` activation and component arrays simultaneously and uses an `einsum` with a dense identity `[d_out,d_out]` even for Euclidean output norms. At native down_proj dimensions (`W[512,1536]`, `N=183*63*2=23058`), those choices can dominate runtime and working memory. Replace with:

1. Orthogonal formulas: `V.T @ W.T` for input-complete, `W.T @ U.T` for output-complete. Validate full-on reconstruction; do not invert the orthogonal basis numerically.
2. Single exact full SVD per W for shared left/right singular bases, preserving the complete `d_in` read basis including null directions.
3. Stream over sample chunks, compute one stable argsort per row, cumulative contribution magnitude `|a_c| ||u_c||_G`, and derive off sets for all three budgets. Preserve ascending-prefix semantics of the existing scorer.
4. In the `G=I` case use direct Euclidean norms and omit dense identity products. For PSD `G`, use a validated Cholesky/eigenspace factor; retain exact module-level seminorm meaning.
5. Aggregate rather than store per-row arrays. Report full-on residual, active/off counts, actual module error, certificate bounds, per-call-type samples/active counts, and a module-output softmax diagnostic explicitly marked as **not** downstream digit-head KL.
6. Add `--max-boards` development cap as a board-level cap only, preventing token- or pass-level pseudoreplication; record selection and keep the native full-roster label restricted to a SHA-verified 183-root capture.

## Acceptance

- On at least 6 deterministic synthetic matrix/activation seeds, chunked and legacy scores differ by at most `1e-10` absolute on scalar metrics; full-on residual `<=1e-11`.
- Exact certificate `actual <= beta_S` and budget `beta_S <= beta0 ||Wx||_G` for every processed sample, every basis and budget (tolerance `1e-10` or relative `1e-11` for floating-point accumulation).
- Re-running with chunk sizes 7 and 64 changes scalar results by `<=1e-10`.
- Phase tags, board grouping, finite checks and output shape admission must fail closed on malformed exported NPZ fixtures.
- Run source-only CPU tests in CI without pretrained weights, capture synthetic timing and peak-memory diagnostics as engineering data only. No real TRM neuron-support conclusion until the local checkpoint capture has run.

Do not modify `REGISTRATION.md`, any existing v2 receipt, or the sealed ARG-03-N test cohort. Preserve the existing `floor_and_basis.py` public function contract and synthetic results while replacing implementation internals.

## Implementation amendment (pre-test, 2026-11-17)

Keep the historical `floor_and_basis.py` evaluator intact as an independent CPU oracle. Introduce the scalable implementation and CLI in separate `basis_stream.py` and `score_exported_basis.py` files; test exact scalar agreement against the historical function on six synthetic seeds. This preserves the previously published synthetic receipt and directs full native 23,058-vector module captures to the streaming implementation. Report the numerical-null write direction count and the non-null active fraction: in a 512×1536 down projection the full right-SVD contains 1,024 structural null-write directions, making raw active counts misleading without the non-null denominator. These null counts are basis geometry, not neural evidence of sparse semantics.
