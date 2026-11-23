# RT-BPD-02 source-only CPU result, 2026-11-23

**Evidence class:** deterministic finite-dimensional NumPy development fixture. No pretrained model, Nanbeige archived state replay, model checkpoint, GPU, training run, formal behavioral comparison or protected 168-case ARG-03-N evaluation was used. The Nanbeige research goal remains paused.

## Locked numerical check

For \`W=diag(4,3,2)\`, \`Sigma=diag(0.01,1,10)\`:

| Arm | Rank | Measured squared weighted error | Algebraic oracle |
| --- | ---: | ---: | ---: |
| Euclidean rank-truncated SVD | 2 | 40 | 40 from dropped third coordinate |
| Reached-state-weighted SVD | 2 | 0.16 | 0.16 tail squared singular value |
| Reached-state rank-1 with reader \`[0,1,0]\` | 1 | 9 downstream | 9 under that reader |
| Fixed-reader-weighted rank-1 | 1 | 0 downstream | 0 tail squared singular value |

The source-only reference additionally covers a singular PSD input moment, rank-deficient reader, zero moment/reader, identity reduction, orthogonal input-coordinate covariance, paired counterfactual moments and invalid inputs.

## Execution receipt

Local isolated CPU run:

\`\`\`text
PYTHONPATH=/mnt/data/rtbpd02 python -m unittest mechinterp.bpd.theory.rtbpd02.test_weighted_rank -v
Ran 8 tests in 0.014s
OK
\`\`\`

NumPy version 2.3.5 in the local verification runtime. The repository's GitHub Actions workflow is pinned to NumPy 1.26.4 for an independent CPU replay; its outcome is recorded by GitHub rather than assumed here.

## Scientific interpretation

This verifies the declared weighted-rank algebra on synthetic arrays and its numerical implementation. The 250-fold reduction in the rank-2 fixture's weighted error is a consequence of a chosen diagonal covariance, not an estimated Nanbeige gain. The published Nanbeige REFIT49/IO18 aggregate includes fitted physical operator comparisons and reached-state cosine summaries, while the raw root-balanced covariance inputs required for a new native fit remain local/unpublished.

Future archive replay requires a separate authorization, root-balanced CAL-only source vectors, fixed full-head reader semantics, and preserved model/precision/occurrence IDs. The original ARG-03-N test remains sealed.
