# Common action-logit bias in the optimizer check

The first remote engineering run at commit
`93da07ef9683d517f3d2c322678cdb5c96186bbb` passed 76 tests and failed one
optimizer parity case. The differing parameter was the scalar `out.bias`.
Its absolute difference after one AdamW update was `0.0007613571`.
The [failed workflow](https://github.com/MoralityLabAI/BPD/actions/runs/45815767566)
and [saved excerpt](FIRST_CI_FAILURE.txt) preserve that outcome.

The score head adds this same scalar to every action logit. For scores
`s_i = f_i + b`, softmax probabilities are independent of `b`. The cross-entropy
teacher objective therefore has an exact zero derivative along that coordinate.
Float32 reductions leave small residues. On AdamW's first update, its adaptive
step contains a factor proportional to `g / (abs(g) + eps)`, which can amplify
such residues near the default epsilon of `1e-8`.

The synthetic optimizer test now excludes only the shared bias from its
optimizer parameter lists while leaving it available to autograd and the
historical update helper. It asserts the held bias stays bitwise fixed and
retains all comparisons for the other parameters and returned logits. New
source and native checks exercise the common-bias shift directly, verifying
the shifted logits, softmax probabilities, action choices and loss. Ordinary
forward and all-parameter gradient tests retain their original tolerances.

This change affects the engineering test contract. The adapter and historical
training source remain byte-for-byte unchanged. Checkpoint parity and the
allocator fixture keep their original contracts. Cross-host replay of a full
trained policy still requires its own casewise numerical audit.
