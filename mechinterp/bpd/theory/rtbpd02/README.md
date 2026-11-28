# RT-BPD-02 reached-state and fixed-reader-weighted low rank

The original CPU formulation is now extended by the [mathematical paper](THEORY.md), the matched-rank [synthetic recurrence](synthetic_gym.py), and a read-only [Nanbeige CAL archive replay](ARCHIVE.md). The [research report](NANBEIGE_RT_BPD_02_REPORT_20261123.md) records the results and limits. The [initial source-only protocol](PROTOCOL.md) and its [eight-test result](RESULTS_SOURCE_ONLY_20261123.md) remain the starting reference.

For a fixed physical reader and a reached second moment, the [NumPy solver](weighted_rank.py) implements the weighted Eckart–Young lift with explicit numerical input and reader supports. The [transport reference](transport.py) covers paired operator and state differences, fixed future readers, delayed-dark rank, directional gain, and signed paths. The [extension registration](EXTENSION_PROTOCOL.md) defines the synthetic and archive families.

## CPU reproduction

From the BPD repository root with Python 3.11 and NumPy 1.26.4:

    python -m unittest discover -s mechinterp/bpd/theory/rtbpd02 -p 'test_*.py' -v
    python mechinterp/bpd/theory/rtbpd02/synthetic_gym.py --out PATH_FOR_SYNTHETIC_RECEIPTS

The output contains an aggregate, one measurement per synthetic root, method, panel and pass, and the synthetic EvalMath input. Both worlds are repeated measurements within a root. The [CPU workflow](../../../../.github/workflows/rtbpd02-source-only.yml) runs the public checks and publishes the generated synthetic receipts. The optional archive replay needs the existing audited COMP14 and IO18 assets and both pinned checkpoint shards. See the [archive contract](ARCHIVE.md) for the command and the exact inputs.

The Nanbeige archive is selection-exposed development evidence. Native causal, GPU, and protected-cohort work require a separately authorized protocol.
