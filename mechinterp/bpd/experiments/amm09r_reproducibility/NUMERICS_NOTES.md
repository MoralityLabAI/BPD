# AMM-09R CPU reproducibility notes

## Scope

This document supports the prospective engineering registration. The initial
comparison uses the historical 64 AMM09 training roots and eight fresh excluded
engineering roots under `baseline32` and `pinned32`. Historical training replay
is a diagnostic comparison. Fresh engineering roots remain outside scientific
training, development, and confirmation populations. All protected confirmation
populations retain their existing seals.

These notes contain source findings and proposed diagnostics. Run receipts must
establish whether a candidate passes. A float64 candidate requires a separate
prospective fallback amendment.

## A specific initialization mechanism to test

The AMM09 physical model initializes `I` and `O` with float32 `torch.randn`;
the BPD fit separately initializes `V` and `U` the same way. Each tensor has
18 elements.

PyTorch v2.10.0's
[CPU distribution source](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cpu/DistributionTemplates.h)
contains distinct implementations for contiguous float32 normal draws with at
least 16 elements. Its AVX2 branch uses vector log/sincos approximations and FMA.
The generic branch uses scalar standard-library transcendental functions.
Different CPU dispatch paths can therefore change floating-point initialization
despite matching generator seeds. This is a candidate explanation for the
historical drift. Attribution requires measured stagewise differences; the
historical receipts lack initial parameter tensors.

The same release's
[dispatch implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/DispatchStub.cpp)
maps `ATEN_CPU_CAPABILITY=default` to `CPUCapability::DEFAULT` and caches the
selected capability. Set the variable before importing PyTorch and start a
fresh process for every run. The
[CPU kernel README](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cpu/README.md)
describes this setting as selecting the oldest supported vector instruction set.

## Controlled float32 recipe

Set these environment variables before the process imports PyTorch:

```text
ATEN_CPU_CAPABILITY=default
MKL_CBWR=COMPATIBLE
MKL_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_DYNAMIC=FALSE
OMP_DYNAMIC=FALSE
OPENBLAS_NUM_THREADS=1
PYTHONHASHSEED=0
```

Before the first computation, apply and record:

```python
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True, warn_only=False)
```

Record the effective thread counts and CPU capability. Environment text records
the requested configuration; cross-worker output comparisons establish the
tested reproducibility scope.

[PyTorch 2.10 deterministic-algorithm documentation](https://docs.pytorch.org/docs/2.10/generated/torch.use_deterministic_algorithms.html)
defines its guarantee for identical inputs on the same software and hardware.
Cross-CPU portability requires the registered empirical check.

[Intel's CNR introduction](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-2/getting-started-with-conditional-numerical.html)
documents `MKL_CBWR=COMPATIBLE` for Intel and compatible processors supporting
SSE2. It selects a shared code path that avoids specified approximate
instructions. Intel's
[reproducibility conditions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-2/reproducibility-conditions.html)
require fixed thread counts and recommend disabling dynamic thread adjustment.
These conditions govern oneMKL routines. PyTorch's ATen operations and the host
math library require their own evidence.

## Receipts and comparison boundaries

Retain raw arrays or canonical tensor-byte hashes, including dtype and shape,
for initial parameters, final physical parameters, and BPD parameters before
canonicalization. Record random-stream fingerprints separately from optimizer
results. Preserve the canonical factors and each rank-one component product.

For the six-mask assay, retain feature arrays and delayed probes together with
all candidate bits. Report each changed label with both signed threshold margins:

```python
margin = 0.18 - (probes - target[:, None, :]).abs().amax(dim=-1)
```

The original success rule remains fixed. Margins support diagnosis and every
case stays in the comparison. Retain raw ridge coefficients alongside the
four-decimal policy so quantization can be examined directly.

Compare numerical payloads independently of provenance fields such as process
IDs and timestamps. Archive those fields in the complete receipt. Tensor-byte
hashes avoid differences introduced by serialized archive metadata. Equality
of final bits alone leaves continuous drift unresolved.

Pin the Python patch release and package wheels, recording wheel hashes and
the container image digest when used. Each receipt should include source SHA256
values, `torch.version.git_version`, `torch.__config__.show()`,
`torch.__config__.parallel_info()`, and the installed package versions. Capture
CPU model and instruction flags together with the hosted runner image version.

Separate hosted jobs establish evidence across workers. A claim about different
CPU models requires observed hardware differences in their receipts. Every
claim should state the tested runtime and worker coverage.

## Canonicalization and a possible float64 amendment

[PyTorch 2.10 SVD documentation](https://docs.pytorch.org/docs/2.10/generated/torch.linalg.svd.html)
explains that singular vectors may differ across hardware or software because
their signs are arbitrary. A simultaneous sign flip preserves a rank-one
component product. Compare those products as well as raw factors so the
diagnostic record identifies this case. CPU SVD uses the CPU linear algebra
backend; the documented `driver` argument applies to CUDA.

The historical AMM09 source explicitly copies `vc.float()` and `uc.float()`
into the factors after SVD. Changing only the default dtype would retain that
float32 rounding step. A prospective full-float64 recipe must specify the dtype
of canonicalization and factor storage.

Changing the random-generation dtype also changes the generated values or
generator consumption. A paired arithmetic diagnostic can freeze float32
initialization and training streams, then cast them to each arithmetic dtype.
A new float64 recipe may instead define its own stream, provided the amendment
states that change before execution.

Passing a controlled recipe would support that recipe within its measured
scope. Historical AMM09 replay requires agreement with its historical locks.
The AMM09R engineering gate and any later scientific preflight leave the
separate confirmation-release requirements in force.
