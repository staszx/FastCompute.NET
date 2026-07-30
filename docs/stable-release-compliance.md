# Stable release compliance

FastCompute 0.5.0 closes the implementation gaps identified during the final
review of the technical specification.

FastCompute 0.6.0 adds the optimized lazy array pipeline and process-wide
preferred-GPU configuration without changing the 0.5.0 execution contracts.

## Supported element types

`float`, `double`, and `int` support one-shot Map, Zip, in-place operations,
Sum, Min, Max, Average, explicit Scalar/Parallel CPU/SIMD/ILGPU execution,
automatic CPU selection, precompilation, prepared Map operations, and
accelerator-resident buffers.

Float expressions use `GpuMath`. Double and integer expressions use arithmetic
operators and supported `System.Math` overloads. Forced backends do not
silently fall back.

## Histogram

Histogram accepts `HistogramOptions`. Finite out-of-range values use `Clamp`
by default or can be ignored with `HistogramOutOfRangeMode.Ignore`. NaN is
always ignored. Scalar, Parallel CPU, and GPU implementations share this
contract.

## Resident buffers

`ComputeBuffer<T>` publicly exposes `Length`, `Context`, and `Location`.
`Location` distinguishes host memory owned by an ILGPU CPU accelerator from
hardware device memory.

## Lazy optimized pipelines

`AsCompute` creates an immutable lazy pipeline for `float`, `double`, and
`int` arrays. Consecutive unary selectors are composed into one expression at
`ToArray`, so backend selection, array traversal, and GPU Map execution happen
once without managed intermediate arrays. `ToArrayInPlace` is the explicit
source-mutating terminal; normal `ToArray` remains branch-safe.

## Memory pool

The transient float-buffer pool is context-local and thread-safe.
`ComputeContextOptions.MemoryPoolLimitBytes` bounds idle retained memory.
Eviction is least-recently-returned, and statistics expose retained bytes,
limit, and eviction count.

## Async-compatible API

`Compute.RunAsync`, `ComputeContext.UploadAsync`, and
`ComputeBuffer.DownloadAsync` propagate cancellation. ILGPU currently exposes
synchronous completion primitives, so these methods return completed tasks
without artificial `Task.Run` scheduling.

## Validation

- Unit and integration tests cover managed CPU, SIMD, ILGPU CPU, and CUDA.
- GPU tests force the selected NVIDIA accelerator and report its name.
- `SpecificationMatrixBenchmarks` covers the required sizes and operations
  across `for`, `Parallel.For`, SIMD, ILGPU CPU, CUDA, and Auto.
- Dedicated benchmarks compare first and repeated GPU runs and bounded versus
  disabled pool retention.
- `PipelineFusionBenchmarks` compares independent Map calls with the fused
  pipeline and reports both execution time and managed allocation.
- The package script builds, tests, packs, verifies strong-name identity, and
  runs a package-only consumer.
- CI runs the no-hardware-GPU package path on Windows and Linux.

Kernel fusion and additional GPU streaming modes remain post-0.5 features, as
allowed by the specification.
