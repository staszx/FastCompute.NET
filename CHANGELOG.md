# Changelog

All notable changes to FastCompute.NET are documented in this file.

## Unreleased

### Changed

- Lazy selector chains followed by `Sum`, `Min`, `Max`, or `Average` now fuse
  Map evaluation into the reduction for Scalar, Parallel CPU, SIMD, and GPU
  backends without materializing an intermediate mapped array.
- Map-reduction fusion supports `float`, `double`, and `int`, including chunked
  GPU execution.
- Lazy pipelines can record one binary `Zip`; selectors before and after it
  fuse into one Scalar, Parallel CPU, SIMD, or GPU Zip operation for `float`,
  `double`, and `int`, including chunked and explicit in-place execution.
- Lazy binary Zip graphs followed by `Sum`, `Min`, `Max`, or `Average` now
  evaluate and reduce in one fused operation without materializing a zipped
  array on any backend, including chunked GPU execution.

## 0.7.0 - 2026-07-30

Backend-neutral math API release.

### Added

- `ComputeMath` expression functions for Scalar CPU, Parallel CPU, SIMD, and
  GPU execution.

### Changed

- Quick Start, samples, tests, benchmarks, and architecture documentation now
  use `ComputeMath`, making it clear that math expressions do not force GPU
  execution.

### Compatibility

- The original `GpuMath` API remains fully supported as an alias of
  `ComputeMath`.

## 0.6.0 - 2026-07-30

Stable lazy-pipeline release.

### Added

- Lazy `AsCompute` array pipelines for `float`, `double`, and `int`.
- Automatic fusion of consecutive unary selectors into one optimized Map.
- Branch-safe `SelectInPlace`, explicit `ToArrayInPlace`, and `Sum`, `Min`,
  `Max`, and `Average` terminals.
- `AsCompute()` overloads for automatic execution, `ComputeOptions`, and a
  reusable `ComputeContext`.
- Process-wide `ComputeDefaults.PreferredGpuAcceleratorIndex` with
  per-operation GPU settings taking precedence.
- Pipeline correctness, CUDA integration, package smoke, and BenchmarkDotNet
  allocation/performance coverage.

### Changed

- The Quick Start now uses the optimized lazy pipeline as the primary API.

## 0.5.0 - 2026-07-28

First stable release.

### Added

- Scalar, Parallel CPU, SIMD, and ILGPU execution backends for `float[]`,
  `double[]`, and `int[]`.
- Automatic backend selection with workload thresholds and diagnostics.
- Preferred hardware GPU selection that does not force GPU execution.
- Explicit-control and LINQ-style array APIs.
- Out-of-place and in-place Map and Zip operations.
- Sum, Min, Max, Average, and equally spaced Histogram operations.
- Configurable Histogram out-of-range handling with `Clamp` and `Ignore`
  modes.
- Memory-budgeted chunked GPU execution and opt-in double-buffered streaming.
- Reusable GPU contexts, kernel precompilation, prepared operations, and
  bounded transient device-memory pooling with eviction diagnostics.
- Task-compatible `RunAsync`, `UploadAsync`, and `DownloadAsync` APIs without
  artificial thread-pool scheduling.
- Public compute-buffer context and host/device location metadata.
- GPU-resident buffers and reductions for all supported element types.
- Full BenchmarkDotNet matrix covering ILGPU CPU, CUDA, first and repeated GPU
  runs, resident memory, allocations, and memory pooling.
- Windows and Linux package validation workflow for machines without a GPU.
- Strong-name signed package assembly with public key token
  `c76a60c96d65300c`.

### Documented limitations

- Arbitrary user delegates are CPU-only; GPU expressions must use the
  supported expression subset. Float expressions use `GpuMath`; double and
  integer expressions use supported `System.Math` overloads.
- GPU streaming currently supports explicit out-of-place unary Map only.
- Automatic GPU selection remains conservative because host-device transfers
  can make CPU execution faster.
- ILGPU currently exposes synchronous completion primitives, so
  task-compatible methods return completed tasks instead of scheduling
  blocking work on the thread pool.
