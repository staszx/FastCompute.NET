# Changelog

All notable changes to FastCompute.NET are documented in this file.

## 0.5.0-alpha.1 - 2026-07-27

First public preview.

### Added

- Scalar, Parallel CPU, AVX SIMD, and ILGPU execution backends for `float[]`.
- Automatic backend selection with workload thresholds and diagnostics.
- Preferred hardware GPU selection that does not force GPU execution.
- Explicit-control and LINQ-style array APIs.
- Out-of-place and in-place Map and Zip operations.
- Sum, Min, Max, Average, and equally spaced Histogram operations.
- Memory-budgeted chunked GPU execution.
- Opt-in double-buffered GPU streaming for unary Map.
- Reusable GPU contexts, kernel precompilation, prepared operations, and
  transient device-memory pooling.
- Lazy GPU-resident execution graphs with copy-on-write in-place operations.
- BenchmarkDotNet performance and backend-comparison scenarios.

### Known limitations

- Array element support is currently limited to `float`.
- Arbitrary user delegates are CPU-only; GPU expressions must use the supported
  expression subset and `GpuMath`.
- GPU streaming currently supports explicit out-of-place unary Map only.
- Automatic GPU selection remains conservative because host-device transfers
  can make CPU execution faster.
