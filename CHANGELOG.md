# Changelog

All notable changes to FastCompute.NET are documented in this file.

## 0.8.0 - 2026-08-14

Native signal, statistics, and image processing primitives release.

### Added

- One- and two-dimensional radix-2 FFT for `Complex32[]` with allocating and
  in-place APIs (`Fft`, `FftInPlace`, `Fft2D`, `Fft2DInPlace`, and inverse
  variants) on Scalar, Parallel CPU, AVX SIMD, and GPU backends.
- `Complex32` as a native composite value with `PowerSpectrum`,
  `MagnitudeSpectrum`, `PhaseSpectrum`, `FindPeaks`, `PeakToMedianRatio`,
  `MeanAbsoluteDifference`, `Percentile`, `Quantile`, `Median`, and window
  functions.
- 1D and 2D convolution (`Convolve1D`, `Convolve2D`) through the same
  `ComputeOptions` contract.
- Statistics primitives: `CalculateStatistics`, `Mean`, `Variance`,
  `StandardDeviation`, `Skewness`, `Kurtosis`, `SumOfSquares`, `Covariance`,
  `Correlation`, `AutoCorrelation`, `LinearRegression`, and `ShannonEntropy`.
- `Threshold` with SIMD/GPU paths, plus `MinMax`, `Normalize`, and
  `SafeDivide` utilities.
- Native SIMD layout load/store kernels and GPU execution for homogeneous
  `byte`-component values with one through four packed components.
- The separate `FastCompute.ImageProcessing` assembly and NuGet package with
  `Rgb24`, `Rgb`, `Gray8`, `GrayF32`, convolution-backed
  Gaussian/Sobel/Laplacian filters, residuals, local contrast and entropy,
  spectrum preparation, deterministic area resize, Bayer CFA sampling,
  demosaicing, camera simulation, and GPU-resident image buffers.
- Lazy selector chains and binary Zip graphs fused with `Sum`, `Min`, `Max`,
  or `Average` reductions without materializing intermediate arrays on every
  backend, including chunked GPU execution.

### Changed

- AI image forensics migrated from private FFT, statistics, convolution,
  Bayer, and camera-simulation implementations to the generic core and image
  processing primitives. The ownership checklist is in
  `docs/ai-image-forensics-algorithm-migration.md`.
- `pack.ps1` now produces both `FastCompute` and
  `FastCompute.ImageProcessing` `.nupkg`/`.snupkg` artifacts, verifies the
  strong-name identity of both assemblies, and the package smoke test consumes
  both packages.
- The Quick Start, samples, and package metadata reference version 0.8.0.

### Compatibility

- The core `FastCompute` package has no image dependency.
- `FastCompute.ImageProcessing` 0.8.0 depends on `FastCompute` 0.8.0 and
  shares the public key token `c76a60c96d65300c`.
- `GpuMath` remains a fully supported alias of `ComputeMath`.
- Local window entropy and phase spectrum have no SIMD implementation;
  explicit SIMD requests are rejected rather than executed by a hidden scalar
  loop. Both operations support Scalar, Parallel CPU, and GPU.
- Percentile/quantile use the runtime in-place sort because FastCompute does
  not yet expose a backend-native ordering primitive.

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
