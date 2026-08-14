# FastCompute.NET Release Notes

## 0.8.0 - 2026-08-14

Native signal, statistics, and image processing primitives. The image and
forensics capabilities now live in their own assembly, and the generic numeric
primitives behind them moved into the core package.

### Added

- One- and two-dimensional radix-2 FFT for `Complex32[]` with allocating and
  in-place APIs (`Fft`, `FftInPlace`, `Fft2D`, `Fft2DInPlace`, inverse
  variants) on Scalar, Parallel CPU, AVX SIMD, and GPU backends.
- `Complex32` native composite value with `PowerSpectrum`,
  `MagnitudeSpectrum`, and `PhaseSpectrum` helpers; `FindPeaks`,
  `PeakToMedianRatio`, `MeanAbsoluteDifference`, `Percentile`, `Quantile`,
  `Median`, and Hann/Hamming/Blackman window functions in
  `Compute.Signal`.
- 1D and 2D convolution (`Convolve1D`, `Convolve2D`) with the same
  `ComputeOptions` contract.
- Statistics: `CalculateStatistics`, `Mean`, `Variance`, `StandardDeviation`,
  `Skewness`, `Kurtosis`, `SumOfSquares`, `Covariance`, `Correlation`,
  `AutoCorrelation`, `LinearRegression`, and `ShannonEntropy`.
- Threshold, `MinMax`, `Normalize`, and `SafeDivide` utilities.
- Homogeneous `byte`-component packed values with native SIMD layout
  load/store kernels and byte-composite GPU execution for one through four
  components.

### Changed

- Image and forensics functionality was moved into the new
  `FastCompute.ImageProcessing` assembly and NuGet package. The core
  `FastCompute` package has no image dependency; it ships with the generic
  primitives above.
- The negative image forensics pipeline now runs on the generic primitives
  instead of its own copies of FFT, statistics, convolution, Bayer handling,
  and camera simulation. See
  `docs/ai-image-forensics-algorithm-migration.md` for the ownership table.
- `Image<TPixel>` gained convolution-backed Gaussian/Sobel/Laplacian filters,
  residuals, local contrast and entropy, spectrum preparation, deterministic
  area resize, Bayer CFA sampling, and demosaicing on Scalar, Parallel CPU,
  SIMD, and GPU.

### Compatibility

- `FastCompute.ImageProcessing` 0.8.0 depends on `FastCompute` 0.8.0 and is
  strongly named with the same public key token `c76a60c96d65300c`.
- The original lazy pipeline, reduction fusion, `ComputeMath` (and the
  `GpuMath` alias), resident buffers, and chunked/streaming GPU execution
  remain unchanged.
- Explicit SIMD requests for local window entropy and phase spectrum are
  rejected instead of falling back to a hidden scalar loop; both operations
  have Scalar, Parallel CPU, and native GPU paths.