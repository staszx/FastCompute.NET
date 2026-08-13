# AiImageForensics algorithm migration

This table is the implementation checklist for separating generic math,
reusable imaging operations, and forensic interpretation. It is not a runtime
contract.

| Method or responsibility | Current location | Target owner | Reason / dependency rule |
|---|---|---|---|
| FFT / inverse FFT / 2D FFT | AiImageForensics `Fft2D` | FastCompute | Generic signal transform; use `ComputeOptions` and native backends. |
| Distribution moments | AiImageForensics `StatisticsMath` | FastCompute | Generic numeric statistics. |
| Correlation / covariance | AiImageForensics `StatisticsMath`, `CameraAnalyzer.PairAccumulator` | FastCompute | Generic paired numeric reduction. |
| Image offset correlation | AiImageForensics `StatisticsMath` | FastCompute.ImageProcessing | Requires width, height, and image-coordinate offsets. |
| Linear regression / R squared | `NoiseAnalyzer.FitSignalModel` | FastCompute | Generic regression; forensic bin selection remains in AiImageForensics. |
| Histogram entropy | `SpatialAnalyzer` | FastCompute | Entropy of a histogram is generic math. |
| Median / percentile / block moments | `AccurateAnalyzer` | FastCompute | Generic aggregation; block policy remains forensic. |
| Magnitude and power spectrum | `FrequencyAnalyzer` | FastCompute | Generic spectral math. |
| FFT crop, DC removal, 2D Hann preparation | `FrequencyAnalyzer` | FastCompute.ImageProcessing | Operates on a two-dimensional image region. |
| Radial spectrum and image frequency bands | `FrequencyAnalyzer` | FastCompute.ImageProcessing | Interprets 2D image-frequency coordinates, but not AI evidence. |
| RGB conversion and linear luminance | `ColorMath`, `AiAnalysisContext` | FastCompute.ImageProcessing | Reusable color/image operation. |
| Residual extraction | `AiAnalysisContext` | FastCompute.ImageProcessing | Generic source-minus-low-pass operation. |
| Gradient, Laplacian, local contrast, edge map | `SpatialAnalyzer`, `AccurateAnalyzer` | FastCompute.ImageProcessing | Reusable spatial image operations. |
| Bayer pattern and parity sampling | AiImageForensics public API / `CameraAnalyzer` | FastCompute.ImageProcessing | General CFA image representation and extraction. |
| Bayer conversion and bilinear demosaicing | `CameraSimulator` | FastCompute.ImageProcessing | General camera/image pipeline. |
| Shot/read/signal-dependent noise | `CameraSimulator` | FastCompute.ImageProcessing | General image simulation. |
| Camera simulation | AiImageForensics `CameraSimulator` | FastCompute.ImageProcessing | Reusable independently of detection. |
| Noise binning and evidence | `NoiseAnalyzer` | AiImageForensics | Binning policy, thresholds, and interpretation are forensic. |
| Frequency peak interpretation / periodicity evidence | `FrequencyAnalyzer` | AiImageForensics | AI-specific thresholds and evidence. |
| CFA hypothesis comparison and evidence | `CameraAnalyzer` | AiImageForensics | Interprets reusable parity measurements. |
| Block size, scales, and aggregation policy | `AccurateAnalyzer` | AiImageForensics | Domain policy; calls lower-layer primitives. |
| Detection score, confidence, feature vector | AiImageForensics | AiImageForensics | Domain meaning must never enter lower layers. |

## Required dependency direction

```text
AiImageForensics -> FastCompute.ImageProcessing -> FastCompute
AiImageForensics -> FastCompute
```

The ImageProcessing assembly is now physically separate. Generic packed-
component SIMD and byte-composite GPU execution remain in core; image GPU
kernels and GPU-resident image buffers live in ImageProcessing and reuse the
core context through a narrow friend-assembly accelerator service. There is no
inverse `FastCompute -> FastCompute.ImageProcessing` assembly reference.
