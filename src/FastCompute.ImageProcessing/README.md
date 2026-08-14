# FastCompute.ImageProcessing

FastCompute.ImageProcessing is a strongly named .NET 8 library for
backend-neutral native image processing. It depends on the
[`FastCompute`](https://github.com/staszx/FastCompute.NET) core package and
shares its Scalar CPU, Parallel CPU, SIMD, and GPU execution model.

## Install

```powershell
dotnet add package FastCompute.ImageProcessing --version 0.8.1
```

The package depends on FastCompute 0.8.1, so the core is restored
automatically. Both assemblies are strongly named with the public key token
`c76a60c96d65300c`.

## Pixel formats

The package provides `Rgb24` (tightly packed 24-bit RGB), floating-point
`Rgb`, `Gray8`, and `GrayF32`, together with separate `Srgb`/`Linear` encoding
metadata.

```csharp
using FastCompute;
using FastCompute.ImageProcessing;

Image<Rgb24> decoded = Image<Rgb24>.Load(pixels, width, height);
Image<Rgb24> view = Image<Rgb24>.Wrap(memory, width, height); // zero-copy
```

`Image<TPixel>` owns an array or wraps contiguous `Memory<TPixel>` without
copying. It provides row spans, `CopyRow`, `Clone`, and `Crop`.

## Conversions

```csharp
Image<GrayF32> linear = decoded.ToGrayscaleF32(ColorEncoding.Linear);
Image<Gray8> compact = decoded.ToGrayscale8();
Image<Rgb> wide = decoded.ToRgbF32(ColorEncoding.Linear);
Image<Rgb24> roundTrip = linear.ToRgb24(ColorEncoding.Srgb);
```

`ToLinear` and `ToSrgb` convert nonlinear/linear light in place of the
supported value types.

## Filters and spatial operations

All filters accept the same `ComputeOptions` contract as the array API:

```csharp
Image<GrayF32> lowPass = linear.BoxBlur(radius: 1);
Image<GrayF32> gaussian = linear.GaussianBlur(radius: 2, sigma: 1.5f);
Image<GrayF32> edges = linear.Sobel();
Image<GrayF32> laplacian = linear.Laplacian();
Image<GrayF32> residual = linear.Subtract(lowPass);
Image<GrayF32> resized = linear.Resize(width: 1024, height: 768);
Image<GrayF32> downsampled = linear.Downsample(width: 256, height: 256);
```

CPU area downsampling vectorizes accumulation across each source interval.
GPU box blur uses parallel per-pixel kernels for small radii and switches to a
linear-time sliding-window pass for radii greater than four. Local contrast,
local window entropy, spectrum preparation, and radial spectra are available
for analysis-style pipelines.

## Bayer CFA and camera simulation

```csharp
float[] mosaic = rgb.ToBayer(BayerPattern.Rggb);
Image<Rgb> demosaiced = mosaicImage.DemosaicBilinear();
Image<Rgb> reconstructed = mosaicImage.Demosaic();

rgb.SimulateCamera(new CameraSimulationOptions
{
    ShotNoise = 0.002f,
    ReadNoise = 0.0005f,
    OpticalBlur = 0.5f,
    Sharpening = 0.15f,
    RandomSeed = 1
});
```

Camera simulation is intended for robustness testing. It implements optical
blur, signal-dependent/read noise, Bayer sampling, basic demosaicing, and
sharpening.

## GPU execution

Explicit GPU execution is available for conversions, transfer functions,
convolution, box blur, subtraction, resize/downsampling, gradients, local
contrast/entropy, radial spectra, Bayer/demosaicing, and noise application:

```csharp
var gpuOptions = new ComputeOptions
{
    Backend = ComputeBackendKind.Gpu,
    GpuContext = gpu
};

Image<GrayF32> linearGpu = decoded.ToGrayscaleF32(
    ColorEncoding.Linear,
    gpuOptions);
Image<GrayF32> lowPassGpu = linearGpu.BoxBlur(
    radius: 1,
    options: gpuOptions);
```

`Auto` follows the normal FastCompute thresholds. Host-backed image operations
use `GpuSimpleThreshold`, which is disabled by default because host/device
transfer often costs more than CPU SIMD. Explicit
`Backend = ComputeBackendKind.Gpu` always requests the GPU path.

For multi-stage GPU processing, upload once and keep intermediate images on
the accelerator:

```csharp
using ImageBuffer<Rgb24> resident = decoded.UploadToGpu(gpu);
using ImageBuffer<GrayF32> luminance = resident.ToGrayscaleF32(
    ColorEncoding.Linear);
using ImageBuffer<GrayF32> blur = luminance.BoxBlur(radius: 1);
using ImageBuffer<GrayF32> residual = luminance.Subtract(blur);

Image<GrayF32> result = residual.Download();
```

`ImageBuffer<TPixel>` owns its device allocation and must be disposed.
Conversion, blur, subtraction, resize, and downsampling operate
device-to-device; only `UploadToGpu` and `Download` cross the host/device
boundary.

## Limitations

- Explicit SIMD requests for local window entropy and phase spectrum are
  rejected rather than executed by a hidden scalar loop; both operations have
  Scalar, Parallel CPU, and native GPU paths.
- Nonlinear `Srgb`/`Linear` transfer conversion must currently use Scalar,
  Parallel CPU, or GPU.
- Percentile/quantile use the runtime in-place sort because FastCompute does
  not yet expose a backend-native ordering primitive.
- Chromatic aberration and vignetting camera simulation options are reserved
  for a future implementation.

## Further documentation

- [Repository README](https://github.com/staszx/FastCompute.NET)
- [FastCompute core package](https://github.com/staszx/FastCompute.NET/blob/main/src/FastCompute/README.md)
- [Release notes](https://github.com/staszx/FastCompute.NET/blob/main/RELEASE_NOTES.md)
- [Release history and known limitations](https://github.com/staszx/FastCompute.NET/blob/main/CHANGELOG.md)