# FastCompute

FastCompute is a strongly named .NET 8 library for fast array processing. It
provides one API for single-threaded CPU, multi-threaded CPU, SIMD, and ILGPU
execution and supports `float`, `double`, and `int` arrays. It has no image
dependency; image processing ships in the separate
[`FastCompute.ImageProcessing`](#image-processing) package.

## Install

```powershell
dotnet add package FastCompute --version 0.8.1
```

The consuming project must target .NET 8 or a compatible later framework.

## Quick start

```csharp
using FastCompute;

float[] source = [0.0f, 0.5f, 1.0f, 1.5f];

float[] result = source
    .AsCompute()
    .Select(value => value * 2.0f)
    .SelectInPlace(value => value + 1.0f)
    .Select(value => ComputeMath.Sin(value))
    .ToArray();
```

`ComputeMath` describes mathematical operations that every compatible backend
can execute. Its use does not select or force GPU execution. Nothing is
executed before `ToArray`: FastCompute fuses the three selectors into one
expression, selects a backend once, and avoids intermediate managed arrays.

## Lazy optimized pipelines

`AsCompute` creates an immutable lazy pipeline for `float[]`, `double[]`, or
`int[]`:

```csharp
ComputePipeline<float> pipeline = source
    .AsCompute()
    .Select(value => value * 2.0f)
    .SelectInPlace(value => value + 1.0f)
    .Select(value => ComputeMath.Clamp(value, 0.0f, 1.0f));

// Planning, optimization, backend selection, and execution happen here.
float[] result = pipeline.ToArray();
```

Consecutive selectors are substituted into one Map expression and execute as
one Parallel/SIMD pass or one GPU Map kernel. A pipeline can record one binary
`Zip` node; selectors before and after it fuse into one Zip operation:

```csharp
float[] combined = left
    .AsCompute()
    .Select(value => value * 2.0f)
    .Zip(right, (first, second) => first + second)
    .Select(value => ComputeMath.Clamp(value, 0.0f, 1.0f))
    .ToArray();
```

The right array remains lazy by reference and must have the same length as the
left array when the terminal operation runs. A second Zip is rejected because
it would require more than two source arrays.

Reduction terminals are fused as well. A selector chain (or a binary Zip
graph) followed by `Sum`, `Min`, `Max`, or `Average` transforms values while
reducing them, without materializing an intermediate array on any backend,
including chunked GPU execution.

`SelectInPlace` allows the optimizer to reuse intermediate storage but does not
change the original managed array, so normal `ToArray` execution is safe to
branch. `ToArrayInPlace` is the explicit mutating terminal:

```csharp
float[] sameArray = source
    .AsCompute()
    .Select(value => value * 2.0f)
    .ToArrayInPlace();

Debug.Assert(ReferenceEquals(source, sameArray));
```

Available terminals are `ToArray`, `ToArrayInPlace`, `Sum`, `Min`, `Max`, and
`Average`.

## Core operations

```csharp
float[] mapped = Compute.Run(
    source,
    value => ComputeMath.Clamp(value * 1.25f, 0.0f, 1.0f));

float[] zipped = Compute.Zip(
    left,
    right,
    (x, y) => x + y); // Both arrays must have the same length.

Compute.RunInPlace(source, value => value * 2.0f + 1.0f);
Compute.ZipInPlace(target: left, right: right, (x, y) => x + y);

float sum = Compute.Sum(source);
float minimum = Compute.Min(source);
float maximum = Compute.Max(source);
float average = Compute.Average(source);

int[] histogram = Compute.Histogram(
    samples,
    binCount: 256,
    minimum: 0.0f,
    maximum: 1.0f,
    new HistogramOptions
    {
        OutOfRangeMode = HistogramOutOfRangeMode.Ignore
    });
```

The same operations are available for `double[]` and `int[]`. `Average`
returns the array element type, so integer average uses integer semantics.

## Supported element types and expressions

| Type | Map/Zip | In-place | Reductions | Histogram | Resident buffer |
| --- | --- | --- | --- | --- | --- |
| `float` | Yes | Yes | Yes | Yes | Yes |
| `double` | Yes | Yes | Yes | No | Yes |
| `int` | Yes | Yes | Yes | No | Yes |

Float expressions support arithmetic, comparisons, conditional expressions,
captured primitive constants, and these `ComputeMath` methods:

- `Abs`, `Min`, `Max`, and `Clamp`;
- `Sqrt`, `Pow`, `Exp`, `Log`, and `Log10`;
- `Sin`, `Cos`, and `Tan`;
- `Floor`, `Ceiling`, and `Round`.

The name is intentionally backend-neutral: `ComputeMath.Sin`, for example, can
run on Scalar CPU, Parallel CPU, SIMD, or GPU. The older `GpuMath` name remains
available as a compatibility alias. Double and integer expressions use
arithmetic and the applicable supported `System.Math` overloads.

Calls through captured reference objects and arbitrary .NET methods are
intentionally rejected because they cannot be translated to SIMD or GPU
instructions. For unrestricted float CPU code, use `Compute.RunDelegate` on
the Scalar or Parallel CPU backend.

## Fast Fourier transform

`Complex32` uses two contiguous single-precision components and is native to
Scalar, Parallel CPU, AVX SIMD, and GPU backends. One- and two-dimensional
radix-2 transforms support both allocating and in-place APIs:

```csharp
Complex32[] spectrum = Compute.Fft(samples, options: computeOptions);
Compute.FftInPlace(spectrum, FourierDirection.Inverse, computeOptions);

Complex32[] spectrum2D = Compute.Fft2D(pixels, width, height, options: computeOptions);
```

Forward transforms are unnormalized. Inverse transforms divide by the complete
element count, so a forward/inverse pair reconstructs the original input.
Dimensions must be positive powers of two.

## Signal processing, statistics, and convolution

```csharp
float[] power = Compute.PowerSpectrum(spectrum);
float[] magnitudes = Compute.MagnitudeSpectrum(spectrum);
float[] phases = Compute.PhaseSpectrum(spectrum); // Scalar, Parallel CPU, or GPU
SignalPeak[] peaks = Compute.FindPeaks(values, minimumValue: 0.5f);

float[] smoothed = Compute.Convolve1D(values, kernel, ConvolutionBoundary.Clamp);
float[] windowed = Compute.ApplyHannWindow(values);
```

Phase spectrum uses `Atan2`, which has no SIMD instruction in the expression
IR, so it runs on Scalar, Parallel CPU, or GPU and rejects explicit SIMD.

Statistics cover moments, covariance, correlation, and regression:

```csharp
StatisticsResult moments = Compute.CalculateStatistics(values);
double covariance = Compute.Covariance(x, y);
double correlation = Compute.Correlation(x, y);
LinearRegressionResult regression = Compute.LinearRegression(x, y);
double entropy = Compute.ShannonEntropy(histogram);
```

Percentile/quantile/median sort a copy of the input because FastCompute does
not yet expose a backend-native ordering primitive.

Thresholding and normalization are first-class as well:

```csharp
float[] binary = Compute.Threshold(values, threshold: 0.5f);
MinMaxResult range = Compute.MinMax(values);
float[] unitRange = Compute.Normalize(values);
float[] safe = Compute.SafeDivide(numerator, denominator, zeroResult: 0.0f);
```

## Composite values

Unmanaged structures can opt into FastCompute by implementing
`IComputeValue<T>`. Descriptors validate a tightly packed homogeneous layout of
`float` or `byte` components. Float-component transformations, including
transformations between different structures, run on Scalar, Parallel CPU,
SIMD, and GPU. Homogeneous `byte`-component values with one through four
components have native SIMD layout load/store kernels and GPU execution.

## Choosing a backend

```csharp
var options = new ComputeOptions
{
    Backend = ComputeBackendKind.Auto,
    MaxDegreeOfParallelism = Environment.ProcessorCount
};
```

| Backend | Behavior |
| --- | --- |
| `Auto` | Selects a compatible backend using expression complexity, array size, transfer cost, and the GPU memory budget. |
| `Scalar` | Runs a conventional single-threaded CPU loop. |
| `ParallelCpu` | Splits work into CPU chunks and processes them on multiple threads. |
| `Simd` | Uses hardware-accelerated CPU vectors and a scalar tail. |
| `Gpu` | Executes through ILGPU on the selected accelerator. |

SIMD is not another form of `Parallel.For`: it processes several values per
CPU instruction on the calling thread. Explicitly selected backends never
silently fall back; use `Auto` when fallback is required.

## GPU execution

Create a reusable context, precompile kernels, and keep multi-step pipelines
resident on the accelerator:

```csharp
using ComputeContext gpu = ComputeContext.Create(
    new ComputeContextOptions
    {
        AcceleratorIndex = preferredGpu.Index
    });

gpu.PrecompileAll();

using ComputeBuffer<float> input = gpu.Upload(source);
using ComputeBuffer<float> scaled =
    input.Select(value => value * 0.75f);
using ComputeBuffer<float> transformed =
    scaled.Select(value => ComputeMath.Sin(value));

float sum = transformed.Sum();
float[] output = new float[transformed.Length];
transformed.Download(output);
```

- `ComputeDefaults.PreferredGpuAcceleratorIndex` sets a process-wide default
  GPU for Auto and explicit GPU operations.
- Arrays larger than available GPU memory run in sequential chunks
  (`GpuChunkElementCount`, `GpuMemoryBudgetBytes`). Opt-in
  `EnableGpuStreaming = true` overlaps transfers and execution for explicit
  out-of-place float Map.
- `Precompile<T>` and `Prepare<T>` move kernel compilation out of the first
  business operation.

## Diagnostics and async

```csharp
ComputeResult<float[]> result = Compute.RunWithDiagnostics(
    source,
    value => value * value,
    new ComputeOptions { Backend = ComputeBackendKind.Auto });

ComputeDiagnostics d = result.Diagnostics;
Console.WriteLine($"Backend:       {d.Backend}");
Console.WriteLine($"Device:        {d.DeviceName ?? "CPU"}");
Console.WriteLine($"Planning:      {d.PlanningTime}");
Console.WriteLine($"Execution:     {d.ExecutionTime}");
Console.WriteLine($"Upload bytes:  {d.UploadedBytes}");
Console.WriteLine($"Download bytes:{d.DownloadedBytes}");
Console.WriteLine($"Chunks:        {d.ChunkCount}");
```

```csharp
using var cancellationSource = new CancellationTokenSource();

float[] mapped = await Compute.RunAsync(
    source,
    value => value * 2.0f,
    new ComputeOptions
    {
        CancellationToken = cancellationSource.Token
    });
```

ILGPU currently exposes synchronous completion primitives, so async methods
return completed tasks and do not hide blocking work in `Task.Run`.

## Common problems

- **GPU is slower than Parallel CPU.** Expected when transfer and compilation
  costs exceed the kernel work. Reuse a context, precompile, keep intermediate
  data in resident buffers, or let Auto choose CPU.
- **An explicit backend throws instead of using CPU.** This is the intended
  contract. Use Auto when fallback is required.
- **An expression cannot be translated.** Use supported arithmetic and math
  methods, or use `RunDelegate` for unrestricted float CPU code.
- **The first GPU call is slow.** The first call includes expression lowering
  and kernel compilation. Reuse the context and call `PrecompileAll`,
  `Precompile<T>`, or `Prepare<T>` during warm-up.

## Image processing

Native image formats, filters, Bayer CFA handling, camera simulation, and
GPU-resident image buffers are not part of this package. Install
[`FastCompute.ImageProcessing`](https://github.com/staszx/FastCompute.NET)
when an application needs `Image<TPixel>` and friends.

## Further documentation

- [Repository README](https://github.com/staszx/FastCompute.NET)
- [FastCompute.ImageProcessing package](https://github.com/staszx/FastCompute.NET/blob/main/src/FastCompute.ImageProcessing/README.md)
- [Release notes](https://github.com/staszx/FastCompute.NET/blob/main/RELEASE_NOTES.md)
- [Release history and known limitations](https://github.com/staszx/FastCompute.NET/blob/main/CHANGELOG.md)
- [Architecture documents](https://github.com/staszx/FastCompute.NET/blob/main/docs/)
