# FastCompute.NET

FastCompute.NET is a strongly named .NET 8 library for fast array processing.
It provides one API for single-threaded CPU, multi-threaded CPU, SIMD, and
ILGPU execution and supports `float`, `double`, and `int` arrays.

Version `0.7.0` is the current stable release. The assembly public key token is
`c76a60c96d65300c`.

## Quick start

### 1. Install the package

```powershell
dotnet add package FastCompute --version 0.7.0
```

The consuming project must target .NET 8 or a compatible later framework.

### 2. Build and execute an optimized pipeline

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
can execute. Its use does not select or force GPU execution.

Nothing is executed before `ToArray`. FastCompute fuses the three selectors
into one expression, selects a backend once, and avoids intermediate managed
arrays.

`AsCompute()` uses Auto by default. FastCompute evaluates the complete
optimized expression and array size and then selects Scalar CPU, Parallel CPU,
SIMD, or GPU. Auto is transfer-conservative: merely having a GPU does not mean
that every pipeline is sent to it.

### 3. Supply a reusable GPU context

```csharp
ComputeDeviceInfo gpuDevice = ComputeContext.GetAccelerators()
    .First(device => !device.AcceleratorType.Contains(
        "CPU",
        StringComparison.OrdinalIgnoreCase));

using ComputeContext context = ComputeContext.Create(
    new ComputeContextOptions
    {
        AcceleratorIndex = gpuDevice.Index
    });

float[] result = source
    .AsCompute(context)
    .Select(value => value * 2.0f)
    .Select(value => ComputeMath.Sin(value))
    .ToArray();
```

Passing a context makes that accelerator available to Auto and reuses its
compiled kernels. It still does not force GPU execution.

### 4. Force the selected GPU when required

```csharp
float[] gpuResult = source
    .AsCompute(
        new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context
        })
    .Select(value => value * 2.0f)
    .Select(value => ComputeMath.Sin(value))
    .ToArray();
```

An explicitly selected backend is a strict request and never silently falls
back. If the expression or machine does not support that backend, the operation
throws an exception.

### 5. Call an existing custom method

Expressions intended for SIMD or GPU must use the supported expression subset.
If the transformation needs unrestricted .NET code, use `RunDelegate` and
select Scalar or Parallel CPU:

```csharp
float GetApplicationCoefficient(float value) =>
    value < 0.5f ? 0.25f : 0.75f;

float CustomTransform(float value) =>
    MathF.Sin(value) + GetApplicationCoefficient(value);

float[] result = Compute.RunDelegate(
    source,
    CustomTransform,
    new ComputeOptions
    {
        Backend = ComputeBackendKind.ParallelCpu
    });
```

## Detailed guide

### Lazy optimized pipelines

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

At the terminal operation, the pipeline optimizer substitutes each selector
into the next selector. Existing expression optimization then performs
constant folding and IEEE-safe simplification. The resulting expression is
executed as one Map operation, which means one Parallel/SIMD pass or one GPU
Map kernel instead of one pass per `Select`.

A pipeline can record one binary `Zip` node. Selectors before and after it are
substituted into one binary expression and execute as one Zip operation:

```csharp
float[] combined = left
    .AsCompute()
    .Select(value => value * 2.0f)
    .Zip(right, (first, second) => first + second)
    .Select(value => ComputeMath.Clamp(value, 0.0f, 1.0f))
    .ToArray();
```

The right array remains lazy by reference and must have the same length as the
left array when the terminal operation runs. `ToArrayInPlace` explicitly
stores the fused Zip result in the left source array. A second Zip is rejected
because it would require a multi-input graph with more than two source arrays.

Reduction terminals are fused as well. A selector chain followed by `Sum`,
`Min`, `Max`, or `Average` transforms values while reducing them, without
materializing a full-size mapped array. GPU execution applies the selector
expression in the first reduction kernel stage, including chunked execution.
Reduction after a Zip graph currently consumes the fused Zip result as a
separate operation.

Pipelines can be configured in three ways:

```csharp
// Auto with library defaults.
ComputePipeline<float> automatic = source.AsCompute();

// Auto or explicit backend with operation settings.
ComputePipeline<float> configured = source.AsCompute(
    new ComputeOptions
    {
        Backend = ComputeBackendKind.Auto,
        PreferredGpuAcceleratorIndex = preferredGpu.Index
    });

// Auto selection with a reusable GPU context when GPU wins.
ComputePipeline<float> withContext = source.AsCompute(context);
```

`SelectInPlace` allows the optimizer to reuse intermediate storage but does not
change the original managed array. This makes normal `ToArray` execution safe
to branch:

```csharp
ComputePipeline<float> root = source.AsCompute();
ComputePipeline<float> doubled =
    root.SelectInPlace(value => value * 2.0f);
ComputePipeline<float> shifted =
    root.Select(value => value + 10.0f);

float[] first = doubled.ToArray();
float[] second = shifted.ToArray();
```

To explicitly overwrite `source`, use the mutating terminal:

```csharp
float[] sameArray = source
    .AsCompute()
    .Select(value => value * 2.0f)
    .SelectInPlace(value => value + 1.0f)
    .ToArrayInPlace();

Debug.Assert(ReferenceEquals(source, sameArray));
```

Available terminals are `ToArray`, `ToArrayInPlace`, `Sum`, `Min`, `Max`, and
`Average`. Selector fusion is currently unary. A reduction after selectors
uses the optimized Map result and then the selected reduction backend;
map-reduction kernel fusion is reserved for a later version.

### Core operations

#### Map

`Run` applies a unary expression to every array element and returns a new
array:

```csharp
float[] mapped = Compute.Run(
    source,
    value => ComputeMath.Clamp(value * 1.25f, 0.0f, 1.0f));
```

#### Zip

`Zip` combines arrays element by element:

```csharp
float[] left = [1.0f, 2.0f, 3.0f];
float[] right = [10.0f, 20.0f, 30.0f];

float[] zipped = Compute.Zip(
    left,
    right,
    (x, y) => x + y);
```

Both arrays must have the same length.

#### In-place processing

Use the in-place variants when the result may overwrite the input array:

```csharp
Compute.RunInPlace(
    source,
    value => value * 2.0f + 1.0f);

Compute.ZipInPlace(
    target: left,
    right: right,
    (x, y) => x + y);
```

The returned reference is the same array as the target. This reduces managed
allocations, but cancellation or an execution failure can leave an in-place
array partially modified.

#### Reductions

```csharp
float sum = Compute.Sum(source);
float minimum = Compute.Min(source);
float maximum = Compute.Max(source);
float average = Compute.Average(source);
```

The same operations are available for `double[]` and `int[]`. `Average` returns
the array element type, so integer average uses integer semantics.

### Supported element types and expressions

| Type | Map/Zip | In-place | Reductions | Resident buffer | Histogram |
| --- | --- | --- | --- | --- | --- |
| `float` | Yes | Yes | Yes | Yes | Yes |
| `double` | Yes | Yes | Yes | Yes | No |
| `int` | Yes | Yes | Yes | Yes | No |

Float expressions support arithmetic, comparisons, conditional expressions,
captured primitive constants, and these `ComputeMath` methods:

- `Abs`, `Min`, `Max`, and `Clamp`;
- `Sqrt`, `Pow`, `Exp`, `Log`, and `Log10`;
- `Sin`, `Cos`, and `Tan`;
- `Floor`, `Ceiling`, and `Round`.

The name is intentionally backend-neutral: `ComputeMath.Sin`, for example, can
run on Scalar CPU, Parallel CPU, SIMD, or GPU. The older `GpuMath` name remains
available as a compatibility alias.

```csharp
float multiplier = 0.75f;
bool clamp = true;

float[] result = Compute.Run(
    source,
    value => clamp
        ? ComputeMath.Clamp(ComputeMath.Sin(value) * multiplier, 0.0f, 1.0f)
        : ComputeMath.Sin(value) * multiplier);
```

Double and integer expressions use arithmetic and the applicable supported
`System.Math` overloads:

```csharp
double[] precise = Compute.Run(
    new[] { 0.0, 0.5, 1.0 },
    value => Math.Sin(value) * Math.Exp(-value));

int[] integers = Compute.Run(
    new[] { 1, 2, 3 },
    value => value * 2 + 1);
```

The expression API converts an expression tree to FastCompute's
backend-independent instruction representation. Captured `float`, `double`,
`int`, and `bool` values are supported. Calls through captured reference
objects and arbitrary .NET methods are intentionally rejected because they
cannot be translated to SIMD or GPU instructions.

### Arbitrary user methods

When a transformation contains unrestricted application code, use
`RunDelegate`. It executes a normal `Func<float, float>` on Scalar or Parallel
CPU without translating it:

```csharp
float GetApplicationCoefficient(float value) =>
    value < 0.5f ? 0.25f : 0.75f;

float CustomCalculation(float value) =>
    MathF.Sin(value) + GetApplicationCoefficient(value);

float[] result = Compute.RunDelegate(
    source,
    CustomCalculation,
    new ComputeOptions
    {
        Backend = ComputeBackendKind.ParallelCpu
    });
```

`RunDelegate` currently supports only `float[]` and the Scalar and Parallel CPU
backends. It cannot execute arbitrary CLR methods on SIMD or GPU hardware.

### Choosing a backend

```csharp
var options = new ComputeOptions
{
    Backend = ComputeBackendKind.Auto,
    MaxDegreeOfParallelism = Environment.ProcessorCount
};
```

The available modes are:

| Backend | Behavior |
| --- | --- |
| `Auto` | Selects a compatible backend using expression complexity, array size, transfer cost, and the GPU memory budget. |
| `Scalar` | Runs a conventional single-threaded CPU loop. |
| `ParallelCpu` | Splits work into CPU chunks and processes them on multiple threads. |
| `Simd` | Uses hardware-accelerated CPU vectors and a scalar tail. |
| `Gpu` | Executes through ILGPU on the selected accelerator. |

SIMD is not another form of `Parallel.For`: it processes several values per CPU
instruction on the calling thread. Parallel CPU and SIMD are separate
backends.

Auto mode is appropriate when the library should make the decision. Use an
explicit backend for reproducible performance tests, a known deployment
environment, or full user control:

```csharp
float[] result = source.RunExplicit(
    value => value * value,
    ComputeBackendKind.ParallelCpu);
```

`RunExplicit` is a LINQ-style extension for `float[]`, `double[]`, and `int[]`.
It rejects `Auto` by design.

### Selecting one of several GPUs

Accelerator indices are assigned by ILGPU and can include CPU accelerators.
Always discover them on the target machine:

```csharp
using FastCompute;

IReadOnlyList<ComputeDeviceInfo> accelerators =
    ComputeContext.GetAccelerators();

foreach (ComputeDeviceInfo device in accelerators)
{
    Console.WriteLine(
        $"{device.Index}: {device.Name} ({device.AcceleratorType})");
}
```

To prefer a particular hardware GPU without forcing its use, pass its index to
Auto mode:

```csharp
ComputeDeviceInfo preferredGpu = ComputeContext.GetAccelerators()
    .First(device => !device.AcceleratorType.Contains(
        "CPU",
        StringComparison.OrdinalIgnoreCase));

ComputeResult<float[]> result = Compute.RunWithDiagnostics(
    source,
    value => ComputeMath.Sin(value) * ComputeMath.Exp(-value * value),
    new ComputeOptions
    {
        Backend = ComputeBackendKind.Auto,
        PreferredGpuAcceleratorIndex = preferredGpu.Index
    });
```

`PreferredGpuAcceleratorIndex` means “use this GPU if the planner decides that
GPU execution is beneficial.” Auto may still select SIMD, Parallel CPU, or
Scalar. If the preferred index is unavailable or does not identify a hardware
GPU, Auto continues with a CPU backend.

To require that accelerator, create a reusable context and explicitly request
GPU execution:

```csharp
using ComputeContext gpu = ComputeContext.Create(
    new ComputeContextOptions
    {
        AcceleratorIndex = preferredGpu.Index
    });

Console.WriteLine($"Selected accelerator: {gpu.DeviceName}");

float[] gpuResult = Compute.Run(
    source,
    value => ComputeMath.Sin(value),
    new ComputeOptions
    {
        Backend = ComputeBackendKind.Gpu,
        GpuContext = gpu
    });
```

`GpuContext` and `PreferredGpuAcceleratorIndex` are mutually exclusive. A
context created without an index prefers a non-CPU accelerator and falls back
to the ILGPU CPU accelerator if no hardware GPU is available.

### Setting the default preferred GPU

For an application-wide preference, set the accelerator once during startup:

```csharp
ComputeDeviceInfo preferredGpu = ComputeContext.GetAccelerators()
    .First(device => !device.AcceleratorType.Contains(
        "CPU",
        StringComparison.OrdinalIgnoreCase));

ComputeDefaults.PreferredGpuAcceleratorIndex = preferredGpu.Index;
```

The default applies to Auto and explicit GPU operations that do not provide
their own GPU setting:

```csharp
// Considers the default GPU, but can still select CPU or SIMD.
float[] automatic = Compute.Run(
    source,
    value => ComputeMath.Sin(value));

// Requires GPU and uses the default preferred accelerator.
float[] forcedGpu = Compute.Run(
    source,
    value => ComputeMath.Sin(value),
    new ComputeOptions
    {
        Backend = ComputeBackendKind.Gpu
    });
```

The precedence order is:

1. `ComputeOptions.GpuContext` for the current operation;
2. `ComputeOptions.PreferredGpuAcceleratorIndex` for the current operation;
3. `ComputeDefaults.PreferredGpuAcceleratorIndex`;
4. FastCompute's automatic accelerator selection.

Set `ComputeDefaults.PreferredGpuAcceleratorIndex = null` to restore automatic
selection. The property is process-wide; configure it during application
startup rather than changing it between concurrent operations.

### When GPU kernels are compiled

For the first operation in a `ComputeContext`, FastCompute:

1. validates and lowers the expression;
2. obtains or compiles the required ILGPU kernel template;
3. caches the lowered expression and compiled template in that context;
4. uploads data, starts the kernel, and downloads the result.

Repeated compatible operations on the same context reuse those caches. This is
why the first GPU call is normally slower than subsequent calls. A new context
has its own caches.

Use `PrecompileAll` during application warm-up to compile every implemented
kernel template:

```csharp
using ComputeContext gpu = ComputeContext.Create(
    new ComputeContextOptions
    {
        AcceleratorIndex = preferredGpu.Index
    });

IReadOnlyList<ComputeCompilationResult> templates = gpu.PrecompileAll();

Console.WriteLine(
    $"Prepared: {templates.Count}; " +
    $"cache hits: {templates.Count(item => item.CacheHit)}");
```

Use `Precompile<T>` to validate, lower, and cache a particular expression:

```csharp
ComputeCompilationResult compilation =
    gpu.Precompile<float>(
        value => ComputeMath.Sin(value) * ComputeMath.Exp(-value));

Console.WriteLine($"Cache hit: {compilation.CacheHit}");
Console.WriteLine($"Compile time: {compilation.CompilationTime}");
```

For an operation that will be called repeatedly, create a prepared operation:

```csharp
PreparedCompute<float> prepared =
    gpu.Prepare<float>(value => ComputeMath.Sin(value) * 2.0f);

float[] first = prepared.Run(source);
float[] second = prepared.Run(otherSource);
```

Precompilation removes kernel compilation from the first business operation.
It does not upload the future input array and cannot remove normal GPU transfer
cost.

### Arrays larger than available GPU memory

GPU Map, Zip, in-place operations, reductions, and Histogram can run in
sequential chunks. Chunking is enabled by default and the effective working
set is limited by the context safety limit and optional operation budget:

```csharp
float[] result = Compute.Run(
    source,
    value => ComputeMath.Sin(value),
    new ComputeOptions
    {
        Backend = ComputeBackendKind.Gpu,
        GpuContext = gpu,
        GpuMemoryBudgetBytes = 512L * 1024 * 1024,
        GpuChunkElementCount = 4_000_000
    });
```

`GpuChunkElementCount` is an optional upper bound. Without it, FastCompute
calculates a chunk size from the memory budget. Setting
`EnableGpuChunking = false` makes insufficient memory a hard error.

Explicit out-of-place float Map can optionally overlap transfers and execution
using two accelerator streams:

```csharp
float[] result = Compute.Run(
    source,
    value => ComputeMath.Sin(value),
    new ComputeOptions
    {
        Backend = ComputeBackendKind.Gpu,
        GpuContext = gpu,
        GpuChunkElementCount = 4_000_000,
        EnableGpuStreaming = true
    });
```

Streaming is opt-in because its benefit depends on the accelerator, bus, and
expression. Other operations use sequential chunks.

### Keeping data on the accelerator

For several consecutive GPU operations, upload once and use
`ComputeBuffer<T>`:

```csharp
using ComputeBuffer<float> input = gpu.Upload(source);
using ComputeBuffer<float> scaled =
    input.Select(value => value * 0.75f);
using ComputeBuffer<float> transformed =
    scaled.Select(value => ComputeMath.Sin(value));

float sum = transformed.Sum();

float[] output = new float[transformed.Length];
transformed.Download(output);

Console.WriteLine(transformed.Context.DeviceName);
Console.WriteLine(transformed.Location); // Host or Device
```

Available resident operations include `Select`, `SelectInPlace`, `Zip`,
`ZipInPlace`, `Sum`, `Min`, `Max`, `Average`, and download to an array or
`Span<T>`. Chained float selections use a lazy, copy-on-write execution graph
where applicable. Dispose buffers and their context to release accelerator
resources.

The context also owns a bounded, thread-safe transient float-buffer pool:

```csharp
using ComputeContext gpu = ComputeContext.Create(
    new ComputeContextOptions
    {
        AcceleratorIndex = preferredGpu.Index,
        MemoryPoolLimitBytes = 256L * 1024 * 1024
    });

Console.WriteLine(gpu.MemoryPoolStatistics.RetainedBytes);
Console.WriteLine(gpu.MemoryPoolStatistics.EvictedBuffers);
```

The limit controls idle memory retained for reuse. Active operations may use
more. Set `MemoryPoolLimitBytes = 0` to disable idle-buffer retention.

### Histogram

Histogram splits a numeric range into equal-width bins:

```csharp
float[] samples = [0.0f, 0.1f, 0.5f, 0.9f, 1.0f];

int[] histogram = Compute.Histogram(
    samples,
    binCount: 256,
    minimum: 0.0f,
    maximum: 1.0f);
```

Finite values outside the range are clamped to the first or last bin by
default. `NaN` is always ignored. To ignore all out-of-range values:

```csharp
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

Histogram supports Scalar, Parallel CPU, and GPU. Automatic GPU selection is
opt-in through `ComputeThresholdOptions.GpuHistogramThreshold`.

### Diagnostics

Diagnostic APIs return the value together with planning and execution details:

```csharp
ComputeResult<float[]> result = Compute.RunWithDiagnostics(
    source,
    value => value * value,
    new ComputeOptions
    {
        Backend = ComputeBackendKind.Auto
    });

ComputeDiagnostics d = result.Diagnostics;

Console.WriteLine($"Backend:       {d.Backend}");
Console.WriteLine($"Device:        {d.DeviceName ?? "CPU"}");
Console.WriteLine($"Planning:      {d.PlanningTime}");
Console.WriteLine($"Compilation:   {d.CompilationTime}");
Console.WriteLine($"Execution:     {d.ExecutionTime}");
Console.WriteLine($"Upload bytes:  {d.UploadedBytes}");
Console.WriteLine($"Download bytes:{d.DownloadedBytes}");
Console.WriteLine($"Chunks:        {d.ChunkCount}");
Console.WriteLine($"Streaming:     {d.IsStreaming}");
Console.WriteLine($"Cache hit:     {d.KernelCacheHit}");
```

Variants include `RunWithDiagnostics`, `RunInPlaceWithDiagnostics`,
`ZipWithDiagnostics`, `ZipInPlaceWithDiagnostics`,
`HistogramWithDiagnostics`, and diagnostic reduction methods such as
`SumWithDiagnostics`.

### Async-compatible API and cancellation

```csharp
using var cancellationSource = new CancellationTokenSource();

float[] mapped = await Compute.RunAsync(
    source,
    value => value * 2.0f,
    new ComputeOptions
    {
        CancellationToken = cancellationSource.Token
    });

using ComputeBuffer<float> buffer =
    await gpu.UploadAsync(source, cancellationSource.Token);

float[] downloaded =
    await buffer.DownloadAsync(cancellationSource.Token);
```

ILGPU currently exposes synchronous completion primitives. These methods
therefore return completed tasks and do not hide blocking work in `Task.Run`.
They are async-compatible API boundaries, not guaranteed non-blocking GPU
execution.

### Performance guidance

- Use Auto for general-purpose calls, but verify the selected backend with
  diagnostics for important workloads.
- A direct loop is normally best for very small arrays.
- GPU execution is most useful when enough computation compensates for upload
  and download cost.
- Reuse `ComputeContext` to reuse compiled kernels.
- Use resident buffers for multi-step GPU pipelines.
- Use in-place methods when overwriting the source is acceptable.
- Benchmark on the deployment hardware; GPU model, memory bandwidth, CPU SIMD,
  and array size all affect the result.

The opt-in performance gate compares Auto with an equivalent single-threaded
loop on large simple, heavy, and in-place Map workloads:

```powershell
dotnet run --project benchmarks/FastCompute.Benchmarks `
  --configuration Release -- `
  --assert-performance
```

It exits with code `1` if FastCompute is more than 5% slower than the loop. Run
it on otherwise idle hardware. The complete backend and operation matrix can
be run with:

```powershell
dotnet run --project benchmarks/FastCompute.Benchmarks `
  --configuration Release -- `
  --filter "*SpecificationMatrixBenchmarks*"
```

### Common problems

**GPU is slower than Parallel CPU.**

This is expected when transfer and compilation costs exceed the kernel work.
Reuse a context, precompile, keep intermediate data in resident buffers, or let
Auto choose CPU.

**The wrong GPU was selected.**

Print `ComputeContext.GetAccelerators()`, select the runtime index, and either
set `PreferredGpuAcceleratorIndex` for Auto or create a context with
`AcceleratorIndex` for strict GPU execution.

**An explicit backend throws instead of using CPU.**

This is the intended contract. Explicit modes never silently fall back. Use
Auto when fallback is required.

**An expression cannot be translated.**

Use supported arithmetic and math methods, or use `RunDelegate` for unrestricted
float CPU code.

**The first GPU call is slow.**

The first call includes expression lowering and kernel compilation. Reuse the
context and call `PrecompileAll`, `Precompile<T>`, or `Prepare<T>` during
warm-up.

## Sample application

The console sample demonstrates Auto selection, diagnostics, captured
constants, precompilation, and a resident GPU pipeline:

```powershell
dotnet run --project samples/FastCompute.Sample.Console `
  --configuration Release
```

## Build, test, and package

```powershell
dotnet build FastCompute.sln --configuration Release
dotnet test FastCompute.sln --configuration Release --no-build
./pack.ps1 -Version 0.7.0
```

`pack.ps1` builds and tests the solution, creates `.nupkg` and `.snupkg`
artifacts, verifies the strong-name identity, and runs a package-only consumer
smoke test. On a Windows or Linux CI machine without a hardware GPU:

```powershell
./pack.ps1 -Version 0.7.0 -SkipGpuTests
```

## Further documentation

- [Stable release compliance](https://github.com/staszx/FastCompute.NET/blob/main/docs/stable-release-compliance.md)
- [Additional technical requirements](https://github.com/staszx/FastCompute.NET/blob/main/docs/additional-requirements.md)
- [Stage 1 architecture](https://github.com/staszx/FastCompute.NET/blob/main/docs/stage-1-architecture.md)
- [Stage 2 architecture](https://github.com/staszx/FastCompute.NET/blob/main/docs/stage-2-architecture.md)
- [Stage 3 GPU implementation and compilation](https://github.com/staszx/FastCompute.NET/blob/main/docs/stage-3-gpu-plan.md)
- [SIMD architecture](https://github.com/staszx/FastCompute.NET/blob/main/docs/simd-architecture.md)
- [Stage 4 reductions and memory pooling](https://github.com/staszx/FastCompute.NET/blob/main/docs/stage-4-reductions-and-pooling.md)
- [Stage 6 execution graph](https://github.com/staszx/FastCompute.NET/blob/main/docs/stage-6-execution-graph-plan.md)
- [Lazy optimized array pipeline](https://github.com/staszx/FastCompute.NET/blob/main/docs/lazy-array-pipeline.md)
- [Release history and known limitations](https://github.com/staszx/FastCompute.NET/blob/main/CHANGELOG.md)

## Authors

- `staszx` — project author and maintainer.
- OpenAI Codex — implementation and documentation assistance.
