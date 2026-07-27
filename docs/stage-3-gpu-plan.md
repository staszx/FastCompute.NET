# Stage 3 GPU implementation

## Context foundation

`ComputeContext` adapts the accelerator discovery and lifetime model from
`D:\Work\Projects\HDR\HDRLib\HDRLib\Gpu\GpuContext.cs` without referencing
HDRLib itself. It creates an ILGPU context, enables all accelerators and the
ILGPU Algorithms intrinsics, selects either an explicit device index or the
preferred accelerator, and owns its disposal.

HDR image processors, image dependencies, console output, and HDR-specific
kernels are intentionally absent.

```csharp
IReadOnlyList<ComputeDeviceInfo> devices = ComputeContext.GetAccelerators();

using ComputeContext context = ComputeContext.Create(
    new ComputeContextOptions { AcceleratorIndex = 0 });
```

When no index is supplied, ILGPU chooses the preferred non-CPU accelerator and
can fall back to its CPU accelerator. The CPU accelerator is also used by the
portable GPU-backend tests.

## What is compiled and when

FastCompute uses two ILGPU kernel templates in stage 3:

- `Map`, which reads one source buffer;
- `Zip`, which reads two source buffers.

GPU in-place Map reuses the Map template and passes the same `ArrayView<float>`
as both source and destination. Each work item reads and overwrites only its
own index, so no cross-thread dependency is introduced and no separate kernel
template is compiled.

The validated FastCompute IR is lowered to a small postfix instruction program.
The kernel interprets that program independently for every output element.
Consequently, arbitrary accepted expression structure does not require
Reflection.Emit or a new .NET method.

There are two separate caches:

1. a structural expression cache stores lowered postfix programs;
2. a thread-safe `ConcurrentDictionary<ComputeKernelKind, Lazy<CompiledKernel>>`
   stores compiled ILGPU kernel delegates.

`Lazy<T>` with `ExecutionAndPublication` guarantees that concurrent callers do
not compile the same kernel template twice.

Creating `ComputeContext` creates the ILGPU context and accelerator but does not
compile a kernel. Compilation happens synchronously when
`LoadAutoGroupedStreamKernel` first loads `Map` or `Zip` for that context and
accelerator. A normal first execution triggers it on a cache miss. Later
executions in the same context reuse the delegate.

The ILGPU compilation cache is context-local and in-memory. Creating another
`ComputeContext` or restarting the process compiles the templates again.

## Forced compilation

The requested single call that compiles every currently implemented template is:

```csharp
using ComputeContext context = ComputeContext.Create();

IReadOnlyList<ComputeCompilationResult> compilation =
    context.PrecompileAll();
```

In stage 3 this compiled `Map` and `Zip`. Stage 4 added the shared Reduction
template, so the current `PrecompileAll()` compiles all three.

Expression-specific preparation validates and lowers the expression as well:

```csharp
ComputeCompilationResult map =
    context.Precompile<float>(
        x => GpuMath.Sin(x) * GpuMath.Exp(x));

ComputeCompilationResult zip =
    context.Precompile<float>(
        (x, y) => x * y + 1.0f);
```

Several expressions can be prepared together:

```csharp
context.Precompile(
    ComputeKernel.Map<float>(x => x * 2.0f),
    ComputeKernel.Map<float>(x => GpuMath.Sin(x)),
    ComputeKernel.Zip<float>((x, y) => x + y));
```

`ComputeCompilationResult.CacheHit` is true only if both the lowered expression
and the required kernel template were already cached. `CompilationTime` measures
the synchronous ILGPU template compilation and is zero on a kernel-cache hit.

## Execution APIs

The one-shot API can force GPU execution. Supplying a reusable context preserves
the caches between calls:

```csharp
float[] result = Compute.Run(
    source,
    x => GpuMath.Sin(x),
    new ComputeOptions
    {
        Backend = ComputeBackendKind.Gpu,
        GpuContext = context
    });
```

If `GpuContext` is omitted, the current implementation uses a lazily created
shared default context. This preserves device selection, compiled kernels, and
the stage-4 transient memory pool between one-shot calls.

A prepared operation additionally skips expression parsing, optimization,
lowering, structural hashing, and cache lookup:

```csharp
PreparedCompute<float> operation =
    context.Prepare<float>(x => GpuMath.Sin(x) * GpuMath.Exp(x));

float[] first = operation.Run(data1);
float[] second = operation.Run(data2);
```

GPU-resident buffers avoid intermediate host transfers:

```csharp
using ComputeBuffer<float> input = context.Upload(source);
using ComputeBuffer<float> mapped =
    input.Select(x => GpuMath.Sin(x));
using ComputeBuffer<float> combined =
    mapped.Zip(input, (x, y) => x + y);

float[] result = combined.Download();
```

Buffers can only interact when they belong to the same context and have the same
length.

## Stage boundary

Stage 3 supports `float` Map/Select and Zip, explicit device selection,
one-shot and reusable contexts, GPU-resident buffers, diagnostics, thread-safe
kernel caching, forced compilation, and prepared map operations.

The later in-place stage added `Compute.RunInPlace` and
`ComputeContext.RunInPlace`. CPU-resident input is uploaded into one pooled
full-length buffer, processed in that buffer, and downloaded into the original
array. The operation reuses the existing Map kernel cache.

The subsequent chunking stage added `Compute.ZipInPlace` and
`ComputeContext.ZipInPlace`. Unary in-place Map uses one device buffer per
chunk. In-place Zip uses two buffers per chunk and passes the target buffer as
both the left input and kernel destination. Both operations download directly
into the corresponding range of the original target array.

Explicit out-of-place unary Map can additionally opt in to double-buffered
streaming:

```csharp
float[] result = Compute.Run(
    source,
    x => GpuMath.Sin(x),
    new ComputeOptions
    {
        Backend = ComputeBackendKind.Gpu,
        GpuContext = context,
        GpuChunkElementCount = 1_000_000,
        EnableGpuStreaming = true
    });
```

This mode uses two ILGPU streams, four pooled device buffers, and two pairs of
page-locked host staging buffers. Upload, kernel execution, and download of
adjacent chunks can overlap. The memory planner therefore accounts for four
device buffers rather than the two used by sequential Map chunking.

Streaming is deliberately explicit and opt-in. `Auto`, Zip, in-place
operations, reductions, Histogram, and delegate execution reject this option.
If the input fits in a normal one-shot allocation and no explicit chunk size
forces splitting, the planner keeps the simpler one-stream execution.
`ComputeDiagnostics.IsStreaming` and `StreamCount` report the actual execution
mode. For a partial final chunk, transfer counters describe the physical
fixed-size staging transfer and can exceed the logical input byte count.

GPU reductions and the reusable transient device-memory pool are implemented in
[Stage 4](stage-4-reductions-and-pooling.md). Kernel fusion and generated
expression-specific kernels remain stage 6 candidates.
