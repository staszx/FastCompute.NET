# FastCompute.NET

FastCompute.NET is a .NET 8 library for array computations with a LINQ-like
expression API and pluggable Scalar CPU, Parallel CPU, SIMD, and GPU backends.

The project currently provides Scalar, multi-threaded CPU, AVX SIMD, and ILGPU
backends for `float` arrays.

```csharp
float[] source = [0.0f, 0.5f, 1.0f];

float[] mapped = Compute.Run(
    source,
    value => GpuMath.Sin(value) * GpuMath.Exp(-value * value));

float[] zipped = Compute.Zip(
    source,
    mapped,
    (left, right) => left + right);

float sum = Compute.Sum(mapped);
float minimum = Compute.Min(mapped);
float maximum = Compute.Max(mapped);
float average = Compute.Average(mapped);
```

## Current capabilities

- unary `Compute.Run`;
- binary `Compute.Zip`;
- `Compute.Sum`;
- `Compute.Min`, `Compute.Max`, and `Compute.Average`;
- arithmetic and the MVP `GpuMath` functions;
- expression validation and backend-independent IR;
- strict constant folding and IEEE-safe simplification;
- single-threaded Scalar CPU execution;
- chunked Parallel CPU execution;
- AVX `Vector256<float>` Map, Zip, Sum, Min, Max, and Average execution;
- automatic Scalar/SIMD/Parallel selection;
- explicit ILGPU Map, Zip, and reduction execution;
- reusable `ComputeContext` adapted from HDRLib's GPU context;
- GPU-resident `ComputeBuffer<float>` pipelines;
- multi-stage GPU reductions without a global atomic;
- context-local transient GPU memory pooling;
- thread-safe kernel and lowered-expression caches;
- `PrecompileAll`, expression-specific precompilation, and prepared operations;
- execution diagnostics;
- BenchmarkDotNet scenarios for the required data sizes;
- cancellation and explicit backend validation.

Captured values and `double`/`int` support are planned for later stages.

See [Stage 1 architecture](docs/stage-1-architecture.md) and
[Stage 2 architecture](docs/stage-2-architecture.md) for the internal flow,
contracts, and known limitations.

The GPU context, exact kernel-compilation lifecycle, forced precompilation, and
prepared-operation API are documented in
[Stage 3 GPU implementation](docs/stage-3-gpu-plan.md).

The supported SIMD expression subset, scalar-tail behavior, and Auto selection
are documented in [SIMD backend architecture](docs/simd-architecture.md).

Reductions, transient GPU memory pooling, CUDA validation, and GPU Auto
selection are documented in
[Stage 4 reductions and pooling](docs/stage-4-reductions-and-pooling.md).

## GPU precompilation

```csharp
using ComputeContext context = ComputeContext.Create();

// Synchronously compile all implemented templates (Map + Zip + Reduction).
context.PrecompileAll();

// Also validate, lower, and cache selected expressions.
context.Precompile<float>(x => GpuMath.Sin(x) * GpuMath.Exp(x));

float[] result = Compute.Run(
    source,
    x => GpuMath.Sin(x),
    new ComputeOptions
    {
        Backend = ComputeBackendKind.Gpu,
        GpuContext = context
    });
```

## Performance gate

The opt-in performance gate compares `Compute.Run` in Auto mode with an
equivalent single-threaded `for` loop on large simple and heavy map workloads:

```powershell
dotnet run --project benchmarks/FastCompute.Benchmarks `
  --configuration Release -- `
  --assert-performance
```

The command exits with code `1` when FastCompute is more than 5% slower than
the loop. Run this gate on stable, otherwise idle hardware. It intentionally
does not assert performance for small arrays, where expression planning and
compilation overhead makes a direct loop faster.
