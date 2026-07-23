# Stage 4 reductions, pooling, and GPU Auto selection

## Reduction API

FastCompute exposes four float reductions:

```csharp
float sum = Compute.Sum(source);
float minimum = Compute.Min(source);
float maximum = Compute.Max(source);
float average = Compute.Average(source);
```

`Sum` returns zero for an empty array. `Min`, `Max`, and `Average` throw
`InvalidOperationException` for an empty array.

Scalar, Parallel CPU, SIMD, and GPU backends implement all four operations.
Floating-point reduction order can differ between backends, so Sum and Average
are compared with a tolerance.

## GPU reduction

The GPU backend uses a multi-stage reduction without a global atomic:

1. each output thread reduces up to 256 consecutive input values;
2. the partial-result buffer becomes the input of the next pass;
3. passes continue until one value remains;
4. Average divides the final Sum by the original element count.

The same reduction template handles Sum, Min, and Max. It is included in
`ComputeContext.PrecompileAll()` and can be prepared explicitly:

```csharp
context.PrecompileReduction<float>(ComputeReductionKind.Sum);

context.Precompile(
    ComputeKernel.Reduction<float>(ComputeReductionKind.Max));
```

CUDA validation uses accelerator index 2 on the development machine:

```text
NVIDIA GeForce GTX 1650 (Cuda)
```

## Transient device-memory pool

Each `ComputeContext` owns a thread-safe pool of transient float buffers keyed
by exact element count. One-shot Map, Zip, and reduction operations rent their
input, output, and intermediate buffers and return them after synchronization
and download.

GPU-resident `ComputeBuffer<T>` instances retain ownership of their buffers and
are not transient pool entries.

Pool behavior can be inspected:

```csharp
ComputeMemoryPoolStatistics statistics = context.MemoryPoolStatistics;
```

The snapshot reports allocations, rentals, successful reuses, and currently
available buffers. Disposing the context disposes every buffer allocated by its
pool before disposing the accelerator.

## Automatic GPU selection

Auto classifies expressions into three groups:

- heavy transcendental expressions use `GpuHeavyThreshold`;
- other function expressions and reductions use `GpuMediumThreshold`;
- arithmetic-only expressions use `GpuSimpleThreshold`.

GPU selection happens before SIMD and Parallel CPU when a hardware accelerator
is available and the applicable threshold is reached. The shared default GPU
context prefers CUDA, retains compiled kernels, and reuses transient memory.

The initial 10-million simple-expression threshold from the specification was
not retained as the default. BenchmarkDotNet showed that a 50-million-element
simple Map on GTX 1650 was slower than AVX after PCIe upload/download. The
default is therefore 100 million while remaining user-configurable. Heavy Map
selects CUDA at 300,000 elements by default.

The measured performance-gate ratios on the development machine were:

```text
Heavy Map Auto / for:  0.146
Simple Map Auto / for: 0.818
Required maximum:       1.050
```
