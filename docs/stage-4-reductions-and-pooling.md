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

GPU-resident buffers expose the same reductions without downloading the input
array:

```csharp
using ComputeBuffer<float> input = context.Upload(source);
using ComputeBuffer<float> transformed =
    input.Select(value => GpuMath.Sin(value));

float sum = transformed.Sum();
float minimum = transformed.Min();
float maximum = transformed.Max();
float average = transformed.Average();
```

The lazy graph is materialized on the accelerator, the existing multi-stage
reduction kernel consumes that device allocation directly, and only the final
scalar is copied to CPU memory.

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
is available, the applicable threshold is reached, and the estimated working
set fits the effective memory budget. The default budget is 75% of the total
accelerator memory reported by ILGPU. `GpuMemoryBudgetBytes` can lower, but not
raise, this safety limit.

The estimate includes all full-length inputs and outputs plus planning
overhead. Map uses two full-length buffers, Zip uses three, and reduction uses
a conservative two-buffer estimate.

The shared default GPU context prefers CUDA, retains compiled kernels, and
reuses transient memory.

The initial 10-million simple-expression threshold from the specification was
not retained as the default. BenchmarkDotNet showed that a 50-million-element
simple Map on GTX 1650 was slower than AVX after PCIe upload/download. A later
1,073,741,824-element test also showed that increasing the input size does not
make a one-add expression suitable for a discrete GPU: transfer grows with the
same data size and the required input/output device buffers exceed 8 GiB.

Automatic GPU selection for arithmetic-only CPU-resident expressions is
therefore disabled by default (`GpuSimpleThreshold = int.MaxValue`). Users can
set a lower threshold to opt in. Heavy Map still selects CUDA at 300,000
elements when its working set fits the memory budget.

`RunWithDiagnostics` reports `BackendSelectionReason`,
`EstimatedGpuMemoryBytes`, and `GpuMemoryBudgetBytes` so the decision is
observable.

The measured performance-gate ratios on the development machine were:

```text
Heavy Map Auto / for:  0.266
In-place Auto / for:    0.901
Simple Map Auto / for: 0.787
Required maximum:       1.050
```

## Captured primitive constants

The expression parser snapshots captured `float`, `double`, and `int` local
values while creating each execution plan. Explicit conversions from captured
`double` and `int` values to `float` become float constants in the
backend-independent IR.

A conditional controlled by a captured `bool` is resolved during planning:

```csharp
float multiplier = 2.0f;
bool negate = false;

float[] result = Compute.Run(
    source,
    value => negate ? -(value * multiplier) : value * multiplier);
```

Changing a captured local affects the next newly planned call. A
`PreparedCompute<T>` snapshots the value when the operation is prepared.

Captured reference objects remain rejected. FastCompute does not invoke
arbitrary property getters or traverse object graphs while planning an
expression.
