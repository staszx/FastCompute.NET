# Stage 4 reductions, pooling, and GPU Auto selection

## Reduction API

FastCompute exposes four float reductions:

```csharp
float sum = Compute.Sum(source);
float minimum = Compute.Min(source);
float maximum = Compute.Max(source);
float average = Compute.Average(source);

ComputeResult<float> sumWithDiagnostics =
    Compute.SumWithDiagnostics(source);
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
    input.Select(value => ComputeMath.Sin(value));

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

When the full CPU-resident input does not fit the effective GPU memory budget,
the same kernel reduces sequential source chunks. One scalar partial result is
downloaded per chunk and combined in chunk order. Sum and Average add partials,
Min and Max preserve the existing NaN propagation, and Average divides only
after all partial sums are combined. GPU-resident `ComputeBuffer<T>`
reductions remain non-chunked because their source allocation is already
resident on the selected accelerator.

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
and download. In-place Map rents one full-length buffer and uses it as both
kernel source and destination.

GPU-resident `ComputeBuffer<T>` instances retain ownership of their buffers and
are not transient pool entries.

Pool behavior can be inspected:

```csharp
ComputeMemoryPoolStatistics statistics = context.MemoryPoolStatistics;
```

Idle retention is bounded by
`ComputeContextOptions.MemoryPoolLimitBytes` (256 MiB by default). When the
limit is exceeded, the least recently returned buffers are disposed. Setting
the limit to zero disables idle retention.

The snapshot reports allocations, rentals, successful reuses, currently
available buffers, retained bytes, the configured limit, and eviction count.
Disposing the context disposes every buffer allocated by its pool before
disposing the accelerator.

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

Applications with multiple GPUs can set
`ComputeOptions.PreferredGpuAcceleratorIndex`. Auto evaluates the normal
complexity, transfer, threshold, and memory rules first. The selected
accelerator context is created lazily only if the planner actually chooses GPU;
otherwise execution remains on SIMD, Parallel CPU, or Scalar. An unavailable
preferred index also leaves Auto on CPU. Explicit GPU execution uses the same
index but never falls back.

`GpuContext` remains the reusable, application-owned alternative for
precompilation and buffer reuse. Supplying both `GpuContext` and
`PreferredGpuAcceleratorIndex` is rejected as ambiguous.

The estimate includes all full-length inputs and outputs plus planning
overhead. Map uses two full-length buffers, Zip uses three, and reduction uses
a conservative two-buffer estimate. In-place Map uses one full-length buffer.

CPU-resident Map and Zip now use sequential GPU chunks when the full working
set exceeds the effective budget. `EnableGpuChunking` controls this behavior,
and `GpuChunkElementCount` can impose a smaller explicit chunk size. Explicit
GPU execution never falls back to CPU: it either builds a valid chunk plan or
throws `ComputeGpuMemoryBudgetExceededException`. Auto may select chunked GPU
for eligible medium/heavy expressions and CPU-resident reductions.

In-place Map and Zip also support sequential chunks. Map requires one
full-length chunk buffer; Zip requires target and right-input chunk buffers and
writes the kernel output back into the target buffer. Completed CPU target
ranges are not rolled back if cancellation or a later GPU operation fails.

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

`RunWithDiagnostics`, `ZipWithDiagnostics`, and their in-place counterparts report
`BackendSelectionReason`, `EstimatedGpuMemoryBytes`, `GpuMemoryBudgetBytes`,
`ChunkCount`, `ChunkElementCount`, `UploadedBytes`, and `DownloadedBytes` so
the decision and transfer volume are observable.

The `GpuChunkingBenchmarks` scenarios compare one full allocation with forced
262,144-element chunks for Map and Zip at 1,000,000 and 10,000,000 elements.
The CUDA Dry-run exercises every scenario; its single cold iteration validates
the benchmark path but is not treated as a performance result.

`GpuInPlaceChunkingBenchmarks` provides the equivalent full-allocation versus
262,144-element chunk comparison for `RunInPlace` and `ZipInPlace`.

`GpuChunkedReductionBenchmarks` compares full-allocation and 262,144-element
chunk execution for Sum and Max.

## Histogram

`Compute.Histogram` counts `float` values in equally sized bins:

```csharp
int[] histogram = Compute.Histogram(
    source,
    binCount: 256,
    minimum: 0.0f,
    maximum: 1.0f,
    options);
```

The configured range is inclusive. `maximum` belongs to the last bin; NaN and
out-of-range values are ignored. Scalar and Parallel CPU are supported. SIMD
is rejected because Histogram requires scattered counter updates.

The GPU kernel uses atomic counter increments. Its histogram buffer remains on
the accelerator while sequential source chunks are uploaded and processed, so
only the final `binCount` counters are downloaded. The memory estimate includes
the persistent counter buffer, one input chunk, and planning overhead.
`HistogramWithDiagnostics` reports chunk and transfer counters.

`PrecompileHistogram<float>()`, `ComputeKernel.Histogram<float>()`, and
`PrecompileAll()` force compilation of the Histogram template.

Automatic GPU selection for CPU-resident Histogram input is disabled by
default (`GpuHistogramThreshold = int.MaxValue`). Set a workload-specific
threshold to opt in after benchmarking the target device. Explicit
`Backend = ComputeBackendKind.Gpu` remains fully supported and never silently
falls back.

`HistogramBenchmarks` compares a direct loop, Scalar, Parallel CPU, one-shot
GPU, and 262,144-element chunked GPU at 1,000,000 and 10,000,000 elements.
The Dry job on the development machine measured 2.73 ms for the direct loop
and 8.76 ms for Parallel CPU at 1,000,000 elements. At 10,000,000 elements,
Parallel CPU edged the direct loop (19.76 ms versus 20.94 ms), while one-shot
GPU took 27.24 ms. Dry is a single cold iteration rather than a stable
performance gate, but it confirms that CPU-to-GPU transfer can erase the
kernel advantage. This is why Histogram does not select GPU automatically
without an explicit threshold.

The measured performance-gate ratios on the development machine were:

```text
Heavy Map Auto / for:  0.266
In-place Auto / for:    0.901
Simple Map Auto / for: 0.787
Required maximum:       1.050
```

A GPU Dry-run also exercised out-of-place and in-place Map from 1,000 through
50,000,000 elements on the development CUDA device. At 50,000,000 elements,
the in-place case allocated about 3 KiB of managed memory per operation,
compared with about 190.7 MiB for the out-of-place result array. Dry-run timing
includes cold-start compilation and is not used as a performance gate.

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
