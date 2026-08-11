# Lazy optimized array pipeline

## Public API

`AsCompute` turns a supported numeric array into an immutable lazy
`ComputePipeline<T>`:

```csharp
float[] result = source
    .AsCompute()
    .Select(value => value * 2.0f)
    .SelectInPlace(value => value + 1.0f)
    .Select(value => ComputeMath.Sin(value))
    .ToArray();
```

One binary `Zip` can be recorded in the same graph:

```csharp
float[] result = left
    .AsCompute()
    .Select(value => value * 2.0f)
    .Zip(right, (first, second) => first + second)
    .Select(value => ComputeMath.Clamp(value, 0.0f, 1.0f))
    .ToArray();
```

Supported element types are `float`, `double`, and `int`. Pipeline expressions
have the same restrictions as the corresponding one-shot `Compute.Run` API.

Creation overloads:

- `source.AsCompute()` uses Auto and process-wide defaults;
- `source.AsCompute(options)` uses the supplied `ComputeOptions`;
- `source.AsCompute(context)` uses Auto and makes the supplied reusable GPU
  context available if the planner selects GPU.

`ComputeContext` remains owned by the caller. Disposing it before a terminal
operation makes a GPU execution that needs it fail normally.

## Lazy graph

Each call to `Select`, `SelectInPlace`, or `Zip` adds immutable graph state.
Adding a node does not parse, compile, upload, or execute anything. Separate
branches therefore have independent operation chains.

The source array is referenced rather than copied when the pipeline is built.
Changes made directly to that array before a terminal call are visible to the
pipeline.

`Zip` also references its right array and validates equal lengths only at the
terminal operation. One pipeline can contain one Zip node. A second Zip is
rejected because the current backend contract supports unary Map and binary
Zip, not kernels with three or more source arrays.

## Optimization

At a terminal call, nodes are restored to source order and their parameter
expressions are substituted into a single unary expression:

```text
Select(x => x * 2)
Select(x => x + 1)
Select(x => Sin(x))

=> x => Sin((x * 2) + 1)
```

That combined expression is passed through the existing type-specific parser
and optimizer. This provides:

- one expression validation pass;
- constant folding and safe algebraic simplification;
- one backend selection;
- one Scalar, Parallel CPU, or SIMD array pass;
- one GPU Map kernel, upload, and download;
- no managed intermediate arrays between fused selectors.

For a binary graph, selectors before Zip are substituted into its left
parameter and selectors after Zip are substituted into its result:

```text
Select(x => x * 2)
Zip(right, (x, y) => x + y)
Select(x => Clamp(x, 0, 1))

=> (x, y) => Clamp((x * 2) + y, 0, 1)
```

The combined binary expression executes as one Scalar, Parallel CPU, SIMD, or
GPU Zip operation. GPU chunking and explicit in-place execution reuse the
existing Zip backend paths.

Kernel compilation and cache behavior are unchanged. A reusable
`ComputeContext`, precompilation, GPU chunking, streaming, and the global
preferred-GPU setting apply to the fused operation through `ComputeOptions`.

## In-place contract

`SelectInPlace` is an intermediate-buffer reuse hint. Normal `ToArray` never
modifies the managed source array, even when the graph contains
`SelectInPlace`. This is required for predictable lazy branches.

`ToArrayInPlace` is the explicit mutating terminal. It applies the fully fused
expression through the existing in-place backend API, returns the original
array reference, and has the same partial-mutation behavior on mid-execution
cancellation or failure as `Compute.RunInPlace`.

## Terminal operations

The current terminals are:

- `ToArray`;
- `ToArrayInPlace`;
- `Sum`;
- `Min`;
- `Max`;
- `Average`.

With no selectors, `ToArray` returns a copy and `ToArrayInPlace` returns the
unchanged source reference. Reductions without selectors run directly on the
source.

Unary selectors are fused into one Map. When `Sum`, `Min`, `Max`, or `Average`
follows one or more selectors, the optimized Map expression is fused into the
reduction. CPU and SIMD backends transform values while accumulating them, and
GPU backends apply the expression in the first reduction kernel stage. No
full-size mapped array is materialized. This applies to `float`, `double`, and
`int` pipelines, including chunked GPU execution.

When a reduction follows a binary Zip graph, the optimized binary expression
is evaluated while accumulating the reduction. Scalar, Parallel CPU, and SIMD
backends perform a single pass over both sources. GPU backends evaluate the
Zip expression in the first reduction stage, including chunked execution. No
full-size zipped array is materialized. This applies to `float`, `double`, and
`int` pipelines.

The allocation and execution-time difference between three independent Map
calls and one fused pipeline can be measured with:

```powershell
dotnet run --project benchmarks/FastCompute.Benchmarks `
  --configuration Release -- `
  --filter "*PipelineFusionBenchmarks*"
```
