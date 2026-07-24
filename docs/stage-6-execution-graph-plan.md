# Stage 6 execution graph

## Stage 6a status

Stage 6a is implemented. `ComputeBuffer<T>.Select` and `Zip` now create lazy,
immutable graph nodes. No accelerator kernel is launched until `Download` or
another operation requests a materialized accelerator view.

Expressions are parsed and optimized when the node is created. This snapshots
captured primitive values at the same point as the previous eager
implementation.

The graph executor is deliberately unfused. A chain of three `Select` nodes
still launches three kernels when materialized, but construction of the chain
does not allocate destination buffers or compile kernels.

## Stage 6 boundary

Stage 6 does not add general C# method support or promise that every graph is
fused.

The public chaining syntax remains unchanged:

```csharp
using ComputeBuffer<float> result = source
    .Select(value => value * 2.0f)
    .Select(value => GpuMath.Sin(value))
    .Select(value => GpuMath.Clamp(value, 0.0f, 1.0f));
```

## Graph model

The implemented internal graph is immutable:

```text
BufferSource
    |
Map(value * 2)
    |
Map(Sin)
    |
Map(Clamp)
```

The node kinds are:

- `BufferSourceNode`: a materialized accelerator allocation;
- `MapNode`: one unary `ComputeExpressionPlan`;
- `ZipNode`: two dependencies and one binary `ComputeExpressionPlan`.

Every node records context identity, element type, length, and its optimized
IR. Captured primitive values are therefore snapshotted when the node is
created, matching the current planning semantics.

## Materialization

`Download`, resident reduction, and any operation requiring an accelerator view are
materialization boundaries. Materialization must:

- happen at most once for a buffer;
- be safe when two threads read the same buffer;
- cache the successful materialized result;
- clean up partial allocations after failure;
- never publish a partially initialized buffer.

The current implementation satisfies these rules with a per-node
materialization lock and a cached materialized allocation. Failed execution
does not release dependencies, so the graph remains valid for cleanup and a
possible retry.

`Download(Span<T>)` copies directly into caller-owned memory. `Sum`, `Min`,
`Max`, and `Average` consume the materialized graph buffer directly and return
only one scalar to CPU memory.

Empty resident buffers are represented without an ILGPU allocation. This
avoids an ILGPU 1.5.3 zero-length CUDA allocation disposal failure and lets
empty Map/Zip graphs, downloads, and Sum complete without launching a kernel.

## Ownership and disposal

Lazy graphs cannot retain raw `MemoryBuffer1D` references owned solely by
another public `ComputeBuffer`. Every graph node is therefore
reference-counted:

- a public buffer handle owns one lease reference;
- each graph dependency acquires its own reference;
- `Dispose` releases the handle without invalidating dependent graphs;
- the accelerator allocation is disposed when the final reference is
  released;
- disposing the context invalidates all remaining operations.

This preserves the useful existing behavior where a derived buffer remains
valid after its source handle has been disposed.

Successful materialization releases dependency references as soon as the
result allocation becomes independent. Explicit `Dispose` is the immediate
release path. A finalizer is a safety net for unnamed intermediate handles in
fluent chains; it is not a replacement for deterministic disposal.

## Fusion boundary

In stage 6b, consecutive `MapNode` expressions
can be composed by substituting the output of one optimized IR tree into the
parameter of the next.

Fusion must stop when:

- the combined program exceeds the GPU instruction or stack-depth limit;
- a node has multiple consumers and recomputation would be more expensive;
- a backend does not support the combined plan;
- diagnostics or an explicit materialization boundary requires a separate
  result.

`ZipNode` fusion is deferred until unary map fusion is measured and stable.

## Validation

Stage 6a tests cover:

- one-, two-, and three-node chains;
- branching from one source into multiple results;
- source disposal before derived-buffer materialization;
- concurrent downloads of one graph;
- context disposal;
- graph execution parity with the current eager path;
- lazy kernel compilation;
- disposal of unused unmaterialized branches.

Remaining stage 6b validation will add:

- exception-path allocation accounting;
- kernel-launch and allocation counts;
- benchmark comparison of unfused and fused graph execution;
- measured unary Map fusion with a conservative split policy.
