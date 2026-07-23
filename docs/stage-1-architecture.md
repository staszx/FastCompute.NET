# Stage 1 architecture

## Scope

Stage 1 establishes the correctness baseline for future Parallel CPU, SIMD, and
ILGPU backends. It intentionally supports only fixed-length one-dimensional
`float` arrays and the Scalar CPU backend.

## Execution flow

1. `Compute.Run` or `Compute.Zip` validates public arguments and options.
2. `ComputeExpressionParser` accepts only typed parameters, `float` constants,
   arithmetic operators, and methods declared by `GpuMath`.
3. The parser produces an ILGPU-independent `ComputeExpressionPlan` containing
   immutable IR nodes.
4. `StrictComputeOptimizer` folds constant-only subtrees and performs only
   transformations that preserve the required IEEE 754 behavior.
5. `CpuExpressionCompiler` lowers the optimized IR to a typed delegate once
   per call.
6. `ScalarComputeBackend` applies that delegate in a normal allocation-bounded
   loop without LINQ in the hot path.

`Compute.Sum` uses the same backend selection boundary but does not require an
expression plan.

## Main design decisions

- The Scalar backend consumes the IR rather than compiling the user's original
  expression. This keeps expression semantics shared with later backends.
- The IR has no dependency on ILGPU or `System.Linq.Expressions`.
- `ComputeBackendKind.Auto` currently selects Scalar. Explicit requests for
  unavailable backends fail instead of silently ignoring the request.
- The Fast optimization mode is present in the public contract but rejects
  execution until its relaxed IEEE rules are specified.
- Strict mode does not simplify `x * 0` because it would change `NaN` and
  infinity results. It also keeps `x + 0` because removing it changes the sign
  of negative zero.
- Reflection is used only once during static Scalar compiler initialization to
  map known `GpuMath` methods. It is not used in element-processing loops.

## Known limitations

- only `float` is supported;
- only Scalar CPU execution is available;
- captured values are rejected, including captured primitive values;
- `ConditionalExpression` is not supported;
- plans and compiled delegates are not cached yet;
- `Min`, `Max`, `Average`, and Histogram reductions are not implemented;
- diagnostics and asynchronous APIs are not implemented;
- `AllowFallback`, parallelism, and diagnostics options are reserved for later
  backends;
- expression compilation uses `Expression.Compile`, so Native AOT is not a
  stage 1 target.

## Completion gate

Stage 1 is complete when the Release solution builds with warnings treated as
errors and all unit tests pass on .NET 8.
