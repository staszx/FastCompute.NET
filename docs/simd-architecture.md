# SIMD backend

## Scope

The SIMD backend implements the MVP contract from section 10 of the technical
specification for `float` arrays:

- `Add`, `Subtract`, `Multiply`, and `Divide`;
- `GpuMath.Min`, `GpuMath.Max`, `GpuMath.Abs`, and `GpuMath.Clamp`;
- unary negation;
- Map, Zip, Sum, Min, Max, and Average;
- scalar processing of the final incomplete vector.

It uses `Vector256<float>` and `System.Runtime.Intrinsics.X86.Avx`. AVX2 does
not define additional floating-point arithmetic instructions; AVX supplies the
required 256-bit float operations.

## Compilation

The backend does not interpret IR inside the element loop. It converts the
backend-independent IR into a specialized expression over
`Vector256<float>`, compiles that expression once per operation, and invokes the
resulting delegate for groups of eight floats.

The scalar CPU compiler builds a second delegate for the zero to seven trailing
elements. This keeps arbitrary array lengths valid without reading beyond a
buffer.

`Min` and `Max` add handling for NaN and signed zero so their result matches the
scalar `GpuMath` contract rather than the raw AVX operand-selection behavior.
`Clamp` also checks invalid minimum/maximum bounds.

## Supported plans and fallback

`SimdComputeBackend.Supports(plan)` returns false when the plan contains
functions without an efficient SIMD implementation, including `Sin`, `Cos`,
`Tan`, `Exp`, `Log`, `Log10`, `Pow`, `Sqrt`, `Floor`, `Ceiling`, and `Round`.

For an explicitly forced SIMD backend, such a plan produces
`ComputeBackendUnavailableException`. In Auto mode, FastCompute skips SIMD and
selects Parallel CPU or Scalar CPU.

On a machine without AVX, the SIMD backend is unavailable. Auto mode falls back
to another CPU backend.

## Automatic selection

For a supported plan, Auto mode selects SIMD when:

```text
elementCount >= ComputeThresholdOptions.SimdThreshold
```

The default threshold is 1,024 elements. SIMD is evaluated before the Parallel
CPU threshold. Unsupported expressions can still select Parallel CPU at its
default threshold of 100,000 elements.

## Validation

The tests cover:

- SIMD/Scalar parity for Map and Zip;
- scalar tails;
- Sum reduction;
- empty and single-element arrays;
- NaN and positive/negative infinity;
- division by zero;
- signed-zero behavior for Min and Max;
- invalid Clamp bounds;
- forced unsupported expressions;
- automatic selection and fallback.

BenchmarkDotNet scenarios for simple Map, simple Zip, and Sum contain explicit
`FastComputeSimd` cases. The performance gate also exercises SIMD through the
simple Auto Map scenario.
