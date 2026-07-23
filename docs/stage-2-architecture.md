# Stage 2 architecture

## Scope

Stage 2 adds multi-threaded CPU execution, automatic Scalar/Parallel selection,
diagnostics, and an executable BenchmarkDotNet suite. SIMD and GPU execution
remain outside this stage.

## Parallel execution

`ParallelComputeBackend` compiles the same validated IR used by the Scalar
backend. Arrays are divided into a limited number of chunks rather than
creating work for every element.

The default target is four chunks per available processor, with a minimum chunk
size of 4,096 elements. `MaxDegreeOfParallelism` is forwarded to
`ParallelOptions`, and cancellation is checked inside long-running chunks.

Parallel `Sum` gives each chunk a private accumulator. Partial sums are combined
sequentially after all chunks finish, avoiding a global lock or atomic update
for every element.

## Automatic planner

For CPU-resident arrays, `Auto` uses:

- Scalar below `ComputeThresholdOptions.ParallelThreshold`;
- Parallel CPU at or above the threshold when more than one worker is allowed.

The default parallel threshold is 100,000 elements. A caller can override it
per operation. Explicit Scalar and Parallel requests bypass the threshold.

## Diagnostics

`Compute.RunWithDiagnostics` reports:

- selected backend;
- planning time;
- delegate compilation time;
- execution time;
- zero upload/download time for CPU backends;
- no device name or kernel cache hit for CPU backends.

Ordinary `Compute.Run` follows a non-diagnostic path: it does not instantiate a
public diagnostics object and does not start timing stopwatches.

## Benchmarks

`FastCompute.Benchmarks` covers the required sizes from 1,000 through
50,000,000 elements and includes:

- simple map;
- heavy map with `Sin` and `Exp`;
- simple and medium `Zip`;
- `Sum` reduction;
- ordinary `for`;
- `Parallel.For`;
- forced Scalar;
- forced Parallel CPU;
- automatic selection.

The benchmark project is compiled during normal validation, but the full suite
is intentionally not run as part of unit tests because it is long-running and
machine-dependent.

### Performance regression gate

Two dedicated `PerformanceGate` benchmarks compare Auto mode against a
single-threaded `for` loop:

- simple map over 50,000,000 elements;
- heavy `Sin`/`Exp` map over 5,000,000 elements.

Run them with:

```powershell
dotnet run --project benchmarks/FastCompute.Benchmarks `
  --configuration Release -- `
  --assert-performance
```

After BenchmarkDotNet completes, `PerformanceGateVerifier` compares statistical
means and returns exit code 1 if `FastComputeAuto / ForLoop` exceeds 1.05. The
5% tolerance prevents ordinary measurement noise from being treated as a
regression.

This is deliberately an opt-in benchmark gate rather than an NUnit test.
Wall-clock assertions inside unit-test runners are strongly affected by JIT,
debuggers, concurrent load, power management, and virtualized CI hosts. The
gate should run on fixed, idle benchmark hardware.

## Known limitations

- only `float` is supported;
- expression delegates are compiled per call and are not cached;
- parallel reduction can differ from Scalar by floating-point rounding because
  values are grouped into chunk sums;
- diagnostics are currently exposed only for unary `Run`;
- `EnableDiagnostics` is reserved for future diagnostic-capable API variants;
- SIMD, ILGPU, memory pools, upload/download, and kernel caching are not present.
