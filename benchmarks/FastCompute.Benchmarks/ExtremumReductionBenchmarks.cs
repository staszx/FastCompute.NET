using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
public class MinReductionBenchmarks : BenchmarkData
{
    private static readonly ComputeOptions SimdOptions =
        new() { Backend = ComputeBackendKind.Simd };

    [Benchmark(Baseline = true)]
    public float ForLoop()
    {
        float result = Source[0];
        for (int index = 1; index < Source.Length; index++)
        {
            result = MathF.Min(result, Source[index]);
        }

        return result;
    }

    [Benchmark]
    public float FastComputeSimd() => Compute.Min(Source, SimdOptions);

    [Benchmark]
    public float FastComputeAuto() => Compute.Min(Source);
}

[MemoryDiagnoser]
public class MaxReductionBenchmarks : BenchmarkData
{
    private static readonly ComputeOptions SimdOptions =
        new() { Backend = ComputeBackendKind.Simd };

    [Benchmark(Baseline = true)]
    public float ForLoop()
    {
        float result = Source[0];
        for (int index = 1; index < Source.Length; index++)
        {
            result = MathF.Max(result, Source[index]);
        }

        return result;
    }

    [Benchmark]
    public float FastComputeSimd() => Compute.Max(Source, SimdOptions);

    [Benchmark]
    public float FastComputeAuto() => Compute.Max(Source);
}
