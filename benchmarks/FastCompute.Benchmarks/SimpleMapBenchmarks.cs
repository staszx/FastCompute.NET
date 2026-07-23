using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
public class SimpleMapBenchmarks : BenchmarkData
{
    private static readonly ComputeOptions ScalarOptions =
        new() { Backend = ComputeBackendKind.Scalar };

    private static readonly ComputeOptions ParallelOptions =
        new() { Backend = ComputeBackendKind.ParallelCpu };

    private static readonly ComputeOptions SimdOptions =
        new() { Backend = ComputeBackendKind.Simd };

    [Benchmark(Baseline = true)]
    public float[] ForLoop()
    {
        var result = new float[Source.Length];
        for (int index = 0; index < Source.Length; index++)
        {
            result[index] = Source[index] * 2.0f + 1.0f;
        }

        return result;
    }

    [Benchmark]
    public float[] ParallelFor()
    {
        var result = new float[Source.Length];
        Parallel.For(
            0,
            Source.Length,
            index => result[index] = Source[index] * 2.0f + 1.0f);
        return result;
    }

    [Benchmark]
    public float[] FastComputeScalar() =>
        Compute.Run(Source, value => value * 2.0f + 1.0f, ScalarOptions);

    [Benchmark]
    public float[] FastComputeParallel() =>
        Compute.Run(Source, value => value * 2.0f + 1.0f, ParallelOptions);

    [Benchmark]
    public float[] FastComputeSimd() =>
        Compute.Run(Source, value => value * 2.0f + 1.0f, SimdOptions);

    [Benchmark]
    public float[] FastComputeAuto() =>
        Compute.Run(Source, value => value * 2.0f + 1.0f);
}
