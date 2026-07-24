using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
public class InPlaceMapBenchmarks : BenchmarkData
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
        for (int index = 0; index < Source.Length; index++)
        {
            Source[index] = Source[index] * 1.000001f + 0.00001f;
        }

        return Source;
    }

    [Benchmark]
    public float[] FastComputeScalar() =>
        Compute.RunInPlace(
            Source,
            value => value * 1.000001f + 0.00001f,
            ScalarOptions);

    [Benchmark]
    public float[] FastComputeParallel() =>
        Compute.RunInPlace(
            Source,
            value => value * 1.000001f + 0.00001f,
            ParallelOptions);

    [Benchmark]
    public float[] FastComputeSimd() =>
        Compute.RunInPlace(
            Source,
            value => value * 1.000001f + 0.00001f,
            SimdOptions);

    [Benchmark]
    public float[] FastComputeAuto() =>
        Compute.RunInPlace(
            Source,
            value => value * 1.000001f + 0.00001f);
}
