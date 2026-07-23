using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
public class HeavyMapBenchmarks : BenchmarkData
{
    private static readonly ComputeOptions ScalarOptions =
        new() { Backend = ComputeBackendKind.Scalar };

    private static readonly ComputeOptions ParallelOptions =
        new() { Backend = ComputeBackendKind.ParallelCpu };

    [Benchmark(Baseline = true)]
    public float[] ForLoop()
    {
        var result = new float[Source.Length];
        for (int index = 0; index < Source.Length; index++)
        {
            float value = Source[index];
            result[index] = MathF.Sin(value) * MathF.Exp(-value * value);
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
            index =>
            {
                float value = Source[index];
                result[index] = MathF.Sin(value) * MathF.Exp(-value * value);
            });
        return result;
    }

    [Benchmark]
    public float[] FastComputeScalar() =>
        Compute.Run(
            Source,
            value => GpuMath.Sin(value) * GpuMath.Exp(-value * value),
            ScalarOptions);

    [Benchmark]
    public float[] FastComputeParallel() =>
        Compute.Run(
            Source,
            value => GpuMath.Sin(value) * GpuMath.Exp(-value * value),
            ParallelOptions);

    [Benchmark]
    public float[] FastComputeAuto() =>
        Compute.Run(
            Source,
            value => GpuMath.Sin(value) * GpuMath.Exp(-value * value));
}
