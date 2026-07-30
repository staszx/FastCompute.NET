using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, warmupCount: 3, iterationCount: 6)]
public class PipelineFusionBenchmarks
{
    private readonly ComputeOptions scalarOptions =
        new()
        {
            Backend = ComputeBackendKind.Scalar
        };
    private float[] source = null!;

    [Params(1_000_000)]
    public int Count { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        source = new float[Count];
        for (int index = 0; index < source.Length; index++)
        {
            source[index] = (index % 10_000) / 10_000.0f;
        }
    }

    [Benchmark(Baseline = true)]
    public float[] SeparateMaps()
    {
        float[] first = Compute.Run(
            source,
            value => value * 2.0f,
            scalarOptions);
        float[] second = Compute.Run(
            first,
            value => value + 1.0f,
            scalarOptions);
        return Compute.Run(
            second,
            value => GpuMath.Clamp(value, 0.0f, 1.0f),
            scalarOptions);
    }

    [Benchmark]
    public float[] FusedPipeline() =>
        source
            .AsCompute(scalarOptions)
            .Select(value => value * 2.0f)
            .SelectInPlace(value => value + 1.0f)
            .Select(value => GpuMath.Clamp(value, 0.0f, 1.0f))
            .ToArray();
}
