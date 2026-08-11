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
            value => ComputeMath.Clamp(value, 0.0f, 1.0f),
            scalarOptions);
    }

    [Benchmark]
    public float[] FusedPipeline() =>
        source
            .AsCompute(scalarOptions)
            .Select(value => value * 2.0f)
            .SelectInPlace(value => value + 1.0f)
            .Select(value => ComputeMath.Clamp(value, 0.0f, 1.0f))
            .ToArray();
}

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, warmupCount: 3, iterationCount: 6)]
public class PipelineReductionFusionBenchmarks
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
    public float MaterializedMapThenSum()
    {
        float[] mapped = Compute.Run(
            source,
            value => ComputeMath.Clamp(value * 2.0f + 1.0f, 0.0f, 2.0f),
            scalarOptions);
        return Compute.Sum(mapped, scalarOptions);
    }

    [Benchmark]
    public float FusedMapReduction() =>
        source
            .AsCompute(scalarOptions)
            .Select(value => value * 2.0f)
            .Select(value => value + 1.0f)
            .Select(value => ComputeMath.Clamp(value, 0.0f, 2.0f))
            .Sum();
}

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, warmupCount: 3, iterationCount: 6)]
public class PipelineZipFusionBenchmarks
{
    private readonly ComputeOptions scalarOptions =
        new()
        {
            Backend = ComputeBackendKind.Scalar
        };
    private float[] left = null!;
    private float[] right = null!;

    [Params(1_000_000)]
    public int Count { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        left = new float[Count];
        right = new float[Count];
        for (int index = 0; index < Count; index++)
        {
            left[index] = (index % 10_000) / 10_000.0f;
            right[index] = ((index * 3) % 10_000) / 10_000.0f;
        }
    }

    [Benchmark(Baseline = true)]
    public float[] SeparateMapZipMap()
    {
        float[] mapped = Compute.Run(
            left,
            value => value * 2.0f,
            scalarOptions);
        float[] zipped = Compute.Zip(
            mapped,
            right,
            (first, second) => first + second,
            scalarOptions);
        return Compute.Run(
            zipped,
            value => ComputeMath.Clamp(value, 0.0f, 1.0f),
            scalarOptions);
    }

    [Benchmark]
    public float[] FusedZipPipeline() =>
        left
            .AsCompute(scalarOptions)
            .Select(value => value * 2.0f)
            .Zip(right, (first, second) => first + second)
            .Select(value => ComputeMath.Clamp(value, 0.0f, 1.0f))
            .ToArray();
}

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, warmupCount: 3, iterationCount: 6)]
public class PipelineZipReductionFusionBenchmarks
{
    private readonly ComputeOptions scalarOptions =
        new()
        {
            Backend = ComputeBackendKind.Scalar
        };
    private float[] left = null!;
    private float[] right = null!;

    [Params(1_000_000)]
    public int Count { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        left = new float[Count];
        right = new float[Count];
        for (int index = 0; index < Count; index++)
        {
            left[index] = (index % 10_000) / 10_000.0f;
            right[index] = ((index * 3) % 10_000) / 10_000.0f;
        }
    }

    [Benchmark(Baseline = true)]
    public float MaterializedZipThenSum()
    {
        float[] zipped = left
            .AsCompute(scalarOptions)
            .Select(value => value * 2.0f)
            .Zip(right, (first, second) => first + second)
            .Select(value => ComputeMath.Clamp(value, 0.0f, 1.0f))
            .ToArray();
        return Compute.Sum(zipped, scalarOptions);
    }

    [Benchmark]
    public float FusedZipReduction() =>
        left
            .AsCompute(scalarOptions)
            .Select(value => value * 2.0f)
            .Zip(right, (first, second) => first + second)
            .Select(value => ComputeMath.Clamp(value, 0.0f, 1.0f))
            .Sum();
}
