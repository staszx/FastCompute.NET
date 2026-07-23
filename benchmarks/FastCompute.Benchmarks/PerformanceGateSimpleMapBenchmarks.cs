using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
[ShortRunJob]
[BenchmarkCategory(PerformanceGateVerifier.Category)]
public class PerformanceGateSimpleMapBenchmarks
{
    private const int Count = 50_000_000;

    private float[] _source = null!;

    [GlobalSetup]
    public void Setup()
    {
        _source = new float[Count];
        for (int index = 0; index < _source.Length; index++)
        {
            _source[index] = (index % 10_000) / 10_000.0f;
        }
    }

    [Benchmark(Baseline = true)]
    public float[] ForLoop()
    {
        var result = new float[_source.Length];
        for (int index = 0; index < _source.Length; index++)
        {
            result[index] = _source[index] * 2.0f + 1.0f;
        }

        return result;
    }

    [Benchmark]
    public float[] FastComputeAuto() =>
        Compute.Run(_source, value => value * 2.0f + 1.0f);
}
