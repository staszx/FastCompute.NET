using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
[ShortRunJob]
[BenchmarkCategory(PerformanceGateVerifier.Category)]
public class PerformanceGateHeavyMapBenchmarks
{
    private const int Count = 5_000_000;

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
            float value = _source[index];
            result[index] = MathF.Sin(value) * MathF.Exp(-value * value);
        }

        return result;
    }

    [Benchmark]
    public float[] FastComputeAuto() =>
        Compute.Run(
            _source,
            value => GpuMath.Sin(value) * GpuMath.Exp(-value * value));
}
