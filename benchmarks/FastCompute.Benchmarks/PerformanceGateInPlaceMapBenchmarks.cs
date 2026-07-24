using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
[ShortRunJob]
[BenchmarkCategory(PerformanceGateVerifier.Category)]
public class PerformanceGateInPlaceMapBenchmarks
{
    private const int Count = 50_000_000;

    private float[] _fastComputeSource = null!;
    private float[] _loopSource = null!;

    [GlobalSetup]
    public void Setup()
    {
        _fastComputeSource = CreateSource();
        _loopSource = CreateSource();
    }

    [Benchmark(Baseline = true)]
    public float[] ForLoop()
    {
        for (int index = 0; index < _loopSource.Length; index++)
        {
            _loopSource[index] =
                _loopSource[index] * 1.000001f + 0.00001f;
        }

        return _loopSource;
    }

    [Benchmark]
    public float[] FastComputeAuto() =>
        Compute.RunInPlace(
            _fastComputeSource,
            value => value * 1.000001f + 0.00001f);

    private static float[] CreateSource()
    {
        var source = new float[Count];
        for (int index = 0; index < source.Length; index++)
        {
            source[index] = (index % 10_000) / 10_000.0f;
        }

        return source;
    }
}
