using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
public class ResidentGraphInPlaceBenchmarks
{
    private const int NvidiaAcceleratorIndex = 2;

    private ComputeContext _context = null!;
    private float[] _source = null!;

    [Params(1_000_000, 10_000_000)]
    public int Count { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _context = ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = NvidiaAcceleratorIndex
            });
        _source = new float[Count];
        for (int index = 0; index < Count; index++)
        {
            _source[index] = (index % 10_000) / 10_000.0f;
        }

        _context.Precompile<float>(
            value =>
                GpuMath.Sin(value) *
                GpuMath.Exp(-value * value));
        _context.PrecompileReduction<float>(ComputeReductionKind.Sum);
    }

    [GlobalCleanup]
    public void Cleanup() => _context.Dispose();

    [Benchmark(Baseline = true)]
    public float OutOfPlace()
    {
        using ComputeBuffer<float> source = _context.Upload(_source);
        using ComputeBuffer<float> result =
            source.Select(
                value =>
                    GpuMath.Sin(value) *
                    GpuMath.Exp(-value * value));
        return result.Sum();
    }

    [Benchmark]
    public float InPlaceExclusive()
    {
        using ComputeBuffer<float> source = _context.Upload(_source);
        source.SelectInPlace(
            value =>
                GpuMath.Sin(value) *
                GpuMath.Exp(-value * value));
        return source.Sum();
    }

    [Benchmark]
    public float InPlaceCopyOnWrite()
    {
        using ComputeBuffer<float> source = _context.Upload(_source);
        using ComputeBuffer<float> oldValueBranch =
            source.Select(value => value);
        source.SelectInPlace(
            value =>
                GpuMath.Sin(value) *
                GpuMath.Exp(-value * value));
        return source.Sum();
    }
}
