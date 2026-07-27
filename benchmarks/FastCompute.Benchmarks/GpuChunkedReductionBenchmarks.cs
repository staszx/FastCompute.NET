using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
public class GpuChunkedReductionBenchmarks
{
    private ComputeContext _context = null!;
    private ComputeOptions _fullOptions = null!;
    private ComputeOptions _chunkedOptions = null!;
    private float[] _source = null!;

    [Params(1_000_000, 10_000_000)]
    public int Count { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _context = ComputeContext.Create();
        _fullOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _context,
            EnableGpuChunking = false
        };
        _chunkedOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _context,
            GpuChunkElementCount = 262_144
        };
        _source = new float[Count];
        for (int index = 0; index < Count; index++)
        {
            _source[index] = (index % 10_000) / 10_000.0f;
        }

        _context.PrecompileReduction<float>(ComputeReductionKind.Sum);
    }

    [GlobalCleanup]
    public void Cleanup() => _context.Dispose();

    [Benchmark]
    public float SumSingleAllocation() =>
        Compute.Sum(_source, _fullOptions);

    [Benchmark]
    public float SumChunked() =>
        Compute.Sum(_source, _chunkedOptions);

    [Benchmark]
    public float MaxSingleAllocation() =>
        Compute.Max(_source, _fullOptions);

    [Benchmark]
    public float MaxChunked() =>
        Compute.Max(_source, _chunkedOptions);
}
