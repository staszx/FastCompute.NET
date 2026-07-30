using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
public class GpuStreamingBenchmarks
{
    private const int NvidiaAcceleratorIndex = 2;

    private ComputeContext _context = null!;
    private ComputeOptions _sequentialOptions = null!;
    private ComputeOptions _streamingOptions = null!;
    private float[] _source = null!;

    [Params(10_000_000, 50_000_000)]
    public int Count { get; set; }

    [Params(262_144, 1_048_576)]
    public int ChunkElementCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _context = ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = NvidiaAcceleratorIndex
            });
        _sequentialOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _context,
            GpuChunkElementCount = ChunkElementCount
        };
        _streamingOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _context,
            GpuChunkElementCount = ChunkElementCount,
            EnableGpuStreaming = true
        };
        _source = new float[Count];
        for (int index = 0; index < Count; index++)
        {
            _source[index] = (index % 10_000) / 10_000.0f;
        }

        _context.Precompile<float>(
            value =>
                ComputeMath.Sin(value) *
                ComputeMath.Exp(-value * value));
    }

    [GlobalCleanup]
    public void Cleanup() => _context.Dispose();

    [Benchmark(Baseline = true)]
    public float[] SequentialChunks() =>
        Compute.Run(
            _source,
            value =>
                ComputeMath.Sin(value) *
                ComputeMath.Exp(-value * value),
            _sequentialOptions);

    [Benchmark]
    public float[] DoubleBuffered() =>
        Compute.Run(
            _source,
            value =>
                ComputeMath.Sin(value) *
                ComputeMath.Exp(-value * value),
            _streamingOptions);
}
