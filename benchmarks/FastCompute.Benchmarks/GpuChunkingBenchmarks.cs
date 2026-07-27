using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
public class GpuChunkingBenchmarks
{
    private ComputeContext _context = null!;
    private ComputeOptions _fullOptions = null!;
    private ComputeOptions _chunkedOptions = null!;
    private float[] _left = null!;
    private float[] _right = null!;

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
        _left = new float[Count];
        _right = new float[Count];
        for (int index = 0; index < Count; index++)
        {
            _left[index] = (index % 10_000) / 10_000.0f;
            _right[index] = 1.0f - _left[index];
        }

        _context.PrecompileAll();
    }

    [GlobalCleanup]
    public void Cleanup() => _context.Dispose();

    [Benchmark]
    public float[] MapSingleAllocation() =>
        Compute.Run(
            _left,
            value => GpuMath.Sin(value) * GpuMath.Exp(-value * value),
            _fullOptions);

    [Benchmark]
    public float[] MapChunked() =>
        Compute.Run(
            _left,
            value => GpuMath.Sin(value) * GpuMath.Exp(-value * value),
            _chunkedOptions);

    [Benchmark]
    public float[] ZipSingleAllocation() =>
        Compute.Zip(
            _left,
            _right,
            (left, right) =>
                left * right + GpuMath.Sqrt(GpuMath.Abs(left - right)),
            _fullOptions);

    [Benchmark]
    public float[] ZipChunked() =>
        Compute.Zip(
            _left,
            _right,
            (left, right) =>
                left * right + GpuMath.Sqrt(GpuMath.Abs(left - right)),
            _chunkedOptions);
}
