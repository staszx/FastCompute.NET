using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
public class GpuInPlaceChunkingBenchmarks
{
    private ComputeContext _context = null!;
    private ComputeOptions _fullOptions = null!;
    private ComputeOptions _chunkedOptions = null!;
    private float[] _seed = null!;
    private float[] _right = null!;
    private float[] _mapTarget = null!;
    private float[] _zipTarget = null!;

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
        _seed = new float[Count];
        _right = new float[Count];
        _mapTarget = new float[Count];
        _zipTarget = new float[Count];
        for (int index = 0; index < Count; index++)
        {
            _seed[index] = (index % 10_000) / 10_000.0f;
            _right[index] = 1.0f - _seed[index];
        }

        _context.PrecompileAll();
    }

    [IterationSetup]
    public void ResetTargets()
    {
        Array.Copy(_seed, _mapTarget, Count);
        Array.Copy(_seed, _zipTarget, Count);
    }

    [GlobalCleanup]
    public void Cleanup() => _context.Dispose();

    [Benchmark]
    public float[] MapSingleAllocation() =>
        Compute.RunInPlace(
            _mapTarget,
            value => ComputeMath.Sin(value) * ComputeMath.Exp(-value * value),
            _fullOptions);

    [Benchmark]
    public float[] MapChunked() =>
        Compute.RunInPlace(
            _mapTarget,
            value => ComputeMath.Sin(value) * ComputeMath.Exp(-value * value),
            _chunkedOptions);

    [Benchmark]
    public float[] ZipSingleAllocation() =>
        Compute.ZipInPlace(
            _zipTarget,
            _right,
            (left, right) =>
                left * right + ComputeMath.Sqrt(ComputeMath.Abs(left - right)),
            _fullOptions);

    [Benchmark]
    public float[] ZipChunked() =>
        Compute.ZipInPlace(
            _zipTarget,
            _right,
            (left, right) =>
                left * right + ComputeMath.Sqrt(ComputeMath.Abs(left - right)),
            _chunkedOptions);
}
