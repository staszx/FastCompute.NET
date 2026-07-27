using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
public class HistogramBenchmarks
{
    private const int BinCount = 256;
    private static readonly ComputeOptions ScalarOptions =
        new() { Backend = ComputeBackendKind.Scalar };
    private static readonly ComputeOptions ParallelOptions =
        new() { Backend = ComputeBackendKind.ParallelCpu };

    private ComputeContext _context = null!;
    private ComputeOptions _gpuFullOptions = null!;
    private ComputeOptions _gpuChunkedOptions = null!;
    private float[] _source = null!;

    [Params(1_000_000, 10_000_000)]
    public int Count { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _context = ComputeContext.Create();
        _gpuFullOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _context,
            EnableGpuChunking = false
        };
        _gpuChunkedOptions = new ComputeOptions
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

        _context.PrecompileHistogram<float>();
    }

    [GlobalCleanup]
    public void Cleanup() => _context.Dispose();

    [Benchmark(Baseline = true)]
    public int[] ForLoop()
    {
        var histogram = new int[BinCount];
        for (int index = 0; index < _source.Length; index++)
        {
            float value = _source[index];
            if (float.IsNaN(value) || value < 0.0f || value > 1.0f)
            {
                continue;
            }

            int binIndex = value == 1.0f
                ? BinCount - 1
                : (int)(value * BinCount);
            if ((uint)binIndex < BinCount)
            {
                histogram[binIndex]++;
            }
        }

        return histogram;
    }

    [Benchmark]
    public int[] FastComputeScalar() =>
        Compute.Histogram(
            _source,
            BinCount,
            0.0f,
            1.0f,
            ScalarOptions);

    [Benchmark]
    public int[] FastComputeParallel() =>
        Compute.Histogram(
            _source,
            BinCount,
            0.0f,
            1.0f,
            ParallelOptions);

    [Benchmark]
    public int[] GpuSingleAllocation() =>
        Compute.Histogram(
            _source,
            BinCount,
            0.0f,
            1.0f,
            _gpuFullOptions);

    [Benchmark]
    public int[] GpuChunked() =>
        Compute.Histogram(
            _source,
            BinCount,
            0.0f,
            1.0f,
            _gpuChunkedOptions);
}
