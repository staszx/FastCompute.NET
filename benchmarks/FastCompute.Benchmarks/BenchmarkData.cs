using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

public abstract class BenchmarkData
{
    [Params(1_000, 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000)]
    public int Count { get; set; }

    protected float[] Source { get; private set; } = null!;

    [GlobalSetup]
    public virtual void Setup()
    {
        Source = new float[Count];
        for (int index = 0; index < Source.Length; index++)
        {
            Source[index] = (index % 10_000) / 10_000.0f;
        }
    }
}
