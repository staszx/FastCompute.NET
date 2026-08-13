namespace FastCompute.ImageProcessing;

/// <summary>Provides deterministic noise generation and backend-native application.</summary>
public static class ImageNoiseOperations
{
    /// <summary>Applies zero-mean Gaussian noise whose variance is <c>a * signal + b</c>.</summary>
    public static float[] ApplySignalDependentNoise(
        ReadOnlySpan<float> source,
        float a,
        float b,
        int randomSeed = 1,
        ComputeOptions? options = null)
    {
        if (!float.IsFinite(a) || a < 0f) throw new ArgumentOutOfRangeException(nameof(a));
        if (!float.IsFinite(b) || b < 0f) throw new ArgumentOutOfRangeException(nameof(b));
        float[] signal = source.ToArray();
        if (signal.Length == 0) return signal;
        var standardNormal = GC.AllocateUninitializedArray<float>(signal.Length);
        FillStandardNormal(standardNormal, randomSeed);
        float[] sigma = Compute.Run(signal, value => ComputeMath.Sqrt(ComputeMath.Max(0f, (a * value) + b)), options);
        float[] perturbation = Compute.Zip(sigma, standardNormal, (scale, random) => scale * random, options);
        return Compute.Zip(signal, perturbation, (value, noise) => ComputeMath.Clamp(value + noise, 0f, 1f), options);
    }

    /// <summary>Applies signal-dependent shot noise.</summary>
    public static float[] ApplyShotNoise(ReadOnlySpan<float> source, float varianceScale, int randomSeed = 1, ComputeOptions? options = null) =>
        ApplySignalDependentNoise(source, varianceScale, 0f, randomSeed, options);

    /// <summary>Applies signal-independent read noise.</summary>
    public static float[] ApplyReadNoise(ReadOnlySpan<float> source, float standardDeviation, int randomSeed = 1, ComputeOptions? options = null)
    {
        if (!float.IsFinite(standardDeviation) || standardDeviation < 0f) throw new ArgumentOutOfRangeException(nameof(standardDeviation));
        return ApplySignalDependentNoise(source, 0f, standardDeviation * standardDeviation, randomSeed, options);
    }

    private static void FillStandardNormal(Span<float> destination, int randomSeed)
    {
        var random = new Random(randomSeed);
        for (int index = 0; index < destination.Length; index += 2)
        {
            double first = Math.Max(double.Epsilon, random.NextDouble());
            double second = random.NextDouble();
            double magnitude = Math.Sqrt(-2d * Math.Log(first));
            double angle = 2d * Math.PI * second;
            destination[index] = (float)(magnitude * Math.Cos(angle));
            if (index + 1 < destination.Length) destination[index + 1] = (float)(magnitude * Math.Sin(angle));
        }
    }
}
