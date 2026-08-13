namespace FastCompute;

public static partial class Compute
{
    /// <summary>Calculates minimum and maximum using the selected reduction backend.</summary>
    public static MinMaxResult MinMax(ReadOnlySpan<float> values, ComputeOptions? options = null)
    {
        if (values.IsEmpty) throw new InvalidOperationException("Sequence contains no elements.");
        float[] data = values.ToArray();
        return new MinMaxResult(Min(data, options), Max(data, options));
    }

    /// <summary>Normalizes values to the inclusive zero-to-one interval.</summary>
    public static float[] Normalize(ReadOnlySpan<float> values, ComputeOptions? options = null)
    {
        if (values.IsEmpty) return [];
        float[] result = values.ToArray();
        MinMaxResult bounds = MinMax(result, options);
        float range = bounds.Maximum - bounds.Minimum;
        if (range <= 1e-30f) return RunInPlace(result, _ => 0f, options);
        float minimum = bounds.Minimum;
        return RunInPlace(result, value => (value - minimum) / range, options);
    }

    /// <summary>Divides corresponding values and substitutes a caller value when the denominator is near zero.</summary>
    public static float[] SafeDivide(
        ReadOnlySpan<float> numerator,
        ReadOnlySpan<float> denominator,
        float zeroResult = 0f,
        float epsilon = 1e-30f,
        ComputeOptions? options = null)
    {
        if (numerator.Length != denominator.Length) throw new ArgumentException("Input spans must have equal lengths.");
        if (!float.IsFinite(epsilon) || epsilon < 0f) throw new ArgumentOutOfRangeException(nameof(epsilon));
        float[] left = numerator.ToArray();
        float[] right = denominator.ToArray();
        float safeEpsilon = epsilon;
        if (safeEpsilon == 0f)
            return Zip(left, right, (first, second) => first / second, options);

        float[] safeMask = Threshold(Run(right, value => ComputeMath.Abs(value), options), safeEpsilon, options);
        float[] unsafeMask = Run(safeMask, value => 1f - value, options);
        float[] safeDenominator = Zip(right, unsafeMask, (value, missing) => value + missing, options);
        float[] quotient = Zip(left, safeDenominator, (first, second) => first / second, options);
        float[] retained = Zip(quotient, safeMask, (value, safe) => value * safe, options);
        float[] replacement = Run(unsafeMask, missing => missing * zeroResult, options);
        return Zip(retained, replacement, (value, fallback) => value + fallback, options);
    }
}
