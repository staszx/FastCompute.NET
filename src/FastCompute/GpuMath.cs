namespace FastCompute;

/// <summary>
/// Provides the original compatibility name for
/// <see cref="ComputeMath"/> functions.
/// </summary>
/// <remarks>
/// New code should use <see cref="ComputeMath"/>. Calling these methods does
/// not force GPU execution.
/// </remarks>
public static class GpuMath
{
    /// <summary>Returns the absolute value of <paramref name="value"/>.</summary>
    public static float Abs(float value) => ComputeMath.Abs(value);

    /// <summary>Returns the smaller of two values.</summary>
    public static float Min(float left, float right) =>
        ComputeMath.Min(left, right);

    /// <summary>Returns the larger of two values.</summary>
    public static float Max(float left, float right) =>
        ComputeMath.Max(left, right);

    /// <summary>Restricts a value to the inclusive interval defined by <paramref name="min"/> and <paramref name="max"/>.</summary>
    public static float Clamp(float value, float min, float max) =>
        ComputeMath.Clamp(value, min, max);

    /// <summary>Returns the square root of <paramref name="value"/>.</summary>
    public static float Sqrt(float value) => ComputeMath.Sqrt(value);

    /// <summary>Returns the sine of <paramref name="value"/>.</summary>
    public static float Sin(float value) => ComputeMath.Sin(value);

    /// <summary>Returns the cosine of <paramref name="value"/>.</summary>
    public static float Cos(float value) => ComputeMath.Cos(value);

    /// <summary>Returns the tangent of <paramref name="value"/>.</summary>
    public static float Tan(float value) => ComputeMath.Tan(value);

    /// <summary>Returns <c>e</c> raised to <paramref name="value"/>.</summary>
    public static float Exp(float value) => ComputeMath.Exp(value);

    /// <summary>Returns the natural logarithm of <paramref name="value"/>.</summary>
    public static float Log(float value) => ComputeMath.Log(value);

    /// <summary>Returns the base-10 logarithm of <paramref name="value"/>.</summary>
    public static float Log10(float value) => ComputeMath.Log10(value);

    /// <summary>Returns <paramref name="value"/> raised to <paramref name="power"/>.</summary>
    public static float Pow(float value, float power) =>
        ComputeMath.Pow(value, power);

    /// <summary>Returns the largest integral value less than or equal to <paramref name="value"/>.</summary>
    public static float Floor(float value) => ComputeMath.Floor(value);

    /// <summary>Returns the smallest integral value greater than or equal to <paramref name="value"/>.</summary>
    public static float Ceiling(float value) => ComputeMath.Ceiling(value);

    /// <summary>Rounds <paramref name="value"/> to the nearest integral value using banker's rounding.</summary>
    public static float Round(float value) => ComputeMath.Round(value);
}
