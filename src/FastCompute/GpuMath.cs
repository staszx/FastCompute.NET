namespace FastCompute;

/// <summary>
/// Provides mathematical functions supported in FastCompute expressions.
/// </summary>
public static class GpuMath
{
    /// <summary>Returns the absolute value of <paramref name="value"/>.</summary>
    public static float Abs(float value) => MathF.Abs(value);

    /// <summary>Returns the smaller of two values.</summary>
    public static float Min(float left, float right) => MathF.Min(left, right);

    /// <summary>Returns the larger of two values.</summary>
    public static float Max(float left, float right) => MathF.Max(left, right);

    /// <summary>Restricts a value to the inclusive interval defined by <paramref name="min"/> and <paramref name="max"/>.</summary>
    public static float Clamp(float value, float min, float max) => Math.Clamp(value, min, max);

    /// <summary>Returns the square root of <paramref name="value"/>.</summary>
    public static float Sqrt(float value) => MathF.Sqrt(value);

    /// <summary>Returns the sine of <paramref name="value"/>.</summary>
    public static float Sin(float value) => MathF.Sin(value);

    /// <summary>Returns the cosine of <paramref name="value"/>.</summary>
    public static float Cos(float value) => MathF.Cos(value);

    /// <summary>Returns the tangent of <paramref name="value"/>.</summary>
    public static float Tan(float value) => MathF.Tan(value);

    /// <summary>Returns <c>e</c> raised to <paramref name="value"/>.</summary>
    public static float Exp(float value) => MathF.Exp(value);

    /// <summary>Returns the natural logarithm of <paramref name="value"/>.</summary>
    public static float Log(float value) => MathF.Log(value);

    /// <summary>Returns the base-10 logarithm of <paramref name="value"/>.</summary>
    public static float Log10(float value) => MathF.Log10(value);

    /// <summary>Returns <paramref name="value"/> raised to <paramref name="power"/>.</summary>
    public static float Pow(float value, float power) => MathF.Pow(value, power);

    /// <summary>Returns the largest integral value less than or equal to <paramref name="value"/>.</summary>
    public static float Floor(float value) => MathF.Floor(value);

    /// <summary>Returns the smallest integral value greater than or equal to <paramref name="value"/>.</summary>
    public static float Ceiling(float value) => MathF.Ceiling(value);

    /// <summary>Rounds <paramref name="value"/> to the nearest integral value using banker's rounding.</summary>
    public static float Round(float value) => MathF.Round(value);
}
