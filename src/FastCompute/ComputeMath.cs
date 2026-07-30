namespace FastCompute;

/// <summary>
/// Provides backend-independent mathematical functions supported in
/// FastCompute expressions.
/// </summary>
/// <remarks>
/// Using these methods does not select or force the GPU backend. The same
/// expression can execute on Scalar CPU, Parallel CPU, SIMD, or GPU.
/// </remarks>
public static class ComputeMath
{
    /// <summary>Returns the absolute value of <paramref name="value"/>.</summary>
    public static float Abs(float value) => MathF.Abs(value);

    /// <summary>Returns the smaller of two values.</summary>
    public static float Min(float left, float right) =>
        MathF.Min(left, right);

    /// <summary>Returns the larger of two values.</summary>
    public static float Max(float left, float right) =>
        MathF.Max(left, right);

    /// <summary>Restricts a value to the inclusive interval.</summary>
    public static float Clamp(float value, float min, float max) =>
        Math.Clamp(value, min, max);

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

    /// <summary>Returns the natural logarithm.</summary>
    public static float Log(float value) => MathF.Log(value);

    /// <summary>Returns the base-10 logarithm.</summary>
    public static float Log10(float value) => MathF.Log10(value);

    /// <summary>Raises <paramref name="value"/> to a power.</summary>
    public static float Pow(float value, float power) =>
        MathF.Pow(value, power);

    /// <summary>Returns the floor of <paramref name="value"/>.</summary>
    public static float Floor(float value) => MathF.Floor(value);

    /// <summary>Returns the ceiling of <paramref name="value"/>.</summary>
    public static float Ceiling(float value) => MathF.Ceiling(value);

    /// <summary>Rounds <paramref name="value"/> to the nearest integer.</summary>
    public static float Round(float value) => MathF.Round(value);
}
