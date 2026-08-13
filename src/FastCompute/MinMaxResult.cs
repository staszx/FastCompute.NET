namespace FastCompute;

/// <summary>Contains minimum and maximum values from one numeric sequence.</summary>
public readonly struct MinMaxResult(float minimum, float maximum)
{
    /// <summary>Gets the minimum value.</summary>
    public float Minimum { get; } = minimum;

    /// <summary>Gets the maximum value.</summary>
    public float Maximum { get; } = maximum;
}
