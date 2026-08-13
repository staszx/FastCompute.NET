namespace FastCompute.ImageProcessing;

/// <summary>Contains separate normalized RGB channel buffers.</summary>
public readonly struct RgbChannels(float[] red, float[] green, float[] blue)
{
    /// <summary>Gets red-channel samples.</summary>
    public float[] Red { get; } = red;
    /// <summary>Gets green-channel samples.</summary>
    public float[] Green { get; } = green;
    /// <summary>Gets blue-channel samples.</summary>
    public float[] Blue { get; } = blue;
}
