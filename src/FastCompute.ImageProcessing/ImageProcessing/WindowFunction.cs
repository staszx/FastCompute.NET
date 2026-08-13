namespace FastCompute.ImageProcessing;

/// <summary>Defines a separable signal window used for image-spectrum preparation.</summary>
public enum WindowFunction
{
    /// <summary>Hann window.</summary>
    Hann,
    /// <summary>Hamming window.</summary>
    Hamming,
    /// <summary>Blackman window.</summary>
    Blackman
}
