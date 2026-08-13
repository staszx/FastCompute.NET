namespace FastCompute;

/// <summary>Defines how convolution samples outside a numeric buffer are handled.</summary>
public enum ConvolutionBoundary
{
    /// <summary>Uses the nearest element inside the buffer.</summary>
    Clamp,

    /// <summary>Uses zero outside the buffer.</summary>
    Zero
}
