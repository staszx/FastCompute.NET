namespace FastCompute;

/// <summary>Specifies the direction of a Fourier transform.</summary>
public enum FourierDirection
{
    /// <summary>Computes the unnormalized forward transform.</summary>
    Forward,

    /// <summary>Computes the inverse transform and normalizes by the element count.</summary>
    Inverse
}
