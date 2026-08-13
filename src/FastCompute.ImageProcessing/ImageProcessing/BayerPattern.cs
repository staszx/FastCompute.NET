namespace FastCompute.ImageProcessing;

/// <summary>Defines a two-by-two Bayer color-filter-array layout.</summary>
public enum BayerPattern
{
    /// <summary>RG/GB layout.</summary>
    Rggb,
    /// <summary>BG/GR layout.</summary>
    Bggr,
    /// <summary>GR/BG layout.</summary>
    Grbg,
    /// <summary>GB/RG layout.</summary>
    Gbrg
}
