using System.Runtime.InteropServices;

namespace FastCompute.ImageProcessing;

/// <summary>Represents one floating-point grayscale pixel.</summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct GrayF32(float value) : IComputeValue<GrayF32>
{
    /// <summary>Gets the native FastCompute component layout.</summary>
    public static ComputeValueDescriptor<GrayF32> ComputeDescriptor { get; } =
        ComputeValueDescriptor<GrayF32>.Create(pixel => pixel.Value);

    /// <summary>Gets the grayscale value.</summary>
    public float Value { get; } = value;
}
