using System.Runtime.InteropServices;

namespace FastCompute.ImageProcessing;

/// <summary>Represents one 8-bit grayscale pixel.</summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly struct Gray8(byte value) : IComputeValue<Gray8>
{
    /// <summary>Gets the native FastCompute byte-component layout.</summary>
    public static ComputeValueDescriptor<Gray8> ComputeDescriptor { get; } =
        ComputeValueDescriptor<Gray8>.Create(pixel => pixel.Value);

    /// <summary>Gets the grayscale value.</summary>
    public byte Value { get; } = value;
}
