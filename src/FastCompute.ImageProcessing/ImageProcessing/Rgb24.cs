using System.Runtime.InteropServices;

namespace FastCompute.ImageProcessing;

/// <summary>Represents one tightly packed 24-bit RGB pixel.</summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly struct Rgb24(byte red, byte green, byte blue)
    : IComputeValue<Rgb24>
{
    /// <summary>Gets the native FastCompute byte-component layout.</summary>
    public static ComputeValueDescriptor<Rgb24> ComputeDescriptor { get; } =
        ComputeValueDescriptor<Rgb24>.Create(
            pixel => pixel.Red,
            pixel => pixel.Green,
            pixel => pixel.Blue);

    /// <summary>Gets the red channel.</summary>
    public byte Red { get; } = red;

    /// <summary>Gets the green channel.</summary>
    public byte Green { get; } = green;

    /// <summary>Gets the blue channel.</summary>
    public byte Blue { get; } = blue;
}
