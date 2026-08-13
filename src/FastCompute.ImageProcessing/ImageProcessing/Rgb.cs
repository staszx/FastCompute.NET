using System.Linq.Expressions;
using System.Runtime.InteropServices;

namespace FastCompute.ImageProcessing;

/// <summary>
/// Represents a pixel with floating-point red, green, and blue channels.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct Rgb : IComputeValue<Rgb>
{
    private const float RedWeight = 0.2126f;
    private const float GreenWeight = 0.7152f;
    private const float BlueWeight = 0.0722f;

    /// <summary>
    /// Gets the native FastCompute component layout.
    /// </summary>
    public static ComputeValueDescriptor<Rgb> ComputeDescriptor { get; } =
        ComputeValueDescriptor<Rgb>.Create(
            pixel => pixel.Red,
            pixel => pixel.Green,
            pixel => pixel.Blue);

    /// <summary>
    /// Gets the backend-portable normalized luminance projection.
    /// </summary>
    public static Expression<Func<Rgb, float>> Luminance { get; } =
        pixel =>
            (RedWeight * pixel.Red) +
            (GreenWeight * pixel.Green) +
            (BlueWeight * pixel.Blue);

    /// <summary>
    /// Gets the backend-portable in-place grayscale transformation.
    /// </summary>
    public static Expression<Func<Rgb, Rgb>> Grayscale { get; } =
        pixel => new Rgb(
            (RedWeight * pixel.Red) +
            (GreenWeight * pixel.Green) +
            (BlueWeight * pixel.Blue),
            (RedWeight * pixel.Red) +
            (GreenWeight * pixel.Green) +
            (BlueWeight * pixel.Blue),
            (RedWeight * pixel.Red) +
            (GreenWeight * pixel.Green) +
            (BlueWeight * pixel.Blue));

    /// <summary>Gets the backend-portable GrayF32 conversion.</summary>
    public static Expression<Func<Rgb, GrayF32>> GrayscaleF32 { get; } =
        pixel => new GrayF32(
            (RedWeight * pixel.Red) +
            (GreenWeight * pixel.Green) +
            (BlueWeight * pixel.Blue));

    /// <summary>
    /// Initializes a new RGB pixel.
    /// </summary>
    public Rgb(float red, float green, float blue)
    {
        Red = red;
        Green = green;
        Blue = blue;
    }

    /// <summary>
    /// Gets or sets the red channel.
    /// </summary>
    public float Red { get; set; }

    /// <summary>
    /// Gets or sets the green channel.
    /// </summary>
    public float Green { get; set; }

    /// <summary>
    /// Gets or sets the blue channel.
    /// </summary>
    public float Blue { get; set; }

    /// <summary>
    /// Subtracts the channels of one pixel from another.
    /// </summary>
    public static Rgb operator -(Rgb left, Rgb right) =>
        new(
            left.Red - right.Red,
            left.Green - right.Green,
            left.Blue - right.Blue);

    /// <summary>
    /// Divides one pixel channel by channel by another.
    /// </summary>
    public static Rgb operator /(Rgb left, Rgb right) =>
        new(
            left.Red / right.Red,
            left.Green / right.Green,
            left.Blue / right.Blue);

    /// <summary>
    /// Divides a scalar by each channel of a pixel.
    /// </summary>
    public static Rgb operator /(float left, Rgb right) =>
        new(left / right.Red, left / right.Green, left / right.Blue);

    /// <summary>
    /// Divides every pixel channel by a scalar.
    /// </summary>
    public static Rgb operator /(Rgb left, float right) =>
        new(left.Red / right, left.Green / right, left.Blue / right);

    /// <summary>
    /// Multiplies two pixels channel by channel.
    /// </summary>
    public static Rgb operator *(Rgb left, Rgb right) =>
        new(
            left.Red * right.Red,
            left.Green * right.Green,
            left.Blue * right.Blue);

    /// <summary>
    /// Multiplies every pixel channel by a scalar.
    /// </summary>
    public static Rgb operator *(Rgb left, float right) =>
        new(left.Red * right, left.Green * right, left.Blue * right);

    /// <summary>
    /// Replaces all channel values.
    /// </summary>
    public void Update(float red, float green, float blue)
    {
        Red = red;
        Green = green;
        Blue = blue;
    }

    /// <inheritdoc />
    public override readonly string ToString() => $"{Red},{Green},{Blue}";
}
