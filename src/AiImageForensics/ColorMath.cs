using FastCompute.ImageProcessing;

namespace AiImageForensics;

/// <summary>Color conversion helpers shared by adapters and analysis.</summary>
public static class ColorMath
{
    /// <summary>Calculates Rec.709 luminance from normalized channels.</summary>
    public static float GetLuminance(in RgbFloat pixel)
    {
        var native = new Rgb(pixel.R, pixel.G, pixel.B);
        return PixelConverter.GetLuminance(in native);
    }

    /// <summary>Converts a normalized sRGB component to linear light.</summary>
    public static float SrgbToLinear(float value) =>
        PixelConverter.SrgbToLinear(Math.Clamp(value, 0f, 1f));
}
