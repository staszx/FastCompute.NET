namespace AiImageForensics;

/// <summary>Color conversion helpers shared by adapters and analysis.</summary>
public static class ColorMath
{
    /// <summary>Calculates Rec.709 luminance from normalized channels.</summary>
    public static float GetLuminance(in RgbFloat pixel) =>
        (0.2126f * pixel.R) + (0.7152f * pixel.G) + (0.0722f * pixel.B);

    /// <summary>Converts a normalized sRGB component to linear light.</summary>
    public static float SrgbToLinear(float value)
    {
        value = Math.Clamp(value, 0f, 1f);
        return value <= 0.04045f
            ? value / 12.92f
            : MathF.Pow((value + 0.055f) / 1.055f, 2.4f);
    }
}
