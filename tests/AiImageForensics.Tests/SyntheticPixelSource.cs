namespace AiImageForensics.Tests;

public sealed class SyntheticPixelSource : IImagePixelSource
{
    private readonly RgbFloat[] pixels;

    public SyntheticPixelSource(int width, int height, Func<int, int, RgbFloat> generator)
    {
        Width = width; Height = height; pixels = new RgbFloat[checked(width * height)];
        for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) pixels[(y * width) + x] = generator(x, y);
    }

    public int Width { get; }
    public int Height { get; }
    public void CopyRow(int y, Span<RgbFloat> destination) => pixels.AsSpan(y * Width, Width).CopyTo(destination);

    public static SyntheticPixelSource Solid(int width, int height, float value) => new(width, height, (_, _) => new RgbFloat(value, value, value));
    public static SyntheticPixelSource Checkerboard(int width, int height) => new(width, height, (x, y) => { float v = ((x + y) & 1) == 0 ? 0 : 1; return new RgbFloat(v, v, v); });
    public static SyntheticPixelSource Sine(int width, int height) => new(width, height, (x, _) => { float v = 0.5f + (0.4f * MathF.Sin(2 * MathF.PI * x / 16)); return new RgbFloat(v, v, v); });
}
