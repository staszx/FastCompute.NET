using FastCompute.ImageProcessing;

namespace AiImageForensics;

/// <summary>Controls deterministic camera-pipeline simulation for robustness testing.</summary>
public sealed class CameraSimulationOptions
{
    /// <summary>Bayer mosaic layout.</summary>
    public BayerPattern BayerPattern { get; set; } = BayerPattern.Rggb;
    /// <summary>Signal-dependent Gaussian noise scale.</summary>
    public float ShotNoise { get; set; }
    /// <summary>Signal-independent Gaussian noise standard deviation.</summary>
    public float ReadNoise { get; set; }
    /// <summary>Optical blur strength in approximate pixels.</summary>
    public float OpticalBlur { get; set; }
    /// <summary>Reserved chromatic-aberration strength; not applied in the initial implementation.</summary>
    public float ChromaticAberration { get; set; }
    /// <summary>Reserved vignetting strength; not applied in the initial implementation.</summary>
    public float Vignetting { get; set; }
    /// <summary>Post-demosaicing unsharp-mask amount.</summary>
    public float Sharpening { get; set; }
    /// <summary>Deterministic random seed.</summary>
    public int RandomSeed { get; set; } = 1;
}

/// <summary>Simulates selected stages of a traditional camera pipeline.</summary>
public static class CameraSimulator
{
    /// <summary>Returns a new image after optical blur, noise, CFA sampling, demosaicing, and sharpening.</summary>
    public static Image<Rgb> SimulateCamera(Image<Rgb> image, CameraSimulationOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(options);
        Validate(options);
        int width = image.Width, height = image.Height, length = image.Length;
        Rgb[] working = image.Pixels.ToArray();
        Rgb[] temporary = new Rgb[length];

        int blurRadius = Math.Min(8, (int)MathF.Ceiling(options.OpticalBlur));
        if (blurRadius > 0)
        {
            Blur(working, temporary, width, height, blurRadius, cancellationToken);
            (working, temporary) = (temporary, working);
        }

        var random = new Random(options.RandomSeed);
        var mosaic = new float[length];
        for (int y = 0; y < height; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int x = 0; x < width; x++)
            {
                int index = (y * width) + x;
                int channel = ChannelAt(options.BayerPattern, x, y);
                Rgb pixel = working[index];
                float signal = channel == 0 ? pixel.Red : channel == 1 ? pixel.Green : pixel.Blue;
                float sigma = MathF.Sqrt(MathF.Max(0, options.ShotNoise * signal) + (options.ReadNoise * options.ReadNoise));
                mosaic[index] = Math.Clamp(signal + (sigma * NextGaussian(random)), 0, 1);
            }
        }

        for (int y = 0; y < height; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int x = 0; x < width; x++)
            {
                temporary[(y * width) + x] = new Rgb(
                    Interpolate(mosaic, width, height, x, y, 0, options.BayerPattern),
                    Interpolate(mosaic, width, height, x, y, 1, options.BayerPattern),
                    Interpolate(mosaic, width, height, x, y, 2, options.BayerPattern));
            }
        }

        if (options.Sharpening > 0)
        {
            Blur(temporary, working, width, height, 1, cancellationToken);
            for (int i = 0; i < length; i++)
            {
                Rgb source = temporary[i], low = working[i];
                working[i] = new Rgb(
                    Math.Clamp(source.Red + (options.Sharpening * (source.Red - low.Red)), 0, 1),
                    Math.Clamp(source.Green + (options.Sharpening * (source.Green - low.Green)), 0, 1),
                    Math.Clamp(source.Blue + (options.Sharpening * (source.Blue - low.Blue)), 0, 1));
            }
        }
        else
        {
            (working, temporary) = (temporary, working);
        }

        return Image<Rgb>.Load(working, width, height, image.Encoding);
    }

    private static void Blur(Rgb[] source, Rgb[] destination, int width, int height, int radius, CancellationToken cancellationToken)
    {
        var horizontal = new Rgb[source.Length];
        for (int y = 0; y < height; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int x = 0; x < width; x++)
            {
                float r = 0, g = 0, b = 0; int count = 0;
                for (int sx = Math.Max(0, x - radius); sx <= Math.Min(width - 1, x + radius); sx++) { Rgb p = source[(y * width) + sx]; r += p.Red; g += p.Green; b += p.Blue; count++; }
                horizontal[(y * width) + x] = new Rgb(r / count, g / count, b / count);
            }
        }
        for (int y = 0; y < height; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int x = 0; x < width; x++)
            {
                float r = 0, g = 0, b = 0; int count = 0;
                for (int sy = Math.Max(0, y - radius); sy <= Math.Min(height - 1, y + radius); sy++) { Rgb p = horizontal[(sy * width) + x]; r += p.Red; g += p.Green; b += p.Blue; count++; }
                destination[(y * width) + x] = new Rgb(r / count, g / count, b / count);
            }
        }
    }

    private static float Interpolate(float[] mosaic, int width, int height, int x, int y, int channel, BayerPattern pattern)
    {
        if (ChannelAt(pattern, x, y) == channel) return mosaic[(y * width) + x];
        float sum = 0; int count = 0;
        for (int yy = Math.Max(0, y - 1); yy <= Math.Min(height - 1, y + 1); yy++)
        for (int xx = Math.Max(0, x - 1); xx <= Math.Min(width - 1, x + 1); xx++)
            if (ChannelAt(pattern, xx, yy) == channel) { sum += mosaic[(yy * width) + xx]; count++; }
        if (count > 0) return sum / count;
        return mosaic[(y * width) + x];
    }

    private static int ChannelAt(BayerPattern pattern, int x, int y)
    {
        int parity = ((y & 1) << 1) | (x & 1);
        return pattern switch
        {
            BayerPattern.Rggb => parity == 0 ? 0 : parity == 3 ? 2 : 1,
            BayerPattern.Bggr => parity == 0 ? 2 : parity == 3 ? 0 : 1,
            BayerPattern.Grbg => parity == 1 ? 0 : parity == 2 ? 2 : 1,
            BayerPattern.Gbrg => parity == 1 ? 2 : parity == 2 ? 0 : 1,
            _ => 1
        };
    }

    private static float NextGaussian(Random random)
    {
        double u1 = Math.Max(double.Epsilon, random.NextDouble());
        double u2 = random.NextDouble();
        return (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
    }

    private static void Validate(CameraSimulationOptions options)
    {
        if (options.ShotNoise < 0 || options.ReadNoise < 0 || options.OpticalBlur < 0 || options.ChromaticAberration < 0 || options.Vignetting < 0 || options.Sharpening < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Simulation strengths must be non-negative.");
        if (!Enum.IsDefined(options.BayerPattern)) throw new ArgumentOutOfRangeException(nameof(options.BayerPattern));
    }
}
