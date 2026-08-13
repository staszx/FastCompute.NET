namespace FastCompute.ImageProcessing;

/// <summary>Provides reusable Bayer sampling and bilinear demosaicing operations.</summary>
public static class BayerOperations
{
    private static readonly float[] InterpolationKernel =
    [
        1f, 1f, 1f,
        1f, 1f, 1f,
        1f, 1f, 1f
    ];

    /// <summary>Splits a native RGB image into backend-projected channel buffers.</summary>
    public static RgbChannels SplitRgbChannels(this Image<Rgb> source, ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        Rgb[] pixels = source.Pixels.ToArray();
        ComputeOptions effective = options ?? ComputeOptions.Default;
        return new RgbChannels(
            pixels.AsCompute(effective).Select(pixel => pixel.Red).ToArray(),
            pixels.AsCompute(effective).Select(pixel => pixel.Green).ToArray(),
            pixels.AsCompute(effective).Select(pixel => pixel.Blue).ToArray());
    }

    /// <summary>Extracts one backend-projected RGB channel.</summary>
    public static float[] ExtractChannel(this Image<Rgb> source, RgbChannel channel, ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (!Enum.IsDefined(channel)) throw new ArgumentOutOfRangeException(nameof(channel));
        Rgb[] pixels = source.Pixels.ToArray();
        ComputeOptions effective = options ?? ComputeOptions.Default;
        return channel switch
        {
            RgbChannel.Red => pixels.AsCompute(effective).Select(pixel => pixel.Red).ToArray(),
            RgbChannel.Green => pixels.AsCompute(effective).Select(pixel => pixel.Green).ToArray(),
            RgbChannel.Blue => pixels.AsCompute(effective).Select(pixel => pixel.Blue).ToArray(),
            _ => throw new ArgumentOutOfRangeException(nameof(channel))
        };
    }

    /// <summary>Combines equally sized channel buffers into a native RGB image.</summary>
    public static Image<Rgb> CombineRgbChannels(
        ReadOnlySpan<float> red,
        ReadOnlySpan<float> green,
        ReadOnlySpan<float> blue,
        int width,
        int height,
        ColorEncoding encoding = ColorEncoding.Linear,
        ComputeOptions? options = null)
    {
        int length = checked(width * height);
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (red.Length < length || green.Length < length || blue.Length < length)
            throw new ArgumentException("Channel buffers are shorter than the image dimensions.");
        float[] redArray = red[..length].ToArray();
        float[] greenArray = green[..length].ToArray();
        float[] blueArray = blue[..length].ToArray();
        var pixels = GC.AllocateUninitializedArray<Rgb>(length);
        ComputeOptions effective = options ?? ComputeOptions.Default;
        Parallel.For(0, length, new ParallelOptions
        {
            CancellationToken = effective.CancellationToken,
            MaxDegreeOfParallelism = effective.MaxDegreeOfParallelism ?? -1
        }, index => pixels[index] = new Rgb(redArray[index], greenArray[index], blueArray[index]));
        return Image<Rgb>.Load(pixels, width, height, encoding);
    }

    /// <summary>Converts RGB samples to a single-channel Bayer mosaic.</summary>
    public static float[] ToBayer(this Image<Rgb> source, BayerPattern pattern = BayerPattern.Rggb, ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (!Enum.IsDefined(pattern)) throw new ArgumentOutOfRangeException(nameof(pattern));
        RgbChannels channels = source.SplitRgbChannels(options);
        (float[] redMask, float[] greenMask, float[] blueMask) = CreateMasks(source.Width, source.Height, pattern);
        float[] red = Compute.Zip(channels.Red, redMask, (value, mask) => value * mask, options);
        float[] green = Compute.Zip(channels.Green, greenMask, (value, mask) => value * mask, options);
        float[] blue = Compute.Zip(channels.Blue, blueMask, (value, mask) => value * mask, options);
        float[] result = Compute.Zip(red, green, (first, second) => first + second, options);
        return Compute.ZipInPlace(result, blue, (first, second) => first + second, options);
    }

    /// <summary>Reconstructs normalized RGB pixels from a Bayer mosaic by local bilinear interpolation.</summary>
    public static Image<Rgb> DemosaicBilinear(
        ReadOnlySpan<float> mosaic,
        int width,
        int height,
        BayerPattern pattern = BayerPattern.Rggb,
        ColorEncoding encoding = ColorEncoding.Linear,
        ComputeOptions? options = null)
    {
        Validate(mosaic, width, height, pattern);
        float[] input = mosaic[..checked(width * height)].ToArray();
        (float[] redMask, float[] greenMask, float[] blueMask) = CreateMasks(width, height, pattern);
        float[] red = InterpolateChannel(input, redMask, width, height, options);
        float[] green = InterpolateChannel(input, greenMask, width, height, options);
        float[] blue = InterpolateChannel(input, blueMask, width, height, options);
        return CombineRgbChannels(red, green, blue, width, height, encoding, options);
    }

    /// <summary>Demosaics a Bayer mosaic using the default bilinear algorithm.</summary>
    public static Image<Rgb> Demosaic(
        ReadOnlySpan<float> mosaic,
        int width,
        int height,
        BayerPattern pattern = BayerPattern.Rggb,
        ColorEncoding encoding = ColorEncoding.Linear,
        ComputeOptions? options = null) =>
        DemosaicBilinear(mosaic, width, height, pattern, encoding, options);

    /// <summary>Extracts four independent buffers for even/even, odd/even, even/odd, and odd/odd positions.</summary>
    public static float[][] ExtractParitySamples(ReadOnlySpan<float> source, int width, int height, ComputeOptions? options = null)
    {
        if (source.Length < checked(width * height)) throw new ArgumentException("Source is shorter than its dimensions.", nameof(source));
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        float[] input = source[..checked(width * height)].ToArray();
        var result = new float[4][];
        ComputeOptions effective = options ?? ComputeOptions.Default;
        Parallel.For(0, 4, new ParallelOptions
        {
            CancellationToken = effective.CancellationToken,
            MaxDegreeOfParallelism = Math.Min(4, effective.MaxDegreeOfParallelism ?? 4)
        }, parity =>
        {
            int parityX = parity & 1;
            int parityY = parity >> 1;
            int parityWidth = (width - parityX + 1) / 2;
            int parityHeight = (height - parityY + 1) / 2;
            var samples = GC.AllocateUninitializedArray<float>(checked(parityWidth * parityHeight));
            int destination = 0;
            for (int y = parityY; y < height; y += 2)
            for (int x = parityX; x < width; x += 2)
                samples[destination++] = input[(y * width) + x];
            result[parity] = samples;
        });
        return result;
    }

    /// <summary>Returns the sampled RGB channel index for one Bayer coordinate.</summary>
    public static int ChannelAt(BayerPattern pattern, int x, int y)
    {
        int parity = ((y & 1) << 1) | (x & 1);
        return pattern switch
        {
            BayerPattern.Rggb => parity == 0 ? 0 : parity == 3 ? 2 : 1,
            BayerPattern.Bggr => parity == 0 ? 2 : parity == 3 ? 0 : 1,
            BayerPattern.Grbg => parity == 1 ? 0 : parity == 2 ? 2 : 1,
            BayerPattern.Gbrg => parity == 1 ? 2 : parity == 2 ? 0 : 1,
            _ => throw new ArgumentOutOfRangeException(nameof(pattern))
        };
    }

    private static float[] InterpolateChannel(float[] mosaic, float[] mask, int width, int height, ComputeOptions? options)
    {
        float[] samples = Compute.Zip(mosaic, mask, (value, sampleMask) => value * sampleMask, options);
        float[] numerator = Compute.Convolve2D(samples, width, height, InterpolationKernel, 3, 3, options: options);
        float[] denominator = Compute.Convolve2D(mask, width, height, InterpolationKernel, 3, 3, options: options);
        return Compute.Zip(numerator, denominator, (sum, count) => sum / ComputeMath.Max(count, 1f), options);
    }

    private static (float[] Red, float[] Green, float[] Blue) CreateMasks(int width, int height, BayerPattern pattern)
    {
        int length = checked(width * height);
        var red = new float[length];
        var green = new float[length];
        var blue = new float[length];
        Parallel.For(0, height, y =>
        {
            for (int x = 0; x < width; x++)
            {
                int index = (y * width) + x;
                switch (ChannelAt(pattern, x, y))
                {
                    case 0: red[index] = 1f; break;
                    case 1: green[index] = 1f; break;
                    default: blue[index] = 1f; break;
                }
            }
        });
        return (red, green, blue);
    }

    private static void Validate(ReadOnlySpan<float> mosaic, int width, int height, BayerPattern pattern)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (mosaic.Length < checked(width * height)) throw new ArgumentException("Mosaic is shorter than its dimensions.", nameof(mosaic));
        if (!Enum.IsDefined(pattern)) throw new ArgumentOutOfRangeException(nameof(pattern));
    }
}
