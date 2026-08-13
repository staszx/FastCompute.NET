namespace FastCompute.ImageProcessing;

/// <summary>Controls deterministic camera-pipeline simulation.</summary>
public sealed class CameraSimulationOptions
{
    /// <summary>Bayer mosaic layout.</summary>
    public BayerPattern BayerPattern { get; set; } = BayerPattern.Rggb;
    /// <summary>Signal-dependent Gaussian noise variance scale.</summary>
    public float ShotNoise { get; set; }
    /// <summary>Signal-independent Gaussian noise standard deviation.</summary>
    public float ReadNoise { get; set; }
    /// <summary>Optical blur strength in approximate pixels.</summary>
    public float OpticalBlur { get; set; }
    /// <summary>Reserved chromatic-aberration strength.</summary>
    public float ChromaticAberration { get; set; }
    /// <summary>Reserved vignetting strength.</summary>
    public float Vignetting { get; set; }
    /// <summary>Post-demosaicing sharpening amount.</summary>
    public float Sharpening { get; set; }
    /// <summary>Deterministic random seed.</summary>
    public int RandomSeed { get; set; } = 1;
    /// <summary>Optional execution settings shared by numerical stages.</summary>
    public ComputeOptions? ComputeOptions { get; set; }
}

/// <summary>Simulates reusable stages of a traditional camera image pipeline.</summary>
public static class CameraSimulator
{
    /// <summary>Returns a new image after optical blur, sensor noise, CFA sampling, demosaicing, and sharpening.</summary>
    public static Image<Rgb> SimulateCamera(Image<Rgb> image, CameraSimulationOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(options);
        Validate(options);
        ComputeOptions computeOptions = MergeOptions(options.ComputeOptions, cancellationToken);
        Image<Rgb> working = image;

        int blurRadius = Math.Min(8, (int)MathF.Ceiling(options.OpticalBlur));
        if (blurRadius > 0)
        {
            RgbChannels channels = working.SplitRgbChannels(computeOptions);
            float[] red = ImageFilters.GaussianBlur(channels.Red, image.Width, image.Height, blurRadius, options: computeOptions);
            float[] green = ImageFilters.GaussianBlur(channels.Green, image.Width, image.Height, blurRadius, options: computeOptions);
            float[] blue = ImageFilters.GaussianBlur(channels.Blue, image.Width, image.Height, blurRadius, options: computeOptions);
            working = BayerOperations.CombineRgbChannels(red, green, blue, image.Width, image.Height, image.Encoding, computeOptions);
        }

        float[] mosaic = working.ToBayer(options.BayerPattern, computeOptions);
        if (options.ShotNoise > 0f || options.ReadNoise > 0f)
        {
            mosaic = ImageNoiseOperations.ApplySignalDependentNoise(
                mosaic,
                options.ShotNoise,
                options.ReadNoise * options.ReadNoise,
                options.RandomSeed,
                computeOptions);
        }
        Image<Rgb> demosaiced = BayerOperations.DemosaicBilinear(mosaic, image.Width, image.Height, options.BayerPattern, image.Encoding, computeOptions);

        if (options.Sharpening <= 0f) return demosaiced;
        RgbChannels reconstructed = demosaiced.SplitRgbChannels(computeOptions);
        float[] sharpenedRed = Clamp(ImageFilters.Sharpen(reconstructed.Red, image.Width, image.Height, options.Sharpening, computeOptions), computeOptions);
        float[] sharpenedGreen = Clamp(ImageFilters.Sharpen(reconstructed.Green, image.Width, image.Height, options.Sharpening, computeOptions), computeOptions);
        float[] sharpenedBlue = Clamp(ImageFilters.Sharpen(reconstructed.Blue, image.Width, image.Height, options.Sharpening, computeOptions), computeOptions);
        return BayerOperations.CombineRgbChannels(sharpenedRed, sharpenedGreen, sharpenedBlue, image.Width, image.Height, image.Encoding, computeOptions);
    }

    private static float[] Clamp(float[] values, ComputeOptions options) =>
        Compute.RunInPlace(values, value => ComputeMath.Clamp(value, 0f, 1f), options);

    private static ComputeOptions MergeOptions(ComputeOptions? source, CancellationToken cancellationToken) => new()
    {
        Backend = source?.Backend ?? ComputeBackendKind.Auto,
        AllowFallback = source?.AllowFallback ?? true,
        CancellationToken = cancellationToken == default ? source?.CancellationToken ?? default : cancellationToken,
        MaxDegreeOfParallelism = source?.MaxDegreeOfParallelism,
        OptimizationMode = source?.OptimizationMode ?? ComputeOptimizationMode.Strict,
        Thresholds = source?.Thresholds ?? new ComputeThresholdOptions(),
        GpuMemoryBudgetBytes = source?.GpuMemoryBudgetBytes,
        EnableGpuChunking = source?.EnableGpuChunking ?? true,
        GpuChunkElementCount = source?.GpuChunkElementCount,
        EnableGpuStreaming = source?.EnableGpuStreaming ?? false,
        PreferredGpuAcceleratorIndex = source?.PreferredGpuAcceleratorIndex,
        GpuContext = source?.GpuContext
    };

    private static void Validate(CameraSimulationOptions options)
    {
        if (options.ShotNoise < 0 || options.ReadNoise < 0 || options.OpticalBlur < 0 || options.ChromaticAberration < 0 || options.Vignetting < 0 || options.Sharpening < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Simulation strengths must be non-negative.");
        if (!Enum.IsDefined(options.BayerPattern)) throw new ArgumentOutOfRangeException(nameof(options.BayerPattern));
    }
}
