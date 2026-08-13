using FastCompute;
using FastCompute.ImageProcessing;

namespace AiImageForensics.Analysis;

internal sealed class AccurateAnalyzer : IAiImageAnalyzer
{
    public AiAnalyzerResult Analyze(IImagePixelSource image, AiAnalysisContext context, CancellationToken cancellationToken)
    {
        ReadOnlyMemory<float> luminance = context.GetLinearLuminanceMemory(cancellationToken);
        int blocksX = (image.Width + 255) / 256, blocksY = (image.Height + 255) / 256;
        int blockCount = checked(blocksX * blocksY);
        var spatial = new double[blockCount];
        var noise = new double[blockCount];
        var frequency = new double[blockCount];
        var periodicity = new double[blockCount];

        Parallel.For(0, blocksY, new ParallelOptions
        {
            MaxDegreeOfParallelism = context.Options.MaxDegreeOfParallelism,
            CancellationToken = cancellationToken
        }, by =>
        {
            for (int bx = 0; bx < blocksX; bx++)
            {
                int x0 = bx * 256, y0 = by * 256;
                int x1 = Math.Min(image.Width, x0 + 256), y1 = Math.Min(image.Height, y0 + 256);
                int blockWidth = x1 - x0;
                int blockHeight = y1 - y0;
                float[] block = ImageRegionOperations.Crop(luminance.Span, image.Width, image.Height, x0, y0, blockWidth, blockHeight);
                var computeOptions = new ComputeOptions
                {
                    Backend = ComputeBackendKind.Auto,
                    CancellationToken = cancellationToken,
                    MaxDegreeOfParallelism = 1
                };
                StatisticsResult statistics = Compute.CalculateStatistics(block, computeOptions);
                float[] laplacian = ImageFilters.Laplacian(block, blockWidth, blockHeight, computeOptions);
                float[] gradient = ImageFilters.GradientMagnitude(block, blockWidth, blockHeight, computeOptions);
                float[][] paritySamples = BayerOperations.ExtractParitySamples(block, blockWidth, blockHeight, computeOptions);
                double[] parityMeans = paritySamples.Select(samples => Compute.Mean(samples, computeOptions)).ToArray();
                int blockIndex = (by * blocksX) + bx;
                spatial[blockIndex] = statistics.Variance;
                noise[blockIndex] = Compute.SumOfSquares(laplacian, computeOptions) / Math.Max(1, laplacian.Length);
                frequency[blockIndex] = Compute.SumOfSquares(gradient, computeOptions) / Math.Max(1, gradient.Length);
                periodicity[blockIndex] = Compute.Variance(parityMeans, computeOptions);
            }
        });

        BlockAggregate spatialAggregate = Aggregate(spatial);
        BlockAggregate noiseAggregate = Aggregate(noise);
        BlockAggregate frequencyAggregate = Aggregate(frequency);
        BlockAggregate cameraAggregate = Aggregate(periodicity);

        int halfWidth = Math.Max(1, image.Width / 2), halfHeight = Math.Max(1, image.Height / 2);
        var half = new float[checked(halfWidth * halfHeight)];
        FastCompute.ImageProcessing.ImageResampler.Downsample(
            luminance.Span, half, image.Width, image.Height, halfWidth, halfHeight, cancellationToken);
        int quarterWidth = Math.Max(1, halfWidth / 2), quarterHeight = Math.Max(1, halfHeight / 2);
        var quarter = new float[checked(quarterWidth * quarterHeight)];
        FastCompute.ImageProcessing.ImageResampler.Downsample(
            half, quarter, halfWidth, halfHeight, quarterWidth, quarterHeight, cancellationToken);
        double gradient1 = MeanGradient(luminance.Span, image.Width, image.Height);
        double gradient2 = MeanGradient(half, halfWidth, halfHeight);
        double gradient3 = MeanGradient(quarter, quarterWidth, quarterHeight);
        double scaleVariation = Math.Abs(gradient2 - gradient1) + Math.Abs(gradient3 - gradient2);

        double blockUniformity = Average(
            Uniformity(spatialAggregate), Uniformity(noiseAggregate),
            Uniformity(frequencyAggregate), Uniformity(cameraAggregate));
        double scaleRegularity = 1 / (1 + (20 * scaleVariation));
        float score = Math.Clamp((float)((0.75 * blockUniformity) + (0.25 * scaleRegularity)), 0, 1);
        float confidence = Math.Clamp(blockCount / 16f, 0.35f, 0.85f);
        return new AiAnalyzerResult
        {
            Score = score,
            Confidence = confidence,
            Evidence = [new AiEvidence
            {
                Type = AiEvidenceType.SpatialStatistics,
                Score = score,
                Confidence = confidence,
                Message = "Accurate mode aggregated mean, median, maximum, 90th percentile, and variance for block noise, frequency, camera-periodicity, and spatial features at 256-pixel blocks, plus 1.0/0.5/0.25 scales."
            }]
        };
    }

    private static BlockAggregate Aggregate(double[] values)
    {
        double mean = Compute.Mean(values);
        double variance = Compute.Variance(values);
        var medianWorking = (double[])values.Clone();
        var p90Working = (double[])values.Clone();
        return new BlockAggregate(
            mean,
            Compute.Median(medianWorking),
            values.Max(),
            Compute.Percentile(p90Working, 90d),
            variance);
    }

    private static double Uniformity(BlockAggregate aggregate) =>
        1 / (1 + (aggregate.Variance / Math.Max(aggregate.Mean * aggregate.Mean, 1e-20)) + ((aggregate.P90 - aggregate.Median) / Math.Max(aggregate.Mean, 1e-10)));

    private static double MeanGradient(ReadOnlySpan<float> values, int width, int height)
    {
        if (width < 2 || height < 2) return 0;
        float[] gradient = ImageFilters.GradientMagnitude(values, width, height);
        return Compute.Mean(gradient);
    }

    private static double Average(double a, double b, double c, double d) => (a + b + c + d) / 4;

    private readonly record struct BlockAggregate(double Mean, double Median, double Maximum, double P90, double Variance);
}
