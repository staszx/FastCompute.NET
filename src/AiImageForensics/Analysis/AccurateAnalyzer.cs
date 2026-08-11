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
            ReadOnlySpan<float> pixels = luminance.Span;
            Span<double> paritySums = stackalloc double[4];
            Span<int> parityCounts = stackalloc int[4];
            Span<double> means = stackalloc double[4];
            for (int bx = 0; bx < blocksX; bx++)
            {
                paritySums.Clear();
                parityCounts.Clear();
                means.Clear();
                int x0 = bx * 256, y0 = by * 256;
                int x1 = Math.Min(image.Width, x0 + 256), y1 = Math.Min(image.Height, y0 + 256);
                double sum = 0, sumSquares = 0, laplacianEnergy = 0, gradientEnergy = 0;
                long count = 0;
                for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++)
                {
                    double value = pixels[(y * image.Width) + x];
                    sum += value; sumSquares += value * value; count++;
                    int parity = ((y & 1) << 1) | (x & 1);
                    paritySums[parity] += value; parityCounts[parity]++;
                    if (x > x0 && x + 1 < x1 && y > y0 && y + 1 < y1)
                    {
                        int index = (y * image.Width) + x;
                        double gx = pixels[index + 1] - pixels[index - 1];
                        double gy = pixels[index + image.Width] - pixels[index - image.Width];
                        double lap = pixels[index - 1] + pixels[index + 1] + pixels[index - image.Width] + pixels[index + image.Width] - (4 * value);
                        gradientEnergy += (gx * gx) + (gy * gy);
                        laplacianEnergy += lap * lap;
                    }
                }

                double safeCount = Math.Max(1, count);
                double mean = sum / safeCount;
                int blockIndex = (by * blocksX) + bx;
                spatial[blockIndex] = Math.Max(0, (sumSquares / safeCount) - (mean * mean));
                noise[blockIndex] = laplacianEnergy / safeCount;
                frequency[blockIndex] = gradientEnergy / safeCount;
                double parityMean = 0;
                for (int p = 0; p < 4; p++) { means[p] = paritySums[p] / Math.Max(1, parityCounts[p]); parityMean += means[p]; }
                parityMean /= 4;
                double parityVariance = 0;
                for (int p = 0; p < 4; p++) { double d = means[p] - parityMean; parityVariance += d * d; }
                periodicity[blockIndex] = parityVariance / 4;
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
        double sum = 0; for (int i = 0; i < values.Length; i++) sum += values[i];
        double mean = sum / values.Length;
        double variance = 0; for (int i = 0; i < values.Length; i++) { double d = values[i] - mean; variance += d * d; }
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        return new BlockAggregate(
            mean,
            sorted[sorted.Length / 2],
            sorted[^1],
            sorted[Math.Min(sorted.Length - 1, (int)(sorted.Length * 0.9))],
            variance / values.Length);
    }

    private static double Uniformity(BlockAggregate aggregate) =>
        1 / (1 + (aggregate.Variance / Math.Max(aggregate.Mean * aggregate.Mean, 1e-20)) + ((aggregate.P90 - aggregate.Median) / Math.Max(aggregate.Mean, 1e-10)));

    private static double MeanGradient(ReadOnlySpan<float> values, int width, int height)
    {
        if (width < 2 || height < 2) return 0;
        double sum = 0; long count = 0;
        for (int y = 0; y < height - 1; y++)
        for (int x = 0; x < width - 1; x++)
        {
            int index = (y * width) + x;
            sum += Math.Abs(values[index + 1] - values[index]) + Math.Abs(values[index + width] - values[index]);
            count += 2;
        }
        return sum / Math.Max(1, count);
    }

    private static double Average(double a, double b, double c, double d) => (a + b + c + d) / 4;

    private readonly record struct BlockAggregate(double Mean, double Median, double Maximum, double P90, double Variance);
}
