namespace AiImageForensics.Analysis;

internal sealed class SpatialAnalyzer : IAiImageAnalyzer
{
    public AiAnalyzerResult Analyze(IImagePixelSource image, AiAnalysisContext context, CancellationToken cancellationToken)
    {
        ReadOnlySpan<float> data = context.GetLinearLuminance(cancellationToken);
        int width = image.Width, height = image.Height;
        double gradientSum = 0, gradientSquares = 0, laplacianSum = 0, laplacianSquares = 0, contrastSum = 0;
        long edgeCount = 0, count = 0;
        Span<int> histogram = stackalloc int[64];
        for (int i = 0; i < data.Length; i++) histogram[Math.Clamp((int)(data[i] * 64), 0, 63)]++;

        for (int y = 1; y < height - 1; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int x = 1; x < width - 1; x++)
            {
                int index = (y * width) + x;
                float gx = 0.5f * (data[index + 1] - data[index - 1]);
                float gy = 0.5f * (data[index + width] - data[index - width]);
                double gradient = Math.Sqrt((gx * gx) + (gy * gy));
                double laplacian = data[index - 1] + data[index + 1] + data[index - width] + data[index + width] - (4 * data[index]);
                float min = 1, max = 0;
                for (int yy = -1; yy <= 1; yy++)
                for (int xx = -1; xx <= 1; xx++) { float value = data[index + (yy * width) + xx]; min = Math.Min(min, value); max = Math.Max(max, value); }
                gradientSum += gradient; gradientSquares += gradient * gradient;
                laplacianSum += laplacian; laplacianSquares += laplacian * laplacian;
                contrastSum += max - min;
                if (gradient > 0.08) edgeCount++;
                count++;
            }
        }

        double safeCount = Math.Max(1, count);
        double gradientMean = gradientSum / safeCount;
        double laplacianMean = laplacianSum / safeCount;
        double gradientVariance = Math.Max(0, (gradientSquares / safeCount) - (gradientMean * gradientMean));
        double laplacianVariance = Math.Max(0, (laplacianSquares / safeCount) - (laplacianMean * laplacianMean));
        double entropy = 0;
        for (int i = 0; i < histogram.Length; i++) if (histogram[i] > 0) { double p = (double)histogram[i] / data.Length; entropy -= p * Math.Log2(p); }

        var details = new SpatialAnalysisResult
        {
            LaplacianVariance = (float)laplacianVariance,
            GradientMean = (float)gradientMean,
            GradientVariance = (float)gradientVariance,
            EdgeDensity = (float)(edgeCount / safeCount),
            LocalContrast = (float)(contrastSum / safeCount),
            LocalEntropy = (float)entropy
        };
        float overlyUniform = Math.Clamp((0.003f - details.LaplacianVariance) / 0.003f, 0, 1);
        float entropyRegularity = Math.Clamp((3.5f - details.LocalEntropy) / 3.5f, 0, 1);
        float score = (0.55f * overlyUniform) + (0.45f * entropyRegularity);
        float confidence = Math.Clamp((float)Math.Log10(Math.Max(10, data.Length)) / 7f, 0.25f, 1f);
        return new AiAnalyzerResult
        {
            Score = score,
            Confidence = confidence,
            Details = details,
            Evidence = [new AiEvidence { Type = AiEvidenceType.SpatialStatistics, Score = score, Confidence = confidence, Message = "Gradient, edge, contrast, entropy, and Laplacian statistics were combined as weak evidence." }]
        };
    }
}
