using FastCompute;
using FastCompute.ImageProcessing;

namespace AiImageForensics.Analysis;

internal sealed class SpatialAnalyzer : IAiImageAnalyzer
{
    public AiAnalyzerResult Analyze(IImagePixelSource image, AiAnalysisContext context, CancellationToken cancellationToken)
    {
        ReadOnlySpan<float> data = context.GetLinearLuminance(cancellationToken);
        int width = image.Width, height = image.Height;
        var computeOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Auto,
            CancellationToken = cancellationToken
        };
        float[] gradient = ImageFilters.GradientMagnitude(data, width, height, computeOptions);
        float[] laplacian = ImageFilters.Laplacian(data, width, height, computeOptions);
        float[] edges = Compute.Threshold(gradient, 0.08f, computeOptions);
        float[] contrast = ImageSpatialOperations.LocalContrast(data, width, height, options: computeOptions);
        float[] localEntropy = ImageSpatialOperations.LocalEntropy(data, width, height, radius: 1, binCount: 16, options: computeOptions);
        StatisticsResult gradientStatistics = Compute.CalculateStatistics(gradient, computeOptions);
        StatisticsResult laplacianStatistics = Compute.CalculateStatistics(laplacian, computeOptions);

        var details = new SpatialAnalysisResult
        {
            LaplacianVariance = (float)laplacianStatistics.Variance,
            GradientMean = (float)gradientStatistics.Mean,
            GradientVariance = (float)gradientStatistics.Variance,
            EdgeDensity = (float)Compute.Mean(edges, computeOptions),
            LocalContrast = (float)Compute.Mean(contrast, computeOptions),
            LocalEntropy = (float)Compute.Mean(localEntropy, computeOptions)
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
