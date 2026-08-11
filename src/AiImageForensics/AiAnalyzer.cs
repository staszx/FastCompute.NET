using AiImageForensics.Analysis;
using AiImageForensics.Scoring;

namespace AiImageForensics;

/// <summary>Provides detailed deterministic image-forensics analysis.</summary>
public static class AiAnalyzer
{
    private static readonly string[] FeatureNames =
    [
        "noise.mean", "noise.variance", "noise.stddev", "noise.skewness", "noise.kurtosis",
        "noise.correlation.x1", "noise.correlation.y1", "noise.correlation.xy1", "noise.correlation.x2", "noise.correlation.y2",
        "noise.signal.a", "noise.signal.b", "noise.signal.r_squared",
        "frequency.low", "frequency.mid", "frequency.high", "frequency.high_to_low", "frequency.peak_ratio", "frequency.peak_count", "frequency.roughness", "frequency.periodicity",
        "camera.cfa_score", "camera.cfa_confidence", "camera.rgb.rg", "camera.rgb.rb", "camera.rgb.gb", "camera.demosaicing",
        "spatial.laplacian_variance", "spatial.gradient_mean", "spatial.gradient_variance", "spatial.edge_density", "spatial.local_contrast", "spatial.entropy",
        "metadata.available", "metadata.explicit_ai"
    ];

    /// <summary>Runs enabled analyzers and returns their detailed features and evidence.</summary>
    public static AiAnalysisResult Analyze(IImagePixelSource image, AiAnalysisOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        options ??= new AiAnalysisOptions();
        return Run(image, options.ToDetectionOptions(), cancellationToken).Analysis;
    }

    /// <summary>Extracts a stable, named feature vector for future models or diagnostics.</summary>
    public static AiFeatureVector ExtractFeatures(IImagePixelSource image, AiDetectionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        options ??= new AiDetectionOptions();
        AiAnalysisResult analysis = Run(image, options, cancellationToken).Analysis;
        return CreateFeatureVector(analysis);
    }

    internal static PipelineResult Run(IImagePixelSource image, AiDetectionOptions options, CancellationToken cancellationToken)
    {
        Validate(image, options);
        cancellationToken.ThrowIfCancellationRequested();
        using var context = new AiAnalysisContext(image, options);
        var weighted = new List<WeightedAnalyzerResult>(6);
        var evidence = new List<AiEvidence>(10);
        var noise = new NoiseAnalysisResult();
        var frequency = new FrequencyAnalysisResult();
        var camera = new CameraAnalysisResult();
        var spatial = new SpatialAnalysisResult();
        var metadata = new MetadataAnalysisResult();

        if (options.AnalyzeNoise)
        {
            AiAnalyzerResult result = new NoiseAnalyzer().Analyze(image, context, cancellationToken);
            noise = (NoiseAnalysisResult)result.Details!;
            Add(result, DefaultDetectionWeights.Noise, weighted, evidence);
        }
        if (options.AnalyzeFrequency)
        {
            AiAnalyzerResult result = new FrequencyAnalyzer().Analyze(image, context, cancellationToken);
            frequency = (FrequencyAnalysisResult)result.Details!;
            Add(result, DefaultDetectionWeights.Frequency, weighted, evidence);
        }
        if (options.AnalyzeCameraPipeline)
        {
            AiAnalyzerResult result = new CameraAnalyzer().Analyze(image, context, cancellationToken);
            camera = (CameraAnalysisResult)result.Details!;
            Add(result, DefaultDetectionWeights.Camera, weighted, evidence);
        }
        if (options.AnalyzeSpatialStatistics)
        {
            AiAnalyzerResult result = new SpatialAnalyzer().Analyze(image, context, cancellationToken);
            spatial = (SpatialAnalysisResult)result.Details!;
            Add(result, DefaultDetectionWeights.Spatial, weighted, evidence);
            if (options.Mode == DetectionMode.Accurate)
            {
                AiAnalyzerResult accurate = new AccurateAnalyzer().Analyze(image, context, cancellationToken);
                Add(accurate, DefaultDetectionWeights.Spatial * 0.5f, weighted, evidence);
            }
        }
        if (options.AnalyzeMetadata)
        {
            AiAnalyzerResult result = new MetadataAnalyzer().Analyze(image, context, cancellationToken);
            metadata = (MetadataAnalysisResult)result.Details!;
            Add(result, DefaultDetectionWeights.Metadata, weighted, evidence);
        }

        (float score, float confidence) = new DetectionScoringModel().Combine(weighted);
        var analysis = new AiAnalysisResult
        {
            AiScore = score, Confidence = confidence, Noise = noise, Frequency = frequency,
            Camera = camera, Spatial = spatial, Metadata = metadata, Evidence = evidence.ToArray()
        };
        return new PipelineResult(analysis);
    }

    private static void Add(AiAnalyzerResult result, float weight, List<WeightedAnalyzerResult> weighted, List<AiEvidence> evidence)
    {
        weighted.Add(new WeightedAnalyzerResult(result, weight));
        for (int i = 0; i < result.Evidence.Count; i++) evidence.Add(result.Evidence[i]);
    }

    private static AiFeatureVector CreateFeatureVector(AiAnalysisResult analysis)
    {
        float[] auto = analysis.Noise.Autocorrelations.Concat(Enumerable.Repeat(0f, 5)).Take(5).ToArray();
        float[] channels = analysis.Camera.ChannelCorrelations.Concat(Enumerable.Repeat(0f, 3)).Take(3).ToArray();
        var values = new float[FeatureNames.Length];
        int i = 0;
        values[i++] = (float)analysis.Noise.Statistics.Mean; values[i++] = (float)analysis.Noise.Statistics.Variance;
        values[i++] = (float)analysis.Noise.Statistics.StandardDeviation; values[i++] = (float)analysis.Noise.Statistics.Skewness; values[i++] = (float)analysis.Noise.Statistics.Kurtosis;
        for (int j = 0; j < 5; j++) values[i++] = auto[j];
        values[i++] = (float)analysis.Noise.SignalModel.A; values[i++] = (float)analysis.Noise.SignalModel.B; values[i++] = (float)analysis.Noise.SignalModel.RSquared;
        values[i++] = analysis.Frequency.LowFrequencyEnergy; values[i++] = analysis.Frequency.MidFrequencyEnergy; values[i++] = analysis.Frequency.HighFrequencyEnergy;
        values[i++] = analysis.Frequency.HighToLow; values[i++] = analysis.Frequency.PeakRatio; values[i++] = analysis.Frequency.StrongPeakCount; values[i++] = analysis.Frequency.SpectrumRoughness; values[i++] = analysis.Frequency.PeriodicityScore;
        values[i++] = analysis.Camera.CfaScore; values[i++] = analysis.Camera.CfaConfidence;
        for (int j = 0; j < 3; j++) values[i++] = channels[j];
        values[i++] = analysis.Camera.DemosaicingScore;
        values[i++] = analysis.Spatial.LaplacianVariance; values[i++] = analysis.Spatial.GradientMean; values[i++] = analysis.Spatial.GradientVariance;
        values[i++] = analysis.Spatial.EdgeDensity; values[i++] = analysis.Spatial.LocalContrast; values[i++] = analysis.Spatial.LocalEntropy;
        values[i++] = analysis.Metadata.IsAvailable ? 1 : 0; values[i] = analysis.Metadata.HasExplicitAiProvenance ? 1 : 0;
        return new AiFeatureVector { Names = (string[])FeatureNames.Clone(), Values = values };
    }

    private static void Validate(IImagePixelSource image, AiDetectionOptions options)
    {
        if (image.Width <= 0) throw new ArgumentOutOfRangeException(nameof(image), "Image width must be positive.");
        if (image.Height <= 0) throw new ArgumentOutOfRangeException(nameof(image), "Image height must be positive.");
        _ = checked(image.Width * image.Height);
        if (options.DetectionThreshold is < 0 or > 1 || float.IsNaN(options.DetectionThreshold)) throw new ArgumentOutOfRangeException(nameof(options.DetectionThreshold));
        if (options.MaxDegreeOfParallelism <= 0) throw new ArgumentOutOfRangeException(nameof(options.MaxDegreeOfParallelism));
        if (!Enum.IsDefined(options.Mode)) throw new ArgumentOutOfRangeException(nameof(options.Mode));
    }

    internal readonly record struct PipelineResult(AiAnalysisResult Analysis);
}
