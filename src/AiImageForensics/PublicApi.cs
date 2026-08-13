using FastCompute;
using FastCompute.ImageProcessing;
using System.Runtime.InteropServices;

namespace AiImageForensics;

/// <summary>Provides normalized RGB pixels to the analysis pipeline.</summary>
public interface IImagePixelSource
{
    /// <summary>Gets the image width.</summary>
    int Width { get; }

    /// <summary>Gets the image height.</summary>
    int Height { get; }

    /// <summary>Copies a row as normalized RGB values in the range 0..1.</summary>
    void CopyRow(int y, Span<RgbFloat> destination);
}

/// <summary>Optionally supplies image metadata without coupling the core to an imaging library.</summary>
public interface IImageMetadataSource
{
    /// <summary>Gets available provenance and capture metadata.</summary>
    ImageMetadataInfo GetMetadata();
}

/// <summary>Optionally identifies whether supplied RGB values are already linear-light.</summary>
public interface IImageColorEncodingSource
{
    /// <summary>Gets whether RGB components are linear rather than sRGB encoded.</summary>
    bool IsLinear { get; }
}

/// <summary>Normalized floating-point RGB pixel.</summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct RgbFloat(float r, float g, float b)
{
    /// <summary>Red channel.</summary>
    public float R { get; } = r;
    /// <summary>Green channel.</summary>
    public float G { get; } = g;
    /// <summary>Blue channel.</summary>
    public float B { get; } = b;
}

/// <summary>Detection detail/performance mode.</summary>
public enum DetectionMode
{
    /// <summary>Low-cost spatial features without FFT.</summary>
    Fast,
    /// <summary>FFT, radial spectrum, noise correlation, and camera features.</summary>
    Balanced,
    /// <summary>Balanced features plus block and multi-scale analysis.</summary>
    Accurate
}

/// <summary>Category of forensic evidence.</summary>
public enum AiEvidenceType
{
    /// <summary>Explicit provenance metadata.</summary>
    Metadata,
    /// <summary>Frequency-domain statistics.</summary>
    Frequency,
    /// <summary>Noise distribution and signal dependence.</summary>
    Noise,
    /// <summary>Spatial noise correlation.</summary>
    NoiseCorrelation,
    /// <summary>Traditional sensor-pipeline consistency.</summary>
    CameraSensor,
    /// <summary>Color-filter-array consistency.</summary>
    Cfa,
    /// <summary>Demosaicing-like correlation.</summary>
    Demosaicing,
    /// <summary>Compression characteristics.</summary>
    Compression,
    /// <summary>Spatial-domain statistics.</summary>
    SpatialStatistics
}

/// <summary>Controls detector analysis.</summary>
public sealed class AiDetectionOptions
{
    /// <summary>Threshold used only to calculate IsLikelyAi.</summary>
    public float DetectionThreshold { get; set; } = 0.65f;
    /// <summary>Analysis depth.</summary>
    public DetectionMode Mode { get; set; } = DetectionMode.Balanced;
    /// <summary>Enables metadata analysis.</summary>
    public bool AnalyzeMetadata { get; set; } = true;
    /// <summary>Enables frequency analysis.</summary>
    public bool AnalyzeFrequency { get; set; } = true;
    /// <summary>Enables noise analysis.</summary>
    public bool AnalyzeNoise { get; set; } = true;
    /// <summary>Enables camera-pipeline analysis.</summary>
    public bool AnalyzeCameraPipeline { get; set; } = true;
    /// <summary>Enables spatial analysis.</summary>
    public bool AnalyzeSpatialStatistics { get; set; } = true;
    /// <summary>Maximum analysis parallelism.</summary>
    public int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;
}

/// <summary>Controls detailed analysis.</summary>
public sealed class AiAnalysisOptions
{
    /// <summary>Analysis depth.</summary>
    public DetectionMode Mode { get; set; } = DetectionMode.Balanced;
    /// <summary>Enables metadata analysis.</summary>
    public bool AnalyzeMetadata { get; set; } = true;
    /// <summary>Enables frequency analysis.</summary>
    public bool AnalyzeFrequency { get; set; } = true;
    /// <summary>Enables noise analysis.</summary>
    public bool AnalyzeNoise { get; set; } = true;
    /// <summary>Enables camera-pipeline analysis.</summary>
    public bool AnalyzeCameraPipeline { get; set; } = true;
    /// <summary>Enables spatial analysis.</summary>
    public bool AnalyzeSpatialStatistics { get; set; } = true;
    /// <summary>Maximum analysis parallelism.</summary>
    public int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;

    internal AiDetectionOptions ToDetectionOptions() => new()
    {
        Mode = Mode,
        AnalyzeMetadata = AnalyzeMetadata,
        AnalyzeFrequency = AnalyzeFrequency,
        AnalyzeNoise = AnalyzeNoise,
        AnalyzeCameraPipeline = AnalyzeCameraPipeline,
        AnalyzeSpatialStatistics = AnalyzeSpatialStatistics,
        MaxDegreeOfParallelism = MaxDegreeOfParallelism
    };
}

/// <summary>One independent evidence item.</summary>
public sealed class AiEvidence
{
    /// <summary>Evidence category.</summary>
    public AiEvidenceType Type { get; init; }
    /// <summary>Normalized detection score.</summary>
    public float Score { get; init; }
    /// <summary>Normalized reliability of this evidence.</summary>
    public float Confidence { get; init; }
    /// <summary>Human-readable, non-conclusive description.</summary>
    public string Message { get; init; } = string.Empty;
}

/// <summary>Top-level detection result.</summary>
public sealed class AiDetectionResult
{
    /// <summary>Normalized heuristic detection score.</summary>
    public float Score { get; init; }
    /// <summary>Whether Score reaches the configured threshold.</summary>
    public bool IsLikelyAi { get; init; }
    /// <summary>Normalized aggregate evidence confidence.</summary>
    public float Confidence { get; init; }
    /// <summary>Independent evidence items.</summary>
    public IReadOnlyList<AiEvidence> Evidence { get; init; } = Array.Empty<AiEvidence>();
}

/// <summary>Signal-dependent noise model.</summary>
public readonly struct NoiseSignalModel
{
    /// <summary>Signal coefficient.</summary>
    public double A { get; init; }
    /// <summary>Constant coefficient.</summary>
    public double B { get; init; }
    /// <summary>Coefficient of determination.</summary>
    public double RSquared { get; init; }
}

/// <summary>Detailed noise features.</summary>
public sealed class NoiseAnalysisResult
{
    /// <summary>Residual distribution.</summary>
    public StatisticsResult Statistics { get; init; }
    /// <summary>Residual correlations in deterministic offset order.</summary>
    public IReadOnlyList<float> Autocorrelations { get; init; } = Array.Empty<float>();
    /// <summary>Signal-dependent variance fit.</summary>
    public NoiseSignalModel SignalModel { get; init; }
}

/// <summary>Detailed frequency-domain features.</summary>
public sealed class FrequencyAnalysisResult
{
    /// <summary>Low-frequency energy fraction.</summary>
    public float LowFrequencyEnergy { get; init; }
    /// <summary>Mid-frequency energy fraction.</summary>
    public float MidFrequencyEnergy { get; init; }
    /// <summary>High-frequency energy fraction.</summary>
    public float HighFrequencyEnergy { get; init; }
    /// <summary>High-to-total energy ratio.</summary>
    public float HighToTotal { get; init; }
    /// <summary>Mid-to-total energy ratio.</summary>
    public float MidToTotal { get; init; }
    /// <summary>High-to-low energy ratio.</summary>
    public float HighToLow { get; init; }
    /// <summary>Normalized radial spectrum.</summary>
    public IReadOnlyList<float> RadialSpectrum { get; init; } = Array.Empty<float>();
    /// <summary>Largest local spectral peak relative to its neighborhood.</summary>
    public float PeakRatio { get; init; }
    /// <summary>Number of statistically strong spectral peaks.</summary>
    public int StrongPeakCount { get; init; }
    /// <summary>Normalized radial spectrum roughness.</summary>
    public float SpectrumRoughness { get; init; }
    /// <summary>Aggregate periodicity feature.</summary>
    public float PeriodicityScore { get; init; }
}

/// <summary>Detailed camera-pipeline features.</summary>
public sealed class CameraAnalysisResult
{
    /// <summary>Estimated Bayer layout, when distinguishable.</summary>
    public BayerPattern? EstimatedPattern { get; init; }
    /// <summary>Evidence consistent with a CFA pipeline.</summary>
    public float CfaScore { get; init; }
    /// <summary>CFA estimate confidence.</summary>
    public float CfaConfidence { get; init; }
    /// <summary>RGB channel correlations.</summary>
    public IReadOnlyList<float> ChannelCorrelations { get; init; } = Array.Empty<float>();
    /// <summary>Neighbor cross-channel correlations.</summary>
    public IReadOnlyList<float> NeighborCorrelations { get; init; } = Array.Empty<float>();
    /// <summary>Demosaicing-like correlation strength.</summary>
    public float DemosaicingScore { get; init; }
}

/// <summary>Detailed spatial-domain features.</summary>
public sealed class SpatialAnalysisResult
{
    /// <summary>Laplacian variance.</summary>
    public float LaplacianVariance { get; init; }
    /// <summary>Mean gradient magnitude.</summary>
    public float GradientMean { get; init; }
    /// <summary>Gradient magnitude variance.</summary>
    public float GradientVariance { get; init; }
    /// <summary>Fraction of significant edge pixels.</summary>
    public float EdgeDensity { get; init; }
    /// <summary>Mean local contrast.</summary>
    public float LocalContrast { get; init; }
    /// <summary>Global fixed-histogram entropy.</summary>
    public float LocalEntropy { get; init; }
}

/// <summary>Metadata features and availability.</summary>
public sealed class MetadataAnalysisResult
{
    /// <summary>Whether metadata was supplied by the source.</summary>
    public bool IsAvailable { get; init; }
    /// <summary>Explicit AI provenance was found.</summary>
    public bool HasExplicitAiProvenance { get; init; }
    /// <summary>Available software identifier.</summary>
    public string? Software { get; init; }
}

/// <summary>Detailed forensic analysis result.</summary>
public sealed class AiAnalysisResult
{
    /// <summary>Normalized heuristic AI score.</summary>
    public float AiScore { get; init; }
    /// <summary>Normalized aggregate confidence.</summary>
    public float Confidence { get; init; }
    /// <summary>Noise features.</summary>
    public NoiseAnalysisResult Noise { get; init; } = new();
    /// <summary>Frequency features.</summary>
    public FrequencyAnalysisResult Frequency { get; init; } = new();
    /// <summary>Camera-pipeline features.</summary>
    public CameraAnalysisResult Camera { get; init; } = new();
    /// <summary>Spatial features.</summary>
    public SpatialAnalysisResult Spatial { get; init; } = new();
    /// <summary>Metadata features.</summary>
    public MetadataAnalysisResult Metadata { get; init; } = new();
    /// <summary>Independent evidence items.</summary>
    public IReadOnlyList<AiEvidence> Evidence { get; init; } = Array.Empty<AiEvidence>();
}

/// <summary>Stable named feature vector suitable for future model input.</summary>
public sealed class AiFeatureVector
{
    /// <summary>Feature values.</summary>
    public float[] Values { get; init; } = Array.Empty<float>();
    /// <summary>Names aligned with Values.</summary>
    public string[] Names { get; init; } = Array.Empty<string>();
}

/// <summary>Imaging-library-neutral metadata.</summary>
public sealed class ImageMetadataInfo
{
    /// <summary>Software field.</summary>
    public string? Software { get; init; }
    /// <summary>Creator tool field.</summary>
    public string? CreatorTool { get; init; }
    /// <summary>Generator/provenance text.</summary>
    public string? Generator { get; init; }
    /// <summary>Camera make, used only as positive camera-pipeline context.</summary>
    public string? CameraMake { get; init; }
    /// <summary>Camera model, used only as positive camera-pipeline context.</summary>
    public string? CameraModel { get; init; }
}

/// <summary>Future classifier abstraction.</summary>
public interface IAiDetectionModel
{
    /// <summary>Predicts from the stable feature sequence.</summary>
    AiModelPrediction Predict(ReadOnlySpan<float> features);
}

/// <summary>Future model output.</summary>
public readonly struct AiModelPrediction
{
    /// <summary>Normalized score.</summary>
    public float Score { get; init; }
    /// <summary>Normalized confidence.</summary>
    public float Confidence { get; init; }
}
