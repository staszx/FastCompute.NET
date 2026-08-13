using FastCompute;
using FastCompute.ImageProcessing;
using System.Runtime.InteropServices;
using AiImageForensics.Analysis;

namespace AiImageForensics;

/// <summary>Forensics operations for native FastCompute images.</summary>
public static class FastComputeImageExtensions
{
    /// <summary>Detects AI evidence in a native RGB24 image.</summary>
    public static AiDetectionResult DetectAi(this Image<Rgb24> image, AiDetectionOptions? options = null, CancellationToken cancellationToken = default) =>
        AiDetector.Detect(new Rgb24Source(image), options, cancellationToken);

    /// <summary>Detects AI evidence in a native floating-point RGB image.</summary>
    public static AiDetectionResult DetectAi(this Image<Rgb> image, AiDetectionOptions? options = null, CancellationToken cancellationToken = default) =>
        AiDetector.Detect(new RgbSource(image), options, cancellationToken);

    /// <summary>Performs detailed analysis of a native RGB24 image.</summary>
    public static AiAnalysisResult AnalyzeAi(this Image<Rgb24> image, AiAnalysisOptions? options = null, CancellationToken cancellationToken = default) =>
        AiAnalyzer.Analyze(new Rgb24Source(image), options, cancellationToken);

    /// <summary>Performs detailed analysis of a native floating-point RGB image.</summary>
    public static AiAnalysisResult AnalyzeAi(this Image<Rgb> image, AiAnalysisOptions? options = null, CancellationToken cancellationToken = default) =>
        AiAnalyzer.Analyze(new RgbSource(image), options, cancellationToken);

    /// <summary>Extracts stable features from a native RGB24 image.</summary>
    public static AiFeatureVector ExtractAiFeatures(this Image<Rgb24> image, AiDetectionOptions? options = null, CancellationToken cancellationToken = default) =>
        AiAnalyzer.ExtractFeatures(new Rgb24Source(image), options, cancellationToken);

    private sealed class Rgb24Source(Image<Rgb24> image) : IImagePixelSource, IImageColorEncodingSource
    {
        public int Width => image.Width;
        public int Height => image.Height;
        public bool IsLinear => image.Encoding == ColorEncoding.Linear;
        public void CopyRow(int y, Span<RgbFloat> destination)
        {
            if (destination.Length < Width) throw new ArgumentException("Destination is too short.", nameof(destination));
            ReadOnlySpan<Rgb24> source = image.GetReadOnlyRowSpan(y);
            PixelConverter.Convert<Rgb24, Rgb>(
                source,
                MemoryMarshal.Cast<RgbFloat, Rgb>(destination[..Width]),
                image.Encoding,
                image.Encoding);
        }
    }

    private sealed class RgbSource(Image<Rgb> image) : IImagePixelSource, IImageColorEncodingSource, ILinearLuminanceSource
    {
        public int Width => image.Width;
        public int Height => image.Height;
        public bool IsLinear => image.Encoding == ColorEncoding.Linear;
        public void CopyRow(int y, Span<RgbFloat> destination)
        {
            if (destination.Length < Width) throw new ArgumentException("Destination is too short.", nameof(destination));
            ReadOnlySpan<Rgb> source = image.GetReadOnlyRowSpan(y);
            source.CopyTo(MemoryMarshal.Cast<RgbFloat, Rgb>(destination[..Width]));
        }

        public bool TryCreateLinearLuminance(CancellationToken cancellationToken, out float[]? luminance)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!IsLinear ||
                !MemoryMarshal.TryGetArray((ReadOnlyMemory<Rgb>)image.Pixels, out ArraySegment<Rgb> segment) ||
                segment.Array is null || segment.Offset != 0 || segment.Count != segment.Array.Length)
            {
                luminance = null;
                return false;
            }

            luminance = segment.Array.AsCompute().Select(Rgb.Luminance).ToArray();
            cancellationToken.ThrowIfCancellationRequested();
            return true;
        }
    }
}
