using System.Buffers;
using System.Runtime.InteropServices;
using FastCompute;
using FastCompute.ImageProcessing;

namespace AiImageForensics.Analysis;

internal sealed class AiAnalysisContext : IDisposable
{
    private readonly IImagePixelSource source;
    private float[]? luminance;
    private float[]? residual;
    private Image<Rgb>? rgbImage;
    private bool disposed;

    public AiAnalysisContext(IImagePixelSource source, AiDetectionOptions options)
    {
        this.source = source;
        Options = options;
        Width = source.Width;
        Height = source.Height;
    }

    public int Width { get; }
    public int Height { get; }
    public AiDetectionOptions Options { get; }

    public ReadOnlySpan<float> GetLinearLuminance(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (luminance is not null) return luminance.AsSpan(0, checked(Width * Height));

        int length = checked(Width * Height);
        if (source is ILinearLuminanceSource optimized &&
            optimized.TryCreateLinearLuminance(cancellationToken, out float[]? optimizedLuminance))
        {
            if (optimizedLuminance is null || optimizedLuminance.Length != length)
                throw new InvalidOperationException("The optimized luminance source returned an invalid buffer.");
            luminance = optimizedLuminance;
            return luminance;
        }

        Image<Rgb> native = GetRgbImage(cancellationToken);
        var gray = GC.AllocateUninitializedArray<GrayF32>(length);
        PixelConverter.Convert<Rgb, GrayF32>(
            native.Pixels.Span,
            gray,
            native.Encoding,
            ColorEncoding.Linear,
            new ComputeOptions
            {
                Backend = ComputeBackendKind.Auto,
                CancellationToken = cancellationToken,
                MaxDegreeOfParallelism = Options.MaxDegreeOfParallelism
            });
        luminance = MemoryMarshal.Cast<GrayF32, float>(gray).ToArray();

        return luminance.AsSpan(0, length);
    }

    public Image<Rgb> GetRgbImage(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (rgbImage is not null) return rgbImage;
        int length = checked(Width * Height);
        var pixels = GC.AllocateUninitializedArray<RgbFloat>(length);
        for (int y = 0; y < Height; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            source.CopyRow(y, pixels.AsSpan(y * Width, Width));
        }
        Rgb[] native = MemoryMarshal.Cast<RgbFloat, Rgb>(pixels).ToArray();
        bool isLinear = source is IImageColorEncodingSource encoding && encoding.IsLinear;
        rgbImage = Image<Rgb>.Load(native, Width, Height, isLinear ? ColorEncoding.Linear : ColorEncoding.Srgb);
        return rgbImage;
    }

    public ReadOnlyMemory<float> GetLinearLuminanceMemory(CancellationToken cancellationToken)
    {
        _ = GetLinearLuminance(cancellationToken);
        return luminance.AsMemory(0, checked(Width * Height));
    }

    public ReadOnlySpan<float> GetResidual(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (residual is not null) return residual.AsSpan(0, checked(Width * Height));

        ReadOnlySpan<float> sourceLuminance = GetLinearLuminance(cancellationToken);
        int length = sourceLuminance.Length;
        residual = ImageFilters.ExtractResidual(
            sourceLuminance,
            Width,
            Height,
            1,
            new ComputeOptions
            {
                Backend = ComputeBackendKind.Auto,
                CancellationToken = cancellationToken,
                MaxDegreeOfParallelism = Options.MaxDegreeOfParallelism
            });
        return residual.AsSpan(0, length);
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        luminance = null;
        residual = null;
        rgbImage = null;
    }
}
