using System.Buffers;
using FastCompute.ImageProcessing;

namespace AiImageForensics.Analysis;

internal sealed class AiAnalysisContext : IDisposable
{
    private readonly IImagePixelSource source;
    private float[]? luminance;
    private float[]? residual;
    private bool luminanceIsPooled;
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
            luminanceIsPooled = false;
            return luminance;
        }

        luminance = ArrayPool<float>.Shared.Rent(length);
        luminanceIsPooled = true;
        RgbFloat[] row = ArrayPool<RgbFloat>.Shared.Rent(Width);
        try
        {
            bool isLinear = source is IImageColorEncodingSource encoding && encoding.IsLinear;
            for (int y = 0; y < Height; y++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                source.CopyRow(y, row.AsSpan(0, Width));
                int offset = y * Width;
                for (int x = 0; x < Width; x++)
                {
                    RgbFloat pixel = row[x];
                    var linear = isLinear
                        ? pixel
                        : new RgbFloat(
                            ColorMath.SrgbToLinear(pixel.R),
                            ColorMath.SrgbToLinear(pixel.G),
                            ColorMath.SrgbToLinear(pixel.B));
                    luminance[offset + x] = ColorMath.GetLuminance(in linear);
                }
            }
        }
        catch
        {
            if (luminanceIsPooled) ArrayPool<float>.Shared.Return(luminance);
            luminance = null;
            throw;
        }
        finally
        {
            ArrayPool<RgbFloat>.Shared.Return(row);
        }

        return luminance.AsSpan(0, length);
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
        residual = ArrayPool<float>.Shared.Rent(length);
        try
        {
            GrayImageOperations.BoxBlur(sourceLuminance, residual.AsSpan(0, length), Width, Height, 1, cancellationToken);
            GrayImageOperations.Subtract(sourceLuminance, residual.AsSpan(0, length), residual.AsSpan(0, length), cancellationToken);
        }
        catch
        {
            ArrayPool<float>.Shared.Return(residual);
            residual = null;
            throw;
        }
        return residual.AsSpan(0, length);
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        if (luminance is not null && luminanceIsPooled) ArrayPool<float>.Shared.Return(luminance);
        if (residual is not null) ArrayPool<float>.Shared.Return(residual);
        luminance = null;
        residual = null;
    }
}
