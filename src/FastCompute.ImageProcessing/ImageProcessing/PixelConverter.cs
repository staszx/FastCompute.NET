using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace FastCompute.ImageProcessing;

/// <summary>Converts between the native FastCompute pixel formats.</summary>
public static class PixelConverter
{
    private const float ByteToUnit = 1f / byte.MaxValue;

    /// <summary>Calculates Rec. 709 luminance from a normalized floating-point RGB value.</summary>
    public static float GetLuminance(in Rgb pixel) =>
        (0.2126f * pixel.Red) + (0.7152f * pixel.Green) + (0.0722f * pixel.Blue);

    /// <summary>Converts an image to another native pixel format.</summary>
    public static Image<TDestination> ConvertTo<TSource, TDestination>(
        this Image<TSource> source,
        ColorEncoding? destinationEncoding = null,
        ComputeOptions? options = null)
        where TSource : unmanaged
        where TDestination : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ColorEncoding targetEncoding = destinationEncoding ?? source.Encoding;
        ComputeOptions effectiveOptions = options ?? ComputeOptions.Default;
        if (ImageGpuExecutor.TryConvert<TSource, TDestination>(
                source.Pixels.Span,
                source.Encoding,
                targetEncoding,
                effectiveOptions,
                out TDestination[] gpuDestination))
        {
            return Image<TDestination>.Load(
                gpuDestination,
                source.Width,
                source.Height,
                targetEncoding);
        }
        var destination = GC.AllocateUninitializedArray<TDestination>(
            source.Pixels.Length);
        Convert<TSource, TDestination>(
            source.Pixels.Span,
            destination,
            source.Encoding,
            targetEncoding,
            effectiveOptions);
        return Image<TDestination>.Load(
            destination,
            source.Width,
            source.Height,
            targetEncoding);
    }

    /// <summary>Converts pixels between native formats without image allocation.</summary>
    public static void Convert<TSource, TDestination>(
        ReadOnlySpan<TSource> source,
        Span<TDestination> destination,
        ColorEncoding sourceEncoding = ColorEncoding.Srgb,
        ColorEncoding destinationEncoding = ColorEncoding.Srgb,
        ComputeOptions? options = null)
        where TSource : unmanaged
        where TDestination : unmanaged
    {
        if (destination.Length < source.Length)
        {
            throw new ArgumentException(
                "The destination span is shorter than the source span.",
                nameof(destination));
        }

        ComputeOptions effectiveOptions = options ?? ComputeOptions.Default;
        if (ImageGpuExecutor.TryConvert(
                source,
                sourceEncoding,
                destinationEncoding,
                effectiveOptions,
                out TDestination[] gpuDestination))
        {
            gpuDestination.CopyTo(destination);
            return;
        }

        if (typeof(TSource) == typeof(TDestination) &&
            sourceEncoding == destinationEncoding)
        {
            MemoryMarshal.AsBytes(source).CopyTo(
                MemoryMarshal.AsBytes(destination));
            return;
        }

        if (effectiveOptions.Backend == ComputeBackendKind.Simd &&
            (sourceEncoding != destinationEncoding || !Avx2.IsSupported))
        {
            throw new ComputeBackendNotSupportedException(
                ComputeBackendKind.Simd,
                sourceEncoding != destinationEncoding
                    ? "pixel conversion with nonlinear color-encoding transfer"
                    : "pixel conversion on the current CPU",
                sourceEncoding != destinationEncoding
                    ? "Scalar, ParallelCpu, Gpu"
                    : "Scalar, ParallelCpu");
        }

        bool useParallel = effectiveOptions.Backend == ComputeBackendKind.ParallelCpu ||
            (effectiveOptions.Backend == ComputeBackendKind.Auto && source.Length >= effectiveOptions.Thresholds.ParallelThreshold);
        if (useParallel && source.Length > 0)
        {
            TSource[] input = source.ToArray();
            var output = GC.AllocateUninitializedArray<TDestination>(source.Length);
            int workerCount = Math.Min(source.Length, effectiveOptions.MaxDegreeOfParallelism ?? Environment.ProcessorCount);
            int rangeSize = (source.Length + workerCount - 1) / workerCount;
            Parallel.For(0, workerCount, new ParallelOptions
            {
                CancellationToken = effectiveOptions.CancellationToken,
                MaxDegreeOfParallelism = effectiveOptions.MaxDegreeOfParallelism ?? -1
            }, worker =>
            {
                int start = worker * rangeSize;
                int end = Math.Min(start + rangeSize, input.Length);
                if (start < end)
                {
                    ConvertCore<TSource, TDestination>(
                        new ReadOnlySpan<TSource>(input, start, end - start),
                        new Span<TDestination>(output, start, end - start),
                        sourceEncoding,
                        destinationEncoding,
                        allowSimd: false);
                }
            });
            output.CopyTo(destination);
            return;
        }

        bool allowSimd = effectiveOptions.Backend == ComputeBackendKind.Simd ||
            (effectiveOptions.Backend == ComputeBackendKind.Auto && source.Length >= effectiveOptions.Thresholds.SimdThreshold && Avx2.IsSupported);
        ConvertCore(source, destination, sourceEncoding, destinationEncoding, allowSimd);
    }

    private static void ConvertCore<TSource, TDestination>(
        ReadOnlySpan<TSource> source,
        Span<TDestination> destination,
        ColorEncoding sourceEncoding,
        ColorEncoding destinationEncoding,
        bool allowSimd)
        where TSource : unmanaged
        where TDestination : unmanaged
    {

        if (typeof(TSource) == typeof(Rgb24))
        {
            ConvertFromRgb24(
                MemoryMarshal.Cast<TSource, Rgb24>(source),
                destination,
                sourceEncoding,
                destinationEncoding,
                allowSimd);
            return;
        }

        if (typeof(TSource) == typeof(Rgb))
        {
            ConvertFromRgb(
                MemoryMarshal.Cast<TSource, Rgb>(source),
                destination,
                sourceEncoding,
                destinationEncoding,
                allowSimd);
            return;
        }

        if (typeof(TSource) == typeof(Gray8))
        {
            ConvertFromGray8(
                MemoryMarshal.Cast<TSource, Gray8>(source),
                destination,
                sourceEncoding,
                destinationEncoding,
                allowSimd);
            return;
        }

        if (typeof(TSource) == typeof(GrayF32))
        {
            ConvertFromGrayF32(
                MemoryMarshal.Cast<TSource, GrayF32>(source),
                destination,
                sourceEncoding,
                destinationEncoding,
                allowSimd);
            return;
        }

        throw Unsupported<TSource, TDestination>();
    }

    /// <summary>Converts a normalized sRGB value to linear light.</summary>
    public static float SrgbToLinear(float value) => value <= 0.04045f
        ? value / 12.92f
        : MathF.Pow((value + 0.055f) / 1.055f, 2.4f);

    /// <summary>Converts a normalized linear-light value to sRGB.</summary>
    public static float LinearToSrgb(float value) => value <= 0.0031308f
        ? value * 12.92f
        : (1.055f * MathF.Pow(value, 1f / 2.4f)) - 0.055f;

    private static void ConvertFromRgb24<TDestination>(
        ReadOnlySpan<Rgb24> source,
        Span<TDestination> destination,
        ColorEncoding sourceEncoding,
        ColorEncoding destinationEncoding,
        bool allowSimd)
        where TDestination : unmanaged
    {
        if (typeof(TDestination) == typeof(Rgb24))
        {
            ConvertRgb24ToRgb24(
                source,
                MemoryMarshal.Cast<TDestination, Rgb24>(destination),
                sourceEncoding,
                destinationEncoding);
        }
        else if (typeof(TDestination) == typeof(Rgb))
        {
            Span<Rgb> output = MemoryMarshal.Cast<TDestination, Rgb>(destination);
            if (sourceEncoding == destinationEncoding && allowSimd)
            {
                PixelConversionKernels.Rgb24ToRgb(source, output);
                return;
            }
            for (int i = 0; i < source.Length; i++)
            {
                Rgb24 pixel = source[i];
                output[i] = new Rgb(
                    ChangeEncoding(pixel.Red * ByteToUnit, sourceEncoding, destinationEncoding),
                    ChangeEncoding(pixel.Green * ByteToUnit, sourceEncoding, destinationEncoding),
                    ChangeEncoding(pixel.Blue * ByteToUnit, sourceEncoding, destinationEncoding));
            }
        }
        else if (typeof(TDestination) == typeof(Gray8))
        {
            Span<Gray8> output = MemoryMarshal.Cast<TDestination, Gray8>(destination);
            if (sourceEncoding == destinationEncoding && allowSimd)
            {
                PixelConversionKernels.Rgb24ToGray8(source, output);
                return;
            }
            for (int i = 0; i < source.Length; i++)
            {
                Rgb24 pixel = source[i];
                float red = ChangeEncoding(pixel.Red * ByteToUnit, sourceEncoding, destinationEncoding);
                float green = ChangeEncoding(pixel.Green * ByteToUnit, sourceEncoding, destinationEncoding);
                float blue = ChangeEncoding(pixel.Blue * ByteToUnit, sourceEncoding, destinationEncoding);
                output[i] = new Gray8(ToByte(Luminance(red, green, blue)));
            }
        }
        else if (typeof(TDestination) == typeof(GrayF32))
        {
            Span<GrayF32> output = MemoryMarshal.Cast<TDestination, GrayF32>(destination);
            if (sourceEncoding == destinationEncoding && allowSimd)
            {
                PixelConversionKernels.Rgb24ToGrayF32(source, output);
                return;
            }
            for (int i = 0; i < source.Length; i++)
            {
                Rgb24 pixel = source[i];
                float red = ChangeEncoding(pixel.Red * ByteToUnit, sourceEncoding, destinationEncoding);
                float green = ChangeEncoding(pixel.Green * ByteToUnit, sourceEncoding, destinationEncoding);
                float blue = ChangeEncoding(pixel.Blue * ByteToUnit, sourceEncoding, destinationEncoding);
                output[i] = new GrayF32(Luminance(red, green, blue));
            }
        }
        else
        {
            throw Unsupported<Rgb24, TDestination>();
        }
    }

    private static void ConvertFromRgb<TDestination>(
        ReadOnlySpan<Rgb> source,
        Span<TDestination> destination,
        ColorEncoding sourceEncoding,
        ColorEncoding destinationEncoding,
        bool allowSimd)
        where TDestination : unmanaged
    {
        if (typeof(TDestination) == typeof(Rgb))
        {
            Span<Rgb> output = MemoryMarshal.Cast<TDestination, Rgb>(destination);
            for (int i = 0; i < source.Length; i++)
            {
                Rgb pixel = source[i];
                output[i] = new Rgb(
                    ChangeEncoding(pixel.Red, sourceEncoding, destinationEncoding),
                    ChangeEncoding(pixel.Green, sourceEncoding, destinationEncoding),
                    ChangeEncoding(pixel.Blue, sourceEncoding, destinationEncoding));
            }
        }
        else if (typeof(TDestination) == typeof(Rgb24))
        {
            Span<Rgb24> output = MemoryMarshal.Cast<TDestination, Rgb24>(destination);
            if (sourceEncoding == destinationEncoding && allowSimd)
            {
                PixelConversionKernels.RgbToRgb24(source, output);
                return;
            }
            for (int i = 0; i < source.Length; i++)
            {
                Rgb pixel = source[i];
                output[i] = new Rgb24(
                    ToByte(ChangeEncoding(pixel.Red, sourceEncoding, destinationEncoding)),
                    ToByte(ChangeEncoding(pixel.Green, sourceEncoding, destinationEncoding)),
                    ToByte(ChangeEncoding(pixel.Blue, sourceEncoding, destinationEncoding)));
            }
        }
        else if (typeof(TDestination) == typeof(Gray8))
        {
            Span<Gray8> output = MemoryMarshal.Cast<TDestination, Gray8>(destination);
            if (sourceEncoding == destinationEncoding && allowSimd)
            {
                PixelConversionKernels.RgbToGray8(source, output);
                return;
            }
            for (int i = 0; i < source.Length; i++)
            {
                Rgb pixel = source[i];
                output[i] = new Gray8(ToByte(Luminance(
                    ChangeEncoding(pixel.Red, sourceEncoding, destinationEncoding),
                    ChangeEncoding(pixel.Green, sourceEncoding, destinationEncoding),
                    ChangeEncoding(pixel.Blue, sourceEncoding, destinationEncoding))));
            }
        }
        else if (typeof(TDestination) == typeof(GrayF32))
        {
            Span<GrayF32> output = MemoryMarshal.Cast<TDestination, GrayF32>(destination);
            if (sourceEncoding == destinationEncoding && allowSimd)
            {
                PixelConversionKernels.RgbToGrayF32(source, output);
                return;
            }
            for (int i = 0; i < source.Length; i++)
            {
                Rgb pixel = source[i];
                output[i] = new GrayF32(Luminance(
                    ChangeEncoding(pixel.Red, sourceEncoding, destinationEncoding),
                    ChangeEncoding(pixel.Green, sourceEncoding, destinationEncoding),
                    ChangeEncoding(pixel.Blue, sourceEncoding, destinationEncoding)));
            }
        }
        else
        {
            throw Unsupported<Rgb, TDestination>();
        }
    }

    private static void ConvertFromGray8<TDestination>(
        ReadOnlySpan<Gray8> source,
        Span<TDestination> destination,
        ColorEncoding sourceEncoding,
        ColorEncoding destinationEncoding,
        bool allowSimd)
        where TDestination : unmanaged
    {
        ConvertGrayBytes(
            MemoryMarshal.Cast<Gray8, byte>(source),
            destination,
            sourceEncoding,
            destinationEncoding,
            allowSimd);
    }

    private static void ConvertFromGrayF32<TDestination>(
        ReadOnlySpan<GrayF32> source,
        Span<TDestination> destination,
        ColorEncoding sourceEncoding,
        ColorEncoding destinationEncoding,
        bool allowSimd)
        where TDestination : unmanaged
    {
        ConvertGrayFloats(
            MemoryMarshal.Cast<GrayF32, float>(source),
            destination,
            sourceEncoding,
            destinationEncoding,
            allowSimd);
    }

    private static void ConvertGrayBytes<TDestination>(
        ReadOnlySpan<byte> source,
        Span<TDestination> destination,
        ColorEncoding sourceEncoding,
        ColorEncoding destinationEncoding,
        bool allowSimd)
        where TDestination : unmanaged
    {
        if (typeof(TDestination) == typeof(Gray8))
        {
            Span<Gray8> output = MemoryMarshal.Cast<TDestination, Gray8>(destination);
            for (int i = 0; i < source.Length; i++)
            {
                output[i] = new Gray8(ToByte(ChangeEncoding(
                    source[i] * ByteToUnit,
                    sourceEncoding,
                    destinationEncoding)));
            }
        }
        else if (typeof(TDestination) == typeof(GrayF32))
        {
            Span<GrayF32> output = MemoryMarshal.Cast<TDestination, GrayF32>(destination);
            if (sourceEncoding == destinationEncoding && allowSimd)
            {
                PixelConversionKernels.Gray8ToGrayF32(source, output);
                return;
            }
            for (int i = 0; i < source.Length; i++)
            {
                output[i] = new GrayF32(ChangeEncoding(
                    source[i] * ByteToUnit,
                    sourceEncoding,
                    destinationEncoding));
            }
        }
        else if (typeof(TDestination) == typeof(Rgb24))
        {
            Span<Rgb24> output = MemoryMarshal.Cast<TDestination, Rgb24>(destination);
            if (sourceEncoding == destinationEncoding && allowSimd)
            {
                PixelConversionKernels.Gray8ToRgb24(source, output);
                return;
            }
            for (int i = 0; i < source.Length; i++)
            {
                byte value = ToByte(ChangeEncoding(
                    source[i] * ByteToUnit,
                    sourceEncoding,
                    destinationEncoding));
                output[i] = new Rgb24(value, value, value);
            }
        }
        else if (typeof(TDestination) == typeof(Rgb))
        {
            Span<Rgb> output = MemoryMarshal.Cast<TDestination, Rgb>(destination);
            if (sourceEncoding == destinationEncoding && allowSimd)
            {
                PixelConversionKernels.Gray8ToRgb(source, output);
                return;
            }
            for (int i = 0; i < source.Length; i++)
            {
                float value = ChangeEncoding(
                    source[i] * ByteToUnit,
                    sourceEncoding,
                    destinationEncoding);
                output[i] = new Rgb(value, value, value);
            }
        }
        else
        {
            throw Unsupported<Gray8, TDestination>();
        }
    }

    private static void ConvertGrayFloats<TDestination>(
        ReadOnlySpan<float> source,
        Span<TDestination> destination,
        ColorEncoding sourceEncoding,
        ColorEncoding destinationEncoding,
        bool allowSimd)
        where TDestination : unmanaged
    {
        if (typeof(TDestination) == typeof(GrayF32))
        {
            Span<GrayF32> output = MemoryMarshal.Cast<TDestination, GrayF32>(destination);
            for (int i = 0; i < source.Length; i++)
            {
                output[i] = new GrayF32(ChangeEncoding(
                    source[i], sourceEncoding, destinationEncoding));
            }
        }
        else if (typeof(TDestination) == typeof(Gray8))
        {
            Span<Gray8> output = MemoryMarshal.Cast<TDestination, Gray8>(destination);
            if (sourceEncoding == destinationEncoding && allowSimd)
            {
                PixelConversionKernels.GrayF32ToGray8(source, output);
                return;
            }
            for (int i = 0; i < source.Length; i++)
            {
                output[i] = new Gray8(ToByte(ChangeEncoding(
                    source[i], sourceEncoding, destinationEncoding)));
            }
        }
        else if (typeof(TDestination) == typeof(Rgb))
        {
            Span<Rgb> output = MemoryMarshal.Cast<TDestination, Rgb>(destination);
            if (sourceEncoding == destinationEncoding && allowSimd)
            {
                PixelConversionKernels.GrayF32ToRgb(source, output);
                return;
            }
            for (int i = 0; i < source.Length; i++)
            {
                float value = ChangeEncoding(
                    source[i], sourceEncoding, destinationEncoding);
                output[i] = new Rgb(value, value, value);
            }
        }
        else if (typeof(TDestination) == typeof(Rgb24))
        {
            Span<Rgb24> output = MemoryMarshal.Cast<TDestination, Rgb24>(destination);
            if (sourceEncoding == destinationEncoding && allowSimd)
            {
                PixelConversionKernels.GrayF32ToRgb24(source, output);
                return;
            }
            for (int i = 0; i < source.Length; i++)
            {
                byte value = ToByte(ChangeEncoding(
                    source[i], sourceEncoding, destinationEncoding));
                output[i] = new Rgb24(value, value, value);
            }
        }
        else
        {
            throw Unsupported<GrayF32, TDestination>();
        }
    }

    private static void ConvertRgb24ToRgb24(
        ReadOnlySpan<Rgb24> source,
        Span<Rgb24> destination,
        ColorEncoding sourceEncoding,
        ColorEncoding destinationEncoding)
    {
        for (int i = 0; i < source.Length; i++)
        {
            Rgb24 pixel = source[i];
            destination[i] = new Rgb24(
                ToByte(ChangeEncoding(pixel.Red * ByteToUnit, sourceEncoding, destinationEncoding)),
                ToByte(ChangeEncoding(pixel.Green * ByteToUnit, sourceEncoding, destinationEncoding)),
                ToByte(ChangeEncoding(pixel.Blue * ByteToUnit, sourceEncoding, destinationEncoding)));
        }
    }

    private static float ChangeEncoding(
        float value,
        ColorEncoding source,
        ColorEncoding destination) => source == destination
            ? value
            : source == ColorEncoding.Srgb
                ? SrgbToLinear(value)
                : LinearToSrgb(value);

    private static float Luminance(float red, float green, float blue) =>
        (0.2126f * red) + (0.7152f * green) + (0.0722f * blue);

    private static byte ToByte(float value) =>
        (byte)Math.Clamp((int)MathF.Round(value * byte.MaxValue), 0, byte.MaxValue);

    private static NotSupportedException Unsupported<TSource, TDestination>() =>
        new(
            $"Pixel conversion from '{typeof(TSource).Name}' to " +
            $"'{typeof(TDestination).Name}' is not supported.");
}
