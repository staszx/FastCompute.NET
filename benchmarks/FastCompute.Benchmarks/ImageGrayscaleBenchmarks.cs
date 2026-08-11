using BenchmarkDotNet.Attributes;
using FastCompute.ImageProcessing;
using SixLabors.ImageSharp.Processing;
using System.Runtime.InteropServices;
using ImageSharpImage = SixLabors.ImageSharp.Image;
using ImageSharpL8 = SixLabors.ImageSharp.PixelFormats.L8;
using ImageSharpRgb24 = SixLabors.ImageSharp.PixelFormats.Rgb24;

namespace FastCompute.Benchmarks;

/// <summary>
/// Compares normalized BT.709 luminance extraction with ImageSharp grayscale
/// conversion on representative large image sizes.
/// </summary>
[MemoryDiagnoser]
public class ImageGrayscaleBenchmarks
{
    private Image<Rgb24> _fastComputeImage = null!;
    private SixLabors.ImageSharp.Image<ImageSharpRgb24> _imageSharpImage = null!;
    private ComputeContext _gpuContext = null!;
    private ComputeOptions _gpuOptions = null!;
    private ImageBuffer<Rgb24> _residentImage = null!;

    /// <summary>
    /// Gets or sets the image size used by the benchmark.
    /// </summary>
    [Params(
        GrayscaleImageSize.FullHd,
        GrayscaleImageSize.TwelveMegapixels,
        GrayscaleImageSize.TwentyFourMegapixels)]
    public GrayscaleImageSize ImageSize { get; set; }

    /// <summary>
    /// Creates equivalent FastCompute and ImageSharp images outside the
    /// measured benchmark operations.
    /// </summary>
    [GlobalSetup]
    public void Setup()
    {
        (int width, int height) = ImageSize switch
        {
            GrayscaleImageSize.FullHd => (1920, 1080),
            GrayscaleImageSize.TwelveMegapixels => (4000, 3000),
            GrayscaleImageSize.TwentyFourMegapixels => (6000, 4000),
            _ => throw new ArgumentOutOfRangeException(nameof(ImageSize))
        };

        var bytes = new byte[checked(width * height * 3)];
        for (int index = 0; index < bytes.Length; index += 3)
        {
            int pixelIndex = index / 3;
            bytes[index] = (byte)(pixelIndex * 17 + 31);
            bytes[index + 1] = (byte)(pixelIndex * 29 + 73);
            bytes[index + 2] = (byte)(pixelIndex * 43 + 127);
        }

        _fastComputeImage = Image<Rgb24>.Load(
            MemoryMarshal.Cast<byte, Rgb24>(bytes).ToArray(),
            width,
            height);
        _imageSharpImage = ImageSharpImage.LoadPixelData<ImageSharpRgb24>(
            bytes,
            width,
            height);
        _gpuContext = ComputeContext.Create();
        _gpuOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _gpuContext
        };
        _residentImage = _fastComputeImage.UploadToGpu(_gpuContext);
    }

    /// <summary>
    /// Disposes the ImageSharp source after all benchmark cases complete.
    /// </summary>
    [GlobalCleanup]
    public void Cleanup()
    {
        _imageSharpImage.Dispose();
        _residentImage.Dispose();
        _gpuContext.Dispose();
    }

    /// <summary>
    /// Runs ImageSharp's standard BT.709 grayscale processor and returns an
    /// RGB24 image.
    /// </summary>
    [Benchmark]
    public byte ImageSharpGrayscaleRgb24()
    {
        using SixLabors.ImageSharp.Image<ImageSharpRgb24> result =
            _imageSharpImage.Clone(context => context.Grayscale());
        return result[result.Width - 1, result.Height - 1].R;
    }

    /// <summary>
    /// Converts ImageSharp RGB24 pixels to its compact L8 representation.
    /// </summary>
    [Benchmark(Baseline = true)]
    public byte ImageSharpL8()
    {
        using SixLabors.ImageSharp.Image<ImageSharpL8> result =
            _imageSharpImage.CloneAs<ImageSharpL8>();
        return result[result.Width - 1, result.Height - 1].PackedValue;
    }

    /// <summary>
    /// Extracts normalized floating-point luminance using FastCompute's image
    /// abstraction.
    /// </summary>
    [Benchmark]
    public float FastComputeGrayscale()
    {
        Image<GrayF32> result = _fastComputeImage.ToGrayscaleF32();
        return result.Pixels.Span[^1].Value;
    }

    /// <summary>
    /// Converts FastCompute floating-point RGB pixels to compact Gray8 pixels.
    /// </summary>
    [Benchmark]
    public byte FastComputeGray8()
    {
        Image<Gray8> result = _fastComputeImage.ToGrayscale8();
        return result.Pixels.Span[^1].Value;
    }

    /// <summary>Extracts floating-point luminance using the GPU backend.</summary>
    [Benchmark]
    public float FastComputeGrayscaleGpu()
    {
        Image<GrayF32> result = _fastComputeImage.ToGrayscaleF32(options: _gpuOptions);
        return result.Pixels.Span[^1].Value;
    }

    /// <summary>Extracts compact grayscale using the GPU backend.</summary>
    [Benchmark]
    public byte FastComputeGray8Gpu()
    {
        Image<Gray8> result = _fastComputeImage.ToGrayscale8(options: _gpuOptions);
        return result.Pixels.Span[^1].Value;
    }

    /// <summary>Runs grayscale with a resident source and one final download.</summary>
    [Benchmark]
    public byte FastComputeGray8ResidentGpu()
    {
        using ImageBuffer<Gray8> result = _residentImage.ToGrayscale8();
        return result.Download().Pixels.Span[^1].Value;
    }
}

/// <summary>
/// Identifies the representative image dimensions used by grayscale
/// benchmarks.
/// </summary>
public enum GrayscaleImageSize
{
    /// <summary>1920 x 1080 pixels.</summary>
    FullHd,

    /// <summary>4000 x 3000 pixels.</summary>
    TwelveMegapixels,

    /// <summary>6000 x 4000 pixels.</summary>
    TwentyFourMegapixels
}
