using System.Runtime.InteropServices;

namespace FastCompute.ImageProcessing;

/// <summary>Represents an image whose pixel storage remains on one GPU accelerator.</summary>
public sealed class ImageBuffer<TPixel> : IDisposable
    where TPixel : unmanaged
{
    private GpuImageStorage? storage;

    internal ImageBuffer(ComputeContext context, GpuImageStorage storage, int width, int height, ColorEncoding encoding)
    {
        Context = context;
        this.storage = storage;
        Width = width;
        Height = height;
        Encoding = encoding;
    }

    /// <summary>Gets the context and accelerator owning the device allocation.</summary>
    public ComputeContext Context { get; }

    /// <summary>Gets the image width.</summary>
    public int Width { get; }

    /// <summary>Gets the image height.</summary>
    public int Height { get; }

    /// <summary>Gets the pixel count.</summary>
    public int Length => checked(Width * Height);

    /// <summary>Gets the color transfer-function metadata.</summary>
    public ColorEncoding Encoding { get; }

    /// <summary>Gets whether this device allocation has been released.</summary>
    public bool IsDisposed => storage is null;

    /// <summary>Converts the GPU-resident image without downloading the source.</summary>
    public ImageBuffer<TDestination> ConvertTo<TDestination>(
        ColorEncoding? destinationEncoding = null,
        CancellationToken cancellationToken = default)
        where TDestination : unmanaged
    {
        cancellationToken.ThrowIfCancellationRequested();
        GpuImageStorage source = GetStorage();
        (bool sourceFloat, int sourceComponents) = ImageGpuExecutor.GetFormat<TPixel>();
        (bool destinationFloat, int destinationComponents) = ImageGpuExecutor.GetFormat<TDestination>();
        if (sourceFloat != source.IsFloat)
            throw new InvalidOperationException("The device pixel storage does not match its declared format.");
        ColorEncoding targetEncoding = destinationEncoding ?? Encoding;
        GpuImageStorage result = Context.GetImageGpuServices().ConvertImageBuffer(
            source,
            Length,
            sourceComponents,
            destinationComponents,
            (int)Encoding,
            (int)targetEncoding,
            destinationFloat);
        cancellationToken.ThrowIfCancellationRequested();
        return new ImageBuffer<TDestination>(Context, result, Width, Height, targetEncoding);
    }

    /// <summary>Downloads the image into a normal host-backed image.</summary>
    public Image<TPixel> Download(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        GpuImageStorage source = GetStorage();
        (bool isFloat, _) = ImageGpuExecutor.GetFormat<TPixel>();
        var pixels = GC.AllocateUninitializedArray<TPixel>(Length);
        if (isFloat)
            Context.GetImageGpuServices().DownloadFloatImage(source, MemoryMarshal.Cast<TPixel, float>(pixels.AsSpan()));
        else
            Context.GetImageGpuServices().DownloadByteImage(source, MemoryMarshal.AsBytes(pixels.AsSpan()));
        cancellationToken.ThrowIfCancellationRequested();
        return Image<TPixel>.Load(pixels, Width, Height, Encoding);
    }

    /// <summary>Releases the GPU allocation owned by this image buffer.</summary>
    public void Dispose()
    {
        GpuImageStorage? allocation = Interlocked.Exchange(ref storage, null);
        allocation?.Dispose();
    }

    internal GpuImageStorage GetStorage() => storage ??
        throw new ObjectDisposedException(GetType().Name);
}

/// <summary>Creates and transforms GPU-resident image buffers.</summary>
public static class ImageBufferExtensions
{
    /// <summary>Uploads a host-backed image once and keeps its pixels on the accelerator.</summary>
    public static ImageBuffer<TPixel> UploadToGpu<TPixel>(
        this Image<TPixel> image,
        ComputeContext context,
        CancellationToken cancellationToken = default)
        where TPixel : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(context);
        cancellationToken.ThrowIfCancellationRequested();
        (bool isFloat, _) = ImageGpuExecutor.GetFormat<TPixel>();
        GpuImageStorage storage = isFloat
            ? context.GetImageGpuServices().UploadImage(MemoryMarshal.Cast<TPixel, float>(image.Pixels.Span))
            : context.GetImageGpuServices().UploadImage(MemoryMarshal.AsBytes(image.Pixels.Span));
        cancellationToken.ThrowIfCancellationRequested();
        return new ImageBuffer<TPixel>(context, storage, image.Width, image.Height, image.Encoding);
    }

    /// <summary>Converts a resident image to compact grayscale on the GPU.</summary>
    public static ImageBuffer<Gray8> ToGrayscale8<TPixel>(
        this ImageBuffer<TPixel> image,
        ColorEncoding? encoding = null,
        CancellationToken cancellationToken = default)
        where TPixel : unmanaged =>
        image.ConvertTo<Gray8>(encoding, cancellationToken);

    /// <summary>Converts a resident image to floating-point grayscale on the GPU.</summary>
    public static ImageBuffer<GrayF32> ToGrayscaleF32<TPixel>(
        this ImageBuffer<TPixel> image,
        ColorEncoding? encoding = null,
        CancellationToken cancellationToken = default)
        where TPixel : unmanaged =>
        image.ConvertTo<GrayF32>(encoding, cancellationToken);

    /// <summary>Applies a box blur without downloading either pass.</summary>
    public static ImageBuffer<GrayF32> BoxBlur(
        this ImageBuffer<GrayF32> image,
        int radius = 1,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (radius < 0) throw new ArgumentOutOfRangeException(nameof(radius));
        cancellationToken.ThrowIfCancellationRequested();
        if (radius == 0) return image.ConvertTo<GrayF32>(cancellationToken: cancellationToken);
        GpuImageStorage result = image.Context.GetImageGpuServices().BoxBlurImageBuffer(image.GetStorage(), image.Width, image.Height, radius);
        cancellationToken.ThrowIfCancellationRequested();
        return new ImageBuffer<GrayF32>(image.Context, result, image.Width, image.Height, image.Encoding);
    }

    /// <summary>Subtracts two resident grayscale images on the same accelerator.</summary>
    public static ImageBuffer<GrayF32> Subtract(
        this ImageBuffer<GrayF32> left,
        ImageBuffer<GrayF32> right,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        if (!ReferenceEquals(left.Context, right.Context))
            throw new ArgumentException("Both image buffers must belong to the same ComputeContext.", nameof(right));
        if (left.Width != right.Width || left.Height != right.Height || left.Encoding != right.Encoding)
            throw new ArgumentException("Image buffers must have matching dimensions and encoding.", nameof(right));
        cancellationToken.ThrowIfCancellationRequested();
        GpuImageStorage result = left.Context.GetImageGpuServices().SubtractImageBuffers(left.GetStorage(), right.GetStorage());
        cancellationToken.ThrowIfCancellationRequested();
        return new ImageBuffer<GrayF32>(left.Context, result, left.Width, left.Height, left.Encoding);
    }

    /// <summary>Downsamples a resident floating-point grayscale image.</summary>
    public static ImageBuffer<GrayF32> Downsample(
        this ImageBuffer<GrayF32> image,
        int width,
        int height,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (width <= 0 || width > image.Width) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0 || height > image.Height) throw new ArgumentOutOfRangeException(nameof(height));
        cancellationToken.ThrowIfCancellationRequested();
        GpuImageStorage result = image.Context.GetImageGpuServices().DownsampleImageBuffer(image.GetStorage(), image.Width, image.Height, width, height);
        cancellationToken.ThrowIfCancellationRequested();
        return new ImageBuffer<GrayF32>(image.Context, result, width, height, image.Encoding);
    }

    /// <summary>Resizes a resident floating-point grayscale image using bilinear interpolation.</summary>
    public static ImageBuffer<GrayF32> Resize(
        this ImageBuffer<GrayF32> image,
        int width,
        int height,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        cancellationToken.ThrowIfCancellationRequested();
        GpuImageStorage result = image.Context.GetImageGpuServices().ResizeImageBuffer(image.GetStorage(), image.Width, image.Height, width, height);
        cancellationToken.ThrowIfCancellationRequested();
        return new ImageBuffer<GrayF32>(image.Context, result, width, height, image.Encoding);
    }
}
