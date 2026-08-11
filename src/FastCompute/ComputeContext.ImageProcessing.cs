using FastCompute.Gpu;
using FastCompute.ImageProcessing;
using ILGPU;
using ILGPU.Runtime;

namespace FastCompute;

public sealed partial class ComputeContext
{
    private Action<AcceleratorStream, Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>? imageByteToByteKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<byte>, ArrayView<float>, int, int, int, int>? imageByteToFloatKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<byte>, int, int, int, int>? imageFloatToByteKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>? imageFloatToFloatKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? imageSubtractKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int, int>? imageBlurHorizontalKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int, int, int>? imageBlurVerticalKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int, int>? imageBlurHorizontalSlidingKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int, int, int>? imageBlurVerticalSlidingKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>? imageDownsampleKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<ByteGpuInstruction>, ArrayView<int>, ArrayView<int>, int, int>? byteCompositeMapKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<byte>, ArrayView<float>, ArrayView<ByteGpuInstruction>, int, int>? byteCompositeProjectKernel;

    internal GpuImageStorage UploadImage(byte[] source) =>
        new GpuByteImageStorage(accelerator.Allocate1D(source));

    internal GpuImageStorage UploadImage(float[] source) =>
        new GpuFloatImageStorage(accelerator.Allocate1D(source));

    internal GpuImageStorage UploadImage(ReadOnlySpan<byte> source)
    {
        ThrowIfDisposed();
        var buffer = accelerator.Allocate1D<byte>(source.Length);
        buffer.View.CopyFromCPU(accelerator.DefaultStream, source);
        accelerator.Synchronize();
        return new GpuByteImageStorage(buffer);
    }

    internal GpuImageStorage UploadImage(ReadOnlySpan<float> source)
    {
        ThrowIfDisposed();
        var buffer = accelerator.Allocate1D<float>(source.Length);
        buffer.View.CopyFromCPU(accelerator.DefaultStream, source);
        accelerator.Synchronize();
        return new GpuFloatImageStorage(buffer);
    }

    internal byte[] ExecuteByteCompositeMap(byte[] source, int valueCount, int sourceComponents, ByteCompositeGpuProgram program)
    {
        ThrowIfDisposed();
        int destinationComponents = program.OutputOffsets.Length;
        using var input = accelerator.Allocate1D(source);
        using var output = accelerator.Allocate1D<byte>(checked(valueCount * destinationComponents));
        using var instructions = accelerator.Allocate1D(program.Instructions);
        using var offsets = accelerator.Allocate1D(program.OutputOffsets);
        using var counts = accelerator.Allocate1D(program.OutputInstructionCounts);
        GetImageKernel(ref byteCompositeMapKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<ByteGpuInstruction>, ArrayView<int>, ArrayView<int>, int, int>(ImageGpuKernels.ByteCompositeMap))(accelerator.DefaultStream, valueCount, input.View, output.View, instructions.View, offsets.View, counts.View, sourceComponents, destinationComponents);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }

    internal float[] ExecuteByteCompositeProjection(byte[] source, int valueCount, int sourceComponents, ByteCompositeGpuProgram program)
    {
        ThrowIfDisposed();
        using var input = accelerator.Allocate1D(source);
        using var output = accelerator.Allocate1D<float>(valueCount);
        using var instructions = accelerator.Allocate1D(program.Instructions);
        GetImageKernel(ref byteCompositeProjectKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<float>, ArrayView<ByteGpuInstruction>, int, int>(ImageGpuKernels.ByteCompositeProject))(accelerator.DefaultStream, valueCount, input.View, output.View, instructions.View, program.Instructions.Length, sourceComponents);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }

    internal byte[] DownloadByteImage(GpuImageStorage storage)
    {
        accelerator.Synchronize();
        return ((GpuByteImageStorage)storage).Buffer.GetAsArray1D();
    }

    internal float[] DownloadFloatImage(GpuImageStorage storage)
    {
        accelerator.Synchronize();
        return ((GpuFloatImageStorage)storage).Buffer.GetAsArray1D();
    }

    internal void DownloadByteImage(GpuImageStorage storage, Span<byte> destination)
    {
        ((GpuByteImageStorage)storage).Buffer.View.CopyToCPU(accelerator.DefaultStream, destination);
        accelerator.Synchronize();
    }

    internal void DownloadFloatImage(GpuImageStorage storage, Span<float> destination)
    {
        ((GpuFloatImageStorage)storage).Buffer.View.CopyToCPU(accelerator.DefaultStream, destination);
        accelerator.Synchronize();
    }

    internal GpuImageStorage ConvertImageBuffer(GpuImageStorage source, int valueCount, int sourceComponents, int destinationComponents, int sourceEncoding, int destinationEncoding, bool destinationFloat)
    {
        ThrowIfDisposed();
        if (source.IsFloat)
        {
            ArrayView<float> input = ((GpuFloatImageStorage)source).Buffer.View;
            if (destinationFloat)
            {
                var output = accelerator.Allocate1D<float>(checked(valueCount * destinationComponents));
                GetImageKernel(ref imageFloatToFloatKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>(ImageGpuKernels.ConvertFloatToFloat))(accelerator.DefaultStream, valueCount, input, output.View, sourceComponents, destinationComponents, sourceEncoding, destinationEncoding);
                accelerator.Synchronize();
                return new GpuFloatImageStorage(output);
            }
            else
            {
                var output = accelerator.Allocate1D<byte>(checked(valueCount * destinationComponents));
                GetImageKernel(ref imageFloatToByteKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<byte>, int, int, int, int>(ImageGpuKernels.ConvertFloatToByte))(accelerator.DefaultStream, valueCount, input, output.View, sourceComponents, destinationComponents, sourceEncoding, destinationEncoding);
                accelerator.Synchronize();
                return new GpuByteImageStorage(output);
            }
        }
        else
        {
            ArrayView<byte> input = ((GpuByteImageStorage)source).Buffer.View;
            if (destinationFloat)
            {
                var output = accelerator.Allocate1D<float>(checked(valueCount * destinationComponents));
                GetImageKernel(ref imageByteToFloatKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<float>, int, int, int, int>(ImageGpuKernels.ConvertByteToFloat))(accelerator.DefaultStream, valueCount, input, output.View, sourceComponents, destinationComponents, sourceEncoding, destinationEncoding);
                accelerator.Synchronize();
                return new GpuFloatImageStorage(output);
            }
            else
            {
                var output = accelerator.Allocate1D<byte>(checked(valueCount * destinationComponents));
                GetImageKernel(ref imageByteToByteKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(ImageGpuKernels.ConvertByteToByte))(accelerator.DefaultStream, valueCount, input, output.View, sourceComponents, destinationComponents, sourceEncoding, destinationEncoding);
                accelerator.Synchronize();
                return new GpuByteImageStorage(output);
            }
        }
    }

    internal GpuImageStorage SubtractImageBuffers(GpuImageStorage left, GpuImageStorage right)
    {
        var leftBuffer = ((GpuFloatImageStorage)left).Buffer;
        var rightBuffer = ((GpuFloatImageStorage)right).Buffer;
        var output = accelerator.Allocate1D<float>((int)leftBuffer.Length);
        GetImageKernel(ref imageSubtractKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(ImageGpuKernels.Subtract))(accelerator.DefaultStream, (int)leftBuffer.Length, leftBuffer.View, rightBuffer.View, output.View);
        accelerator.Synchronize();
        return new GpuFloatImageStorage(output);
    }

    internal GpuImageStorage BoxBlurImageBuffer(GpuImageStorage source, int width, int height, int radius)
    {
        var input = ((GpuFloatImageStorage)source).Buffer;
        var temporary = accelerator.Allocate1D<float>((int)input.Length);
        var output = accelerator.Allocate1D<float>((int)input.Length);
        try
        {
            LaunchImageBlur(input.View, temporary.View, output.View, width, height, radius);
            accelerator.Synchronize();
            return new GpuFloatImageStorage(output);
        }
        catch
        {
            output.Dispose();
            throw;
        }
        finally
        {
            temporary.Dispose();
        }
    }

    internal GpuImageStorage DownsampleImageBuffer(GpuImageStorage source, int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight)
    {
        var input = ((GpuFloatImageStorage)source).Buffer;
        int destinationLength = checked(destinationWidth * destinationHeight);
        var output = accelerator.Allocate1D<float>(destinationLength);
        try
        {
            GetImageKernel(ref imageDownsampleKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>(ImageGpuKernels.Downsample))(accelerator.DefaultStream, destinationLength, input.View, output.View, sourceWidth, sourceHeight, destinationWidth, destinationHeight);
            accelerator.Synchronize();
            return new GpuFloatImageStorage(output);
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    internal byte[] ExecuteImageConversion(byte[] source, int valueCount, int sourceComponents, int destinationComponents, int sourceEncoding, int destinationEncoding)
    {
        ThrowIfDisposed();
        using MemoryBuffer1D<byte, Stride1D.Dense> input = accelerator.Allocate1D(source);
        using MemoryBuffer1D<byte, Stride1D.Dense> output = accelerator.Allocate1D<byte>(checked(valueCount * destinationComponents));
        GetImageKernel(ref imageByteToByteKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(ImageGpuKernels.ConvertByteToByte))(accelerator.DefaultStream, valueCount, input.View, output.View, sourceComponents, destinationComponents, sourceEncoding, destinationEncoding);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }

    internal float[] ExecuteImageConversion(byte[] source, int valueCount, int sourceComponents, int destinationComponents, int sourceEncoding, int destinationEncoding, bool floatDestination)
    {
        ThrowIfDisposed();
        using MemoryBuffer1D<byte, Stride1D.Dense> input = accelerator.Allocate1D(source);
        using MemoryBuffer1D<float, Stride1D.Dense> output = accelerator.Allocate1D<float>(checked(valueCount * destinationComponents));
        GetImageKernel(ref imageByteToFloatKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<float>, int, int, int, int>(ImageGpuKernels.ConvertByteToFloat))(accelerator.DefaultStream, valueCount, input.View, output.View, sourceComponents, destinationComponents, sourceEncoding, destinationEncoding);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }

    internal byte[] ExecuteImageConversion(float[] source, int valueCount, int sourceComponents, int destinationComponents, int sourceEncoding, int destinationEncoding)
    {
        ThrowIfDisposed();
        using MemoryBuffer1D<float, Stride1D.Dense> input = accelerator.Allocate1D(source);
        using MemoryBuffer1D<byte, Stride1D.Dense> output = accelerator.Allocate1D<byte>(checked(valueCount * destinationComponents));
        GetImageKernel(ref imageFloatToByteKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<byte>, int, int, int, int>(ImageGpuKernels.ConvertFloatToByte))(accelerator.DefaultStream, valueCount, input.View, output.View, sourceComponents, destinationComponents, sourceEncoding, destinationEncoding);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }

    internal float[] ExecuteImageConversion(float[] source, int valueCount, int sourceComponents, int destinationComponents, int sourceEncoding, int destinationEncoding, bool floatDestination)
    {
        ThrowIfDisposed();
        using MemoryBuffer1D<float, Stride1D.Dense> input = accelerator.Allocate1D(source);
        using MemoryBuffer1D<float, Stride1D.Dense> output = accelerator.Allocate1D<float>(checked(valueCount * destinationComponents));
        GetImageKernel(ref imageFloatToFloatKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>(ImageGpuKernels.ConvertFloatToFloat))(accelerator.DefaultStream, valueCount, input.View, output.View, sourceComponents, destinationComponents, sourceEncoding, destinationEncoding);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }

    internal float[] ExecuteImageSubtract(float[] left, float[] right)
    {
        ThrowIfDisposed();
        using MemoryBuffer1D<float, Stride1D.Dense> leftBuffer = accelerator.Allocate1D(left);
        using MemoryBuffer1D<float, Stride1D.Dense> rightBuffer = accelerator.Allocate1D(right);
        using MemoryBuffer1D<float, Stride1D.Dense> output = accelerator.Allocate1D<float>(left.Length);
        GetImageKernel(ref imageSubtractKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(ImageGpuKernels.Subtract))(accelerator.DefaultStream, left.Length, leftBuffer.View, rightBuffer.View, output.View);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }

    internal float[] ExecuteImageBoxBlur(float[] source, int width, int height, int radius)
    {
        ThrowIfDisposed();
        using MemoryBuffer1D<float, Stride1D.Dense> input = accelerator.Allocate1D(source);
        using MemoryBuffer1D<float, Stride1D.Dense> temporary = accelerator.Allocate1D<float>(source.Length);
        using MemoryBuffer1D<float, Stride1D.Dense> output = accelerator.Allocate1D<float>(source.Length);
        LaunchImageBlur(input.View, temporary.View, output.View, width, height, radius);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }

    internal float[] ExecuteImageDownsample(float[] source, int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight)
    {
        ThrowIfDisposed();
        int destinationLength = checked(destinationWidth * destinationHeight);
        using MemoryBuffer1D<float, Stride1D.Dense> input = accelerator.Allocate1D(source);
        using MemoryBuffer1D<float, Stride1D.Dense> output = accelerator.Allocate1D<float>(destinationLength);
        GetImageKernel(ref imageDownsampleKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>(ImageGpuKernels.Downsample))(accelerator.DefaultStream, destinationLength, input.View, output.View, sourceWidth, sourceHeight, destinationWidth, destinationHeight);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }

    private IReadOnlyList<ComputeCompilationResult> PrecompileImageKernels()
    {
        return
        [
            CompileImageKernel(() => imageByteToByteKernel is not null, () => _ = GetImageKernel(ref imageByteToByteKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(ImageGpuKernels.ConvertByteToByte))),
            CompileImageKernel(() => imageByteToFloatKernel is not null, () => _ = GetImageKernel(ref imageByteToFloatKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<float>, int, int, int, int>(ImageGpuKernels.ConvertByteToFloat))),
            CompileImageKernel(() => imageFloatToByteKernel is not null, () => _ = GetImageKernel(ref imageFloatToByteKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<byte>, int, int, int, int>(ImageGpuKernels.ConvertFloatToByte))),
            CompileImageKernel(() => imageFloatToFloatKernel is not null, () => _ = GetImageKernel(ref imageFloatToFloatKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>(ImageGpuKernels.ConvertFloatToFloat))),
            CompileImageKernel(() => imageSubtractKernel is not null, () => _ = GetImageKernel(ref imageSubtractKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(ImageGpuKernels.Subtract))),
            CompileImageKernel(() => imageBlurHorizontalKernel is not null, () => _ = GetImageKernel(ref imageBlurHorizontalKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int>(ImageGpuKernels.BlurHorizontal))),
            CompileImageKernel(() => imageBlurVerticalKernel is not null, () => _ = GetImageKernel(ref imageBlurVerticalKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int, int>(ImageGpuKernels.BlurVertical))),
            CompileImageKernel(() => imageBlurHorizontalSlidingKernel is not null, () => _ = GetImageKernel(ref imageBlurHorizontalSlidingKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int>(ImageGpuKernels.BlurHorizontalSliding))),
            CompileImageKernel(() => imageBlurVerticalSlidingKernel is not null, () => _ = GetImageKernel(ref imageBlurVerticalSlidingKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int, int>(ImageGpuKernels.BlurVerticalSliding))),
            CompileImageKernel(() => imageDownsampleKernel is not null, () => _ = GetImageKernel(ref imageDownsampleKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>(ImageGpuKernels.Downsample))),
            CompileImageKernel(() => byteCompositeMapKernel is not null, () => _ = GetImageKernel(ref byteCompositeMapKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<ByteGpuInstruction>, ArrayView<int>, ArrayView<int>, int, int>(ImageGpuKernels.ByteCompositeMap))),
            CompileImageKernel(() => byteCompositeProjectKernel is not null, () => _ = GetImageKernel(ref byteCompositeProjectKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<float>, ArrayView<ByteGpuInstruction>, int, int>(ImageGpuKernels.ByteCompositeProject)))
        ];
    }

    private void LaunchImageBlur(
        ArrayView<float> input,
        ArrayView<float> temporary,
        ArrayView<float> output,
        int width,
        int height,
        int radius)
    {
        const int slidingRadiusThreshold = 4;
        if (radius > slidingRadiusThreshold)
        {
            GetImageKernel(ref imageBlurHorizontalSlidingKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int>(ImageGpuKernels.BlurHorizontalSliding))(accelerator.DefaultStream, height, input, temporary, width, radius);
            GetImageKernel(ref imageBlurVerticalSlidingKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int, int>(ImageGpuKernels.BlurVerticalSliding))(accelerator.DefaultStream, width, temporary, output, width, height, radius);
            return;
        }

        int length = checked(width * height);
        GetImageKernel(ref imageBlurHorizontalKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int>(ImageGpuKernels.BlurHorizontal))(accelerator.DefaultStream, length, input, temporary, width, radius);
        GetImageKernel(ref imageBlurVerticalKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int, int>(ImageGpuKernels.BlurVertical))(accelerator.DefaultStream, length, temporary, output, width, height, radius);
    }

    private ComputeCompilationResult CompileImageKernel(Func<bool> isCached, Action compile)
    {
        bool cacheHit = isCached();
        long started = System.Diagnostics.Stopwatch.GetTimestamp();
        compile();
        return new ComputeCompilationResult(
            cacheHit,
            TimeSpan.Zero,
            cacheHit ? TimeSpan.Zero : System.Diagnostics.Stopwatch.GetElapsedTime(started),
            ComputeBackendKind.Gpu,
            accelerator.Name);
    }

    private static TKernel GetImageKernel<TKernel>(ref TKernel? cache, Func<TKernel> compile)
        where TKernel : class
    {
        TKernel? existing = Volatile.Read(ref cache);
        if (existing is not null) return existing;
        TKernel candidate = compile();
        return Interlocked.CompareExchange(ref cache, candidate, null) ?? candidate;
    }
}
