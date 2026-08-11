using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using FastCompute.Expressions;
using FastCompute.ImageProcessing;

namespace FastCompute.Backends;

internal static class ByteComputeSimdExecutor
{
    internal static TDestination[] Transform<TSource, TDestination>(TSource[] source, ByteComputeProgram program, Func<TSource, TDestination> scalar, CancellationToken cancellationToken)
        where TSource : unmanaged, IComputeValue<TSource>
        where TDestination : unmanaged, IComputeValue<TDestination>
    {
        var destination = GC.AllocateUninitializedArray<TDestination>(source.Length);
        ExecuteMap(
            MemoryMarshal.AsBytes(source.AsSpan()),
            MemoryMarshal.AsBytes(destination.AsSpan()),
            source.Length,
            TSource.ComputeDescriptor.ComponentCount,
            TDestination.ComputeDescriptor.ComponentCount,
            program,
            cancellationToken);
        int tailStart = source.Length - (source.Length % Vector<int>.Count);
        for (int index = tailStart; index < source.Length; index++) destination[index] = scalar(source[index]);
        return destination;
    }

    internal static T[] Map<T>(T[] source, ByteComputeProgram program, Func<T, T> scalar, bool inPlace, CancellationToken cancellationToken)
        where T : unmanaged, IComputeValue<T>
    {
        T[] destination = inPlace ? source : GC.AllocateUninitializedArray<T>(source.Length);
        ExecuteMap(
            MemoryMarshal.AsBytes(source.AsSpan()),
            MemoryMarshal.AsBytes(destination.AsSpan()),
            source.Length,
            T.ComputeDescriptor.ComponentCount,
            T.ComputeDescriptor.ComponentCount,
            program,
            cancellationToken);
        int tailStart = source.Length - (source.Length % Vector<int>.Count);
        for (int index = tailStart; index < source.Length; index++) destination[index] = scalar(source[index]);
        return destination;
    }

    internal static float[] Project<T>(T[] source, ByteComputeProgram program, Func<T, float> scalar, CancellationToken cancellationToken)
        where T : unmanaged, IComputeValue<T>
    {
        var destination = GC.AllocateUninitializedArray<float>(source.Length);
        ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(source.AsSpan());
        int lanes = Vector<int>.Count;
        int vectorizedLength = source.Length - (source.Length % lanes);
        var components = new Vector<int>[T.ComputeDescriptor.ComponentCount];
        Span<int> gathered = stackalloc int[T.ComputeDescriptor.ComponentCount * lanes];
        for (int index = 0; index < vectorizedLength; index += lanes)
        {
            if ((index & 0xFFFF) == 0) cancellationToken.ThrowIfCancellationRequested();
            Gather(bytes, index, T.ComputeDescriptor.ComponentCount, components, gathered);
            Vector.ConvertToSingle(Evaluate(program.Outputs[0], components)).CopyTo(destination, index);
        }
        for (int index = vectorizedLength; index < source.Length; index++) destination[index] = scalar(source[index]);
        return destination;
    }

    private static void ExecuteMap(ReadOnlySpan<byte> source, Span<byte> destination, int valueCount, int sourceComponents, int destinationComponents, ByteComputeProgram program, CancellationToken cancellationToken)
    {
        int lanes = Vector<int>.Count;
        int vectorizedLength = valueCount - (valueCount % lanes);
        var components = new Vector<int>[sourceComponents];
        Span<int> gathered = stackalloc int[sourceComponents * lanes];
        Span<int> output = stackalloc int[destinationComponents * lanes];
        for (int index = 0; index < vectorizedLength; index += lanes)
        {
            if ((index & 0xFFFF) == 0) cancellationToken.ThrowIfCancellationRequested();
            Gather(source, index, sourceComponents, components, gathered);
            for (int component = 0; component < destinationComponents; component++)
                Evaluate(program.Outputs[component], components).CopyTo(output.Slice(component * lanes, lanes));
            if (lanes == 8 && Avx2.IsSupported && destinationComponents == 1)
                PixelConversionKernels.StoreByte1(output[..lanes], destination, index);
            else if (lanes == 8 && Avx2.IsSupported && destinationComponents == 2)
                PixelConversionKernels.InterleaveByte2(output[..lanes], output.Slice(lanes, lanes), destination, index);
            else if (lanes == 8 && Avx2.IsSupported && destinationComponents == 3)
                PixelConversionKernels.InterleaveByte3(output[..lanes], output.Slice(lanes, lanes), output.Slice(lanes * 2, lanes), destination, index);
            else if (lanes == 8 && Avx2.IsSupported && destinationComponents == 4)
                PixelConversionKernels.InterleaveByte4(output[..lanes], output.Slice(lanes, lanes), output.Slice(lanes * 2, lanes), output.Slice(lanes * 3, lanes), destination, index);
            else
                for (int lane = 0; lane < lanes; lane++)
                    for (int component = 0; component < destinationComponents; component++)
                        destination[((index + lane) * destinationComponents) + component] = (byte)output[(component * lanes) + lane];
        }
    }

    private static void Gather(ReadOnlySpan<byte> source, int valueIndex, int componentCount, Vector<int>[] components, Span<int> gathered)
    {
        int lanes = Vector<int>.Count;
        if (lanes == 8 && Avx2.IsSupported && componentCount == 1)
        {
            Span<int> values = gathered[..lanes];
            ulong packed = MemoryMarshal.Read<ulong>(source.Slice(valueIndex, 8));
            Avx2.ConvertToVector256Int32(Vector128.CreateScalar(packed).AsByte()).CopyTo(values);
            components[0] = new Vector<int>(values);
            return;
        }
        if (lanes == 8 && Avx2.IsSupported && componentCount == 3)
        {
            PixelConversionKernels.DeinterleaveByte3(
                source,
                valueIndex,
                gathered[..lanes],
                gathered.Slice(lanes, lanes),
                gathered.Slice(lanes * 2, lanes));
            components[0] = new Vector<int>(gathered[..lanes]);
            components[1] = new Vector<int>(gathered.Slice(lanes, lanes));
            components[2] = new Vector<int>(gathered.Slice(lanes * 2, lanes));
            return;
        }
        if (lanes == 8 && Avx2.IsSupported && componentCount == 2)
        {
            PixelConversionKernels.DeinterleaveByte2(
                source,
                valueIndex,
                gathered[..lanes],
                gathered.Slice(lanes, lanes));
            components[0] = new Vector<int>(gathered[..lanes]);
            components[1] = new Vector<int>(gathered.Slice(lanes, lanes));
            return;
        }
        if (lanes == 8 && Avx2.IsSupported && componentCount == 4)
        {
            PixelConversionKernels.DeinterleaveByte4(
                source,
                valueIndex,
                gathered[..lanes],
                gathered.Slice(lanes, lanes),
                gathered.Slice(lanes * 2, lanes),
                gathered.Slice(lanes * 3, lanes));
            for (int component = 0; component < 4; component++)
                components[component] = new Vector<int>(gathered.Slice(component * lanes, lanes));
            return;
        }
        for (int component = 0; component < componentCount; component++)
        {
            Span<int> values = gathered.Slice(component * lanes, lanes);
            for (int lane = 0; lane < lanes; lane++)
                values[lane] = source[((valueIndex + lane) * componentCount) + component];
            components[component] = new Vector<int>(values);
        }
    }

    private static Vector<int> Evaluate(ByteComputeNode node, IReadOnlyList<Vector<int>> components) => node switch
    {
        ByteComponentNode component => components[component.Index],
        ByteConstantNode constant => new Vector<int>(constant.Value),
        ByteNegateNode negate => -Evaluate(negate.Operand, components),
        ByteNarrowNode narrow => Evaluate(narrow.Operand, components) & new Vector<int>(255),
        ByteBinaryNode binary => binary.Operation switch
        {
            ByteComputeOperation.Add => Evaluate(binary.Left, components) + Evaluate(binary.Right, components),
            ByteComputeOperation.Subtract => Evaluate(binary.Left, components) - Evaluate(binary.Right, components),
            ByteComputeOperation.Multiply => Evaluate(binary.Left, components) * Evaluate(binary.Right, components),
            ByteComputeOperation.Divide => Evaluate(binary.Left, components) / Evaluate(binary.Right, components),
            _ => throw new ArgumentOutOfRangeException()
        },
        _ => throw new NotSupportedException($"Unknown byte compute node '{node.GetType().Name}'.")
    };
}
