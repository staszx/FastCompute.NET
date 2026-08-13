using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FastCompute.Backends;

/// <summary>Provides image-agnostic packed AoS component layout kernels.</summary>
internal static class PackedComponentKernels
{
    internal static void DeinterleaveFloat2(ReadOnlySpan<float> input, int valueIndex, Span<float> first, Span<float> second)
    {
        ref float inputReference = ref MemoryMarshal.GetReference(input);
        nuint offset = (nuint)(valueIndex * 2);
        Vector256<float> firstBlock = Vector256.LoadUnsafe(ref inputReference, offset);
        Vector256<float> secondBlock = Vector256.LoadUnsafe(ref inputReference, offset + 8);
        Vector256<int> even = Vector256.Create(0, 2, 4, 6, 0, 2, 4, 6);
        Vector256<int> odd = Vector256.Create(1, 3, 5, 7, 1, 3, 5, 7);
        Avx.Blend(Avx2.PermuteVar8x32(firstBlock, even), Avx2.PermuteVar8x32(secondBlock, even), 0xF0).CopyTo(first);
        Avx.Blend(Avx2.PermuteVar8x32(firstBlock, odd), Avx2.PermuteVar8x32(secondBlock, odd), 0xF0).CopyTo(second);
    }

    internal static void DeinterleaveFloat3(ReadOnlySpan<float> input, int valueIndex, Span<float> first, Span<float> second, Span<float> third)
    {
        ref float inputReference = ref MemoryMarshal.GetReference(input);
        DeinterleaveFloat3(ref inputReference, valueIndex, out Vector256<float> firstVector, out Vector256<float> secondVector, out Vector256<float> thirdVector);
        firstVector.CopyTo(first);
        secondVector.CopyTo(second);
        thirdVector.CopyTo(third);
    }

    internal static void InterleaveFloat2(ReadOnlySpan<float> first, ReadOnlySpan<float> second, Span<float> output, int valueIndex)
    {
        Vector256<float> firstVector = Vector256.Create(first);
        Vector256<float> secondVector = Vector256.Create(second);
        Vector256<float> low = Avx.UnpackLow(firstVector, secondVector);
        Vector256<float> high = Avx.UnpackHigh(firstVector, secondVector);
        ref float outputReference = ref MemoryMarshal.GetReference(output);
        nuint offset = (nuint)(valueIndex * 2);
        Avx.Permute2x128(low, high, 0x20).StoreUnsafe(ref outputReference, offset);
        Avx.Permute2x128(low, high, 0x31).StoreUnsafe(ref outputReference, offset + 8);
    }

    internal static void InterleaveFloat3(ReadOnlySpan<float> first, ReadOnlySpan<float> second, ReadOnlySpan<float> third, Span<float> output, int valueIndex)
    {
        InterleaveFloat3(Vector256.Create(first), Vector256.Create(second), Vector256.Create(third), out Vector256<float> firstBlock, out Vector256<float> secondBlock, out Vector256<float> thirdBlock);
        ref float outputReference = ref MemoryMarshal.GetReference(output);
        nuint offset = (nuint)(valueIndex * 3);
        firstBlock.StoreUnsafe(ref outputReference, offset);
        secondBlock.StoreUnsafe(ref outputReference, offset + 8);
        thirdBlock.StoreUnsafe(ref outputReference, offset + 16);
    }

    internal static void DeinterleaveByte2(ReadOnlySpan<byte> input, int valueIndex, Span<int> first, Span<int> second)
    {
        ref byte source = ref MemoryMarshal.GetReference(input);
        Vector128<byte> packed = Vector128.LoadUnsafe(ref source, (nuint)(valueIndex * 2));
        Vector128<byte> mask = Vector128.Create((byte)0, 0x80, 0x80, 0x80, 2, 0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80);
        Vector256.Create(Ssse3.Shuffle(packed, mask).AsInt32(), Ssse3.Shuffle(packed, Sse2.Add(mask, Vector128.Create((byte)8))).AsInt32()).CopyTo(first);
        mask = Sse2.Add(mask, Vector128.Create((byte)1));
        Vector256.Create(Ssse3.Shuffle(packed, mask).AsInt32(), Ssse3.Shuffle(packed, Sse2.Add(mask, Vector128.Create((byte)8))).AsInt32()).CopyTo(second);
    }

    internal static void DeinterleaveByte3(ReadOnlySpan<byte> input, int valueIndex, Span<int> first, Span<int> second, Span<int> third)
    {
        GatherByte3(input, valueIndex, out Vector256<int> firstVector, out Vector256<int> secondVector, out Vector256<int> thirdVector);
        firstVector.CopyTo(first);
        secondVector.CopyTo(second);
        thirdVector.CopyTo(third);
    }

    internal static void DeinterleaveByte4(ReadOnlySpan<byte> input, int valueIndex, Span<int> first, Span<int> second, Span<int> third, Span<int> fourth)
    {
        ref byte source = ref MemoryMarshal.GetReference(input);
        nuint offset = (nuint)(valueIndex * 4);
        Vector128<byte> low = Vector128.LoadUnsafe(ref source, offset);
        Vector128<byte> high = Vector128.LoadUnsafe(ref source, offset + 16);
        Vector128<byte> mask = Vector128.Create((byte)0, 0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80, 8, 0x80, 0x80, 0x80, 12, 0x80, 0x80, 0x80);
        DeinterleaveByte4Component(low, high, mask, first);
        mask = Sse2.Add(mask, Vector128.Create((byte)1));
        DeinterleaveByte4Component(low, high, mask, second);
        mask = Sse2.Add(mask, Vector128.Create((byte)1));
        DeinterleaveByte4Component(low, high, mask, third);
        mask = Sse2.Add(mask, Vector128.Create((byte)1));
        DeinterleaveByte4Component(low, high, mask, fourth);
    }

    internal static void StoreByte1(ReadOnlySpan<int> values, Span<byte> output, int valueIndex)
    {
        ref byte outputReference = ref MemoryMarshal.GetReference(output);
        StoreEightBytes(PackIntegers(Vector256.Create(values)), ref outputReference, valueIndex);
    }

    internal static void InterleaveByte2(ReadOnlySpan<int> first, ReadOnlySpan<int> second, Span<byte> output, int valueIndex)
    {
        Vector128<byte> packed = Sse2.UnpackLow(PackIntegers(Vector256.Create(first)), PackIntegers(Vector256.Create(second)));
        ref byte destination = ref MemoryMarshal.GetReference(output);
        packed.StoreUnsafe(ref destination, (nuint)(valueIndex * 2));
    }

    internal static void InterleaveByte3(ReadOnlySpan<int> first, ReadOnlySpan<int> second, ReadOnlySpan<int> third, Span<byte> output, int valueIndex)
    {
        Vector128<byte> firstBytes = PackIntegers(Vector256.Create(first));
        Vector128<byte> secondBytes = PackIntegers(Vector256.Create(second));
        Vector128<byte> thirdBytes = PackIntegers(Vector256.Create(third));
        for (int lane = 0; lane < 8; lane++)
        {
            int offset = ((valueIndex + lane) * 3);
            output[offset] = firstBytes.GetElement(lane);
            output[offset + 1] = secondBytes.GetElement(lane);
            output[offset + 2] = thirdBytes.GetElement(lane);
        }
    }

    internal static void InterleaveByte4(ReadOnlySpan<int> first, ReadOnlySpan<int> second, ReadOnlySpan<int> third, ReadOnlySpan<int> fourth, Span<byte> output, int valueIndex)
    {
        Vector128<byte> firstSecond = Sse2.UnpackLow(PackIntegers(Vector256.Create(first)), PackIntegers(Vector256.Create(second)));
        Vector128<byte> thirdFourth = Sse2.UnpackLow(PackIntegers(Vector256.Create(third)), PackIntegers(Vector256.Create(fourth)));
        Vector128<byte> low = Sse2.UnpackLow(firstSecond.AsUInt16(), thirdFourth.AsUInt16()).AsByte();
        Vector128<byte> high = Sse2.UnpackHigh(firstSecond.AsUInt16(), thirdFourth.AsUInt16()).AsByte();
        ref byte destination = ref MemoryMarshal.GetReference(output);
        nuint offset = (nuint)(valueIndex * 4);
        low.StoreUnsafe(ref destination, offset);
        high.StoreUnsafe(ref destination, offset + 16);
    }

    private static void DeinterleaveFloat3(ref float input, int valueIndex, out Vector256<float> first, out Vector256<float> second, out Vector256<float> third)
    {
        nuint offset = (nuint)(valueIndex * 3);
        Vector256<float> block0 = Vector256.LoadUnsafe(ref input, offset);
        Vector256<float> block1 = Vector256.LoadUnsafe(ref input, offset + 8);
        Vector256<float> block2 = Vector256.LoadUnsafe(ref input, offset + 16);
        first = Avx.Blend(Avx.Blend(Avx2.PermuteVar8x32(block0, Vector256.Create(0, 3, 6, 0, 0, 0, 0, 0)), Avx2.PermuteVar8x32(block1, Vector256.Create(0, 0, 0, 1, 4, 7, 0, 0)), 0x38), Avx2.PermuteVar8x32(block2, Vector256.Create(0, 0, 0, 0, 0, 0, 2, 5)), 0xC0);
        second = Avx.Blend(Avx.Blend(Avx2.PermuteVar8x32(block0, Vector256.Create(1, 4, 7, 0, 0, 0, 0, 0)), Avx2.PermuteVar8x32(block1, Vector256.Create(0, 0, 0, 2, 5, 0, 0, 0)), 0x18), Avx2.PermuteVar8x32(block2, Vector256.Create(0, 0, 0, 0, 0, 0, 3, 6)), 0xE0);
        third = Avx.Blend(Avx.Blend(Avx2.PermuteVar8x32(block0, Vector256.Create(2, 5, 0, 0, 0, 0, 0, 0)), Avx2.PermuteVar8x32(block1, Vector256.Create(0, 0, 0, 3, 6, 0, 0, 0)), 0x1C), Avx2.PermuteVar8x32(block2, Vector256.Create(0, 0, 0, 0, 0, 1, 4, 7)), 0xE0);
    }

    private static void InterleaveFloat3(Vector256<float> first, Vector256<float> second, Vector256<float> third, out Vector256<float> block0, out Vector256<float> block1, out Vector256<float> block2)
    {
        block0 = Avx.Blend(Avx.Blend(Avx2.PermuteVar8x32(first, Vector256.Create(0, 0, 0, 1, 0, 0, 2, 0)), Avx2.PermuteVar8x32(second, Vector256.Create(0, 0, 0, 0, 1, 0, 0, 2)), 0x92), Avx2.PermuteVar8x32(third, Vector256.Create(0, 0, 0, 0, 0, 1, 0, 0)), 0x24);
        block1 = Avx.Blend(Avx.Blend(Avx2.PermuteVar8x32(third, Vector256.Create(2, 0, 0, 3, 0, 0, 4, 0)), Avx2.PermuteVar8x32(first, Vector256.Create(0, 3, 0, 0, 4, 0, 0, 5)), 0x92), Avx2.PermuteVar8x32(second, Vector256.Create(0, 0, 3, 0, 0, 4, 0, 0)), 0x24);
        block2 = Avx.Blend(Avx.Blend(Avx2.PermuteVar8x32(second, Vector256.Create(5, 0, 0, 6, 0, 0, 7, 0)), Avx2.PermuteVar8x32(third, Vector256.Create(0, 5, 0, 0, 6, 0, 0, 7)), 0x92), Avx2.PermuteVar8x32(first, Vector256.Create(0, 0, 6, 0, 0, 7, 0, 0)), 0x24);
    }

    private static void GatherByte3(ReadOnlySpan<byte> bytes, int index, out Vector256<int> first, out Vector256<int> second, out Vector256<int> third)
    {
        ref byte input = ref MemoryMarshal.GetReference(bytes);
        nuint offset = (nuint)(index * 3);
        Vector128<byte> low = Vector128.LoadUnsafe(ref input, offset);
        Vector128<byte> high = Vector128.LoadUnsafe(ref input, offset + 8);
        Vector128<byte> lowMask = Vector128.Create((byte)0, 0x80, 0x80, 0x80, 3, 0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80, 9, 0x80, 0x80, 0x80);
        Vector128<byte> highMask = Vector128.Create((byte)4, 0x80, 0x80, 0x80, 7, 0x80, 0x80, 0x80, 10, 0x80, 0x80, 0x80, 13, 0x80, 0x80, 0x80);
        first = Vector256.Create(Ssse3.Shuffle(low, lowMask).AsInt32(), Ssse3.Shuffle(high, highMask).AsInt32());
        lowMask = Sse2.Add(lowMask, Vector128.Create((byte)1)); highMask = Sse2.Add(highMask, Vector128.Create((byte)1));
        second = Vector256.Create(Ssse3.Shuffle(low, lowMask).AsInt32(), Ssse3.Shuffle(high, highMask).AsInt32());
        lowMask = Sse2.Add(lowMask, Vector128.Create((byte)1)); highMask = Sse2.Add(highMask, Vector128.Create((byte)1));
        third = Vector256.Create(Ssse3.Shuffle(low, lowMask).AsInt32(), Ssse3.Shuffle(high, highMask).AsInt32());
    }

    private static void DeinterleaveByte4Component(Vector128<byte> low, Vector128<byte> high, Vector128<byte> mask, Span<int> destination) =>
        Vector256.Create(Ssse3.Shuffle(low, mask).AsInt32(), Ssse3.Shuffle(high, mask).AsInt32()).CopyTo(destination);

    private static Vector128<byte> PackIntegers(Vector256<int> values)
    {
        Vector128<ushort> words = Sse41.PackUnsignedSaturate(values.GetLower(), values.GetUpper());
        return Sse2.PackUnsignedSaturate(words.AsInt16(), Vector128<short>.Zero);
    }

    private static void StoreEightBytes(Vector128<byte> values, ref byte destination, int offset) =>
        Unsafe.WriteUnaligned(ref Unsafe.Add(ref destination, offset), values.AsUInt64().GetElement(0));
}
