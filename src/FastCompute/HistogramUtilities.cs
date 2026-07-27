using System.Runtime.CompilerServices;

namespace FastCompute;

internal static class HistogramUtilities
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int GetBinIndex(
        float value,
        int binCount,
        float minimum,
        float maximum,
        float scale)
    {
        if (float.IsNaN(value) ||
            value < minimum ||
            value > maximum)
        {
            return -1;
        }

        if (value == maximum)
        {
            return binCount - 1;
        }

        int binIndex = (int)((value - minimum) * scale);
        return (uint)binIndex < (uint)binCount
            ? binIndex
            : -1;
    }
}
