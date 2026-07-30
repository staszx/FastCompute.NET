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
        float scale,
        HistogramOutOfRangeMode outOfRangeMode)
    {
        if (float.IsNaN(value))
        {
            return -1;
        }

        if (value < minimum)
        {
            return outOfRangeMode == HistogramOutOfRangeMode.Clamp ? 0 : -1;
        }

        if (value > maximum)
        {
            return outOfRangeMode == HistogramOutOfRangeMode.Clamp
                ? binCount - 1
                : -1;
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
