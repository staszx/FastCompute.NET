namespace FastCompute;

internal readonly record struct GpuChunkPlan(
    int ElementCount,
    int ChunkElementCount,
    int ChunkCount,
    long FullWorkingSetBytes,
    long BudgetBytes)
{
    internal const long PlanningOverheadBytes = 1024 * 1024;

    internal bool IsChunked => ChunkCount > 1;

    internal static GpuChunkPlan CreateMap(
        int elementCount,
        long budgetBytes,
        bool enableChunking,
        int? requestedChunkElementCount,
        bool enableStreaming)
    {
        if (!enableStreaming)
        {
            return Create(
                elementCount,
                fullLengthBufferCount: 2,
                budgetBytes,
                enableChunking,
                requestedChunkElementCount);
        }

        bool requestedChunkForcesSplit =
            requestedChunkElementCount is int requested &&
            requested < elementCount;
        long standardWorkingSet =
            EstimateWorkingSetBytes(
                elementCount,
                fullLengthBufferCount: 2);
        if (!requestedChunkForcesSplit &&
            standardWorkingSet <= budgetBytes)
        {
            return Create(
                elementCount,
                fullLengthBufferCount: 2,
                budgetBytes,
                enableChunking,
                requestedChunkElementCount);
        }

        return Create(
            elementCount,
            fullLengthBufferCount: 4,
            budgetBytes,
            enableChunking,
            requestedChunkElementCount);
    }

    internal static GpuChunkPlan Create(
        int elementCount,
        int fullLengthBufferCount,
        long budgetBytes,
        bool enableChunking,
        int? requestedChunkElementCount)
        => Create(
            elementCount,
            checked((long)sizeof(float) * fullLengthBufferCount),
            fixedWorkingSetBytes: 0,
            budgetBytes,
            enableChunking,
            requestedChunkElementCount);

    internal static GpuChunkPlan Create(
        int elementCount,
        long bytesPerElement,
        long fixedWorkingSetBytes,
        long budgetBytes,
        bool enableChunking,
        int? requestedChunkElementCount)
    {
        if (bytesPerElement <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(bytesPerElement));
        }

        if (fixedWorkingSetBytes < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(fixedWorkingSetBytes));
        }

        if (elementCount == 0)
        {
            return new GpuChunkPlan(0, 0, 0, 0, budgetBytes);
        }

        long fullWorkingSetBytes =
            EstimateWorkingSetBytes(
                elementCount,
                bytesPerElement,
                fixedWorkingSetBytes);
        long availableElementBytes =
            budgetBytes -
            PlanningOverheadBytes -
            fixedWorkingSetBytes;
        long maximumElementsByBudget =
            availableElementBytes > 0
                ? availableElementBytes / bytesPerElement
                : 0;

        if (!enableChunking)
        {
            if (fullWorkingSetBytes > budgetBytes)
            {
                throw new ComputeGpuMemoryBudgetExceededException(
                    fullWorkingSetBytes,
                    budgetBytes);
            }

            return new GpuChunkPlan(
                elementCount,
                elementCount,
                1,
                fullWorkingSetBytes,
                budgetBytes);
        }

        long requestedOrFull =
            requestedChunkElementCount is int requestedChunk
                ? Math.Min(requestedChunk, elementCount)
                : elementCount;
        long chunkElementCount =
            Math.Min(requestedOrFull, maximumElementsByBudget);

        if (requestedChunkElementCount is not null &&
            requestedOrFull > maximumElementsByBudget)
        {
            throw new ComputeGpuMemoryBudgetExceededException(
                EstimateWorkingSetBytes(
                    checked((int)requestedOrFull),
                    bytesPerElement,
                    fixedWorkingSetBytes),
                budgetBytes);
        }

        if (chunkElementCount <= 0)
        {
            throw new ComputeGpuMemoryBudgetExceededException(
                EstimateWorkingSetBytes(
                    1,
                    bytesPerElement,
                    fixedWorkingSetBytes),
                budgetBytes);
        }

        int effectiveChunkElementCount =
            checked((int)Math.Min(chunkElementCount, int.MaxValue));
        int chunkCount =
            checked(
                (int)(((long)elementCount +
                    effectiveChunkElementCount - 1) /
                    effectiveChunkElementCount));

        return new GpuChunkPlan(
            elementCount,
            effectiveChunkElementCount,
            chunkCount,
            fullWorkingSetBytes,
            budgetBytes);
    }

    internal static long EstimateWorkingSetBytes(
        int elementCount,
        int fullLengthBufferCount)
        => EstimateWorkingSetBytes(
            elementCount,
            checked((long)sizeof(float) * fullLengthBufferCount),
            fixedWorkingSetBytes: 0);

    internal static long EstimateWorkingSetBytes(
        int elementCount,
        long bytesPerElement,
        long fixedWorkingSetBytes)
    {
        if (elementCount == 0)
        {
            return fixedWorkingSetBytes == 0
                ? 0
                : checked(PlanningOverheadBytes + fixedWorkingSetBytes);
        }

        return checked(
            (long)elementCount *
            bytesPerElement +
            PlanningOverheadBytes +
            fixedWorkingSetBytes);
    }
}
