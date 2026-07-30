namespace FastCompute.Tests;

public sealed class ComputeBufferContractTests
{
    [Test]
    public async Task Buffer_ExposesContextLocationAndAsyncTransfers()
    {
        ComputeDeviceInfo cpu = ComputeContext.GetAccelerators()
            .First(device => device.AcceleratorType.Contains(
                "CPU",
                StringComparison.OrdinalIgnoreCase));
        using ComputeContext context = ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = cpu.Index
            });

        using ComputeBuffer<float> buffer =
            await context.UploadAsync(new[] { 1.0f, 2.0f, 3.0f });
        float[] result = await buffer.DownloadAsync();

        Assert.Multiple(() =>
        {
            Assert.That(buffer.Context, Is.SameAs(context));
            Assert.That(buffer.Location, Is.EqualTo(ComputeMemoryLocation.Host));
            Assert.That(result, Is.EqualTo(new[] { 1.0f, 2.0f, 3.0f }));
        });
    }

    [Test]
    public void AsyncTransfers_ObservePreCancelledTokens()
    {
        ComputeDeviceInfo cpu = ComputeContext.GetAccelerators()
            .First(device => device.AcceleratorType.Contains(
                "CPU",
                StringComparison.OrdinalIgnoreCase));
        using ComputeContext context = ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = cpu.Index
            });
        using ComputeBuffer<float> buffer = context.Upload([1.0f]);
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        Assert.Multiple(() =>
        {
            Assert.Throws<OperationCanceledException>(
                () => context.UploadAsync(
                    new[] { 1.0f },
                    cancellation.Token));
            Assert.Throws<OperationCanceledException>(
                () => buffer.DownloadAsync(cancellation.Token));
        });
    }

    [Test]
    public void MemoryPool_EvictsIdleBuffersAboveConfiguredLimit()
    {
        ComputeDeviceInfo cpu = ComputeContext.GetAccelerators()
            .First(device => device.AcceleratorType.Contains(
                "CPU",
                StringComparison.OrdinalIgnoreCase));
        using ComputeContext context = ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = cpu.Index,
                MemoryPoolLimitBytes = 0
            });

        _ = context.Run([1.0f, 2.0f], value => value + 1.0f);
        ComputeMemoryPoolStatistics statistics = context.MemoryPoolStatistics;

        Assert.Multiple(() =>
        {
            Assert.That(statistics.LimitBytes, Is.Zero);
            Assert.That(statistics.RetainedBytes, Is.Zero);
            Assert.That(statistics.AvailableBuffers, Is.Zero);
            Assert.That(statistics.EvictedBuffers, Is.GreaterThan(0));
        });
    }
}
