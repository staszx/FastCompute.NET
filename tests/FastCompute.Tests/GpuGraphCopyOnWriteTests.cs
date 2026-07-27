namespace FastCompute.Tests;

[TestFixture]
[Category("GPU")]
public sealed class GpuGraphCopyOnWriteTests
{
    private const int NvidiaAcceleratorIndex = 2;

    [Test]
    public void SelectInPlace_ReusesExclusivelyOwnedAllocation()
    {
        using ComputeContext context = CreateCudaContext();
        using ComputeBuffer<float> buffer =
            context.Upload(new[] { 1.0f, 2.0f, 3.0f });

        ComputeBuffer<float> returned =
            buffer.SelectInPlace(value => value * 2.0f + 1.0f);

        Assert.That(returned, Is.SameAs(buffer));
        Assert.That(context.GraphInPlaceReuseCount, Is.Zero);

        Assert.Multiple(() =>
        {
            Assert.That(
                buffer.Download(),
                Is.EqualTo(new[] { 3.0f, 5.0f, 7.0f }).Within(1e-5f));
            Assert.That(context.GraphInPlaceReuseCount, Is.EqualTo(1));
            Assert.That(context.GraphCopyOnWriteCount, Is.Zero);
        });
    }

    [Test]
    public void SelectInPlace_CopiesWhenOldValueHasAnotherConsumer()
    {
        using ComputeContext context = CreateCudaContext();
        using ComputeBuffer<float> source =
            context.Upload(new[] { 1.0f, 2.0f, 3.0f });
        using ComputeBuffer<float> oldValueBranch =
            source.Select(value => value * 10.0f);

        source.SelectInPlace(value => value + 1.0f);

        Assert.Multiple(() =>
        {
            Assert.That(
                source.Download(),
                Is.EqualTo(new[] { 2.0f, 3.0f, 4.0f }).Within(1e-5f));
            Assert.That(
                oldValueBranch.Download(),
                Is.EqualTo(new[] { 10.0f, 20.0f, 30.0f }).Within(1e-5f));
            Assert.That(context.GraphCopyOnWriteCount, Is.EqualTo(1));
            Assert.That(context.GraphInPlaceReuseCount, Is.Zero);
        });
    }

    [Test]
    public void ZipInPlace_ReusesTargetAndPreservesRightBuffer()
    {
        using ComputeContext context = CreateCudaContext();
        using ComputeBuffer<float> target =
            context.Upload(new[] { 1.0f, 2.0f, 3.0f });
        using ComputeBuffer<float> right =
            context.Upload(new[] { 4.0f, 5.0f, 6.0f });

        ComputeBuffer<float> returned =
            target.ZipInPlace(right, (left, value) => left + value);

        Assert.Multiple(() =>
        {
            Assert.That(returned, Is.SameAs(target));
            Assert.That(
                target.Download(),
                Is.EqualTo(new[] { 5.0f, 7.0f, 9.0f }).Within(1e-5f));
            Assert.That(
                right.Download(),
                Is.EqualTo(new[] { 4.0f, 5.0f, 6.0f }));
            Assert.That(context.GraphInPlaceReuseCount, Is.EqualTo(1));
            Assert.That(context.GraphCopyOnWriteCount, Is.Zero);
        });
    }

    [Test]
    public void ZipInPlace_AliasedRightUsesCopyOnWrite()
    {
        using ComputeContext context = CreateCudaContext();
        using ComputeBuffer<float> source =
            context.Upload(new[] { 1.0f, 2.0f, 3.0f });

        source.ZipInPlace(source, (left, right) => left + right);

        Assert.Multiple(() =>
        {
            Assert.That(
                source.Download(),
                Is.EqualTo(new[] { 2.0f, 4.0f, 6.0f }).Within(1e-5f));
            Assert.That(context.GraphCopyOnWriteCount, Is.EqualTo(1));
            Assert.That(context.GraphInPlaceReuseCount, Is.Zero);
        });
    }

    [Test]
    public void ConsecutiveSelectInPlaceOperationsRemainLazyAndReuseAllocation()
    {
        using ComputeContext context = CreateCudaContext();
        using ComputeBuffer<float> source =
            context.Upload(new[] { 1.0f, 2.0f, 3.0f });

        source
            .SelectInPlace(value => value * 2.0f)
            .SelectInPlace(value => value + 1.0f);

        Assert.That(context.GraphInPlaceReuseCount, Is.Zero);
        Assert.Multiple(() =>
        {
            Assert.That(
                source.Download(),
                Is.EqualTo(new[] { 3.0f, 5.0f, 7.0f }).Within(1e-5f));
            Assert.That(context.GraphInPlaceReuseCount, Is.EqualTo(2));
            Assert.That(context.GraphCopyOnWriteCount, Is.Zero);
        });
    }

    [Test]
    public void ZipInPlace_LengthMismatchLeavesTargetUnchanged()
    {
        using ComputeContext context = CreateCudaContext();
        using ComputeBuffer<float> target =
            context.Upload(new[] { 1.0f, 2.0f });
        using ComputeBuffer<float> right =
            context.Upload(new[] { 3.0f });

        Assert.Throws<ComputeBufferMismatchException>(
            () => target.ZipInPlace(
                right,
                (left, value) => left + value));
        Assert.That(
            target.Download(),
            Is.EqualTo(new[] { 1.0f, 2.0f }));
    }

    [Test]
    public void InPlaceGraphOperationsSupportEmptyBuffers()
    {
        using ComputeContext context = CreateCudaContext();
        using ComputeBuffer<float> target =
            context.Upload(Array.Empty<float>());
        using ComputeBuffer<float> right =
            context.Upload(Array.Empty<float>());

        target
            .SelectInPlace(value => value + 1.0f)
            .ZipInPlace(right, (left, value) => left + value);

        Assert.Multiple(() =>
        {
            Assert.That(target.Download(), Is.Empty);
            Assert.That(context.GraphInPlaceReuseCount, Is.Zero);
            Assert.That(context.GraphCopyOnWriteCount, Is.Zero);
        });
    }

    private static ComputeContext CreateCudaContext()
    {
        ComputeDeviceInfo device = ComputeContext.GetAccelerators()
            .Single(item => item.Index == NvidiaAcceleratorIndex);
        Assert.That(
            device.AcceleratorType,
            Does.Contain("Cuda").IgnoreCase);
        ComputeContext context =
            ComputeContext.Create(
                new ComputeContextOptions
                {
                    AcceleratorIndex = NvidiaAcceleratorIndex
                });
        TestContext.Out.WriteLine(
            $"Graph copy-on-write accelerator: {context.DeviceName}");
        return context;
    }
}
