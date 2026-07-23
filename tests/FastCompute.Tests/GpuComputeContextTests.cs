using System.Linq.Expressions;
using NUnit.Framework;

namespace FastCompute.Tests;

[TestFixture]
public sealed class GpuComputeContextTests
{
    [Test]
    public void Run_ExecutesMapOnIlGpuAccelerator()
    {
        using ComputeContext context = CreateTestContext();
        float[] source = [0f, 0.25f, 1f, 2f];

        float[] result = context.Run(
            source,
            value => GpuMath.Sin(value) * GpuMath.Exp(-value));

        float[] expected = source
            .Select(value => GpuMath.Sin(value) * GpuMath.Exp(-value))
            .ToArray();
        Assert.That(result, Is.EqualTo(expected).Within(1e-5f));
    }

    [Test]
    public void Zip_ExecutesBinaryExpressionOnIlGpuAccelerator()
    {
        using ComputeContext context = CreateTestContext();
        float[] left = [1f, 2f, 3f, 4f];
        float[] right = [5f, 6f, 7f, 8f];

        float[] result = context.Zip(
            left,
            right,
            (x, y) => GpuMath.Clamp(x * y + 1f, 0f, 25f));

        Assert.That(result, Is.EqualTo(new[] { 6f, 13f, 22f, 25f }).Within(1e-5f));
    }

    [Test]
    public void Precompile_ReportsCacheHitOnSecondCall()
    {
        using ComputeContext context = CreateTestContext();
        Expression<Func<float, float>> expression =
            value => GpuMath.Sqrt(value) + 2f;

        ComputeCompilationResult first = context.Precompile(expression);
        ComputeCompilationResult second = context.Precompile(expression);

        Assert.Multiple(() =>
        {
            Assert.That(first.Backend, Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(first.CacheHit, Is.False);
            Assert.That(second.CacheHit, Is.True);
            Assert.That(second.CompilationTime, Is.EqualTo(TimeSpan.Zero));
        });
    }

    [Test]
    public void Prepare_ReusesPlannedOperation()
    {
        using ComputeContext context = CreateTestContext();
        PreparedCompute<float> operation =
            context.Prepare<float>(value => value * value + 1f);

        Assert.Multiple(() =>
        {
            Assert.That(operation.Run([1f, 2f]), Is.EqualTo(new[] { 2f, 5f }));
            Assert.That(operation.Run([3f, 4f]), Is.EqualTo(new[] { 10f, 17f }));
        });
    }

    [Test]
    public void PreparedOperation_ThrowsAfterContextIsDisposed()
    {
        ComputeContext context = CreateTestContext();
        PreparedCompute<float> operation =
            context.Prepare<float>(value => value + 1f);
        context.Dispose();

        Assert.That(
            () => operation.Run([1f]),
            Throws.TypeOf<ObjectDisposedException>());
    }

    [Test]
    public void BatchPrecompile_PreparesMapAndZip()
    {
        using ComputeContext context = CreateTestContext();

        IReadOnlyList<ComputeCompilationResult> results = context.Precompile(
            ComputeKernel.Map<float>(value => value * 2f),
            ComputeKernel.Zip<float>((left, right) => left + right),
            ComputeKernel.Reduction<float>(ComputeReductionKind.Sum));

        Assert.That(results, Has.Count.EqualTo(3));
        Assert.That(results, Has.All.Property(nameof(ComputeCompilationResult.Backend))
            .EqualTo(ComputeBackendKind.Gpu));
    }

    [Test]
    public void PrecompileAll_CompilesEveryImplementedTemplate()
    {
        using ComputeContext context = CreateTestContext();

        IReadOnlyList<ComputeCompilationResult> first = context.PrecompileAll();
        IReadOnlyList<ComputeCompilationResult> second = context.PrecompileAll();

        Assert.Multiple(() =>
        {
            Assert.That(first, Has.Count.EqualTo(3));
            Assert.That(first, Has.All.Property(nameof(ComputeCompilationResult.CacheHit)).False);
            Assert.That(second, Has.All.Property(nameof(ComputeCompilationResult.CacheHit)).True);
        });
    }

    [Test]
    public void ComputeRun_UsesExplicitGpuBackendAndReusableContext()
    {
        using ComputeContext context = CreateTestContext();
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context
        };

        var result = Compute.RunWithDiagnostics(
            [1f, 2f, 3f],
            value => value * 3f - 1f,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.EqualTo(new[] { 2f, 5f, 8f }));
            Assert.That(result.Diagnostics.Backend, Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(result.Diagnostics.DeviceName, Is.EqualTo(context.DeviceName));
        });
    }

    [Test]
    public void ComputeRun_CanUseSharedDefaultGpuContext()
    {
        float[] result = Compute.Run(
            [1f, 2f, 3f],
            value => value + 0.5f,
            new ComputeOptions { Backend = ComputeBackendKind.Gpu });

        Assert.That(result, Is.EqualTo(new[] { 1.5f, 2.5f, 3.5f }).Within(1e-5f));
    }

    [Test]
    public void ResidentBuffers_ChainMapAndZipWithoutIntermediateDownload()
    {
        using ComputeContext context = CreateTestContext();
        using ComputeBuffer<float> left = context.Upload(new[] { 1f, 2f, 3f });
        using ComputeBuffer<float> right = context.Upload(new[] { 4f, 5f, 6f });
        using ComputeBuffer<float> doubled = left.Select(value => value * 2f);
        using ComputeBuffer<float> result =
            doubled.Zip(right, (x, y) => x + y);

        Assert.That(result.Download(), Is.EqualTo(new[] { 6f, 9f, 12f }));
    }

    [Test]
    public void ResidentBuffers_RejectDifferentContexts()
    {
        using ComputeContext firstContext = CreateTestContext();
        using ComputeContext secondContext = CreateTestContext();
        using ComputeBuffer<float> first = firstContext.Upload(new[] { 1f });
        using ComputeBuffer<float> second = secondContext.Upload(new[] { 2f });

        Assert.That(
            () => first.Zip(second, (x, y) => x + y),
            Throws.TypeOf<ComputeBufferMismatchException>());
    }

    [Test]
    public void TransientMemoryPool_ReusesMapBuffers()
    {
        using ComputeContext context = CreateTestContext();
        float[] source = Enumerable.Range(0, 1_003)
            .Select(index => (float)index)
            .ToArray();

        _ = context.Run(source, value => value * 2f);
        ComputeMemoryPoolStatistics afterFirst = context.MemoryPoolStatistics;
        _ = context.Run(source, value => value * 3f);
        ComputeMemoryPoolStatistics afterSecond = context.MemoryPoolStatistics;

        Assert.Multiple(() =>
        {
            Assert.That(afterFirst.AllocatedBuffers, Is.EqualTo(2));
            Assert.That(afterFirst.AvailableBuffers, Is.EqualTo(2));
            Assert.That(afterSecond.AllocatedBuffers, Is.EqualTo(2));
            Assert.That(afterSecond.Reuses, Is.EqualTo(2));
            Assert.That(afterSecond.Rentals, Is.EqualTo(4));
        });
    }

    [Test]
    public void KernelCacheAndMemoryPool_SupportConcurrentCalls()
    {
        using ComputeContext context = CreateTestContext();
        float[] source = Enumerable.Range(0, 4_096)
            .Select(index => index / 10f)
            .ToArray();

        Task<float[]>[] operations = Enumerable.Range(0, 8)
            .Select(_ => Task.Run(
                () => context.Run(source, value => value * 2f + 1f)))
            .ToArray();
        Task.WaitAll(operations);

        Assert.That(
            operations.Select(task => task.Result[^1]),
            Is.All.EqualTo(source[^1] * 2f + 1f).Within(1e-5f));
    }

    private static ComputeContext CreateTestContext()
    {
        IReadOnlyList<ComputeDeviceInfo> devices = ComputeContext.GetAccelerators();
        ComputeDeviceInfo? cpu = devices.FirstOrDefault(
            device => device.AcceleratorType.Contains("CPU", StringComparison.OrdinalIgnoreCase));

        return ComputeContext.Create(new ComputeContextOptions
        {
            AcceleratorIndex = cpu?.Index
        });
    }
}
