namespace FastCompute.Tests;

public sealed class NumericTypeTests
{
    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    public void Double_MapZipAndReductions_WorkOnCpuBackends(
        ComputeBackendKind backend)
    {
        double[] source = [1d, 2d, 3d, 4d];
        double[] right = [4d, 3d, 2d, 1d];
        var options = new ComputeOptions { Backend = backend };

        double[] map = Compute.Run(
            source,
            value => value * 2d + 0.5d,
            options);
        double[] zip = Compute.Zip(
            source,
            right,
            (left, value) => left * value + 1d,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(map, Is.EqualTo(new[] { 2.5d, 4.5d, 6.5d, 8.5d }));
            Assert.That(zip, Is.EqualTo(new[] { 5d, 7d, 7d, 5d }));
            Assert.That(Compute.Sum(source, options), Is.EqualTo(10d));
            Assert.That(Compute.Min(source, options), Is.EqualTo(1d));
            Assert.That(Compute.Max(source, options), Is.EqualTo(4d));
            Assert.That(Compute.Average(source, options), Is.EqualTo(2.5d));
        });
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    public void Int_MapZipAndReductions_WorkOnCpuBackends(
        ComputeBackendKind backend)
    {
        int[] source = [1, 2, 3, 4];
        int[] right = [4, 3, 2, 1];
        var options = new ComputeOptions { Backend = backend };

        int[] map = Compute.Run(
            source,
            value => value * 2 + 1,
            options);
        int[] zip = Compute.Zip(
            source,
            right,
            (left, value) => left * value + 1,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(map, Is.EqualTo(new[] { 3, 5, 7, 9 }));
            Assert.That(zip, Is.EqualTo(new[] { 5, 7, 7, 5 }));
            Assert.That(Compute.Sum(source, options), Is.EqualTo(10));
            Assert.That(Compute.Min(source, options), Is.EqualTo(1));
            Assert.That(Compute.Max(source, options), Is.EqualTo(4));
            Assert.That(Compute.Average(source, options), Is.EqualTo(2));
        });
    }

    [Test]
    public void Double_SystemMath_IsSupportedOutsideSimd()
    {
        double[] source = [0d, 0.5d, 1d];
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Scalar
        };

        double[] result = Compute.Run(
            source,
            value => Math.Sin(value) * Math.Exp(-value),
            options);

        Assert.That(
            result,
            Is.EqualTo(
                    source.Select(
                        value => Math.Sin(value) * Math.Exp(-value)))
                .Within(1e-12));
    }

    [Test]
    public void InPlaceNumericOperations_PreserveArrayIdentity()
    {
        int[] target = [1, 2, 3];
        int[] right = [3, 2, 1];

        int[] mapResult = Compute.RunInPlace(
            target,
            value => value + 1,
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });
        int[] zipResult = Compute.ZipInPlace(
            target,
            right,
            (left, value) => left * value,
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        Assert.Multiple(() =>
        {
            Assert.That(mapResult, Is.SameAs(target));
            Assert.That(zipResult, Is.SameAs(target));
            Assert.That(target, Is.EqualTo(new[] { 6, 6, 4 }));
        });
    }

    [Test]
    public void NumericOperations_RunOnIlgpuCpuAccelerator()
    {
        ComputeDeviceInfo cpu = ComputeContext.GetAccelerators()
            .First(device => device.AcceleratorType.Contains(
                "CPU",
                StringComparison.OrdinalIgnoreCase));
        using ComputeContext context = ComputeContext.Create(
            new ComputeContextOptions { AcceleratorIndex = cpu.Index });
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context
        };

        double[] doubles = Compute.Run(
            [0d, 0.5d, 1d],
            value => Math.Sin(value) + 2d,
            options);
        int[] integers = Compute.Zip(
            [1, 2, 3],
            [3, 2, 1],
            (left, right) => left * right + 1,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(
                doubles,
                Is.EqualTo(
                        new[]
                        {
                            2d,
                            Math.Sin(0.5d) + 2d,
                            Math.Sin(1d) + 2d
                        })
                    .Within(1e-12));
            Assert.That(integers, Is.EqualTo(new[] { 4, 5, 4 }));
            Assert.That(Compute.Sum(new[] { 1d, 2d, 3d }, options), Is.EqualTo(6d));
            Assert.That(Compute.Max(new[] { 1, 9, 3 }, options), Is.EqualTo(9));
        });
    }

    [Test]
    public async Task NumericResidentBuffers_MapZipReduceWithoutHostApiChanges()
    {
        ComputeDeviceInfo cpu = ComputeContext.GetAccelerators()
            .First(device => device.AcceleratorType.Contains(
                "CPU",
                StringComparison.OrdinalIgnoreCase));
        using ComputeContext context = ComputeContext.Create(
            new ComputeContextOptions { AcceleratorIndex = cpu.Index });
        using ComputeBuffer<double> left =
            await context.UploadAsync(new[] { 1d, 2d, 3d });
        using ComputeBuffer<double> right =
            context.Upload(new[] { 3d, 2d, 1d });
        using ComputeBuffer<double> mapped =
            left.Select(value => value * 2d);
        using ComputeBuffer<double> zipped =
            mapped.Zip(right, (first, second) => first + second);

        double[] result = await zipped.DownloadAsync();

        Assert.Multiple(() =>
        {
            Assert.That(result, Is.EqualTo(new[] { 5d, 6d, 7d }));
            Assert.That(zipped.Sum(), Is.EqualTo(18d));
            Assert.That(zipped.Min(), Is.EqualTo(5d));
            Assert.That(zipped.Max(), Is.EqualTo(7d));
            Assert.That(zipped.Average(), Is.EqualTo(6d));
        });
    }

    [Test]
    public void NumericKernels_CanBePrecompiledAndPrepared()
    {
        ComputeDeviceInfo cpu = ComputeContext.GetAccelerators()
            .First(device => device.AcceleratorType.Contains(
                "CPU",
                StringComparison.OrdinalIgnoreCase));
        using ComputeContext context = ComputeContext.Create(
            new ComputeContextOptions { AcceleratorIndex = cpu.Index });

        ComputeCompilationResult first =
            context.Precompile<double>(value => value * 2d);
        ComputeCompilationResult second =
            context.Precompile<double>(value => value + 1d);
        ComputeCompilationResult reduction =
            context.PrecompileReduction<int>(ComputeReductionKind.Sum);
        PreparedCompute<int> prepared =
            context.Prepare<int>(value => value * 3);
        int[] result = prepared.Run([1, 2, 3]);

        Assert.Multiple(() =>
        {
            Assert.That(first.CacheHit, Is.False);
            Assert.That(second.CacheHit, Is.True);
            Assert.That(reduction.CacheHit, Is.False);
            Assert.That(result, Is.EqualTo(new[] { 3, 6, 9 }));
        });
    }
}
