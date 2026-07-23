using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using FastCompute.Benchmarks;

bool assertPerformance = args.Contains(
    "--assert-performance",
    StringComparer.OrdinalIgnoreCase);

string[] benchmarkArguments = args
    .Where(argument => !string.Equals(
        argument,
        "--assert-performance",
        StringComparison.OrdinalIgnoreCase))
    .ToArray();

if (assertPerformance)
{
    benchmarkArguments =
    [
        .. benchmarkArguments,
        "--anyCategories",
        PerformanceGateVerifier.Category
    ];
}

Summary[] summaries = BenchmarkSwitcher
    .FromAssembly(typeof(Program).Assembly)
    .Run(benchmarkArguments)
    .ToArray();

if (assertPerformance &&
    !PerformanceGateVerifier.Passed(summaries, Console.Error))
{
    Environment.ExitCode = 1;
}
