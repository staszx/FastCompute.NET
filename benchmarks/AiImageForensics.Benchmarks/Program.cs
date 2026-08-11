using BenchmarkDotNet.Running;
using AiImageForensics.Benchmarks;

BenchmarkSwitcher
    .FromAssembly(typeof(ForensicsBenchmarks).Assembly)
    .Run(args);
