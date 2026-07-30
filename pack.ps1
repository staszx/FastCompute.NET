[CmdletBinding()]
param(
    [string]$Configuration = "Release",
    [string]$Version = "0.7.0",
    [string]$OutputDirectory = "artifacts",
    [switch]$SkipGpuTests
)

$ErrorActionPreference = "Stop"
$repositoryRoot = $PSScriptRoot
$solutionPath = Join-Path $repositoryRoot "FastCompute.sln"
$projectPath = Join-Path $repositoryRoot "src/FastCompute/FastCompute.csproj"
$packageOutput = Join-Path $repositoryRoot $OutputDirectory

function Invoke-DotNet {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    & dotnet @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet $($Arguments -join ' ') failed with exit code $LASTEXITCODE."
    }
}

New-Item -ItemType Directory -Path $packageOutput -Force | Out-Null

Invoke-DotNet restore $solutionPath
Invoke-DotNet build $solutionPath --configuration $Configuration --no-restore
if ($SkipGpuTests) {
    Invoke-DotNet test $solutionPath `
        --configuration $Configuration `
        --no-build `
        --no-restore `
        --filter "TestCategory!=GPU"
}
else {
    Invoke-DotNet test $solutionPath `
        --configuration $Configuration `
        --no-build `
        --no-restore
}
Invoke-DotNet pack $projectPath `
    --configuration $Configuration `
    --no-build `
    --no-restore `
    --output $packageOutput `
    -p:PackageVersion=$Version

$packagePath = Join-Path $packageOutput "FastCompute.$Version.nupkg"
$symbolPath = Join-Path $packageOutput "FastCompute.$Version.snupkg"
if (-not (Test-Path -LiteralPath $packagePath)) {
    throw "Expected package was not created: $packagePath"
}

if (-not (Test-Path -LiteralPath $symbolPath)) {
    throw "Expected symbol package was not created: $symbolPath"
}

$assemblyPath = Join-Path `
    $repositoryRoot `
    "src/FastCompute/bin/$Configuration/net8.0/FastCompute.dll"
$assemblyName = [System.Reflection.AssemblyName]::GetAssemblyName($assemblyPath)
$publicKeyToken = (
    $assemblyName.GetPublicKeyToken() |
        ForEach-Object { $_.ToString("x2") }) -join ""
if ($publicKeyToken -ne "c76a60c96d65300c") {
    throw "Unexpected FastCompute public key token: $publicKeyToken"
}

$smokeProject = Join-Path `
    $repositoryRoot `
    "tests/FastCompute.PackageSmokeTest/FastCompute.PackageSmokeTest.csproj"
$smokeConfig = Join-Path `
    $repositoryRoot `
    "tests/FastCompute.PackageSmokeTest/NuGet.config"
$smokePackages = Join-Path `
    $packageOutput `
    "package-smoke-cache-$Version"
if (Test-Path -LiteralPath $smokePackages) {
    Remove-Item -LiteralPath $smokePackages -Recurse -Force
}

Invoke-DotNet restore $smokeProject `
    --configfile $smokeConfig `
    --packages $smokePackages `
    --no-cache `
    --force-evaluate `
    -p:FastComputePackageVersion=$Version
Invoke-DotNet run `
    --project $smokeProject `
    --configuration $Configuration `
    --no-restore `
    -p:FastComputePackageVersion=$Version

Write-Host "Package ready: $packagePath"
Write-Host "Symbols ready: $symbolPath"
