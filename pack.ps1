[CmdletBinding()]
param(
    [string]$Configuration = "Release",
    [string]$Version = "0.8.1",
    [string]$OutputDirectory = "artifacts",
    [switch]$SkipGpuTests
)

$ErrorActionPreference = "Stop"
$repositoryRoot = $PSScriptRoot
$solutionPath = Join-Path $repositoryRoot "FastCompute.sln"
$projectPath = Join-Path $repositoryRoot "src/FastCompute/FastCompute.csproj"
$imageProcessingProjectPath = Join-Path `
    $repositoryRoot `
    "src/FastCompute.ImageProcessing/FastCompute.ImageProcessing.csproj"
$packageOutput = Join-Path $repositoryRoot $OutputDirectory

function Invoke-DotNet {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    & dotnet @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet $($Arguments -join ' ') failed with exit code $LASTEXITCODE."
    }
}

function Clear-PackageArtifacts {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [string]$CurrentVersion
    )

    $resolvedPath = (Resolve-Path -LiteralPath $Path).Path
    $packageFiles = Get-ChildItem -LiteralPath $resolvedPath -File |
        Where-Object {
            $_.Name -like "FastCompute.*.nupkg" -or
            $_.Name -like "FastCompute.*.snupkg"
        }

    foreach ($file in $packageFiles) {
        Write-Host "Removing old package artifact: $($file.Name)"
        Remove-Item -LiteralPath $file.FullName -Force
    }

    $smokeCaches = Get-ChildItem -LiteralPath $resolvedPath -Directory |
        Where-Object {
            $_.Name -like "package-smoke-cache-*" -and
            $_.Name -ne "package-smoke-cache-$CurrentVersion"
        }

    foreach ($directory in $smokeCaches) {
        Write-Host "Removing old package smoke cache: $($directory.Name)"
        Remove-Item -LiteralPath $directory.FullName -Recurse -Force
    }
}

New-Item -ItemType Directory -Path $packageOutput -Force | Out-Null
Clear-PackageArtifacts -Path $packageOutput -CurrentVersion $Version

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
Invoke-DotNet pack $imageProcessingProjectPath `
    --configuration $Configuration `
    --no-build `
    --no-restore `
    --output $packageOutput `
    -p:PackageVersion=$Version

$packagePath = Join-Path $packageOutput "FastCompute.$Version.nupkg"
$symbolPath = Join-Path $packageOutput "FastCompute.$Version.snupkg"
$imageProcessingPackagePath = Join-Path `
    $packageOutput `
    "FastCompute.ImageProcessing.$Version.nupkg"
$imageProcessingSymbolPath = Join-Path `
    $packageOutput `
    "FastCompute.ImageProcessing.$Version.snupkg"
if (-not (Test-Path -LiteralPath $packagePath)) {
    throw "Expected package was not created: $packagePath"
}

if (-not (Test-Path -LiteralPath $symbolPath)) {
    throw "Expected symbol package was not created: $symbolPath"
}

if (-not (Test-Path -LiteralPath $imageProcessingPackagePath)) {
    throw `
        "Expected image processing package was not created: $imageProcessingPackagePath"
}

if (-not (Test-Path -LiteralPath $imageProcessingSymbolPath)) {
    throw `
        "Expected image processing symbol package was not created: $imageProcessingSymbolPath"
}

function Assert-PublicKeyToken {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DllPath,
        [string]$AssemblyLabel,
        [string]$ExpectedToken = "c76a60c96d65300c"
    )

    $assemblyName = [System.Reflection.AssemblyName]::GetAssemblyName($DllPath)
    $publicKeyToken = (
        $assemblyName.GetPublicKeyToken() |
            ForEach-Object { $_.ToString("x2") }) -join ""
    if ($publicKeyToken -ne $ExpectedToken) {
        throw `
            "Unexpected $AssemblyLabel public key token: $publicKeyToken"
    }
}

$assemblyPath = Join-Path `
    $repositoryRoot `
    "src/FastCompute/bin/$Configuration/net8.0/FastCompute.dll"
Assert-PublicKeyToken -DllPath $assemblyPath -AssemblyLabel "FastCompute"

$imageProcessingAssemblyPath = Join-Path `
    $repositoryRoot `
    "src/FastCompute.ImageProcessing/bin/$Configuration/net8.0/FastCompute.ImageProcessing.dll"
Assert-PublicKeyToken `
    -DllPath $imageProcessingAssemblyPath `
    -AssemblyLabel "FastCompute.ImageProcessing"

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
Write-Host "Image processing package ready: $imageProcessingPackagePath"
Write-Host "Image processing symbols ready: $imageProcessingSymbolPath"
