<#
.SYNOPSIS
Build LightGBM with GPU (OpenCL) support and create a Python wheel.

.DESCRIPTION
Sets BOOST_ROOT, clones the LightGBM repository if needed, builds the
library using OpenCL via USE_GPU=1, and generates a wheel in
`python-package/dist`.

.PARAMETER BoostRoot
Path to a Boost installation.

.PARAMETER SourceDir
Where LightGBM will be cloned and built.

.PARAMETER OpenCLInclude
Optional path to the OpenCL headers.

.PARAMETER OpenCLLibrary
Optional path to the OpenCL library.
#>
param(
    [string]$BoostRoot = 'C:\\local\\boost_1_82_0',
    [string]$SourceDir = "$PSScriptRoot\\LightGBM",
    [string]$OpenCLInclude = '',
    [string]$OpenCLLibrary = ''
)

$env:BOOST_ROOT = $BoostRoot

if (-not (Test-Path $SourceDir)) {
    git clone --recursive https://github.com/microsoft/LightGBM $SourceDir
}

Push-Location $SourceDir

if (-not (Test-Path 'build')) {
    New-Item -ItemType Directory 'build' | Out-Null
}

Push-Location 'build'

$cmakeArgs = @('..', '-DUSE_GPU=1', '-DCMAKE_BUILD_TYPE=Release')
if ($OpenCLInclude) { $cmakeArgs += "-DOpenCL_INCLUDE_DIR=$OpenCLInclude" }
if ($OpenCLLibrary) { $cmakeArgs += "-DOpenCL_LIBRARY=$OpenCLLibrary" }

cmake @cmakeArgs
cmake --build . --config Release
Pop-Location

Push-Location 'python-package'
python setup.py bdist_wheel
Pop-Location

Pop-Location

