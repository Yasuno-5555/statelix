$ErrorActionPreference = "Stop"

Write-Host "--- Statelix Build & Verify Tool ---" -ForegroundColor Cyan

# 1. Clean
Write-Host "Cleaning artifacts..."
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyd" | Remove-Item -Force

# 2. Config
Write-Host "Configuring CMake..."
# Explicitly use local vendor dirs and Release mode
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DEIGEN3_INCLUDE_DIR="$PSScriptRoot/../vendor/eigen"

# 3. Build
Write-Host "Building C++ Core..."
cmake --build build --config Release --parallel 4

# 4. Install/Locate
Write-Host "Locating compiled extension..."
$pyd = Get-ChildItem -Path "build" -Recurse -Filter "*.pyd" | Select-Object -First 1

if ($pyd) {
    Write-Host "Found extension: $($pyd.FullName)"
    # Copy to statelix_py/core to ensure local import works
    Copy-Item $pyd.FullName -Destination "statelix_py/core/" -Force
    Write-Host "Copied to statelix_py/core/" -ForegroundColor Green
} else {
    Write-Error "Build failed or .pyd not found!"
}

Write-Host "Build Complete."
