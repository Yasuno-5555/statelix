$ErrorActionPreference = "Stop"

Write-Host "--- Statelix Self-Contained Verify ---" -ForegroundColor Cyan

# 1. Build Image (This compiles the code inside)
Write-Host "Building Container (Compiling Code)..."
# --no-cache to ensure we pick up latest local file changes
docker build -t statelix-verify-full .

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker Build Failed!"
}

# 2. Run Benchmarks
Write-Host "Running Benchmarks..."
docker run --rm statelix-verify-full

if ($LASTEXITCODE -eq 0) {
    Write-Host "Verification Success!" -ForegroundColor Green
}
else {
    Write-Error "Verification Failed!"
}
