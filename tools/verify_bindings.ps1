$ErrorActionPreference = "Stop"

Write-Host "--- Statelix Binding Verification ---" -ForegroundColor Cyan

# 1. Build Image
Write-Host "Building Container..."
docker build -t statelix-bindings .

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker Build Failed!"
}

# 2. Run Smoke Tests
Write-Host "Running Binding Smoke Tests..."
# We set PYTHONPATH to include current dir so statelix_py package is found
# calling the test script
docker run --rm -e PYTHONPATH=/statelix statelix-bindings python3 tests/python/test_bindings_smoke.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "Binding Verification Success!" -ForegroundColor Green
}
else {
    Write-Error "Binding Verification Failed!"
}
