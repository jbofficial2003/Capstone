# train.ps1 - Complete training orchestration script for federated learning
# Usage: .\train.ps1

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Federated Learning Training Hub" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Clean previous metric artifacts to keep each run consistent.
Write-Host "[CHECK] Resetting previous metric files..." -ForegroundColor Yellow
if (-not (Test-Path ".\results")) {
    New-Item -ItemType Directory -Path ".\results" | Out-Null
}

Get-ChildItem -Path ".\results" -Filter "*_metrics.csv" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path ".\results" -Filter "*_predictions.csv" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path ".\results" -Filter "summary_report.*" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path ".\results" -Filter "label_mapping.json" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host "[OK]    Metrics reset complete" -ForegroundColor Green
Write-Host ""

# Pre-run cleanup: stop stale FL python processes that can keep port 8080 busy.
Write-Host "[CHECK] Cleaning stale federated processes..." -ForegroundColor Yellow
$staleProcs = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and
    $_.CommandLine -and
    ($_.CommandLine -match "server.py" -or $_.CommandLine -match "client_.*\.py" -or $_.CommandLine -match "run_all_clients.py")
}

if ($staleProcs) {
    foreach ($proc in $staleProcs) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
            Write-Host "[CLEAN] Stopped stale PID $($proc.ProcessId)" -ForegroundColor DarkYellow
        }
        catch {
            Write-Host "[WARN]  Could not stop PID $($proc.ProcessId)" -ForegroundColor Yellow
        }
    }
}
else {
    Write-Host "[OK]    No stale federated processes found" -ForegroundColor Green
}

Write-Host ""

# Check dependencies
Write-Host "[CHECK] Verifying dependencies..." -ForegroundColor Yellow

$required_modules = @("flwr", "torch", "pandas", "sklearn")

# Use venv python if available, otherwise system python
$python_cmd = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python.exe" }

$pip_result = & $python_cmd -m pip list --format=json | ConvertFrom-Json
$installed_names = @{
    "flwr" = "flwr"
    "torch" = "torch"
    "pandas" = "pandas"
    "sklearn" = "scikit-learn"
}

$missing = @()
foreach ($module in $required_modules) {
    $package_name = $installed_names[$module]
    if ($pip_result | Where-Object { $_.name -eq $package_name }) {
        Write-Host "[OK]    $module is installed" -ForegroundColor Green
    }
    else {
        Write-Host "[MISSING] $module not found" -ForegroundColor Yellow
        $missing += $module
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "[WARN]  Missing packages: $($missing -join ', ')" -ForegroundColor Yellow
    Write-Host "[HINT]  Run: pip install flwr torch pandas scikit-learn" -ForegroundColor Yellow
    Write-Host ""
    $confirm = Read-Host "Continue anyway? (y/n)"
    if ($confirm -ne "y") {
        Write-Host "[ABORT] Installation cancelled." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "[START] Launching federated training..." -ForegroundColor Yellow
Write-Host "[INFO]  Check console output for training progress" -ForegroundColor Cyan
Write-Host ""

# Launch training
& $python_cmd run_all_clients.py

Write-Host ""
Write-Host "[COMPLETE] Training finished" -ForegroundColor Green
Write-Host "[INFO]    Generating consolidated metrics report..." -ForegroundColor Yellow

# Generate report
& $python_cmd report_metrics.py

Write-Host ""
Write-Host "[SUCCESS] Training and reporting complete!" -ForegroundColor Green
Write-Host "[OUTPUT]  Check results/ folder for detailed metrics:" -ForegroundColor Cyan
Write-Host "          - results/server_metrics.csv" -ForegroundColor Cyan
Write-Host "          - results/*_metrics.csv (per-client)" -ForegroundColor Cyan
Write-Host "          - results/summary_report.json" -ForegroundColor Cyan
Write-Host "          - results/summary_report.csv" -ForegroundColor Cyan
Write-Host ""
