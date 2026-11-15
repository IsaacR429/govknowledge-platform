$repo = "C:\Users\isaac\govknowledge-platform"
$pidFile = Join-Path $repo ".runtime\backend.pid"

Write-Host ""
Write-Host "============================================"
Write-Host " GOVKNOWLEDGE BACKEND - STOP"
Write-Host "============================================"
Write-Host ""

if (Test-Path $pidFile) {

    $backendPid = Get-Content $pidFile | Select-Object -First 1

    $process = Get-Process -Id $backendPid -ErrorAction SilentlyContinue

    if ($process) {
        Write-Host "Stopping backend PID $backendPid..."
        Stop-Process -Id $backendPid -Force

        Start-Sleep -Seconds 1

        Write-Host "Backend: STOPPED"
    }
    else {
        Write-Host "Stored backend process is no longer running."
    }

    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}
else {

    $connection = Get-NetTCPConnection `
        -LocalAddress 127.0.0.1 `
        -LocalPort 8000 `
        -State Listen `
        -ErrorAction SilentlyContinue

    if ($connection) {
        Write-Host "Backend found on port 8000."
        Write-Host "Stopping PID $($connection.OwningProcess)..."

        Stop-Process `
            -Id $connection.OwningProcess `
            -Force

        Write-Host "Backend: STOPPED"
    }
    else {
        Write-Host "Backend is already stopped."
    }
}