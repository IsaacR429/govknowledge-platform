$ErrorActionPreference = "Stop"

$repo = "C:\Users\isaac\govknowledge-platform"
$backend = Join-Path $repo "backend"
$python = Join-Path $repo ".runtime\venvs\backend\Scripts\python.exe"

$container = "govknowledge-qa-postgres"
$systemId = "7686228710959620134"

Write-Host ""
Write-Host "============================================"
Write-Host " GOVKNOWLEDGE BACKEND - START"
Write-Host "============================================"
Write-Host ""

# --------------------------------------------------
# Check if backend is already running
# --------------------------------------------------

$existing = Get-NetTCPConnection `
    -LocalAddress 127.0.0.1 `
    -LocalPort 8000 `
    -State Listen `
    -ErrorAction SilentlyContinue

if ($existing) {
    Write-Host "Backend is already running on:"
    Write-Host "http://127.0.0.1:8000"
    exit 0
}

# --------------------------------------------------
# Check Docker
# --------------------------------------------------

cmd /c "docker info >nul 2>nul"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Starting Docker Desktop..."

    $dockerDesktop = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"

    if (-not (Test-Path $dockerDesktop)) {
        throw "Docker Desktop could not be found."
    }

    Start-Process $dockerDesktop

    $dockerReady = $false

    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 2

        cmd /c "docker info >nul 2>nul"

        if ($LASTEXITCODE -eq 0) {
            $dockerReady = $true
            break
        }
    }

    if (-not $dockerReady) {
        throw "Docker Desktop did not become available."
    }
}

Write-Host "Docker: READY"

# --------------------------------------------------
# Start PostgreSQL
# --------------------------------------------------

$running = docker inspect -f "{{.State.Running}}" $container 2>$null

if ($LASTEXITCODE -ne 0) {
    throw "Container '$container' was not found."
}

if ($running -ne "true") {
    Write-Host "Starting PostgreSQL..."
    docker start $container | Out-Null
}

Write-Host "PostgreSQL: READY"

# --------------------------------------------------
# Build database URL from container configuration
# --------------------------------------------------

$passwordLine = docker inspect `
    --format "{{range .Config.Env}}{{println .}}{{end}}" `
    $container |
    Select-String "^POSTGRES_PASSWORD=" |
    Select-Object -First 1

if (-not $passwordLine) {
    throw "POSTGRES_PASSWORD could not be read."
}

$password = $passwordLine.ToString().Substring("POSTGRES_PASSWORD=".Length)
$encodedPassword = [uri]::EscapeDataString($password)

$env:DATABASE_URL = "postgresql+psycopg://govknowledge_qa:$encodedPassword@127.0.0.1:15432/govknowledge_qa"
$env:GOVKNOWLEDGE_QA_SYSTEM_ID = $systemId

# --------------------------------------------------
# Start FastAPI
# --------------------------------------------------

Write-Host ""
Write-Host "============================================"
Write-Host " GOVKNOWLEDGE BACKEND"
Write-Host " http://127.0.0.1:8000"
Write-Host " Press Ctrl+C to stop"
Write-Host "============================================"
Write-Host ""

Set-Location $backend

& $python -m uvicorn app.main:app `
    --host 127.0.0.1 `
    --port 8000 `
    --log-level info `
    --access-log

# --------------------------------------------------
# Wait until FastAPI actually responds
# --------------------------------------------------

$ready = $false

for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1

    try {
        $response = Invoke-WebRequest `
            "http://127.0.0.1:8000/docs" `
            -UseBasicParsing `
            -TimeoutSec 2

        if ($response.StatusCode -eq 200) {
            $ready = $true
            break
        }
    }
    catch {
    }
}

if (-not $ready) {
    Write-Host ""
    Write-Host "Backend failed to become ready."
    Write-Host "Check:"
    Write-Host $stderrLog
    exit 1
}

Write-Host ""
Write-Host "============================================"
Write-Host " BACKEND READY"
Write-Host " http://127.0.0.1:8000"
Write-Host " PID: $($process.Id)"
Write-Host "============================================"