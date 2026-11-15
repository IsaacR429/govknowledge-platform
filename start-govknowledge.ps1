$ErrorActionPreference = "Stop"

$repo = "C:\Users\isaac\govknowledge-platform"
$backend = Join-Path $repo "backend"
$frontend = Join-Path $repo "frontend"
$python = Join-Path $repo ".runtime\venvs\backend\Scripts\python.exe"

$container = "govknowledge-qa-postgres"
$systemId = "7686228710959620134"

Write-Host ""
Write-Host "============================================"
Write-Host " GOVKNOWLEDGE - START"
Write-Host "============================================"
Write-Host ""

# --------------------------------------------------
# 1. Make sure Docker Desktop is running
# --------------------------------------------------

# --------------------------------------------------
# 1. Make sure Docker Desktop is running
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
# 2. Start QA PostgreSQL
# --------------------------------------------------

$running = docker inspect -f "{{.State.Running}}" $container 2>$null

if ($LASTEXITCODE -ne 0) {
    throw "Container '$container' was not found."
}

if ($running -ne "true") {
    Write-Host "Starting QA PostgreSQL..."
    docker start $container | Out-Null
}

Write-Host "PostgreSQL QA: READY"

# --------------------------------------------------
# 3. Recover QA database password automatically
# --------------------------------------------------

$passwordLine = docker inspect `
    --format "{{range .Config.Env}}{{println .}}{{end}}" `
    $container |
    Select-String "^POSTGRES_PASSWORD=" |
    Select-Object -First 1

if (-not $passwordLine) {
    throw "POSTGRES_PASSWORD could not be read from the QA container."
}

$password = $passwordLine.ToString().Substring("POSTGRES_PASSWORD=".Length)
$encodedPassword = [uri]::EscapeDataString($password)

$databaseUrl = "postgresql+psycopg://govknowledge_qa:$encodedPassword@127.0.0.1:15432/govknowledge_qa"

# --------------------------------------------------
# 4. Start backend
# --------------------------------------------------

$backendCommand = @"
`$env:DATABASE_URL = '$databaseUrl'
`$env:GOVKNOWLEDGE_QA_SYSTEM_ID = '$systemId'

Set-Location '$backend'

Write-Host ''
Write-Host '============================================'
Write-Host ' GOVKNOWLEDGE BACKEND'
Write-Host ' http://127.0.0.1:8000'
Write-Host '============================================'
Write-Host ''

& '$python' -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"@

Start-Process powershell.exe -ArgumentList `
    "-NoExit",
    "-NoProfile",
    "-Command",
    $backendCommand

Write-Host "Backend: STARTING"

# --------------------------------------------------
# 5. Start frontend
# --------------------------------------------------

$frontendCommand = @"
Set-Location '$frontend'

Write-Host ''
Write-Host '============================================'
Write-Host ' GOVKNOWLEDGE FRONTEND'
Write-Host ' http://127.0.0.1:5173'
Write-Host '============================================'
Write-Host ''

npm run dev -- --host 127.0.0.1
"@

Start-Process powershell.exe -ArgumentList `
    "-NoExit",
    "-NoProfile",
    "-Command",
    $frontendCommand

Write-Host "Frontend: STARTING"

# --------------------------------------------------
# 6. Open GovKnowledge
# --------------------------------------------------

Start-Sleep -Seconds 3

Start-Process "http://127.0.0.1:5173"

Write-Host ""
Write-Host "============================================"
Write-Host " GOVKNOWLEDGE STARTED"
Write-Host "============================================"