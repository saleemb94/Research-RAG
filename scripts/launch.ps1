<#
    Start Research RAG and open it in the browser.

    The app needs two services that are not part of it: Weaviate in Docker and
    Ollama for generation. Starting app.py on its own when either is down gives
    a stack trace, so this brings them up first, waits until they actually
    answer, and only then starts the app.

    Nothing here is specific to one machine - the repo root is derived from the
    script's own location and the interpreter is looked up rather than assumed -
    so the desktop shortcut keeps working if the project moves.

    Close the window (or Ctrl+C) to stop the app. Weaviate is left running.
#>

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$port      = 7860
$appUrl    = "http://127.0.0.1:$port"
$weaviate  = "http://localhost:8081/v1/.well-known/ready"
$ollama    = "http://localhost:11434/api/tags"

function Test-Endpoint($url) {
    try {
        Invoke-WebRequest -Uri $url -TimeoutSec 3 -UseBasicParsing | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Wait-Endpoint($url, $label, $seconds) {
    $waited = 0
    while (-not (Test-Endpoint $url)) {
        if ($waited -ge $seconds) {
            Write-Host "  $label did not come up after ${seconds}s." -ForegroundColor Red
            return $false
        }
        Start-Sleep -Seconds 2
        $waited += 2
        Write-Host "  waiting for $label... ${waited}s" -ForegroundColor DarkGray
    }
    Write-Host "  $label is ready." -ForegroundColor Green
    return $true
}

function Test-Docker {
    # Judged by exit code, with cmd doing the redirection. PowerShell 5.1 turns
    # a native command's redirected stderr into an ErrorRecord, which under
    # ErrorActionPreference=Stop aborts the script even when the command
    # succeeded - and "docker compose up" writes its progress to stderr.
    cmd /c "docker version >nul 2>&1"
    return ($LASTEXITCODE -eq 0)
}

function Find-Python {
    # A virtualenv in the repo wins, then an explicit override, then whatever
    # conda env holds the dependencies, then PATH.
    $candidates = @()
    $candidates += (Join-Path $root ".venv\Scripts\python.exe")
    if ($env:RESEARCH_RAG_PYTHON) { $candidates += $env:RESEARCH_RAG_PYTHON }
    $candidates += (Join-Path $env:USERPROFILE "anaconda3\envs\wenv\python.exe")
    $candidates += (Join-Path $env:USERPROFILE "miniconda3\envs\wenv\python.exe")
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    return $null
}

Write-Host ""
Write-Host "  Research RAG" -ForegroundColor Cyan
Write-Host "  $root" -ForegroundColor DarkGray
Write-Host ""

# -- already running? just open it ------------------------------------------
#
# The page is probed, not only the health endpoint. An instance started from a
# folder that has since been moved or renamed still answers /api/health, which
# only reads the database, while every request for the page fails: it serves
# static/index.html relative to a working directory that no longer exists.
# Checking health alone made that look healthy and opened a browser onto a 500.
if (Test-Endpoint "$appUrl/api/health") {
    if (Test-Endpoint $appUrl) {
        Write-Host "  Already running - opening the browser." -ForegroundColor Green
        Start-Process $appUrl
        Start-Sleep -Seconds 2
        exit 0
    }
    Write-Host "  Port $port is in use, but not by a working copy of the app." -ForegroundColor Red
    Write-Host "  This is usually an older instance started from a folder that has" -ForegroundColor Red
    Write-Host "  since moved. Close that window, or stop it with:" -ForegroundColor Red
    Write-Host ""
    Write-Host "    Get-NetTCPConnection -LocalPort $port -State Listen |" -ForegroundColor Yellow
    Write-Host "      ForEach-Object { Stop-Process -Id `$_.OwningProcess -Force }" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "  Press Enter to close"
    exit 1
}

$python = Find-Python
if (-not $python) {
    Write-Host "  No Python found. Create a venv in the project folder:" -ForegroundColor Red
    Write-Host "    python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    Read-Host "`n  Press Enter to close"
    exit 1
}
Write-Host "  Python: $python" -ForegroundColor DarkGray

# -- Docker, then Weaviate ---------------------------------------------------
Write-Host ""
Write-Host "  Vector database" -ForegroundColor Cyan
if (-not (Test-Docker)) {
    Write-Host "  Docker is not running - starting Docker Desktop..." -ForegroundColor Yellow
    $dd = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Start-Process $dd } else {
        Write-Host "  Docker Desktop not found at $dd" -ForegroundColor Red
    }
    $waited = 0
    while ($waited -lt 180) {
        if (Test-Docker) { break }
        Start-Sleep -Seconds 3
        $waited += 3
        Write-Host "  waiting for Docker... ${waited}s" -ForegroundColor DarkGray
    }
    if (-not (Test-Docker)) {
        Write-Host "  Docker did not start. Start Docker Desktop and try again." -ForegroundColor Red
        Read-Host "`n  Press Enter to close"
        exit 1
    }
}

cmd /c "docker compose up -d 2>&1" | ForEach-Object {
    Write-Host "  $_" -ForegroundColor DarkGray
}
if (-not (Wait-Endpoint $weaviate "Weaviate" 120)) {
    Read-Host "`n  Press Enter to close"
    exit 1
}

# -- Ollama ------------------------------------------------------------------
Write-Host ""
Write-Host "  Language model" -ForegroundColor Cyan
if (-not (Test-Endpoint $ollama)) {
    Write-Host "  Ollama is not answering - starting it..." -ForegroundColor Yellow
    $ollamaExe = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaExe) {
        Start-Process $ollamaExe.Source -ArgumentList "serve" -WindowStyle Hidden
        Wait-Endpoint $ollama "Ollama" 60 | Out-Null
    } else {
        Write-Host "  Ollama is not installed or not on PATH. Answers will fail" -ForegroundColor Red
        Write-Host "  until it is running: https://ollama.com" -ForegroundColor Red
    }
} else {
    Write-Host "  Ollama is ready." -ForegroundColor Green
}

# -- hand the console to the app --------------------------------------------
# app.py opens the browser itself once it answers, so there is deliberately no
# second opener here - having both is what made two tabs open.
Write-Host ""
Write-Host "  Starting the app (first run loads the embedding models)..." -ForegroundColor Cyan
Write-Host "  $appUrl   (close this window to stop)" -ForegroundColor Green
Write-Host ""
& $python "app.py"

Write-Host ""
Write-Host "  The app has stopped. Weaviate is still running;" -ForegroundColor DarkGray
Write-Host "  stop it with 'docker compose down' if you want to." -ForegroundColor DarkGray
Read-Host "  Press Enter to close"
