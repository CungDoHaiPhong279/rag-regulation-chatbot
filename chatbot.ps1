<#
Bật / tắt chatbot. GPU chỉ bị dùng khi API đang chạy: "stop" là GPU về 0.

    powershell -ExecutionPolicy Bypass -File chatbot.ps1 start    # Qdrant + API + giao diện, mở trình duyệt
    powershell -ExecutionPolicy Bypass -File chatbot.ps1 stop     # tắt hết, giải phóng GPU
    powershell -ExecutionPolicy Bypass -File chatbot.ps1 status

Log nằm trong .run\ (api.log, ui.log).
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('start', 'stop', 'status')]
    [string]$Action
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$RunDir = Join-Path $Root '.run'
$Compose = Join-Path $Root 'docker-compose.yml'

# Mỗi dịch vụ: cổng và chuỗi nhận diện trong dòng lệnh (để không tắt nhầm ứng dụng khác dùng cùng cổng).
$Services = [ordered]@{
    api = @{ Port = 8000; Marker = 'src.api.main:app'; Args = @('-W', 'ignore', '-m', 'uvicorn', 'src.api.main:app', '--host', '127.0.0.1', '--port', '8000') }
    ui  = @{ Port = 8502; Marker = 'src/ui/app.py'; Args = @('-m', 'streamlit', 'run', 'src/ui/app.py', '--server.port', '8502', '--server.headless', 'true') }
}

function Get-ServiceProcess($Service) {
    # Tìm theo dòng lệnh, không theo cổng: mỗi dịch vụ gồm 2 tiến trình (launcher của .venv + python thật),
    # và chỉ tiến trình có đúng chuỗi nhận diện của dự án mới bị tắt.
    @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*$($Service.Marker)*" -and $_.CommandLine -like "*port $($Service.Port)*" })
}

function Get-Health {
    try { return Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 5 } catch { return $null }
}

function Show-Gpu {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $used = (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader) -join ' '
        Write-Host "GPU đang dùng: $used"
    }
}

function Invoke-Docker([string]$Verb) {
    # Docker lỗi chỉ cảnh báo: có thể đang dùng Qdrant local mode (QDRANT_PATH).
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    docker compose -f $Compose $Verb 2>&1 | Out-Null
    $ok = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $previous
    return $ok
}

function Start-Chatbot {
    if (-not (Test-Path $Python)) { throw "Chưa có .venv: xem mục 3 trong CLAUDE.md để cài môi trường." }
    New-Item -ItemType Directory -Force $RunDir | Out-Null
    $env:PYTHONIOENCODING = 'utf-8'

    if (Invoke-Docker 'up -d') { Write-Host 'Qdrant (Docker): đang chạy' }
    else { Write-Warning 'Không bật được Qdrant Docker. Mở Docker Desktop rồi chạy lại (bỏ qua nếu đang dùng QDRANT_PATH).' }

    foreach ($name in $Services.Keys) {
        $service = $Services[$name]
        if ((Get-ServiceProcess $service).Count) { Write-Host "$name đã chạy sẵn (cổng $($service.Port))"; continue }
        Start-Process -FilePath $Python -ArgumentList $service.Args -WorkingDirectory $Root -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $RunDir "$name.out.log") -RedirectStandardError (Join-Path $RunDir "$name.log") | Out-Null
        Write-Host "Đang bật $name (cổng $($service.Port))..."
    }

    Write-Host 'Đang tải model (lần đầu có thể mất 1-2 phút)...'
    $deadline = (Get-Date).AddMinutes(4)
    while (-not ($health = Get-Health)) {
        if ((Get-Date) -gt $deadline) { throw "API không lên sau 4 phút; xem $RunDir\api.log" }
        Start-Sleep -Seconds 3
    }
    Write-Host "API: $($health | ConvertTo-Json -Compress)"
    if (-not $health.collection_ready) {
        Write-Warning 'Chưa có dữ liệu: chạy  .venv\Scripts\python -m src.ingestion.indexer --pdf-dir data/pdfs'
    }
    Show-Gpu
    Start-Process 'http://localhost:8502'
    Write-Host 'Mở http://localhost:8502. Dùng xong chạy: chatbot.ps1 stop'
}

function Stop-Chatbot {
    foreach ($name in $Services.Keys) {
        $procs = Get-ServiceProcess $Services[$name]
        if ($procs.Count) {
            $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            Write-Host "Đã tắt $name"
        }
        else { Write-Host "$name không chạy" }
    }
    if (Invoke-Docker 'stop') { Write-Host 'Đã dừng Qdrant (dữ liệu vẫn giữ nguyên)' }
    Start-Sleep -Seconds 2
    Show-Gpu
}

function Show-Status {
    foreach ($name in $Services.Keys) {
        $state = if ((Get-ServiceProcess $Services[$name]).Count) { 'đang chạy' } else { 'tắt' }
        Write-Host "${name}: $state"
    }
    $health = Get-Health
    if ($health) { Write-Host "API: $($health | ConvertTo-Json -Compress)" }
    Show-Gpu
}

switch ($Action) {
    'start' { Start-Chatbot }
    'stop' { Stop-Chatbot }
    'status' { Show-Status }
}
