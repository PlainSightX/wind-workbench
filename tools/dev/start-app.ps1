param(
    [ValidateRange(1024, 65535)][int]$Port,
    [switch]$Build
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if ($PSBoundParameters.ContainsKey('Port')) { $env:WIND_HTTP_PORT = [string]$Port }
. (Join-Path $PSScriptRoot 'compose-context.ps1')
$config = Get-WindComposeConfig
Assert-WindPortAvailable $config 'api' 8000
$httpPort = $config.services.api.ports[0].published

# 基础脚本只负责密码和 PG/Redis；不会运行训练，也不会覆盖已有卷密码。
& (Join-Path $PSScriptRoot 'start-infra.ps1')
$image = docker image ls -q $config.services.api.image
if ($Build -or -not $image) {
    docker compose --profile app build api
    if ($LASTEXITCODE -ne 0) { throw 'Application image build failed; existing services retained.' }
}
$recreate = @(Get-WindSecretRecreateServices $config @('migrate', 'api', 'dispatcher', 'worker'))
if ($recreate.Count -gt 0) {
    docker compose --profile app up -d --force-recreate --no-deps @recreate
    if ($LASTEXITCODE -ne 0) { throw 'Current application secret mount rebinding failed; volumes retained.' }
}
docker compose --profile app up -d --wait --wait-timeout 120
if ($LASTEXITCODE -ne 0) { throw 'Application startup failed. Inspect docker compose logs.' }
if (@(Get-WindSecretRecreateServices $config @('postgres', 'migrate', 'api', 'dispatcher', 'worker')).Count -gt 0) {
    throw 'Startup returned but actual secret mounts still differ from effective Compose configuration.'
}
Write-Host "Wind workbench: http://127.0.0.1:$httpPort/"
Write-Host "API documentation: http://127.0.0.1:$httpPort/docs"
Write-Host 'No training is triggered by startup. Submit an experiment explicitly.'
