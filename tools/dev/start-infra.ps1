param([switch]$Pull)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $projectRoot

. (Join-Path $PSScriptRoot 'compose-context.ps1')
$config = Get-WindComposeConfig
Assert-WindPortAvailable $config 'postgres' 5432

$secrets = Get-WindSecretFiles $config
$existingVolumes = @(docker volume ls --format '{{.Name}}')
if ($LASTEXITCODE -ne 0) { throw 'Docker engine is not available. Start Docker Desktop first.' }
foreach ($name in @('postgres_password', 'app_password')) {
    $path = $secrets[$name]
    if (-not (Test-Path -LiteralPath $path)) {
        if ($existingVolumes -contains $config.volumes.postgres_data.name) {
            throw "Existing database volume requires the original $name. Restore it; do not generate another password."
        }
        # 只为初次创建生成密码；重启不轮换，避免与持久卷里的账户失配。
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $path) -Force
        $bytes = New-Object byte[] 32
        $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
        try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
        $password = [Convert]::ToBase64String($bytes)
        [IO.File]::WriteAllText($path, $password, [Text.UTF8Encoding]::new($false))
    }
}

docker compose config --quiet
if ($LASTEXITCODE -ne 0) { throw 'Compose configuration validation failed.' }
if ($Pull) {
    docker compose --progress quiet pull
    if ($LASTEXITCODE -ne 0) { throw 'Image download failed; existing volumes were retained.' }
}
$recreate = @(Get-WindSecretRecreateServices $config @('postgres'))
if ($recreate.Count -gt 0) {
    docker compose up -d --force-recreate --no-deps --wait --wait-timeout 120 postgres
    if ($LASTEXITCODE -ne 0) { throw 'Current PostgreSQL secret mount rebinding failed; volumes retained.' }
}
docker compose up -d --wait --wait-timeout 120 postgres redis
if ($LASTEXITCODE -ne 0) { throw 'Infrastructure did not become healthy. Inspect docker compose logs.' }
docker compose ps
