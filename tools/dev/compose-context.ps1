# 只加载函数；实际调用时才读取 Compose 配置，不创建服务或读取密码内容。
function Assert-WindComposeIdentity($Config) {
    if ($Config.name -eq 'wind-workbench') {
        throw 'The wind-workbench/8000 stack is retired. Restore it explicitly from recovery; it is not an ordinary launch target.'
    }
}

function Get-WindSecretFiles($Config) {
    # 使用 Compose 已消费 .env 后的实际路径，不能再次从 shell 环境猜一遍。
    $paths = @{}
    foreach ($name in @('postgres_password', 'app_password')) {
        $path = $Config.secrets.$name.file
        if (-not $path) { throw "Compose secret file is missing: $name" }
        $paths[$name] = [IO.Path]::GetFullPath($path)
    }
    return $paths
}

function Set-WindHostEnvironment($Config) {
    $secrets = Get-WindSecretFiles $Config
    $env:WIND_DB_HOST = '127.0.0.1'
    $env:WIND_DB_PORT = [string](@($Config.services.postgres.ports | Where-Object target -eq 5432)[0].published)
    $env:WIND_DB_PASSWORD_FILE = $secrets.app_password
    $env:WIND_DB_ADMIN_PASSWORD_FILE = $secrets.postgres_password
    $httpPort = @($Config.services.api.ports | Where-Object target -eq 8000)[0].published
    $env:WIND_TEST_BASE_URL = "http://127.0.0.1:$httpPort"
    $env:WIND_API_PORT = [string]$httpPort
    # 消费 Compose 实际挂载，避免 .env 与 shell 优先级被再次猜测。
    $assistant = @($Config.services.api.volumes | Where-Object target -eq '/app/assistant-runtime')
    if ($assistant.Count -ne 1 -or $assistant[0].type -ne 'bind') {
        throw 'Compose assistant runtime must be one bind mount.'
    }
    $env:WIND_ASSISTANT_RUNTIME = [IO.Path]::GetFullPath($assistant[0].source)
}

function Get-WindComposeConfig {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        $dockerBin = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin'
        if (-not (Test-Path -LiteralPath (Join-Path $dockerBin 'docker.exe'))) {
            throw 'Docker Desktop CLI was not found. Install or start Docker Desktop first.'
        }
        $env:PATH = "$dockerBin;$env:PATH"
    }
    $json = docker compose --profile app config --format json
    if ($LASTEXITCODE -ne 0) { throw 'Compose configuration validation failed.' }
    $config = $json | ConvertFrom-Json
    Assert-WindComposeIdentity $config
    # project 名相同不代表属于此 checkout；误用旧 .env 时禁止接管原栈。
    $containers = @(docker ps -a -q --filter "label=com.docker.compose.project=$($config.name)")
    if ($LASTEXITCODE -ne 0) { throw 'Docker engine is not available.' }
    foreach ($container in $containers) {
        $owner = docker inspect --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}' $container
        if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect Compose ownership.' }
        if (-not $owner -or [IO.Path]::GetFullPath($owner) -ne [IO.Path]::GetFullPath((Get-Location).Path)) {
            throw "Compose project $($config.name) belongs to another checkout ($owner). Set a unique COMPOSE_PROJECT_NAME in .env."
        }
    }
    return $config
}

function Assert-WindPortAvailable($Config, [string]$Service, [int]$TargetPort) {
    $binding = @($Config.services.$Service.ports | Where-Object target -eq $TargetPort)[0]
    $port = [int]$binding.published
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)
    if ($listeners.Count -eq 0) { return }
    $container = docker compose --profile app ps -q $Service
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect $Service ownership." }
    $actual = if ($container) { docker compose --profile app port $Service $TargetPort } else { '' }
    if ($actual -ne "127.0.0.1:$port") {
        throw "Port $port is occupied outside this Compose service. Choose another port in .env."
    }
}

function Test-WindSecretMounts($Config, [string]$Service, $Mounts) {
    # Compose up 的成功退出不能证明全局 secret 源路径已经重新绑定。
    foreach ($secret in @($Config.services.$Service.secrets)) {
        if (-not $secret) { continue }
        $target = if ($secret.target) { $secret.target } else { $secret.source }
        $expected = [IO.Path]::GetFullPath($Config.secrets.($secret.source).file)
        $destination = if ($target.StartsWith('/')) { $target } else { "/run/secrets/$target" }
        $actual = @($Mounts | Where-Object Destination -eq $destination)
        if ($actual.Count -ne 1 -or [IO.Path]::GetFullPath($actual[0].Source) -ne $expected) { return $false }
    }
    return $true
}

function Get-WindSecretRecreateServices($Config, [string[]]$Services) {
    foreach ($service in $Services) {
        $containers = @(docker compose --profile app ps -a -q $service)
        if ($LASTEXITCODE -ne 0) { throw "Cannot inspect current service: $service" }
        foreach ($container in $containers) {
            $mounts = docker inspect --format '{{json .Mounts}}' $container | ConvertFrom-Json
            if ($LASTEXITCODE -ne 0) { throw "Cannot inspect current secret mounts: $service" }
            if (-not (Test-WindSecretMounts $Config $service $mounts)) { $service; break }
        }
    }
}
