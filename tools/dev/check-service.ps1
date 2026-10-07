param(
    [string[]]$TestPath = @('tests/unit', 'tests/integration', 'tests/e2e'),
    [ValidatePattern('^[a-z][a-z0-9-]*$')]
    [string]$ReportName = 'service-tests',
    [switch]$OriginalReplay
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $projectRoot
if (-not (Get-Command uv -CommandType Application -ErrorAction SilentlyContinue)) {
    throw 'uv is required for locked test dependency export. Install uv and retry check-service.ps1.'
}
. (Join-Path $PSScriptRoot 'compose-context.ps1')
$config = Get-WindComposeConfig
# 只接受仓库内测试路径/单个测试名，避免把选择器拼成任意shell命令。
foreach ($item in $TestPath) {
    if ($item -notmatch '^tests/(unit|integration|e2e)(/[A-Za-z0-9_.-]+)*(::[A-Za-z0-9_]+)?$' -or $item.Contains('..')) {
        throw "Invalid test selector: $item"
    }
    if (-not (Test-Path -LiteralPath ($item -split '::')[0])) {
        throw "Missing test path: $item"
    }
}
if ($TestPath.Count -eq 0) { throw 'At least one test path is required.' }
$selectors = ($TestPath | ForEach-Object { "'$_'" }) -join ' '
$apiContainer = docker compose --profile app ps -q api
if ($LASTEXITCODE -ne 0 -or -not $apiContainer) {
    throw 'Start the application with start-app.ps1 -Build before verification.'
}
$image = docker inspect --format '{{.Image}}' $apiContainer
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the running application image.' }

# 测试依赖仍从同一 uv.lock 导出；临时容器不安装 CUDA，也不改运行中服务的依赖。
$secrets = Get-WindSecretFiles $config
foreach ($path in $secrets.Values) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw 'Current Compose secret file is missing; restore it before verification.' }
}
$null = New-Item -ItemType Directory -Path (Join-Path $projectRoot '.local/runtime') -Force
uv export --locked --only-group dev --no-emit-project --output-file .local/runtime/test-requirements.txt *> .local/runtime/test-export.log
if ($LASTEXITCODE -ne 0) { throw 'Locked test dependency export failed; inspect test-export.log.' }
if ($IsLinux) {
    # CI 的 runtime 为 app group 的 setgid 目录；只补共享组读取，不公开凭据。
    chmod g+r .local/runtime/test-requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Cannot make test dependency export readable to the application group.' }
}

# 共用 API 的网络 namespace，使 loopback e2e 无需放宽 Host 校验；PG 测试仍只清理随机专用库。
$group = if ($OriginalReplay) { 'original_replay' } else { 'not original_replay' }
$replayArguments = @()
if ($OriginalReplay) {
    foreach ($name in @('WIND_ENGIE_DEVELOPMENT_SOURCE','WIND_ENGIE_FINAL_SOURCE')) {
        $path = [Environment]::GetEnvironmentVariable($name)
        if (-not $path -or -not (Test-Path -LiteralPath $path -PathType Container)) {
            throw "Prepare original attachments and set $name before selecting OriginalReplay."
        }
        $target = "/replay/$name"
        $replayArguments += @('--volume', "$([IO.Path]::GetFullPath($path)):${target}:ro", '--env', "${name}=$target")
    }
}
# 测量器的指标解析依赖显式固定，不能借宿主 Jupyter 的间接安装掩盖缺项。
$check = 'pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --target /tmp/wind-test-deps --require-hashes -r .local/runtime/test-requirements.txt -r infra/inference-tools-requirements.txt > /tmp/test-install.log 2>&1 || { cat /tmp/test-install.log; exit 1; }; python -m pytest ' + $selectors + " -m '$group' --junitxml=.local/runtime/$ReportName.xml"
docker run --rm --network "container:$apiContainer" `
    --label "com.docker.compose.project.working_dir=$projectRoot" --label 'wind.role=verification' `
    --volume "${projectRoot}:/workspace" `
    --volume "$($secrets.app_password):/runtime-secrets/app-password.txt:ro" `
    --volume "$($secrets.postgres_password):/workspace/.local/runtime/postgres-password.txt:ro" --workdir /workspace `
    --env 'PYTHONPATH=/tmp/wind-test-deps:/workspace/src' `
    --env 'WIND_DB_HOST=postgres' --env 'WIND_DB_PORT=5432' `
    --env 'WIND_DB_PASSWORD_FILE=/runtime-secrets/app-password.txt' `
    --env 'POWER_FORECAST_DATA=/workspace/data/sample/wind_2019_q1.csv' `
    --env 'WIND_TEST_BASE_URL=http://127.0.0.1:8000' `
    @replayArguments $image sh -c $check 2>&1 | Tee-Object -FilePath ".local/runtime/$ReportName.log"
if ($LASTEXITCODE -ne 0) { throw "Selected verification failed; inspect $ReportName.log." }
Write-Host "Selected checks passed: $($TestPath -join ', '). Only selected e2e checks verify the running image."
