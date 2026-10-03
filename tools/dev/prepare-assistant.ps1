param([switch]$Vector, [switch]$ProbeProvider)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
. (Join-Path $PSScriptRoot 'compose-context.ps1')
$config = Get-WindComposeConfig
Set-WindHostEnvironment $config
# 准备容器使用 CPU 依赖，并且仅此次可写助手目录；API 仍只读。
$assistant = $env:WIND_ASSISTANT_RUNTIME
$tools = (Join-Path (Get-Location).Path 'tools').Replace('\', '/')
function Invoke-AssistantTool([string[]]$Arguments) {
    docker compose --profile app run --rm --no-deps --volume "${tools}:/tools:ro" `
        --volume "${assistant}:/app/assistant-prepare:rw" `
        --env 'WIND_ASSISTANT_RUNTIME=/app/assistant-prepare' `
        --env 'HF_HOME=/app/assistant-prepare/.cache/huggingface' `
        --env 'HF_XET_CACHE=/app/assistant-prepare/.cache/huggingface/xet' `
        --env DEEPSEEK_API_KEY api python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'Assistant preparation failed.' }
}
Invoke-AssistantTool @('/tools/diagnostics/assistant_probe.py', 'configure')
if ($Vector) {
    Invoke-AssistantTool @('/tools/diagnostics/assistant_probe.py', 'prepare-embedding')
    Invoke-AssistantTool @('/tools/diagnostics/assistant_probe.py', 'vector')
    Invoke-AssistantTool @('/tools/dev/index-assistant.py')
}
# 默认关键词模式不下载 embedding、不建向量索引、不发送 provider 请求。
if ($ProbeProvider) { Invoke-AssistantTool @('/tools/diagnostics/assistant_probe.py', 'provider') }
