param([Parameter(Mandatory, ValueFromRemainingArguments)][string[]]$Command)

# 本机工具消费与 Compose 相同的端口和凭据位置；不自动同步开发环境。
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
. (Join-Path $PSScriptRoot 'compose-context.ps1')
$config = Get-WindComposeConfig
Set-WindHostEnvironment $config
$program = $Command[0]
$arguments = @($Command | Select-Object -Skip 1)
& $program @arguments
if ($LASTEXITCODE -ne 0) { throw "Local command failed with exit code $LASTEXITCODE." }
