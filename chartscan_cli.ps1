# ChartScan ZeroMQ CLIENT launcher（在 GolemQ env 启动；CLI 自动切 GolemAI）
#   chartscan_cli.ps1 fast 6666
#   chartscan_cli.ps1 etf  7777
#   chartscan_cli.ps1 etc_1 8888
#   chartscan_cli.ps1 etc_2 9999
# 路径全部推导，换机器不用改：
#   * 仓库根 = 本脚本所在目录（$PSScriptRoot）
#   * conda   = $env:CONDA_EXE → PATH 里的 conda
# 前提：conda env 列表里有 GolemQ 与 GolemAI（torch/ultralytics 只在后者）。
$ErrorActionPreference = 'Stop'

Set-Location -LiteralPath $PSScriptRoot

$Conda = $env:CONDA_EXE
if (-not $Conda) {
    $c = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($c) { $Conda = $c.Source }
}
if (-not $Conda) {
    $c = Get-Command conda -ErrorAction SilentlyContinue
    if ($c) { $Conda = $c.Source }
}
if (-not $Conda) {
    throw '找不到 conda。请安装 Miniconda/Anaconda 并把 conda 放进 PATH。'
}

$Base = (& $Conda info --base).Trim()
$Hook = Join-Path $Base 'shell\condabin\conda-hook.ps1'
if (-not (Test-Path -LiteralPath $Hook)) {
    throw "找不到 conda hook：$Hook"
}
& $Hook
conda activate GolemQ

# ⚠️ $n 从未自增 ⇒ 这是**有意的**无限 supervisor 重启循环（崩了 1 秒后拉起）。
#    若想只重启固定次数，加 $n++。
$n = 1
while ($n -lt 6) {
    python -m GolemQ.cli --scan ChartScan --eval $args[0] --port $args[1]
    Start-Sleep -Seconds 1
}
