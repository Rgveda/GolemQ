# ChartScan 一键启动 —— Windows Terminal 里开 1 个服务端标签页 + 4 个客户端标签页。
#
#   powershell -ExecutionPolicy Bypass -File <repo>\chartscan.ps1
#
# ⚠️ 两个曾经踩过的坑（2026-10-17 实测）：
#
# 1) `new-tab` **不是** PowerShell 命令，它是 `wt.exe` 的子命令。
#    写成 `wt new-tab ...; new-tab ...` 时，**只有第一条**是 wt 的，
#    后面每条都会让 PowerShell 去找名为 `new-tab` 的外部程序 ⇒
#    「术语 'new-tab' 不会被识别为 cmdlet、函数、脚本文件或可执行程序的名称」。
#    ⇒ **每一行都要以 `wt.exe` 开头。**
#
# 2) `--title` 属于 `new-tab` 的**选项区**（子命令之后、被启动的命令行之前）。
#    写在最后会跟着 `powershell ...` 一起被当成 powershell 的参数。
#
# 环境：`chartscan_svr.ps1` / `chartscan_cli.ps1` 都在 **GolemQ** env 里起，
# 由 CLI 自己切到 GolemAI（`cli/commands/chartscan.py`，`DECISIONS.md` D51 五之二）。

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Svr  = Join-Path $Root 'chartscan_svr.ps1'
$Cli  = Join-Path $Root 'chartscan_cli.ps1'

foreach ($f in @($Svr, $Cli)) {
    if (-not (Test-Path -LiteralPath $f)) { throw "找不到启动脚本：$f" }
}

$wt = (Get-Command wt.exe -ErrorAction SilentlyContinue).Source
if (-not $wt) {
    throw '没找到 wt.exe（Windows Terminal）。请装 Windows Terminal，或直接用两个普通 PowerShell 窗口分别跑 chartscan_svr.ps1 / chartscan_cli.ps1。'
}

# `-w 0` = 复用**当前**窗口（不在 Windows Terminal 里时则新开一个）。
# 之后每个 `wt -w 0 new-tab` 都挂到同一个窗口的标签页上。

& $wt -w 0 new-tab --title 'ChartScanSvr'    powershell -ExecutionPolicy Bypass -NoExit -File $Svr '6666,7777,8888,9999'
& $wt -w 0 new-tab --title 'ChartScanCli6666' powershell -ExecutionPolicy Bypass -NoExit -File $Cli fast  6666
& $wt -w 0 new-tab --title 'ChartScanCli7777' powershell -ExecutionPolicy Bypass -NoExit -File $Cli etf   7777
& $wt -w 0 new-tab --title 'ChartScanCli8888' powershell -ExecutionPolicy Bypass -NoExit -File $Cli etc_1 8888
& $wt -w 0 new-tab --title 'ChartScanCli9999' powershell -ExecutionPolicy Bypass -NoExit -File $Cli etc_2 9999
