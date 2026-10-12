# ChartScan ZeroMQ CLIENT launcher.
#   chartscan_cli.ps1 fast 6666
#   chartscan_cli.ps1 etf  7777
#   chartscan_cli.ps1 etc_1 8888
#   chartscan_cli.ps1 etc_2 9999
#
# Scopes: fast / block / etf / etc / etc_1 / etc_2  (see markets/StockCN/scopes.py)
# MUST run in the GolemAI conda env (pyzmq). See chartscan_svr.ps1 for the why.
$ErrorActionPreference = 'Continue'

C:\ProgramData\miniconda3\shell\condabin\conda-hook.ps1
conda activate 'C:\ProgramData\miniconda3'
conda activate GolemAI

Set-Location Y:\Projects\GolemQ

$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:NUMEXPR_NUM_THREADS = '4'

$n = 1
while ($n -lt 6) {
    python -m GolemQ.markets.StockCN.chartscan.client --eval $args[0] --port $args[1]
    Start-Sleep -Seconds 1
}
