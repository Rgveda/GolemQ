# ChartScan ZeroMQ SERVER launcher.
#   chartscan_svr.ps1 '6666,7777,8888,9999'    # one process, one thread per port
#   chartscan_svr.ps1 6666                     # single port (main thread)
#
# MUST run in the GolemAI conda env: torch / ultralytics / pyzmq live there, and
# the GolemQ env is NoMKL (BLAS conflict) so it cannot run torch at all.
# Deliberately NOT `python -m GolemQ.cli` -- the main CLI hard-gates on
# talib / polars>=1.24 which the GolemAI env does not have (exit 1).
$ErrorActionPreference = 'Continue'

C:\ProgramData\miniconda3\shell\condabin\conda-hook.ps1
conda activate 'C:\ProgramData\miniconda3'
conda activate GolemAI

Set-Location Y:\Projects\GolemQ

# Set these BEFORE python starts: GolemQ/__init__.py pins them to 1 via os.environ
# setdefault (the BLAS guard), which would cap torch/MKL intra-op threads.
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:NUMEXPR_NUM_THREADS = '4'

# Restart loop. NOTE: the original never incremented its counter, so this is an
# infinite supervisor-restart loop -- kept on purpose (a crashed server is
# respawned after 1s). Add `$n++` if exactly-5 restarts was the intent.
$n = 1
while ($n -lt 6) {
    python -m GolemQ.markets.StockCN.chartscan.server --port $args[0]
    Start-Sleep -Seconds 1
}
