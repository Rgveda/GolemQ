# Open the whole ChartScan fleet as Windows Terminal tabs:
#   server on 6666, server on 7777, client fast->6666, client etc_2->7777
# Same 4-tab shape as the legacy GolemAI launcher; ports/scopes are the defaults.
$root = $PSScriptRoot

wt new-tab "powershell" -executionPolicy bypass -noexit `
   -command "$root\chartscan_svr.ps1 6666,7777,8888,9999" --title "ChartScanSvr" `; `
   new-tab "powershell" -executionPolicy bypass -noexit `
   -command "$root\chartscan_cli.ps1 fast 6666" --title "ChartScanCli-fast-6666" `; `
   new-tab "powershell" -executionPolicy bypass -noexit `
   -command "$root\chartscan_cli.ps1 etf 7777" --title "ChartScanCli-etf-7777" `; `
   new-tab "powershell" -executionPolicy bypass -noexit `
   -command "$root\chartscan_cli.ps1 etc_1 8888" --title "ChartScanCli-etc_1-8888" `; `
   new-tab "powershell" -executionPolicy bypass -noexit `
   -command "$root\chartscan_cli.ps1 etc_2 9999" --title "ChartScanCli-etc_2-9999"
