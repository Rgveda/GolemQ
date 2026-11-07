# coding: utf-8
"""Watch the 60min ChipDist backfill, then resume --review automatically.

Operational helper (not part of the library):

1. Poll the three backfill processes (local + Gilgamesh + Nightwatcher);
2. when all are gone, report how many review codes still miss `cost15PrzD`;
3. launch `run_review_shard.py` on Gilgamesh (shard 2) and Nightwatcher
   (shard 3), skipping rows already in the DB.

Run detached:

    nohup python tools/watch_chip_then_review.py > _chip_watchdog.log 2>&1 &

Logs: `_chip_watchdog.log` (this file), plus remote `_review_resume{2,3}.log`.
"""
from __future__ import annotations

import base64
import subprocess
import time
from datetime import datetime

HOSTS = {'gilgamesh': '192.168.50.204', 'nightwatcher': '192.168.50.210'}
POLL_SECONDS = 300
MAX_HOURS = 12


def log(msg: str) -> None:
    print('[{}] {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg),
          flush=True)


def ps(cmd: str, timeout: int = 180) -> str:
    """Run a PowerShell command via -EncodedCommand (avoids Bash quoting)."""
    b64 = base64.b64encode(cmd.encode('utf-16-le')).decode()
    p = subprocess.run(
        ['powershell.exe', '-NoProfile', '-EncodedCommand', b64],
        capture_output=True, timeout=timeout)
    # PowerShell on this host emits ANSI/cp936; the values we parse are ASCII.
    return ((p.stdout or b'').decode('utf-8', errors='replace') +
            (p.stderr or b'').decode('utf-8', errors='replace'))


def _count(pattern: str, host: str = '') -> int:
    remote = ("Invoke-Command -ComputerName '{}' -ScriptBlock {{ "
              "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
              "Where-Object {{ $_.CommandLine -match '{}' }}).Count }}"
              .format(HOSTS[host], pattern)) if host else (
        "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
        "Where-Object {{ $_.CommandLine -match '{}' }}).Count".format(pattern))
    out = ps(remote).strip().splitlines()
    for line in reversed(out):
        line = line.strip()
        if line.isdigit():
            return int(line)
    return -1


def launch_review(host: str, code_list: str, log_name: str) -> str:
    cmd = ("cmd.exe /c cd /d C:\\Users\\azai\\GQ_review_20261103 && "
           "C:\\ProgramData\\miniconda3\\envs\\GolemQ\\python.exe "
           "tools\\run_review_shard.py {} --pct-of-cores 50 "
           "--review-timeout 14400 > {}".format(code_list, log_name))
    script = ("$r = Invoke-CimMethod -ComputerName '{}' -ClassName Win32_Process "
              "-MethodName Create -Arguments @{{CommandLine='{}'}}; "
              "Write-Output $r.ProcessId".format(HOSTS[host], cmd))
    out = ps(script).strip().splitlines()
    for line in reversed(out):
        if line.strip().isdigit():
            return line.strip()
    return '?'


def missing_chip() -> tuple[int, int]:
    from GolemQ import get_active_market
    db = get_active_market().get_database()
    c = db['stock_reality_feat_60min']
    rev = set(c.distinct('code', {'zDLagR': {'$ne': None}}))
    chip = set(c.distinct('code', {'cost15PrzD': {'$ne': None}}))
    return len(rev), len(rev - chip)


def main() -> None:
    start = time.time()
    log('watch start; poll every {}s'.format(POLL_SECONDS))
    while time.time() - start < MAX_HOURS * 3600:
        counts = {'local': _count('backfill_hour_extra')}
        for h in HOSTS:
            counts[h] = _count('backfill_hour_extra', h)
        log('backfill processes: {}'.format(counts))
        if counts['local'] == 0 and all(counts[h] == 0 for h in HOSTS):
            break
        time.sleep(POLL_SECONDS)
    else:
        log('MAX_HOURS reached; launching review anyway')

    time.sleep(60)
    try:
        total, miss = missing_chip()
        log('review codes={} missing cost15PrzD={}'.format(total, miss))
    except Exception as exc:                 # noqa: BLE001
        log('DB coverage check failed: {!r}'.format(exc))

    for host, code_list, log_name in (
            ('gilgamesh', '_review_rebal2.txt', '_review_resume2.log'),
            ('nightwatcher', '_review_rebal3.txt', '_review_resume3.log')):
        running = _count('run_review_shard', host)
        log('{} review running={}'.format(host, running))
        if running == 0:
            pid = launch_review(host, code_list, log_name)
            log('{} review launched pid={}'.format(host, pid))
        else:
            log('{} review already running; skip'.format(host))
    log('watch done')


if __name__ == '__main__':
    main()
