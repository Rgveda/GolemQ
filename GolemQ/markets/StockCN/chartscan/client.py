# coding:utf-8
"""ChartScan 的 **ZeroMQ REQ 客户端** —— 按 scope 解析出任务、逐条发给服务端。

    python -m GolemQ.cli --scan ChartScan --eval fast --port 6666   # ← 首选（自动切 env）
    python -m GolemQ.markets.StockCN.chartscan.client --eval etf  --port 7777

⚠️ 同 `server.py`：要在 `GolemAI` env 里跑（要 pyzmq；且它读的账本在服务端那侧）。
走 CLI 那条时**不需要**手动切 —— `cli/commands/chartscan.py` 会 re-exec 过去（D51）。

搬自旧树 `GolemAI/GolemQ/cli/__main__.py:247-392`。**没搬**两条旧入口
（用户 2026-10-13 定「只搬要用的」）：`--eval` 给一段 notebook 文本、以及给一长串
原始代码（`len(eval) > 12`）—— 都属旧研究的残迹，且后者那条路有 `NameError` 缺陷
（`market_type` 只在 `else` 分支被赋值）。

⚠️ **`market_type` 逐码定**（这是对旧 CLI 的**刻意更正**）：旧 CLI 给整个 `etf`
scope 传 `INDEX_CN`（`__main__.py:229-232`），而那张 506 个码的表里**混着 ETF 与
真指数** ⇒ 传 `INDEX_CN` 会让 ETF 去读 `index_60min` 而**静默读空**。
这里改用 `is_stock_cn(code)[1]` —— 它就是 `MARKET_TYPE` 的取值
（`'stock_cn'` / `'etf_cn'` / `'index_cn'`），读口与结果族**两边都**吃它。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime as dt
from datetime import timedelta

import pandas as pd

from GolemQ.markets.StockCN.symbol import is_stock_cn

from .predict import calc_ChartScaned_persistence

#: 客户端等回包的秒数（旧树 `__main__.py:330`）。⚠️ 与**服务端一次完整推理的耗时**标定 ——
#: 别在「优化取数」时把它一起动掉（见 `predict.py` 末尾的告警）。
TIMEOUT_SECONDS = 45


def _last_day():
    from GolemQ.markets.StockCN.date_utils import GQ_util_get_last_day
    return GQ_util_get_last_day(ts=dt.today())


def _if_tradetime():
    from GolemQ.markets.StockCN.date_utils import GQ_util_if_tradetime
    return GQ_util_if_tradetime(dt.now())


def market_type_of(code: str) -> str:
    """逐码定市场类型 —— 见模块 docstring 的 ⚠️。

    >>> market_type_of('600519'), market_type_of('510300'), market_type_of('399001')
    ('stock_cn', 'etf_cn', 'index_cn')
    """
    return is_stock_cn(code)[1]


def build_jobs(code_list, verbose=True):
    """逐码问「要重扫什么」→ 任务列表 `(ticker, market_type, mode, start, end)`。

    逐字搬自旧树那一段（含 mode 1 的**按月切块**、mode 0 跳过、以及 `>28 天` 的
    `Fooo` 提示）。
    """
    from tqdm import tqdm
    code_job = []
    for code in tqdm(code_list, desc='Processing symbols', unit='code', disable=not verbose):
        req = calc_ChartScaned_persistence(code, market_type=market_type_of(code))
        if req[2] == 1:
            start_date, end_date = req[3], req[4]
            cur = start_date
            while cur <= end_date:
                nxt = min(cur + pd.DateOffset(months=1), end_date)
                code_job.append((req[0], req[1], req[2], cur, nxt))
                cur = nxt + pd.DateOffset(days=1)
            req = code_job[-1]
        elif req[2] == 0:
            continue
        else:
            if (req[2] == 2) and (pd.to_datetime(req[4]) - pd.to_datetime(req[3]) > timedelta(days=28)):
                if verbose:
                    tqdm.write(f'Fooo: {code}, {req[3]} {req[4]}')
            code_job.append(req)
        if verbose and (len(code_job) % 63 == 7) and (len(code_list) > 500):
            tqdm.write(f'Total: {len(code_job)} {req}')
    return code_job


def _remove_outdated_jobs(code_job):
    """丢掉「全扫且远端（>31 天）」的任务（旧树 `:302-304`）。"""
    last_day = _last_day()
    return [j for j in code_job
            if not (j[2] == 1 and (last_day - pd.to_datetime(j[4])) > timedelta(days=31))]


def _throttle(code_job, code_list, scope, verbose=True):
    """跑完之后的**休市节流**（旧树 `:351-392` 逐字）。

    低于阈值（占 `len(code_list)` 的 0.168 / 0.382 / 0.618；`etf` 另有一条 0.832）
    且**非交易时段**时就挂起 —— 4h（周末）/ 2h（夜间）/ 1h / 30min / 15min。
    """
    import time
    say = print if verbose else (lambda *a, **k: None)
    if _if_tradetime():
        return
    n, total = len(code_job), len(code_list)
    weekend = dt.now().weekday() >= 5
    night = dt.now().hour >= 22 or dt.now().hour < 6
    if (n < total * 0.168) or (scope == 'etf' and n < total * 0.832):
        hours, label = (4, '周末') if weekend else ((2, '夜间') if night else (1, '休市'))
    elif n < total * 0.382:
        hours = 4 if weekend else (2 if night else 0.5)
        label = '周末' if weekend else ('夜间' if night else '休市')
    elif n < total * 0.618:
        hours = 4 if weekend else (2 if night else 0.25)
        label = '周末' if weekend else ('夜间' if night else '休市')
    else:
        return
    say(f'{label}阶段: 将会挂起 {hours} 小时....')
    time.sleep(hours * 3600)


def run_client(scope, port, verbose=True):
    """解析 scope → 建任务 → 逐条 REQ。返回 `(code_list, code_job)`（便于测试）。"""
    from GolemQ.analysis.cv.scopes import perpar_symbol_range
    from tqdm import tqdm
    import zmq                              # 只在 GolemAI env 有（见模块 docstring）

    say = print if verbose else (lambda *a, **k: None)
    say(f'[{dt.now().strftime("%Y-%m-%d %H:%M:%S")}]: ChartScan 扫描 (scope:{scope} port:{port}) ...')
    # scope 解析在**跨市场**的 `analysis/cv/scopes.py`，数据落点由本站的 `scan_dataset()`
    # 注入 —— 那个模块不 import 任何市场包（2026-10-13 归位，见 D44）。
    from GolemQ.core.market_registry import get_active_market
    code_list = perpar_symbol_range(scope, dataset=get_active_market().scan_dataset(),
                                    verbose=verbose)
    if code_list is False:
        say(f'错误: 无法获取股票列表 [{scope}]。可先跑 `--save tdx` 灌 stock_list。')
        sys.exit(1)

    code_job = build_jobs(code_list, verbose=verbose)
    if (len(code_job) < len(code_list) * 2.68) and _if_tradetime():
        code_job = _remove_outdated_jobs(code_job)
    say(f'Total: {len(code_job)}')

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f'tcp://localhost:{port}')
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    with tqdm(code_job, desc='Processing requests', unit='request', disable=not verbose) as pbar:
        for ticker, market_type, _, start_date, end_date in pbar:
            request = {'ticker': ticker, 'market_type': market_type,
                       'start_date': start_date.strftime('%Y-%m-%d'),
                       'end_date': end_date.strftime('%Y-%m-%d')}
            try:
                socket.send_json(request)
                if poller.poll(TIMEOUT_SECONDS * 1000):
                    response = socket.recv_json()
                    pbar.set_postfix_str(f'Ticker: {ticker}')
                    if response.get('status') == 'error':
                        pbar.write(f"Error for {ticker}: {response.get('message', 'Unknown error')}")
                else:
                    pbar.write(f'Timeout for {ticker}: No response received within {TIMEOUT_SECONDS} seconds')
                    # 超时后 socket 处于未知状态 ⇒ 关掉重连（旧树 `:339-343`）
                    socket.close()
                    socket = context.socket(zmq.REQ)
                    socket.connect(f'tcp://localhost:{port}')
                    poller.unregister(socket)
                    poller.register(socket, zmq.POLLIN)
            except KeyError as exc:
                say(f'[{dt.now().strftime("%Y-%m-%d %H:%M:%S")}]: ZeroMQ Request: {request} got a Error! {exc}')
    socket.close()
    context.term()
    _throttle(code_job, code_list, scope, verbose=verbose)
    return code_list, code_job


def main(argv=None):
    parser = argparse.ArgumentParser(description='ChartScan ZeroMQ 客户端（REQ）')
    parser.add_argument('--eval', type=str, default='',
                        help='扫描范围。可用: fast / block / etf / etc / etc_1 / etc_2')
    parser.add_argument('--port', type=str, default='5555', help='服务端端口')
    parser.add_argument('-q', '--quiet', action='store_true', default=False)
    args = parser.parse_args(argv)
    run_client(args.eval, args.port, verbose=not args.quiet)


if __name__ == '__main__':
    main()
