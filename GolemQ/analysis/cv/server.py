# coding:utf-8
"""ChartScan 的 **ZeroMQ REP 服务端**。

    python -m GolemQ.analysis.cv.server --port 6666
    python -m GolemQ.analysis.cv.server --port "6666,7777"

⚠️ **必须用 `GolemAI` conda env 跑**（torch / ultralytics / pyzmq 在那里；
GolemQ env 是 NoMKL 跑不了 torch）。

**首选入口是 CLI**：``python -m GolemQ.cli --server ChartScan --port 6666``
—— 它会**自动切到 GolemAI env** 再起本模块（`cli/commands/chartscan.py`）。
直接 ``python -m`` 跑本模块也照旧可用（那时必须**自己先** `conda activate GolemAI`），
`chartscan_svr.ps1` 走的就是这条路。

协议（**与旧树逐字相同**，客户端照旧树写）
=====================================
* 收 `send_json({'ticker','market_type','start_date':'%Y-%m-%d','end_date':'%Y-%m-%d'})`
  （可选 `enforce`：强制重扫）
* 回 `{'status':'success'}` 或 `{'status':'error','message':…}` —— **每个请求都必回**，
  否则 REQ/REP 的锁步就断了，客户端只能靠 45 秒超时脱身。

多端口
======
`--port "6666,7777"` → **一个端口一个线程**，每个线程**各载一份 52MB 模型**
（旧树如此；代价是常驻内存随线程数线性涨）。`--port 6666` 则跑在主线程上
（只有主线程能装 `SIGINT`）。
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
import traceback

from .model import load_model
from .predict import calc_ChartScanAI_reality


def _signal_handler(sig, frame):
    print('\nCtrl+C pressed. Shutting down the server...')
    sys.exit(0)


def run_server(port, is_thread=False):
    """在 `port` 上起一个 REP 服务端并**永不返回**（直到 ZMQ 错误 / Ctrl-C）。"""
    import zmq                              # 只在 GolemAI env 有（见模块 docstring）
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f'tcp://*:{port}')
    if not is_thread:
        signal.signal(signal.SIGINT, _signal_handler)
        print('Server is running. Press Ctrl+C to stop.')
    else:
        print(f'Server thread is running on port {port}.')

    from GolemQ.core.market_registry import get_active_market
    db = get_active_market().get_database()
    model = load_model()                    # 权重缺失会抛 —— 让它抛（静默重试只会更难查）
    while True:
        try:
            message = socket.recv_json()
            print(f'Received request: {message}')
            try:
                # ⚠️ 协议传的是**字符串**（`%Y-%m-%d`），而下游一律按 Timestamp 用
                # ⇒ 必须在这里转（旧树 `cli/__main__.py:95-96` 就是这么做的；
                # 漏了会在 `calc_ChartScaned_persistence` 里 `str.strftime` AttributeError）。
                import pandas as pd
                calc_ChartScanAI_reality(
                    ticker=message['ticker'],
                    market_type=message['market_type'],
                    start_date=pd.to_datetime(message['start_date']),
                    end_date=pd.to_datetime(message['end_date']),
                    enforce=message.get('enforce', None),
                    model=model, db=db,
                )
                socket.send_json({'status': 'success'})
            except Exception as exc:        # noqa: BLE001 照旧：打栈后**照样回** error
                traceback.print_exc()
                print(f'Error processing request: {exc}')
                socket.send_json({'status': 'error', 'message': str(exc)})
        except zmq.ZMQError as exc:
            print(f'ZMQ Error: {exc}')
            break
        except KeyboardInterrupt:
            print('\nCtrl+C pressed. Shutting down the server...')
            break
        except Exception as exc:            # noqa: BLE001 连收包都炸：回 error 保住锁步
            traceback.print_exc()
            print(f'Error recv_json: {exc}')
            socket.send_json({'status': 'error', 'message': str(exc)})
    socket.close()
    context.term()


def main(argv=None):
    parser = argparse.ArgumentParser(description='ChartScan ZeroMQ 服务端（REP）')
    parser.add_argument('--port', type=str, default='5555',
                        help='端口；逗号分隔可一次起多个（每个一线程、各载一份模型）')
    args = parser.parse_args(argv)
    ports = [p.strip() for p in str(args.port).split(',') if p.strip()]
    if len(ports) == 1:
        run_server(ports[0])
        return
    threads = []
    for p in ports:
        if not p.isdigit():
            print(f'[server] 非法端口 {p!r}（只接受数字）', file=sys.stderr)
            sys.exit(2)
    for p in ports:
        threads.append(threading.Thread(target=run_server, args=(p, True), daemon=True))
    for t in threads:
        t.start()
        time.sleep(1)                        # 旧树如此：错开各线程的模型加载
    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
