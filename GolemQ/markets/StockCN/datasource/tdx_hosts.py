# coding:utf-8
"""通达信行情服务器池：候选收集 → **真协议探活** → 排序 → 缓存。

为什么需要它
============
`DEFAULT_HOSTS` 是**手写的 4 台**。但这类服务器**会时好时坏**，而且手写表
既漏（pytdx 包里自带 64 台）又旧（反映不了当下状态）。2026-10-09 实测同一台
``123.125.108.14``：一次探针**稳定超时 10.055s**（连测 5 次都超），几十分钟后
再测 **6/6 成功、中位 0.233s** —— 它是**间歇性**的，不是死的。

所以「探一次」分辨不出好机器。本模块的做法是**每台多测几次、取中位**。

机制（默认每周一次）
====================
* **候选**：pytdx 包内自带的 `pytdx.util.best_ip.stock_ip`（**64 台**，由上游维护）
  ∪ 我们的 `DEFAULT_HOSTS`。**不自己去爬网页** —— 包里那张表就是别人维护好的，
  再造一个爬虫是第二个轮子，还得跟着网页改版维护。
* **探活**：**真协议调用**，既不是 ICMP ping 也不是 TCP 连通。
  ⚠️ **这一条是承重的**：本项目 `new_api()` 的探针是 ``get_security_count(0)``，
  而 pytdx 的 ``best_ip.ping()`` 用的是 ``get_security_list(0, 0)`` —— **两者不等价**。
  实测那台间歇服**在 `get_security_count` 上超时、在 `get_security_list` 上正常**。
  所以**不能拿 `select_best_ip()` 顶替**：它按另一个调用排序，会把这台判成好的。
  「探针必须跑**生产实际用的那个调用**」是这里的核心纪律。
* **重测**：每台 ``ATTEMPTS`` 次，按**成功次数的中位延迟**排序，失败率超阈值剔除。
* **缓存**：落 ``~/.GolemQ/settings/tdx_hosts.json``。**用文件不用 Mongo**：
  本模块在 `datasource/`（数据源适配层），而 `CLAUDE.md` 明令 DB 操作只能进
  `services/`。顺带一个好处：**文件的 mtime 就是「上次刷新时刻」**，
  每周一次的门就不用再引一套签到表。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor

#: 缓存文件。放在 `~/.GolemQ/settings/`（与 `config.ini` 同处）。
CACHE_NAME = 'tdx_hosts.json'

#: 刷新周期。**缓存文件的 mtime 就是起算点**。
REFRESH_DAYS = 7

#: 每台测几次。**1 次分辨不出「间歇」** —— 实测同一台能在 0.2s 与 >10s 之间跳。
ATTEMPTS = 3

#: 单次探针的超时（秒）。
PROBE_TIMEOUT = 5

#: 并发探活线程数。池子 60+ 台，串行要几分钟。
PROBE_WORKERS = 16

#: 失败率上限：超过就不收进结果。3 次里允许坏 1 次。
MAX_FAIL_RATE = 0.34


def cache_path() -> str:
    """缓存文件全路径（`~/.GolemQ/settings/tdx_hosts.json`）。"""
    from GolemQ.core.path import setting_path
    return os.path.join(setting_path, CACHE_NAME)


def _probe_once(ip, port, timeout=PROBE_TIMEOUT):
    """一次**真协议**探针。返回墙钟秒数；失败返回 ``None``。

    ⚠️ 用的必须是 ``get_security_count(0)`` —— 与 `TdxSource.new_api()` 一致。
    换成 `get_security_list` 会放过「只在 count 上卡住」的那种间歇服。
    """
    import time

    from pytdx.hq import TdxHq_API
    api = TdxHq_API(heartbeat=False)
    started = time.time()
    try:
        api.connect(ip, port, time_out=timeout)
        count = api.get_security_count(0)
        # ⚠️ **必须判返回值，不能只判"没抛异常"。**
        # pytdx 连不上时**不抛异常**：`connect()` 内部重试约 7s 后返回，
        # 而 `get_security_count()` 返回 **`None`**（`PITFALLS.md` P3b 的同一种形态 ——
        # 这个库用「返回空」代替「抛异常」）。只判异常会把**死服判成好服**，
        # 且它带着 ~7s 的"延迟"混进结果里。实测：`127.0.0.1:1` 就是这么被判成
        # 7.03s「成功」的 —— 是单测抓出来的。
        if not count:
            return None
        return time.time() - started
    except Exception:      # noqa: BLE001 探活失败就是失败，不区分原因
        return None
    finally:
        try:
            api.disconnect()
        except Exception:  # noqa: BLE001
            pass


def is_loopback(ip) -> bool:
    """`127.0.0.0/8` / `::1` / `0.0.0.0` —— **行情服务器不可能是本机**。

    >>> is_loopback('127.0.0.1'), is_loopback('0.0.0.0'), is_loopback('::1')
    (True, True, True)
    >>> is_loopback('115.238.90.165')
    False
    """
    import ipaddress
    try:
        addr = ipaddress.ip_address(str(ip))
    except ValueError:      # 不是 IP 字面量（域名）→ 不是回环
        return False
    return addr.is_loopback or addr.is_unspecified


def probe(ip, port, attempts=ATTEMPTS):
    """对一台测 ``attempts`` 次，返回 ``{'ip','port','ok','runs','median','fail_rate'}``。

    ⚠️ **回环/未指定地址直接判不可用**，连探都不探（见 :func:`is_loopback`）——
    不能靠"反正连不上"碰巧判对：本机若真有东西监听 7709，探针会把它当成 TDX 服务器。
    """
    if is_loopback(ip):
        return {'ip': ip, 'port': port, 'ok': False, 'runs': 0,
                'median': None, 'fail_rate': 1.0}

    runs = [_probe_once(ip, port) for _ in range(attempts)]
    good = [r for r in runs if r is not None]
    fail_rate = 1.0 - (len(good) / attempts if attempts else 1.0)
    return {
        'ip': ip, 'port': port,
        'ok': bool(good) and fail_rate <= MAX_FAIL_RATE,
        'runs': len(good),
        'median': statistics.median(good) if good else None,
        'fail_rate': fail_rate,
    }


def candidates(include_builtin_pool=True):
    """候选服务器：pytdx 包内池（64 台）∪ 我们的 `DEFAULT_HOSTS`，去重。

    ⚠️ `DEFAULT_HOSTS` **函数级导入**：`pytdx_source` 会 import 本模块拿缓存，
    模块级互导会成环。
    """
    # 本模块被 `pytdx_source` import，故这里**只能是函数级导入**（避免循环）
    from .pytdx_source import DEFAULT_HOSTS       # noqa: PLC0415
    out, seen = [], set()
    pool = []
    if include_builtin_pool:
        try:
            from pytdx.util.best_ip import stock_ip
            pool = [(h['ip'], h['port']) for h in stock_ip]
        except Exception:      # noqa: BLE001 包内池拿不到就只用自带的
            pool = []
    for ip, port in list(pool) + list(DEFAULT_HOSTS):
        if (ip, port) in seen:
            continue
        seen.add((ip, port))
        out.append((ip, port))
    return out


def probe_pool(pool=None, attempts=ATTEMPTS, workers=PROBE_WORKERS):
    """并行探活整个池子，返回**按中位延迟升序**的可用列表。

    :returns: ``[{'ip','port','median','runs','fail_rate'}, ...]``（只含可用者）
    """
    pool = list(pool) if pool is not None else candidates()
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        rows = list(ex.map(lambda hp: probe(hp[0], hp[1], attempts), pool))
    alive = [r for r in rows if r['ok']]
    alive.sort(key=lambda r: r['median'])
    return alive


def load(now=None):
    """读缓存；**文件不存在 / 过期 / 坏了**都返回 ``None``（调用方退回 `DEFAULT_HOSTS`）。

    :returns: ``[(ip, port), ...]``（按缓存的顺序，即上次探活的中位延迟升序）
    """
    path = cache_path()
    try:
        if not os.path.exists(path):
            return None
        age_days = ((now or dt.datetime.now()).timestamp()
                    - os.path.getmtime(path)) / 86400.0
        if age_days > REFRESH_DAYS:
            return None
        with open(path, encoding='utf-8') as fh:
            doc = json.load(fh)
        hosts = [(h['ip'], h['port']) for h in doc.get('hosts', [])]
        return hosts or None
    except Exception:      # noqa: BLE001 缓存坏了就当没有 —— 绝不能因此跑不起来
        return None


def save(alive, now=None):
    """写缓存。``alive`` 是 :func:`probe_pool` 的返回。"""
    path = cache_path()
    doc = {
        'probed_at': (now or dt.datetime.now()).isoformat(timespec='seconds'),
        'hosts': [{'ip': r['ip'], 'port': r['port'],
                   'median_ms': round(r['median'] * 1000, 1),
                   'runs': r['runs'], 'fail_rate': round(r['fail_rate'], 2)}
                  for r in alive],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    return path


def refresh(force=False, verbose=False, pool=None):
    """**每周一次**的刷新：不到期就直接用缓存；到期才真探。

    :returns: ``(hosts, did_probe)``；``hosts`` 是 ``[(ip, port), ...]``。
    """
    if not force:
        cached = load()
        if cached:
            if verbose:
                print('[tdx_hosts] 缓存未过期（< {} 天），跳过探活：{} 台'.format(
                    REFRESH_DAYS, len(cached)))
            return cached, False

    alive = probe_pool(pool=pool)
    if not alive:
        # ⚠️ **探不到就保留旧缓存**，别把可用列表清空 —— 网络抖动时清空等于自毁
        if verbose:
            print('[tdx_hosts] ⚠️ 本轮一台都没探到，**保留现有缓存/默认表**')
        return (load() or []), True
    save(alive)
    if verbose:
        for r in alive[:8]:
            print('[tdx_hosts] {:>6.1f}ms  {}:{}  ({}/{} 次成功)'.format(
                r['median'] * 1000, r['ip'], r['port'], r['runs'], ATTEMPTS))
        if len(alive) > 8:
            print('[tdx_hosts] … 共 {} 台可用'.format(len(alive)))
    return [(r['ip'], r['port']) for r in alive], True


__all__ = ['ATTEMPTS', 'CACHE_NAME', 'MAX_FAIL_RATE', 'PROBE_TIMEOUT',
           'PROBE_WORKERS', 'REFRESH_DAYS', 'cache_path', 'candidates', 'load',
           'probe', 'probe_pool', 'refresh', 'save']
