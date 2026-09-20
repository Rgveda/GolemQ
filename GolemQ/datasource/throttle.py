# coding:utf-8
"""按源的最小请求间隔节流器。

为什么不复用 `supervisor/function_checkin.py`
==========================================
`checkin_function(name, expired_time)` 是 **Mongo 级、按 (函数名, 调用方IP)** 的
「到期前拒绝」闸门 —— 语义是「这个功能今天跑过没有」，最小粒度在 15 分钟量级，
且它 **拒绝（denied）** 而不是 **等待（wait）**，调用方要自己写降级分支。

而数据源适配器需要的是相反的语义：**在两次请求之间补足剩余间隔**，让一个遍历
5000+ 代码的循环不要打爆上游。那是节流（throttle），不是访问闸门（gate）。

两者正交，各自保留：`function_checkin` 继续管「别重复跑整个昂贵函数」，
本模块管「每次请求之间歇够」。

默认 30 秒，逐源可覆盖
======================
baostock / akshare / eastmoney 均有访问频率控制。默认间隔 30s，可经
`~/.GolemQ/settings/config.ini` 的 `[DATASOURCE]` 段逐源覆盖 —— 见
`interval_of()`。配置缺失时用 `default_interval`，不报错。
"""
from __future__ import annotations

import threading
import time


class SourceThrottle:
    """进程内、按源名维护 last-request 时间。

    进程内状态意味着**多进程并行取数时各自节流**，合计频率会被放大。当前取数
    以单进程为主，故不引入跨进程协调；若日后改为 joblib 多进程，需要在取数入口
    改为按进程分片，或把状态移到 Mongo。
    """

    def __init__(self, default_interval: float = 30.0, per_source: dict = None):
        self.default_interval = float(default_interval)
        self.overrides = {k: float(v) for k, v in (per_source or {}).items()}
        self._last: dict = {}
        self._lock = threading.Lock()

    def interval(self, name: str) -> float:
        """某源的最小间隔：按源覆盖优先，否则用默认（30s）。

        >>> t = SourceThrottle(default_interval=30.0, per_source={'pytdx': 0})
        >>> t.interval('pytdx')          # 显式覆盖
        0.0

        >>> t.interval('akshare')        # 未覆盖 → 用默认
        30.0

        >>> SourceThrottle().interval('任何未配置的源')
        30.0
        """
        return self.overrides.get(name, self.default_interval)

    def wait(self, name: str) -> float:
        """睡到距上次请求满 `interval(name)` 秒。返回实际睡了多少秒。"""
        target = self.interval(name)
        with self._lock:
            elapsed = time.monotonic() - self._last.get(name, float('-inf'))
            remaining = target - elapsed
        slept = 0.0
        if remaining > 0:
            time.sleep(remaining)
            slept = remaining
        with self._lock:
            self._last[name] = time.monotonic()
        return slept

    def reset(self, name: str = None):
        """清掉节流状态（测试与手动触发用）。"""
        with self._lock:
            if name is None:
                self._last.clear()
            else:
                self._last.pop(name, None)


def from_settings(section: str = 'DATASOURCE') -> SourceThrottle:
    """从配置构造节流器。

    读 `[DATASOURCE]` 段的:
      * ``default_interval`` —— 默认 30
      * ``<源名>_interval``  —— 逐源覆盖，如 ``pytdx_interval = 5``

    配置读取失败一律回落到默认值 —— 取数不该因为配置缺失而起不来。
    """
    default = 30.0
    overrides: dict = {}
    try:
        from GolemQ.core.settings import GQSETTING
        raw = GQSETTING.get_config(section, 'default_interval', None)
        if raw not in (None, ''):
            default = float(raw)
        # 逐源覆盖：凡以 _interval 结尾且不是 default_interval 的键
        try:
            section_map = GQSETTING.get_config(section, None, None)
        except Exception:
            section_map = None
        if isinstance(section_map, dict):
            for key, val in section_map.items():
                if key.endswith('_interval') and key != 'default_interval':
                    try:
                        overrides[key[: -len('_interval')]] = float(val)
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass
    return SourceThrottle(default_interval=default, per_source=overrides)
