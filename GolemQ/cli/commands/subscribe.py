# coding:utf-8
"""`--sub <KEY>`：跑一个第三方行情订阅器。

订阅器从 `GQSUBSCRIBER` 按 key 取，**必须能零参调用**（`PITFALLS.md` P10）。
"""
from __future__ import annotations

import sys

from GolemQ import GQSUBSCRIBER

from ._registry import Command, EXIT_FAILURE, usage_error


def add_subscribe_arguments(parser) -> None:
    parser.add_argument('--sub',
                        help="执行指定的第三方行情订阅功能",
                        type=str,
                        metavar="SUBSCRIBER_KEY",
                        default=None)


def run_subscribe(args) -> None:
    subscriber_key = args.sub
    if subscriber_key in GQSUBSCRIBER:
        try:
            print("执行第三方行情订阅器: {}".format(subscriber_key))
            subscriber_func = GQSUBSCRIBER[subscriber_key]
            subscriber_func()
            print("第三方行情订阅器 {} 执行完成".format(subscriber_key))
        except Exception as e:      # noqa: BLE001 订阅器内部异常统一转人话
            # ⚠️ **运行期失败**（订阅器真跑了但炸了）→ 保持 1，**不是**用法错
            print("执行第三方行情订阅器 {} 时发生错误: {}".format(subscriber_key, e))
            sys.exit(EXIT_FAILURE)
    else:
        # 键不在**运行期注册表**里 = 取值非法 → 用法错，与 `--save-collections` 同类
        usage_error("argument --sub: 订阅器 '{}' 不存在".format(subscriber_key),
                    hint='可用的第三方行情订阅器: {}'.format(
                        ', '.join(sorted(GQSUBSCRIBER.keys()))))


# ⚠️ `needs_db=False`：订阅器**自己要什么自己取**，别在入口处先卡一道 ——
# 否则一个纯行情源的订阅器会因为「运维库连不上」而整个跑不起来。
SUBSCRIBE = Command('subscribe', ('sub',), add_subscribe_arguments,
                    run_subscribe, needs_db=False)
