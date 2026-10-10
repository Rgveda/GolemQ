# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import sys
import importlib
import inspect
from pathlib import Path
from GolemQ.core.market_registry import GQMARKETS, register_market


def auto_register_markets(verbose: bool = False):
    """自动注册所有市场模块到 GQMARKETS（只注册尚未注册的市场）

    :param verbose: ``False``（**默认**）时**一个字都不打**。本函数是
        `get_active_market()` 兜底路径上的一环（`core/market_registry.py`），
        也就是说**任何一次取数**都可能顺手跑到它 —— 无条件打印会让每个
        调用方（含 streamlit 演示）的开屏多出几行 `[ok] 自动注册市场: X`。

        ⚠️ **四条消息（含三条 `[warn]`）是**一起**被 `verbose` 关掉的**，
        不是只关 `[ok]`。取舍在这里：注册失败**不会因此被静默** ——
        `market_registry` 那边对「市场不在注册表」有明确报错
        （`core/market_registry.py` 的 `KeyError: 激活市场 'X' 不在注册表内`），
        症状会带着原因重新浮上来，只是不再多打一行中间态。
        要排查注册过程本身，就 `--purge -v` 或显式传 `verbose=True`。
    """
    markets_dir = Path(__file__).parent.parent / "markets"

    for market_dir in markets_dir.iterdir():
        if market_dir.is_dir() and not market_dir.name.startswith('.'):
            market_name = market_dir.name

            # 如果市场已经注册，跳过
            if market_name in GQMARKETS:
                continue

            init_file = market_dir / "__init__.py"

            if init_file.exists():
                try:
                    # 动态导入市场模块
                    module = importlib.import_module(f"GolemQ.markets.{market_name}")

                    # 查找继承自 BaseMarket 的类
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and
                                hasattr(obj, '__module__') and
                                obj.__module__ == module.__name__ and
                                hasattr(obj, 'purge_historical_collections')):

                            # 检查是否是抽象类
                            if inspect.isabstract(obj):
                                if verbose:
                                    print(f"[warn] 跳过抽象类 {market_name}.{name}")
                                continue

                            try:
                                # 实例化并注册（register_market 默认不覆盖）
                                market_instance = obj()
                                register_market(market_name, market_instance)
                                if verbose:
                                    print(f"[ok] 自动注册市场: {market_name}")
                                break
                            except Exception as e:
                                if verbose:
                                    print(f"[warn] 实例化市场 {market_name} 时出错: {e}")
                                break

                except Exception as e:
                    if verbose:
                        print(f"[warn] 自动注册市场 {market_name} 时出错: {e}")


def purge_mongodb_database(verbose: bool = False) -> None:
    """清理MongoDB数据库 - 通过各个市场实例的清理方法"""
    try:
        # 自动注册所有市场（`-v` 时把注册过程也打出来，否则静默）
        auto_register_markets(verbose=verbose)
        
        if verbose:
            print("开始通过市场实例清理MongoDB数据库...")
            print(f"找到 {len(GQMARKETS)} 个注册的市场: {list(GQMARKETS.keys())}")
        
        # 调用每个市场实例的清理方法
        for market_name, market_instance in GQMARKETS.items():
            if verbose:
                print(f"清理 {market_name} 市场数据...")
            
            if hasattr(market_instance, 'purge_historical_collections'):
                try:
                    collections = market_instance.purge_historical_collections()
                except Exception as e:
                    print(f"[warn] 清理 {market_name} 数据时出错: {e}")
                else:
                    # ⚠️ 这里是**唯一**的打印处。市场那边（`markets/StockCN/tools.py`）
                    # 只返回删了哪些、不打印 —— 否则同一条命令有两处打印分属两个模块，
                    # 测试只能 patch 到一个，另一个照样漏到屏上。
                    if verbose and collections:
                        print(f"  清理了 {len(collections)} 个集合: {collections}")
            else:
                print(f"[warn] {market_name} 市场没有实现 purge_historical_collections 方法")
        
        if verbose:
            print("数据库清理完成!")
        else:
            print("MongoDB数据库已通过市场实例清理")
            
    except Exception as e:
        print(f"清理数据库时出错: {e}")
        sys.exit(1)