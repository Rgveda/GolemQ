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
from GolemQ import GQMARKETS


def auto_register_markets():
    """自动注册所有市场模块到 GQMARKETS（只注册尚未注册的市场）"""
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
                                print(f"[warn] 跳过抽象类 {market_name}.{name}")
                                continue
                                
                            try:
                                # 实例化并注册
                                market_instance = obj()
                                GQMARKETS[market_name] = market_instance
                                print(f"[ok] 自动注册市场: {market_name}")
                                break
                            except Exception as e:
                                print(f"[warn] 实例化市场 {market_name} 时出错: {e}")
                                break
                            
                except Exception as e:
                    print(f"[warn] 自动注册市场 {market_name} 时出错: {e}")


def purge_mongodb_database(verbose: bool = False) -> None:
    """清理MongoDB数据库 - 通过各个市场实例的清理方法"""
    try:
        # 自动注册所有市场
        auto_register_markets()
        
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
                    if verbose and collections:
                        print(f"  清理了 {len(collections)} 个集合: {collections}")
                except Exception as e:
                    print(f"[warn] 清理 {market_name} 数据时出错: {e}")
            else:
                print(f"[warn] {market_name} 市场没有实现 purge_historical_collections 方法")
        
        if verbose:
            print("数据库清理完成!")
        else:
            print("MongoDB数据库已通过市场实例清理")
            
    except Exception as e:
        print(f"清理数据库时出错: {e}")
        sys.exit(1)