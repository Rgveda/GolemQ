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

import datetime
import re
from typing import List
from GolemQ.core.settings import GOLEMQ as DATABASE_GolemQ
from GolemQ.core.constants import DATASOURCE


def parse_symbols(symbols_str: str) -> List[str]:
    """
    解析股票代码字符串，支持逗号或换行分割
    支持多种格式：
    - 6位数字代码：000001
    - 带市场后缀：000001.SZ, 600000.SH
    - ETF代码：159919.SZ
    - 指数代码：000300.SH
    """
    if not symbols_str:
        return []
    
    # 替换换行符为逗号，然后按逗号分割
    symbols_str = symbols_str.replace('\n', ',').replace('\r', '')
    symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
    
    valid_symbols = []
    for symbol in symbols:
        # 支持多种格式验证
        if re.match(r'^\d{6}$', symbol):
            # 6位数字代码：000001
            valid_symbols.append(symbol)
        elif re.match(r'^\d{6}\.(SH|SZ)$', symbol):
            # 带市场后缀的代码：000001.SZ, 600000.SH
            valid_symbols.append(symbol)
        elif re.match(r'^(SH|SZ)\d{6}$', symbol):
            # 前缀格式：SH600000, SZ000001
            # 转换为后缀格式
            market = symbol[:2]
            code = symbol[2:]
            valid_symbols.append(f"{code}.{market}")
        else:
            print(f"警告: 股票代码格式不正确，已忽略: {symbol}")
            print("  支持格式: 000001 | 000001.SZ | 600000.SH | SH600000 | SZ000001")
    
    return valid_symbols


def add_symbols_to_watchlist(symbols: List[str], verbose: bool = False) -> int:
    """
    添加股票代码到关注列表
    
    Args:
        symbols: 股票代码列表
        verbose: 是否显示详细信息
        
    Returns:
        成功添加的数量
    """
    if not symbols:
        print("没有有效的股票代码需要添加")
        return 0
    
    collection = DATABASE_GolemQ.StockCN_watchdog_eneloop
    success_count = 0
    current_time = datetime.datetime.now()
    current_timestamp_int32 = int(current_time.timestamp())
    
    for symbol in symbols:
        try:
            # 检查是否已存在
            existing = collection.find_one({'symbol': symbol})
            
            if existing:
                if verbose:
                    print(f"股票代码 {symbol} 已在关注列表中")
                continue
            
            # 创建新记录
            record = {
                'symbol': symbol,
                'added_at': current_timestamp_int32,
                'status': 'active',
                'source': 'cli_manual',
                'updated_at': current_timestamp_int32
            }
            
            result = collection.insert_one(record)
            if result.inserted_id:
                success_count += 1
                if verbose:
                    print(f"✓ 成功添加股票代码: {symbol}")
            
        except Exception as e:
            print(f"✗ 添加股票代码 {symbol} 时出错: {e}")
    
    # 创建索引
    try:
        collection.create_index('symbol')
        collection.create_index('status')
        collection.create_index('added_at')
    except Exception as e:
        if verbose:
            print(f"创建索引时出错: {e}")
    
    print(f"成功添加 {success_count}/{len(symbols)} 个股票代码到关注列表")
    return success_count


def remove_symbols_from_watchlist(symbols: List[str], verbose: bool = False) -> int:
    """
    从关注列表删除股票代码并移动到归档库
    
    Args:
        symbols: 股票代码列表
        verbose: 是否显示详细信息
        
    Returns:
        成功删除的数量
    """
    if not symbols:
        print("没有有效的股票代码需要删除")
        return 0
    
    collection = DATABASE_GolemQ.StockCN_watchdog_eneloop
    archive_collection = DATABASE_GolemQ.StockCN_watchdog_eneloop_archive
    success_count = 0
    current_time = datetime.datetime.now()
    current_timestamp_int32 = int(current_time.timestamp())
    
    for symbol in symbols:
        try:
            # 查找要删除的记录
            record = collection.find_one({'symbol': symbol})
            
            if not record:
                if verbose:
                    print(f"股票代码 {symbol} 不在关注列表中")
                continue
            
            # 准备归档记录
            archive_record = record.copy()
            archive_record['removed_at'] = current_timestamp_int32
            archive_record['status'] = 'archived'
            archive_record['updated_at'] = current_timestamp_int32
            
            # 插入到归档库
            archive_result = archive_collection.insert_one(archive_record)
            
            if archive_result.inserted_id:
                # 从原库删除
                delete_result = collection.delete_one({'symbol': symbol})
                if delete_result.deleted_count > 0:
                    success_count += 1
                    if verbose:
                        print(f"✓ 成功删除并归档股票代码: {symbol}")
                else:
                    # 如果删除失败，也要从归档库删除刚插入的记录
                    archive_collection.delete_one({'_id': archive_result.inserted_id})
                    if verbose:
                        print(f"✗ 删除股票代码 {symbol} 失败")
            
        except Exception as e:
            print(f"✗ 删除股票代码 {symbol} 时出错: {e}")
    
    # 创建归档库索引
    try:
        archive_collection.create_index('symbol')
        archive_collection.create_index('status')
        archive_collection.create_index('removed_at')
    except Exception as e:
        if verbose:
            print(f"创建归档库索引时出错: {e}")
    
    print(f"成功删除并归档 {success_count}/{len(symbols)} 个股票代码")
    return success_count


def list_watchlist_symbols(verbose: bool = False) -> List[str]:
    """
    列出当前关注列表中的所有股票代码
    只查询最近一年的数据，对重复的股票代码只保留更新时间最新的记录
    
    Args:
        verbose: 是否显示详细信息
        
    Returns:
        股票代码列表
    """
    collection = DATABASE_GolemQ.StockCN_watchdog_eneloop
    archive_collection = DATABASE_GolemQ.StockCN_watchdog_eneloop_archive
    
    try:
        # 计算一年前的时间戳
        one_year_ago = datetime.datetime.now() - datetime.timedelta(days=365)
        one_year_ago_timestamp = int(one_year_ago.timestamp())
        
        # 查询最近一年的记录，不限制status字段
        # 优先显示手动添加的active状态，然后显示xtquant同步的数据
        query = {
            '$and': [
                {
                    '$or': [
                        {'status': 'active'},  # 手动添加的记录
                        {'status': {'$exists': False}},  # xtquant同步的记录（没有status字段）
                        {'source': DATASOURCE.QMT}  # 明确来自xtquant的记录
                    ]
                },
                {
                    '$or': [
                        {'updated_at': {'$gte': one_year_ago_timestamp}},
                        {'added_at': {'$gte': one_year_ago_timestamp}},
                        {'updated_at': {'$exists': False}},
                        {'added_at': {'$exists': False}}
                    ]
                }
            ]
        }
        
        cursor = collection.find(query, {
            'symbol': 1,
            'added_at': 1,
            'updated_at': 1,
            'source': 1,
            'status': 1,
            'volume': 1,
            'market_value': 1
        }).sort('symbol', 1)
        
        # 使用字典来存储每个symbol的最新记录
        symbol_records = {}
        
        for record in cursor:
            symbol = record['symbol']
            updated_at = record.get('updated_at', 0) or record.get('added_at', 0)
            
            # 如果symbol已存在，比较更新时间，只保留最新的
            if symbol in symbol_records:
                existing_updated_at = symbol_records[symbol].get('updated_at', 0) or symbol_records[symbol].get('added_at', 0)
                if updated_at > existing_updated_at:
                    symbol_records[symbol] = record
            else:
                symbol_records[symbol] = record
        
        # 查询归档库中的历史持仓
        archive_query = {
            '$or': [
                {'source': DATASOURCE.QMT},
                {'source': 'xtquant'}
            ],
            'removed_at': {'$gte': one_year_ago_timestamp}
        }
        
        archive_cursor = archive_collection.find(archive_query, {
            'symbol': 1,
            'removed_at': 1,
            'volume': 1,
            'market_value': 1
        }).sort('symbol', 1)
        
        # 统计历史持仓
        historical_positions = {}
        for record in archive_cursor:
            symbol = record['symbol']
            if symbol not in historical_positions:
                historical_positions[symbol] = record
        
        symbols = []
        manual_count = 0
        xtquant_current_count = 0
        xtquant_historical_count = len(historical_positions)
        
        print("当前关注列表 (StockCN_watchdog_eneloop):")
        # 按symbol排序输出当前持仓
        for symbol in sorted(symbol_records.keys()):
            record = symbol_records[symbol]
            symbols.append(symbol)
            source = record.get('source', '未知')
            
            if source == 'cli_manual':
                manual_count += 1
                status_label = "[手动添加]"
            elif source == 'xtquant' or source == DATASOURCE.QMT:
                xtquant_current_count += 1
                status_label = "[XTQuant持仓]"
            else:
                status_label = f"[{source}]"
            
            if verbose:
                # 显示详细信息
                added_at = record.get('added_at', 0)
                updated_at = record.get('updated_at', 0)
                volume = record.get('volume', 0)
                market_value = record.get('market_value', 0)
                
                info_parts = [f"{symbol} {status_label}"]
                
                if added_at:
                    added_time = datetime.datetime.fromtimestamp(added_at)
                    info_parts.append(f"添加: {added_time.strftime('%Y-%m-%d %H:%M:%S')}")
                elif updated_at:
                    updated_time = datetime.datetime.fromtimestamp(updated_at)
                    info_parts.append(f"更新: {updated_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if volume > 0:
                    info_parts.append(f"持仓: {volume}股")
                if market_value > 0:
                    info_parts.append(f"市值: {market_value:.2f}元")
                
                print(f"  {' | '.join(info_parts)}")
            else:
                print(f"  {symbol} {status_label}")
        
        # 输出历史持仓（一年内交易过的）
        if historical_positions and verbose:
            print("\n历史持仓 (一年内交易过):")
            for symbol in sorted(historical_positions.keys()):
                record = historical_positions[symbol]
                removed_at = record.get('removed_at', 0)
                volume = record.get('volume', 0)
                market_value = record.get('market_value', 0)
                
                info_parts = [f"{symbol} [历史持仓]"]
                
                if removed_at:
                    removed_time = datetime.datetime.fromtimestamp(removed_at)
                    info_parts.append(f"移除: {removed_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if volume > 0:
                    info_parts.append(f"原持仓: {volume}股")
                if market_value > 0:
                    info_parts.append(f"原市值: {market_value:.2f}元")
                
                print(f"  {' | '.join(info_parts)}")
        
        if not symbols and not historical_positions:
            print("  (空)")
        else:
            total_count = len(symbols) + len(historical_positions)
            print(f"\n总计: {total_count} 个股票")
            if manual_count > 0:
                print(f"  手动添加: {manual_count} 个")
            if xtquant_current_count > 0 or xtquant_historical_count > 0:
                print(f"  XTQuant持仓: {xtquant_current_count + xtquant_historical_count} 个")
                print(f"    当前持仓: {xtquant_current_count} 个")
                print(f"    历史持仓: {xtquant_historical_count} 个")
        
        return symbols
        
    except Exception as e:
        print(f"查询关注列表时出错: {e}")
        return []

def migrate_eneloop_watchlist(verbose: bool = True, source_uri: str = None) -> dict:
    """把 4.4 的**关注列表**搬到 8.3 的 `GOLEMQ`（**一次性**，见 `core/migrate44.py`）。

    为什么必须搬：`--eneloop-*` 是活 CLI，而改绑之后它的两张表在 8.3 是**空的** ——
    只改代码不搬数据，`--eneloop-list` 会**静默变空**（数据没丢，还在 4.4，但新树看不见）。

    两张表的幂等键**不同**（别图省事用同一个）：
    * 活动表 `StockCN_watchdog_eneloop` → ``['symbol']``（写入侧就是 ``find_one({'symbol':…})``）
    * 归档表 `StockCN_watchdog_eneloop_archive` → ``['symbol', 'removed_at']``
      —— 归档是 append-only，同一个 symbol 可以有多行（每删一次加一行）

    ⚠️ **归档表不能用 `writer.save_collection`**：它对缺 key 的行是
    ``if all(k in r …)`` **静默跳过**（`datasource/writer.py:71-77`），而归档行
    可能是早期版本写的、未必都有 `removed_at` —— 那会**静默丢行**。
    所以这里手写 `ReplaceOne`，并把跳过数打进报告。

    :returns: ``{'src':…, 'dst':…, 'skipped':…}``（两张表各一份计数）
    """
    from pymongo import ReplaceOne, DESCENDING  # noqa: F401
    from GolemQ.core.migrate44 import db44

    src_db = db44('golemq', uri=source_uri)
    out = {}
    for name, keys in (('StockCN_watchdog_eneloop', ['symbol']),
                       ('StockCN_watchdog_eneloop_archive', ['symbol', 'removed_at'])):
        src = src_db[name]
        dst = DATABASE_GolemQ[name]
        if name not in src_db.list_collection_names():
            raise RuntimeError(
                '源集合不存在：4.4 golemq.{0} —— 这条搬运就是为了避免 '
                '`--eneloop-list` 静默变空，所以宁可直接失败'.format(name))
        rows = list(src.find({}, {'_id': 0}))
        stats = {'src': len(rows), 'dst': 0, 'skipped': 0}
        ops, skipped = [], 0
        for r in rows:
            if not all(k in r for k in keys):
                skipped += 1
                continue
            ops.append(ReplaceOne({k: r[k] for k in keys}, r, upsert=True))
        if ops:
            dst.bulk_write(ops, ordered=False)
        stats['dst'] = dst.count_documents({})
        stats['skipped'] = skipped
        out[name] = stats
        if verbose:
            print('[migrate:eneloop] {0}: 源 {1} 行 → 目标 {2} 行（跳过 {3}）'.format(
                name, stats['src'], stats['dst'], stats['skipped']))
    return out
