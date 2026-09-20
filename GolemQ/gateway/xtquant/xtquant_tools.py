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

import pandas as pd
import datetime
from typing import List, Dict, Any, Set
import traceback
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
from .trader import xtQmtTrader
from GolemQ.markets.StockCN.date_utils import GQ_util_if_trade
from GolemQ.gateway.xtquant.config import get_xtquant_config
from GolemQ.core.settings import DATABASE as DATABASE_GolemQ
from GolemQ.core.constants import DATASOURCE
from GolemQ.markets.StockCN.date_utils import (
    get_15min_aligned_timestamp,
)
from GolemQ.supervisor.heartbeat import HeartbeatModule
from GolemQ.supervisor.messenger import send_alert
import time

# 连续获取持仓数据失败的次数
_EXPORT_POSITIONS_FAILURE_COUNT = 0


def get_xtquant_account_asset(xt_trader: XtQuantTrader, acc: StockAccount) -> Dict[str, float]:
    """
    获取XTQuant账户资金信息
    """
    try:
        asset = xt_trader.query_stock_asset(acc)
        return {
            'cash': getattr(asset, 'cash', 0),
            'cash_in_transit': getattr(asset, 'cash_in_transit', 0),
            'total_assets': getattr(asset, 'total_asset', 0)
        }
    except Exception as e:
        print(f"获取账户资金信息时出错: {e}")
        return {'cash': 0, 'cash_in_transit': 0, 'total_assets': 0}


def get_xtquant_positions(xt_trader: XtQuantTrader, acc: StockAccount) -> List[Dict[str, Any]]:
    """
    获取XTQuant账户持仓数据
    """
    try:
        positions = xt_trader.query_stock_positions(acc)
        positions_data = []
        
        for pos in positions:
            position_info = {
                'stock_code': pos.stock_code,
                'stock_name': getattr(pos, 'stock_name', ''),
                'volume': pos.volume,
                'avail_vol': pos.can_use_volume,
                'open_price': pos.open_price,
                'market_value': pos.market_value,
                'current_price': getattr(pos, 'current_price', 0),
                'cost_price': getattr(pos, 'cost_price', pos.open_price),
                'profit': getattr(pos, 'profit', 0),
                'profit_ratio': getattr(pos, 'profit_ratio', 0)
            }
            positions_data.append(position_info)
            
        return positions_data
        
    except Exception as e:
        print(f"获取持仓数据时出错: {e}")
        return []


def positions_to_dataframe(positions_data: List[Dict[str, Any]], account_id: str = None) -> pd.DataFrame:
    """
    将持仓数据转换为DataFrame
    """
    df = pd.DataFrame(positions_data)
    
    # 计算挂单数量（总持仓量 - 可用数量）
    df['pending_volume'] = df['volume'] - df['avail_vol']
    
    # 添加额外字段
    current_time = datetime.datetime.now()
    time_stamp = get_15min_aligned_timestamp(current_time.time())
    datetime_str = pd.to_datetime(time_stamp).strftime("%Y-%m-%d %H:%M:%S")
    
    # 获取当前时间的int32格式时间戳
    current_timestamp_int32 = int(current_time.timestamp())
    
    df['source'] = DATASOURCE.QMT
    df['account_id'] = account_id if account_id else ''
    # 使用int32格式的时间戳（Unix时间戳）
    df['updated_at'] = current_timestamp_int32
    df['ckop_at'] = current_timestamp_int32
    df['time_stamp'] = time_stamp
    df['datetime'] = datetime_str
    
    # 重命名列以符合数据库要求
    df = df.rename(columns={
        'stock_code': 'symbol',
        'cost_price': 'cost_price',
        'open_price': 'open_price'
    })
    
    return df


def calculate_position_stats(positions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算持仓统计信息
    """
    total_volume = sum(pos['volume'] for pos in positions_data)
    total_market_value = sum(pos['market_value'] for pos in positions_data)
    
    return {
        'position_count': len(positions_data),
        'total_volume': total_volume,
        'total_market_value': total_market_value
    }


def save_positions_to_mongodb(
        df: pd.DataFrame,
        collection=DATABASE_GolemQ.StockCN_watchdog_eneloop,
        archive_collection=DATABASE_GolemQ.StockCN_watchdog_eneloop_archive
):
    """
    将持仓数据保存到MongoDB，并将不在当前持仓中的XTQuant记录移动到归档库
    """
    try:
        # 转换为字典列表
        records = df.to_dict('records')
        
        # 获取当前时间戳
        current_time = datetime.datetime.now()
        current_timestamp_int32 = int(current_time.timestamp())
        
        # 获取当前XTQuant持仓的所有symbol
        current_xtquant_symbols = {record['symbol'] for record in records}
        
        # 查找所有XTQuant来源的记录
        xtquant_records = collection.find({
            '$or': [
                {'source': DATASOURCE.QMT},
                {'source': 'xtquant'}
            ]
        })
        
        # 处理不在当前持仓中的XTQuant记录
        moved_to_archive_count = 0
        for old_record in xtquant_records:
            symbol = old_record['symbol']
            if symbol not in current_xtquant_symbols:
                # 准备归档记录
                archive_record = old_record.copy()
                archive_record['removed_at'] = current_timestamp_int32
                archive_record['status'] = 'archived'
                archive_record['updated_at'] = current_timestamp_int32
                
                # 插入到归档库
                archive_collection.insert_one(archive_record)
                
                # 从原库删除
                collection.delete_one({'_id': old_record['_id']})
                moved_to_archive_count += 1
                print(f"✓ 移动不在持仓中的股票到归档库: {symbol}")
        
        # 使用update_one进行更新或插入当前持仓
        success_count = 0
        if records:
            for record in records:
                # 使用symbol作为唯一标识进行更新或插入
                result = collection.update_one(
                    {'symbol': record['symbol']},
                    {'$set': record},
                    upsert=True
                )
                if result.modified_count > 0 or result.upserted_id is not None:
                    success_count += 1
            
            print(f"成功处理 {success_count}/{len(records)} 条持仓记录")
        else:
            print("没有持仓数据需要保存")
            
        if moved_to_archive_count > 0:
            print(f"成功移动 {moved_to_archive_count} 个不在持仓中的股票到归档库")
            
        # 创建联合索引 (symbol + time_stamp)
        collection.create_index([('symbol', 1), ('time_stamp', 1)])
        collection.create_index('time_stamp')
        
        return success_count > 0
        
    except Exception as e:
        print(f"保存数据到MongoDB时出错: {e}")
        return False


def save_sync_summary_to_mongodb(
        positions_data: List[Dict[str, Any]],
        asset_info: Dict[str, float],
        collection=DATABASE_GolemQ.StockCN_xtquant_synchronized
):
    """
    保存同步汇总信息到MongoDB
    """
    try:
        # 计算持仓统计
        stats = calculate_position_stats(positions_data)
        
        # 获取当前时间
        current_time = datetime.datetime.now()
        time_stamp = get_15min_aligned_timestamp(current_time.time())
        datetime_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建汇总记录
        summary_record = {
            'datetime': datetime_str,
            'time_stamp': time_stamp,
            'position_count': stats['position_count'],
            'total_volume': stats['total_volume'],
            'total_market_value': stats['total_market_value'],
            'cash_available': asset_info['cash'],
            'cash_in_transit': asset_info['cash_in_transit'],
            'total_assets': asset_info['total_assets'],
            'source': DATASOURCE.QMT,
            'updated_at': time_stamp
        }
        
        # 插入汇总记录
        collection.insert_one(summary_record)
        
        # 创建索引
        collection.create_index('time_stamp')
        collection.create_index('datetime')
        
        print("成功保存汇总信息到StockCN_xtquant_synchronized表")
        return True
        
    except Exception as e:
        print(f"保存汇总信息到MongoDB时出错: {e}")
        return False


def export_xtquant_positions_to_mongodb(
    xt_trader,
    account,
    acc,
):
    """
    导出XTQuant持仓数据到MongoDB的主函数
    """
    print("开始导出XTQuant持仓数据到MongoDB...")
    
    try:
        global _EXPORT_POSITIONS_FAILURE_COUNT
            
        # 获取持仓数据
        positions_data = get_xtquant_positions(xt_trader, acc)
        
        if not positions_data:
            print("没有获取到持仓数据")
            _EXPORT_POSITIONS_FAILURE_COUNT += 1
            print(f"连续第 {_EXPORT_POSITIONS_FAILURE_COUNT} 次获取持仓数据失败")
            
            # 检查是否连续5次失败
            if _EXPORT_POSITIONS_FAILURE_COUNT >= 5:
                alert_title = "XTQuant MiniQMT客户端可能需输入交易密码"
                alert_message = (
                    "连续5次未能获取到XTQuant持仓数据和账户资金信息，\n"
                    "怀疑MiniQMT客户端需要输入交易密码验证。\n"
                    "请检查MiniQMT客户端状态并重新登录。"
                )
                send_alert(title=alert_title, message=alert_message, level="warning")
                print("已发送Server酱提醒：MiniQMT客户端可能需要输入交易密码")
            
            return False
        else:
            # 重置失败计数器
            _EXPORT_POSITIONS_FAILURE_COUNT = 0
            
        # 获取账户资金信息
        asset_info = get_xtquant_account_asset(xt_trader, acc)
        
        # 计算持仓统计信息
        stats = calculate_position_stats(positions_data)
        
        # 转换为DataFrame
        df = positions_to_dataframe(positions_data, account)
        
        # 打印统计信息
        print(f"总持仓数量: {stats['position_count']} 只股票")
        print(f"总持仓股数: {stats['total_volume']} 股")
        print(f"总市值: {stats['total_market_value']:.2f} 元")
        print(f"可用资金: {asset_info['cash']:.2f} 元")
        print(f"在途资金: {asset_info['cash_in_transit']:.2f} 元")
        print(f"总资产: {asset_info['total_assets']:.2f} 元")
        
        print(f"\n获取到 {len(df)} 个持仓详情:")
        print(df[['symbol', 'volume', 'avail_vol', 'cost_price', 'market_value']])
        
        # 打印可用数量统计
        total_available_volume = sum(pos['avail_vol'] for pos in positions_data)
        print("\n可用数量统计:")
        print(f"总持仓数量: {stats['total_volume']} 股")
        print(f"总可用数量: {total_available_volume} 股")
        if (stats['total_volume'] > 1e-12):
            print(f"可用比例: {total_available_volume/stats['total_volume']*100:.2f}%")
        
        # 保存持仓数据到MongoDB
        success1 = save_positions_to_mongodb(
            df,
            DATABASE_GolemQ.StockCN_watchdog_eneloop,
            DATABASE_GolemQ.StockCN_watchdog_eneloop_archive
        )
        
        # 保存汇总信息到MongoDB
        success2 = save_sync_summary_to_mongodb(positions_data, asset_info)

        return success1 and success2
        
    except Exception as e:
        print(f"导出持仓数据时出错: {e}")
        traceback.print_exc()
        return False
    

def watchdog_xtquant_positions_checkpoint(
    verbose: bool = False,
):
    print(f"{datetime.datetime.now():%Y-%m-%d}持仓个股监控")
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
        # xtquant_historical_count = len(historical_positions)  # 未使用
        
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

    except Exception as e:
        print(f"查询关注列表时出错: {e}")
        return []
    

def get_xtquant_orders(xt_trader: XtQuantTrader, acc: StockAccount) -> List[Dict[str, Any]]:
    """
    获取XTQuant账户委托订单（挂单）数据
    """
    try:
        orders = xt_trader.query_stock_orders(acc)
        orders_data = []
        
        for order in orders:
            # 只关注未完全成交的挂单（委托状态为已报、部分成交等）
            order_status = getattr(order, 'order_status', 0)
            status_msg = getattr(order, 'status_msg', '')
            
            # 使用报单时间作为时间戳，转换为int32
            order_time = getattr(order, 'order_time', 0)
            time_stamp = int(order_time) if order_time else int(datetime.datetime.now().timestamp())
            
            order_info = {
                'order_id': order.order_id,
                'stock_code': order.stock_code,
                'order_type': order.order_type,  # 委托类型：买入/卖出
                'order_volume': order.order_volume,  # 委托数量
                'traded_volume': order.traded_volume,  # 成交数量
                'price': order.price,  # 委托价格
                'order_time': order_time,  # 报单时间
                'time_stamp': time_stamp,  # 时间戳，用于唯一标识和标记
                'order_status': order_status,
                'status_msg': status_msg,
                'strategy_name': getattr(order, 'strategy_name', ''),
                'order_remark': getattr(order, 'order_remark', ''),
                'source': 'xtquant',  # 数据来源
                'account_id': acc.account_id,  # 账户ID
                'updated_at': int(datetime.datetime.now().timestamp()),  # 更新时间戳
                'pushed': 0  # 推送标记，0表示未推送
            }
            orders_data.append(order_info)
            
        return orders_data
        
    except Exception as e:
        print(f"获取委托订单数据时出错: {e}")
        return []


def save_orders_to_database(orders_data: List[Dict[str, Any]],
                            collection=DATABASE_GolemQ.StockCN_xtquant_orders) -> int:
    """
    将挂单数据保存到数据库
    
    Args:
        orders_data: 订单数据列表
        collection: 数据库集合
        
    Returns:
        成功保存的记录数量
    """
    try:
        if not orders_data:
            print("没有挂单数据需要保存")
            return 0
        
        saved_count = 0
        current_time = int(datetime.datetime.now().timestamp())
        
        for order in orders_data:
            # 构建查询条件：使用time_stamp作为唯一标识（因为order_id可能为零）
            time_stamp = order.get('time_stamp')
            if time_stamp is None:
                print(f"警告：订单缺少time_stamp字段，跳过保存: {order.get('order_id')}")
                continue
                
            query = {
                'time_stamp': time_stamp,
                'source': 'xtquant'
            }
            
            # 准备更新数据
            update_data = order.copy()
            update_data['last_checked'] = current_time
            
            # 如果订单已存在，保留原有的pushed标记
            existing_order = collection.find_one(query)
            if existing_order:
                # 保留原有的pushed标记
                update_data['pushed'] = existing_order.get('pushed', 0)
                # 保留首次创建时间
                if 'created_at' in existing_order:
                    update_data['created_at'] = existing_order['created_at']
                else:
                    update_data['created_at'] = current_time
            else:
                # 新订单，设置创建时间
                update_data['created_at'] = current_time
                update_data['pushed'] = 0  # 新订单默认未推送
            
            # 更新或插入订单数据
            result = collection.update_one(
                query,
                {'$set': update_data},
                upsert=True
            )
            
            if result.modified_count > 0 or result.upserted_id is not None:
                saved_count += 1
        
        print(f"成功保存 {saved_count}/{len(orders_data)} 条挂单记录到数据库")
        return saved_count
        
    except Exception as e:
        print(f"保存挂单数据到数据库时出错: {e}")
        traceback.print_exc()
        return 0


def get_pushed_order_ids(collection=DATABASE_GolemQ.StockCN_xtquant_orders) -> Set[int]:
    """
    获取已推送的订单时间戳集合
    
    Args:
        collection: 数据库集合
        
    Returns:
        已推送的订单时间戳集合
    """
    try:
        # 创建索引以提高查询性能
        collection.create_index([('source', 1), ('pushed', 1), ('time_stamp', -1)])
        collection.create_index('time_stamp')
        
        # 计算24小时前的时间戳（秒）
        current_timestamp = int(time.time())
        twenty_four_hours_ago = current_timestamp - 86400  # 24 * 60 * 60
        
        # 查询已推送的订单，限制在24小时内
        pushed_orders = collection.find({
            'source': 'xtquant',
            'pushed': 1,
            'time_stamp': {'$gte': twenty_four_hours_ago}
        }, {'time_stamp': 1})
        
        pushed_time_stamps = {int(order['time_stamp']) for order in pushed_orders if 'time_stamp' in order}
        print(f"从数据库加载了 {len(pushed_time_stamps)} 个已推送的订单时间戳（最近24小时内）")
        return pushed_time_stamps
        
    except Exception as e:
        print(f"获取已推送订单时间戳时出错: {e}")
        return set()


def mark_order_as_pushed(time_stamp: int, collection=DATABASE_GolemQ.StockCN_xtquant_orders) -> bool:
    """
    标记订单为已推送
    
    Args:
        time_stamp: 订单时间戳
        collection: 数据库集合
        
    Returns:
        是否成功标记
    """
    try:
        result = collection.update_one(
            {
                'time_stamp': time_stamp,
                'source': 'xtquant'
            },
            {
                '$set': {
                    'pushed': 1,
                    'pushed_at': int(datetime.datetime.now().timestamp())
                }
            }
        )
        
        if result.modified_count > 0:
            print(f"✓ 订单时间戳 {time_stamp} 已标记为已推送")
            return True
        else:
            print(f"✗ 未找到订单时间戳 {time_stamp} 或标记失败")
            return False
            
    except Exception as e:
        print(f"标记订单为已推送时出错: {e}")
        return False


def check_new_orders_and_alert(xt_trader: XtQuantTrader, acc: StockAccount,
                               known_order_ids: Set[int], sync_count: int) -> Set[int]:
    """
    检查新增挂单并发送提醒
    
    Args:
        xt_trader: XTQuant交易实例
        acc: 账户对象
        known_order_ids: 已知的订单时间戳集合（现在从数据库加载）
        sync_count: 当前同步次数
        
    Returns:
        更新后的已知订单时间戳集合
    """
    try:
        # 查询当前所有委托订单
        current_orders = get_xtquant_orders(xt_trader, acc)
        
        if not current_orders:
            print("当前没有委托订单")
            # 仍然保存空订单列表到数据库以更新检查时间
            save_orders_to_database(current_orders)
            return known_order_ids
        
        # 保存订单数据到数据库
        save_orders_to_database(current_orders)
        
        # 从数据库加载已推送的订单时间戳
        pushed_time_stamps = get_pushed_order_ids()
        
        # 获取当前订单时间戳
        current_time_stamps = {order['time_stamp'] for order in current_orders if 'time_stamp' in order}
        
        # 检测未推送的新订单（在数据库中但未标记为已推送）
        new_time_stamps = current_time_stamps - pushed_time_stamps
        
        if new_time_stamps:
            # 获取新增订单的详细信息
            new_orders = [order for order in current_orders
                          if order.get('time_stamp') in new_time_stamps]
            
            print(f"检测到 {len(new_orders)} 个未推送的挂单")
            
            # 为每个新增订单发送提醒
            for order in new_orders:
                stock_code = order['stock_code']
                order_type = "买入" if order['order_type'] == 23 else "卖出"  # 23是买入，24是卖出
                order_volume = order['order_volume']
                price = order['price']
                order_time = datetime.datetime.fromtimestamp(order['order_time']).strftime("%Y-%m-%d %H:%M:%S")
                time_stamp = order.get('time_stamp', 0)
                
                # 构建消息内容
                title = f"XTQuant新增挂单提醒 - 第{sync_count}次同步"
                message = (
                    f"检测到新增挂单:\n"
                    f"股票代码: {stock_code}\n"
                    f"委托类型: {order_type}\n"
                    f"委托数量: {order_volume}股\n"
                    f"委托价格: {price}元\n"
                    f"报单时间: {order_time}\n"
                    f"时间戳: {time_stamp}\n"
                    f"订单ID: {order['order_id']}\n"
                    f"策略名称: {order['strategy_name']}\n"
                    f"备注: {order['order_remark']}"
                )
                
                # 发送提醒
                alert_sent = send_alert(
                    title=title,
                    message=message,
                    level="info"
                )
                
                if alert_sent:
                    # 标记订单为已推送
                    mark_success = mark_order_as_pushed(time_stamp)
                    if mark_success:
                        print(f"✓ 已发送新增挂单提醒并标记为已推送: {stock_code} {order_type} {order_volume}股 @ {price}元")
                    else:
                        print(f"✓ 已发送新增挂单提醒但标记失败: {stock_code} {order_type} {order_volume}股 @ {price}元")
                else:
                    print(f"✗ 发送挂单提醒失败: {stock_code}")
        
        # 更新已知订单时间戳集合（合并新旧订单）
        updated_known_time_stamps = known_order_ids.union(current_time_stamps)
        
        # 打印当前挂单统计
        pending_orders = [order for order in current_orders
                          if order['order_status'] in [0, 1, 2]]  # 0:已报, 1:部分成交, 2:未成交
        if pending_orders:
            print(f"当前共有 {len(pending_orders)} 个未完成挂单")
            for order in pending_orders[:5]:  # 只显示前5个
                stock_code = order['stock_code']
                order_type = "买入" if order['order_type'] == 23 else "卖出"
                traded_ratio = order['traded_volume'] / order['order_volume'] * 100 if order['order_volume'] > 0 else 0
                print(f"  {stock_code} {order_type} {order['order_volume']}股 "
                      f"(已成交: {order['traded_volume']}股, {traded_ratio:.1f}%)")
            if len(pending_orders) > 5:
                print(f"  ... 还有 {len(pending_orders) - 5} 个挂单未显示")
        
        return updated_known_time_stamps
        
    except Exception as e:
        print(f"检查新增挂单时出错: {e}")
        traceback.print_exc()
        return known_order_ids


def xtquant_sync_during_trading_hours():
    """
    在交易时间内循环执行XTQuant同步，包含心跳签到机制
    """
    
    # 创建心跳监控实例
    module = HeartbeatModule(
        module_name="xtquant_sync_loop",
        instance_id=f"xtquant_sync_loop_{int(time.time())}",
        timeout_seconds=60,  # 1分钟超时
    )
    
    if (module.mutex(verbose=True)):
        return False
    else:
        # 开始模块执行记录
        module.start(
            initial_message="开始交易日内的XTQuant持仓同步循环"
        )
    
    try:
        # 获取当前日期和交易时间
        current_date = datetime.datetime.now()
        
        # 设置交易结束时间（15:15收盘）
        start_time = current_date.replace(hour=9, minute=15, second=0, microsecond=0)
        end_time = current_date.replace(hour=15, minute=40, second=0, microsecond=0)
        get_once = True
        print(f"开始交易日内的XTQuant持仓同步循环，将在 {end_time.strftime('%H:%M:%S')} 结束")
        
        sync_count = 0
        # 获取XTQuant配置
        xtquant_config = get_xtquant_config()
        min_path = xtquant_config['min_path']
        account = xtquant_config['account']
        
        # 创建交易实例
        session_id = int(datetime.datetime.now().timestamp())
        xt_trader = XtQuantTrader(min_path, session_id)
        acc = StockAccount(account, 'STOCK')
        
        # 启动交易线程
        xt_trader.start()
        
        # 建立连接
        connect_result = xt_trader.connect()
        if connect_result != 0:
            module.complete(
                exit_code=1,
                completion_message=f"连接XTQuant失败: {connect_result}"
            )
            print(f"连接XTQuant失败: {min_path} {account} {connect_result}")
            return False
        
        # 从数据库加载已推送的订单时间戳
        known_order_ids: Set[int] = get_pushed_order_ids()
        print(f"从数据库加载了 {len(known_order_ids)} 个已推送的订单时间戳")
        
        # 在交易时间内循环执行
        while (start_time < datetime.datetime.now() < end_time) or get_once:
            current_time = datetime.datetime.now()
            
            # 心跳签到
            module.checkin(
                message=f"第{sync_count + 1}次同步，下次同步时间: {(current_time + datetime.timedelta(minutes=5)).strftime('%H:%M:%S')}"
            )
            
            # 执行同步
            print(f"\n=== 第{sync_count + 1}次同步 ===")
            success = export_xtquant_positions_to_mongodb(
                xt_trader,
                account,
                acc,
            )

            if success:
                sync_count += 1
                print(f"✓ 第{sync_count}次同步成功完成")
            else:
                print("✗ 同步失败，将在5分钟后重试")
            
            # 检查新增挂单并发送提醒
            known_order_ids = check_new_orders_and_alert(
                xt_trader, acc, known_order_ids, sync_count
            )
            
            watchdog_xtquant_positions_checkpoint()

            if (get_once):
                get_once = False
                continue
            elif not (GQ_util_if_trade(current_time.strftime("%Y-%m-%d"))):
                break

            # 等待5分钟再进行下一次同步
            time.sleep(119)  # 5分钟
        
        # 停止交易实例
        xt_trader.stop()
        
        # 交易日结束，标记模块完成
        module.complete(
            completion_message=f"交易日同步完成，共执行{sync_count}次同步"
        )
        
        print(f"\n交易日同步结束，共完成 {sync_count} 次同步")
        return True
        
    except Exception as e:
        # 发生异常，标记错误
        module.complete(
            exit_code=1,
            completion_message=f"交易日同步异常: {str(e)}"
        )
        print(f"交易日同步异常: {e}")
        return False


def export_xtquant_positions_to_mongodb_v2(
    trader: xtQmtTrader,
    account: str = None,
):
    """
    导出XTQuant持仓数据到MongoDB的主函数（使用xtQmtTrader类）
    """
    print("开始导出XTQuant持仓数据到MongoDB (v2)...")
    
    try:
        # 获取持仓数据
        position_df = trader.get_position()
        
        if position_df.empty:
            print("没有获取到持仓数据")
            return False
            
        # 获取账户资金信息
        balance_dict = trader.get_balance()
        
        # 将DataFrame转换为与现有函数兼容的格式
        positions_data = []
        for _, row in position_df.iterrows():
            if pd.isna(row['证券代码']):
                continue
                
            position_info = {
                'stock_code': str(row['证券代码']),
                'stock_name': '',
                'volume': float(row['持仓数量']) if not pd.isna(row['持仓数量']) else 0,
                'avail_vol': float(row['可用数量']) if not pd.isna(row['可用数量']) else 0,
                'open_price': float(row['平均建仓成本']) if not pd.isna(row['平均建仓成本']) else 0,
                'market_value': float(row['市值']) if not pd.isna(row['市值']) else 0,
                'current_price': 0,
                'cost_price': float(row['平均建仓成本']) if not pd.isna(row['平均建仓成本']) else 0,
                'profit': 0,
                'profit_ratio': 0
            }
            positions_data.append(position_info)
        
        # 转换资金信息格式
        asset_info = {
            'cash': float(balance_dict.get('可用金额', 0)) if not pd.isna(balance_dict.get('可用金额', 0)) else 0,
            'cash_in_transit': float(balance_dict.get('冻结金额', 0)) if not pd.isna(balance_dict.get('冻结金额', 0)) else 0,
            'total_assets': float(balance_dict.get('总资产', 0)) if not pd.isna(balance_dict.get('总资产', 0)) else 0
        }
        
        # 计算持仓统计信息
        stats = calculate_position_stats(positions_data)
        
        # 转换为DataFrame
        df = positions_to_dataframe(positions_data, account)
        
        # 打印统计信息
        print(f"总持仓数量: {stats['position_count']} 只股票")
        print(f"总持仓股数: {stats['total_volume']} 股")
        print(f"总市值: {stats['total_market_value']:.2f} 元")
        print(f"可用资金: {asset_info['cash']:.2f} 元")
        print(f"在途资金: {asset_info['cash_in_transit']:.2f} 元")
        print(f"总资产: {asset_info['total_assets']:.2f} 元")
        
        print(f"\n获取到 {len(df)} 个持仓详情:")
        print(df[['symbol', 'volume', 'avail_vol', 'cost_price', 'market_value']])
        
        # 打印可用数量统计
        total_available_volume = sum(pos['avail_vol'] for pos in positions_data)
        print("\n可用数量统计:")
        print(f"总持仓数量: {stats['total_volume']} 股")
        print(f"总可用数量: {total_available_volume} 股")
        if (stats['total_volume'] > 1e-12):
            print(f"可用比例: {total_available_volume/stats['total_volume']*100:.2f}%")
        
        # 保存持仓数据到MongoDB
        success1 = save_positions_to_mongodb(
            df,
            DATABASE_GolemQ.StockCN_watchdog_eneloop,
            DATABASE_GolemQ.StockCN_watchdog_eneloop_archive
        )
        
        # 保存汇总信息到MongoDB
        success2 = save_sync_summary_to_mongodb(positions_data, asset_info)

        return success1 and success2
        
    except Exception as e:
        print(f"导出持仓数据时出错: {e}")
        traceback.print_exc()
        return False


def xtquant_sync_during_trading_hours_v2():
    """
    在交易时间内循环执行XTQuant同步，包含心跳签到机制（使用xtQmtTrader类）
    """
    
    # 创建心跳监控实例
    module = HeartbeatModule(
        module_name="xtquant_sync_loop_v2",
        instance_id=f"xtquant_sync_loop_v2_{int(time.time())}",
        timeout_seconds=60,  # 1分钟超时
    )
    
    if (module.mutex(verbose=True)):
        return False
    else:
        # 开始模块执行记录
        module.start(
            initial_message="开始交易日内的XTQuant持仓同步循环 (v2)"
        )
    
    try:
        # 获取当前日期和交易时间
        current_date = datetime.datetime.now()
        
        # 设置交易结束时间（15:15收盘）
        start_time = current_date.replace(hour=9, minute=15, second=0, microsecond=0)
        end_time = current_date.replace(hour=15, minute=40, second=0, microsecond=0)
        get_once = True
        print(f"开始交易日内的XTQuant持仓同步循环 (v2)，将在 {end_time.strftime('%H:%M:%S')} 结束")
        
        sync_count = 0
        
        # 创建xtQmtTrader实例
        trader = xtQmtTrader()
        # 连接QMT
        trader.connect()
        
        # 在交易时间内循环执行
        while (start_time < datetime.datetime.now() < end_time) or get_once:
            current_time = datetime.datetime.now()
            
            # 心跳签到
            module.checkin(
                message=f"第{sync_count + 1}次同步，下次同步时间: {(current_time + datetime.timedelta(minutes=5)).strftime('%H:%M:%S')}"
            )
            
            # 执行同步
            print(f"\n=== 第{sync_count + 1}次同步 (v2) ===")
            success = export_xtquant_positions_to_mongodb_v2(
                trader,
                trader.account,
            )

            if success:
                sync_count += 1
                print(f"✓ 第{sync_count}次同步成功完成")
            else:
                print("✗ 同步失败，将在5分钟后重试")
            
            watchdog_xtquant_positions_checkpoint()

            if (get_once):
                get_once = False
                continue
            elif not (GQ_util_if_trade(current_time.strftime("%Y-%m-%d"))):
                break

            # 等待5分钟再进行下一次同步
            time.sleep(119)  # 5分钟
        
        # 交易日结束，标记模块完成
        module.complete(
            completion_message=f"交易日同步完成，共执行{sync_count}次同步"
        )
        
        print(f"\n交易日同步结束，共完成 {sync_count} 次同步")
        return True
        
    except Exception as e:
        # 发生异常，标记错误
        module.complete(
            exit_code=1,
            completion_message=f"交易日同步异常: {str(e)}"
        )
        print(f"交易日同步异常: {e}")
        return False


if __name__ == '__main__':
    # 直接运行测试
    success = export_xtquant_positions_to_mongodb()
    if success:
        print("持仓数据导出成功!")
    else:
        print("持仓数据导出失败!")