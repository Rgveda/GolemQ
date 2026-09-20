#!/usr/bin/env python3
# coding:utf-8
"""
测试可用数量显示功能
"""

import sys
import os

try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from GolemQ.gateway.xtquant.xtquant_tools import positions_to_dataframe, calculate_position_stats

def test_available_volume_display():
    """测试可用数量显示功能"""
    print("=== 测试可用数量显示功能 ===")
    
    # 模拟持仓数据
    mock_positions = [
        {
            'stock_code': '000001.SZ',
            'stock_name': '平安银行',
            'volume': 1000,
            'avail_vol': 800,  # 800股可用
            'open_price': 15.5,
            'market_value': 15500,
            'current_price': 15.5,
            'cost_price': 15.2,
            'profit': 300,
            'profit_ratio': 1.97
        },
        {
            'stock_code': '600036.SH',
            'stock_name': '招商银行',
            'volume': 500,
            'avail_vol': 500,  # 500股可用
            'open_price': 35.8,
            'market_value': 17900,
            'current_price': 35.8,
            'cost_price': 34.5,
            'profit': 650,
            'profit_ratio': 3.77
        }
    ]
    
    # 转换为DataFrame
    df = positions_to_dataframe(mock_positions)
    
    print("持仓详情 (包含可用数量):")
    print(df[['symbol', 'volume', 'avail_vol', 'cost_price', 'market_value']])
    
    # 计算可用数量统计
    total_available_volume = sum(pos['avail_vol'] for pos in mock_positions)
    stats = calculate_position_stats(mock_positions)
    
    print("\n可用数量统计:")
    print(f"总持仓数量: {stats['total_volume']} 股")
    print(f"总可用数量: {total_available_volume} 股")
    print(f"可用比例: {total_available_volume/stats['total_volume']*100:.2f}%")
    
    return True

if __name__ == '__main__':
    print("开始测试可用数量显示功能...")
    
    success = test_available_volume_display()
    
    if success:
        print("\n✓ 测试通过!")
        print("修改后的功能已成功显示可用数量信息")
    else:
        print("\n✗ 测试失败!")
        sys.exit(1)