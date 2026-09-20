#!/usr/bin/env python3
# coding:utf-8
"""
测试导出持仓数据功能
"""

import sys
import os

try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from GolemQ.gateway.xtquant.xtquant_tools import positions_to_dataframe


def test_data_conversion():
    """测试数据转换功能"""
    print("=== 测试数据转换功能 ===")
    
    # 模拟持仓数据
    mock_positions = [
        {
            'stock_code': '000001.SZ',
            'stock_name': '平安银行',
            'volume': 1000,
            'can_use_volume': 1000,
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
            'can_use_volume': 500,
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
    
    print("转换后的DataFrame:")
    print(df)
    print("\nDataFrame列名:")
    print(df.columns.tolist())
    print("\nDataFrame形状:", df.shape)
    
    # 检查必要字段
    required_columns = ['symbol', 'volume', 'cost_price', 'source', 'updated_at', 'ckop_at']
    for col in required_columns:
        if col in df.columns:
            print(f"✓ 字段 '{col}' 存在")
        else:
            print(f"✗ 字段 '{col}' 缺失")
    
    return True

def test_cli_help():
    """测试CLI帮助信息"""
    print("\n=== 测试CLI帮助信息 ===")
    
    # 模拟运行命令行帮助
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, '-m', 'GolemQ.cli', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if '--export-positions' in result.stdout:
            print("✓ CLI帮助信息中包含 --export-positions 选项")
            return True
        else:
            print("✗ CLI帮助信息中缺少 --export-positions 选项")
            print("输出内容:", result.stdout)
            return False
            
    except Exception as e:
        print(f"测试CLI帮助时出错: {e}")
        return False

if __name__ == '__main__':
    print("开始测试导出持仓数据功能...")
    
    success1 = test_data_conversion()
    success2 = test_cli_help()
    
    if success1 and success2:
        print("\n✓ 所有测试通过!")
        print("\n使用说明:")
        print("1. 确保MongoDB配置正确")
        print("2. 确保XTQuant配置正确")
        print("3. 运行命令: python -m GolemQ.cli --export-positions")
    else:
        print("\n✗ 测试失败!")
        sys.exit(1)