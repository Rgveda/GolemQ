#!/usr/bin/env python3
"""
GolemQ 测试运行脚本
运行所有单元测试并生成测试报告
"""

import unittest
import sys
import os


def run_tests():
    """运行所有测试"""
    
    # 添加项目根目录到Python路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # 发现并运行所有测试
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(os.path.abspath(__file__)), pattern='test_*.py')
    
    # 运行测试
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # 返回测试结果
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)