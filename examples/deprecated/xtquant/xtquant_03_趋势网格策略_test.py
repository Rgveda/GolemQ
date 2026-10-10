# coding:utf-8
#
# 趋势网格策略测试脚本
#

class MockTrendGridStrategy:
    """模拟趋势网格策略用于测试"""
    
    def __init__(self):
        self.grid_levels = {}
        self.base_prices = {}
    
    def mock_is_trend_up(self, ma_values: dict) -> bool:
        """
        模拟趋势判断
        ma_values: {'ma5': value, 'ma10': value, 'ma20': value, 'ma30': value, 'ma60': value}
        """
        ma5 = ma_values['ma5']
        ma10 = ma_values['ma10']
        ma20 = ma_values['ma20']
        ma30 = ma_values['ma30']
        ma60 = ma_values['ma60']
        
        # 判断多头排列条件
        condition1 = (ma5 > ma10 > ma20 > ma30 > ma60)
        condition2 = (ma10 > ma5 > ma20 > ma30 > ma60)
        
        return condition1 or condition2
    
    def initialize_grid(self, stock_code: str, current_price: float):
        """初始化网格"""
        if stock_code not in self.grid_levels:
            self.grid_levels[stock_code] = 0
            self.base_prices[stock_code] = current_price
            print(f"初始化 {stock_code} 网格，基准价格: {current_price}")
    
    def calculate_grid_level(self, stock_code: str, current_price: float) -> int:
        """计算网格层级"""
        if stock_code not in self.base_prices:
            return 0
            
        base_price = self.base_prices[stock_code]
        price_change_percent = (current_price - base_price) / base_price * 100
        
        # 每2%为一个网格
        grid_size = 2.0
        grid_level = int(price_change_percent / grid_size)
        
        return grid_level
    
    def test_trend_conditions(self):
        """测试趋势判断条件"""
        print("=== 测试趋势判断条件 ===")
        
        # 测试用例1: MA5 > MA10 > MA20 > MA30 > MA60
        test_case1 = {
            'ma5': 15.0, 'ma10': 14.0, 'ma20': 13.0, 'ma30': 12.0, 'ma60': 11.0
        }
        result1 = self.mock_is_trend_up(test_case1)
        print(f"测试用例1 (MA5>MA10>MA20>MA30>MA60): {result1}")
        
        # 测试用例2: MA10 > MA5 > MA20 > MA30 > MA60
        test_case2 = {
            'ma5': 14.0, 'ma10': 15.0, 'ma20': 13.0, 'ma30': 12.0, 'ma60': 11.0
        }
        result2 = self.mock_is_trend_up(test_case2)
        print(f"测试用例2 (MA10>MA5>MA20>MA30>MA60): {result2}")
        
        # 测试用例3: 不满足条件
        test_case3 = {
            'ma5': 11.0, 'ma10': 12.0, 'ma20': 13.0, 'ma30': 14.0, 'ma60': 15.0
        }
        result3 = self.mock_is_trend_up(test_case3)
        print(f"测试用例3 (不满足多头排列): {result3}")
    
    def test_grid_logic(self):
        """测试网格逻辑"""
        print("\n=== 测试网格逻辑 ===")
        
        stock_code = "000001.SZ"
        base_price = 10.0
        
        # 初始化网格
        self.initialize_grid(stock_code, base_price)
        
        # 测试不同价格下的网格层级
        test_prices = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
        
        for price in test_prices:
            grid_level = self.calculate_grid_level(stock_code, price)
            change_percent = (price - base_price) / base_price * 100
            print(f"价格: {price:.2f}, 涨跌幅: {change_percent:+.1f}%, 网格层级: {grid_level}")
    
    def test_trading_conditions(self):
        """测试交易条件"""
        print("\n=== 测试交易条件 ===")
        
        stock_code = "000001.SZ"
        base_price = 10.0
        self.initialize_grid(stock_code, base_price)
        
        # 测试加仓条件 (价格低于MA30且网格层级<-1)
        test_cases_buy = [
            {'price': 9.0, 'ma30_15min': 9.5, 'should_buy': True},    # 价格<MA30, 网格=-5
            {'price': 9.5, 'ma30_15min': 9.0, 'should_buy': False},   # 价格>MA30
            {'price': 9.8, 'ma30_15min': 10.0, 'should_buy': False},  # 网格=-1 (不满足<-1)
            {'price': 8.0, 'ma30_15min': 9.0, 'should_buy': True},    # 价格<MA30, 网格=-10
        ]
        
        print("加仓条件测试:")
        for i, case in enumerate(test_cases_buy, 1):
            grid_level = self.calculate_grid_level(stock_code, case['price'])
            condition1 = case['price'] < case['ma30_15min']
            condition2 = grid_level < -1
            should_buy = condition1 and condition2
            print(f"  用例{i}: 价格{case['price']} < MA30{case['ma30_15min']}, "
                  f"网格{grid_level}, 预期{case['should_buy']}, 实际{should_buy}")
        
        # 测试减仓条件 (价格高于MA30且网格层级>2)
        test_cases_sell = [
            {'price': 11.0, 'ma30_15min': 10.5, 'should_sell': False},  # 网格=5, 但价格>MA30
            {'price': 10.5, 'ma30_15min': 11.0, 'should_sell': False},  # 价格<MA30
            {'price': 11.0, 'ma30_15min': 10.0, 'should_sell': True},   # 价格>MA30, 网格=5
            {'price': 10.2, 'ma30_15min': 10.0, 'should_sell': False},  # 网格=1 (不满足>2)
        ]
        
        print("\n减仓条件测试:")
        for i, case in enumerate(test_cases_sell, 1):
            grid_level = self.calculate_grid_level(stock_code, case['price'])
            condition1 = case['price'] > case['ma30_15min']
            condition2 = grid_level > 2
            should_sell = condition1 and condition2
            print(f"  用例{i}: 价格{case['price']} > MA30{case['ma30_15min']}, "
                  f"网格{grid_level}, 预期{case['should_sell']}, 实际{should_sell}")


if __name__ == '__main__':
    strategy = MockTrendGridStrategy()
    strategy.test_trend_conditions()
    strategy.test_grid_logic()
    strategy.test_trading_conditions()
    
    print("\n=== 测试完成 ===")
    print("这个测试脚本验证了趋势网格策略的核心逻辑:")
    print("1. 趋势判断: MA多头排列条件")
    print("2. 网格计算: 基于基准价格的网格层级")
    print("3. 交易条件: 加仓和减仓的判断逻辑")
    print("\n在实际运行前，请确保:")
    print("- XTQuant配置正确")
    print("- 交易账户有足够资金")
    print("- 选择的股票符合多头趋势条件")