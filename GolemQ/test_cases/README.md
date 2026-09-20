# GolemQ 测试说明

## 测试文件组织结构

本项目采用模块化的测试组织结构，每个主要功能模块都有对应的测试文件：

### 现有测试文件

1. **`test_messenger.py`** - 测试消息通知功能（钉钉、Server酱）
2. **`test_stock_cn.py`** - 测试A股市场基础功能

### 新增测试文件

3. **`test_market_tools.py`** - 测试市场工具功能（数据清理等）
4. **`test_market_align.py`** - 测试数据对齐功能  
5. **`test_market_crawler.py`** - 测试数据爬取功能
6. **`test_market_quotes.py`** - 测试行情数据功能

## 测试运行

### 运行所有测试
```bash
python run_tests.py
```

### 运行特定测试模块
```bash
python -m unittest GolemQ.tests.test_messenger -v
```

### 运行单个测试类
```bash
python -m unittest GolemQ.tests.test_messenger.TestDingtalkConfig -v
```

### 运行单个测试方法
```bash
python -m unittest GolemQ.tests.test_messenger.TestDingtalkConfig.test_check_config_success -v
```

## 测试编写规范

1. **命名规范**: 测试文件以 `test_` 开头，测试类以 `Test` 开头，测试方法以 `test_` 开头
2. **Mock使用**: 外部依赖（数据库、API等）必须使用mock进行隔离测试
3. **断言清晰**: 每个测试应该有明确的断言和错误信息
4. **独立运行**: 测试之间不应该有依赖关系，可以独立运行

## 注意事项

- 测试文件应该放在 `GolemQ/tests/` 目录下
- 避免在测试中访问真实的外部服务
- 使用适当的mock来模拟外部依赖
- 确保测试覆盖主要业务逻辑