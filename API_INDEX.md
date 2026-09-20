# GolemQ API 索引

> **本文件由 `tools/gen_api_index.py` 自动生成，请勿手工编辑。**
> 它是**导航索引**，不是文档 —— 只出「名称 + 首行摘要」。
> 完整说明在各模块的 docstring 与 `PITFALLS.md` / `DECISIONS.md` / `GLOSSARY.md`。

`[dt]` = 该 docstring 含 doctest，由 `GolemQ/test_cases/test_doctests.py` 收集执行。

## GolemQ.agents

### `messenger`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `check_dingtalk_config()` | 检查钉钉配置是否存在 |
| f | `setup_dingtalk_config()` | 交互式设置钉钉配置 |
| f | `get_dingtalk_config()` | 获取钉钉配置 |
| C | `DingtalkAccessToken` | 钉钉访问令牌管理类 |
| f | `DingtalkAccessToken.create_client()` | 使用 Token 初始化账号Client |
| f | `DingtalkAccessToken.main(appkey, appsecret)` |  |
| C | `DingReminder` | 钉钉消息提醒类 |
| f | `DingReminder.create_client()` | 使用 Token 初始化账号Client |
| f | `DingReminder.send_message(robot_code, user_id_list, content, appkey, appsecret)` |  |
| f | `setup_dingtalk_config_interactive()` | 交互式设置钉钉配置 |
| f | `send_dingtalk_test_message()` | 发送钉钉测试消息 |
| f | `check_serverchan_config()` | 检查Server酱配置是否存在 |
| f | `setup_serverchan_config()` | 交互式设置Server酱配置 |
| f | `get_serverchan_config()` | 获取Server酱配置 |
| f | `sc_send(sendkey, title, desp, options)` | 发送Server酱消息 |
| f | `send_serverchan_message(title, content, options)` | 发送Server酱消息 |
| f | `setup_serverchan_config_interactive()` | 交互式设置Server酱配置 |
| f | `send_serverchan_test_message()` | 发送Server酱测试消息 |

## GolemQ.analysis

### `timeseries`
Time series analysis stubs (to be populated).

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `Timeline_duration(series)` | Calculate timeline duration (stub). |
| f | `align_kline_timeline(kline_data, freq, annual)` | Align kline timeline to standard timestamps (stub). |

## GolemQ.cli

### `__main__`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `main()` | 主函数：处理命令行参数 |

### `tools`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `auto_register_markets()` | 自动注册所有市场模块到 GQMARKETS（只注册尚未注册的市场） |
| f | `purge_mongodb_database(verbose)` | 清理MongoDB数据库 - 通过各个市场实例的清理方法 |

### `watchdog_manager`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `parse_symbols(symbols_str)` | 解析股票代码字符串，支持逗号或换行分割 |
| f | `add_symbols_to_watchlist(symbols, verbose)` | 添加股票代码到关注列表 |
| f | `remove_symbols_from_watchlist(symbols, verbose)` | 从关注列表删除股票代码并移动到归档库 |
| f | `list_watchlist_symbols(verbose)` | 列出当前关注列表中的所有股票代码 |

## GolemQ.core

### `base`
Core base utilities (stub — to be populated).

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_util_get_last_day(ts, n)` | Get the last trading day (stub). |
| f | `set_cpu_affinity_even()` | Set CPU affinity to even-numbered cores (stub). |

### `constants`
GolemQ Constants Module

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `MARKET_TYPE` | 市场种类 |
| C | `BROKER_TYPE` | 执行环境 |
| C | `EVENT_TYPE` | [summary] |
| C | `MARKET_EVENT` | 交易前置事件 |
| C | `ENGINE_EVENT` | 引擎事件 |
| C | `ACCOUNT_EVENT` | 账户事件 |
| C | `BROKER_EVENT` | BROKER事件 |
| C | `ORDER_EVENT` | 订单事件 |
| C | `FREQUENCE` | 查询的级别 |
| C | `CURRENCY_TYPE` | 货币种类 |
| C | `DATASOURCE` | 数据来源 |
| C | `AKA` | A singleton class to manage all global constants in GolemQ. |
| C | `FIELD` | A singleton class to manage all global constants in GolemQ. |
| C | `FEATURES` | Feature column name constants (stub — to be populated). |
| C | `TREND_STATUS` | Trend status constants (stub — to be populated). |
| C | `STATE` | State constants (stub — to be populated). |

### `market_registry`
市场注册表与「当前激活市场」。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `register_market(name, instance, replace)` | 把市场实例登记进注册表。 `[dt]` |
| f | `register_subscriber(key, func, replace)` | 登记订阅者。replace 语义同 :func:register_market。 |
| f | `active_market_name()` | 当前激活的市场名。 |
| f | `set_active_market(name)` | 切换激活市场。 `[dt]` |
| f | `get_active_market()` | 返回当前激活的市场实例。 |
| f | `get_market(name)` | 按名取市场实例（不影响激活状态）。 |

### `mongo`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_util_mongodb_client(uri)` | explanation: |
| f | `GQ_util_mongodb_client_async(uri)` | explanation: |

### `path`
这里定义的是一些本地目录

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `get_python_version_suffix()` | 获取Python版本后缀，如 'py38' |
| f | `get_pickle_filename(base_name, suffix)` | 生成带版本后缀的pickle文件名 |
| f | `cache_path(dirname, portable, prefix)` | 返回本地用户目录下的'.GolemQ'为根目录的缓存临时文件目录，如果 portable 参数等于 True， |
| f | `mkdirs_user(dirname)` |  |
| f | `mkdirs(dirname)` |  |
| f | `load_snapshot_cache(dirpath, filename)` |  |
| f | `save_snapshot_cache(dirpath, filename, metadata)` |  |

### `preprocessing`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `mask_sensitive_info(value, sensitive)` | 对敏感信息进行脱敏处理 |
| f | `GQ_util_to_json_from_pandas(data)` | explanation: |
| f | `winsorize_quantile(factor, up, down)` | 参考 scipy.stats.mstats.winsorize(a, limits=None) |
| f | `winsorize_med(factor)` | 实现3倍中位数绝对偏差去极值 |
| f | `winsorize_threesigma(factor)` | 自实现正态分布去极值 |
| f | `standardize(s, ty)` | 标准化函数 |
| f | `normalize(x)` | 标准化函数 |
| f | `fill_zero_features(features, all_columns, all_columns_dtype)` | 将所有DataFrame中的缺失列统一补齐。 |
| f | `prefill_null_columns(features_list, columns, columns_dtype)` | 填满NULL列，否则pd.concat()将会导致转换成 np.float64 类型。 |
| f | `concat_return_features(codelist_candidate, eval_range, ret_features)` |  |

### `presentation`
这里定义字符界面输出进度条等UI互动元素

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `tqdm_joblib(tqdm_object, label_template, step)` | Context manager to patch joblib to report into tqdm progress bar given as argument |
| C | `suppress_stdout_stderr` | A context manager for doing a "deep suppression" of stdout and stderr in |
| f | `pandas_display_formatter()` | 将pandas.DataFrame的打印效果设置为中文unicode优化 |
| f | `stock_length_less_than_required(code, ohlc_data, log_msg, verbose, ret_ohlc, ret_meta)` | 停止运行，处理错误股票代码，提交给 RabbitMQ 或者记录到日志文件 |

### `settings`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `GQ_Setting` |  |
| f | `GQ_Setting.get_mongo()` |  |
| f | `GQ_Setting.get_config(section, option, default_value)` | [summary] |
| f | `GQ_Setting.set_config(section, option, default_value)` | [summary] |
| f | `GQ_Setting.get_or_set_section(config, section, option, DEFAULT_VALUE, method)` | [summary] |
| f | `GQ_Setting.env_config()` |  |
| f | `GQ_Setting.client()` |  |
| f | `GQ_Setting.client_async()` |  |
| f | `GQ_Setting.change(ip, port)` |  |
| f | `test_mongodb_connection(uri)` | 测试MongoDB连接 |
| f | `setup_mongodb_config()` | 交互式设置MongoDB配置 |

### `symbol`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_util_code_tostr(code)` | explanation: |
| f | `GQ_util_code_tolist(code, auto_fill)` | explanation: |

## GolemQ.datasource

### `base`
数据源适配层 —— 基类与注册机制。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `register()` | 类装饰器：把数据源登记进注册表。以 cls.name 为键。 `[dt]` |
| f | `registry()` |  |
| C | `DataSourceNotAvailable` | 源不可用（依赖缺失、无凭证、网络不通）。 |
| C | `UnsupportedCollection` | 该源不提供此集合。与「取回空结果」严格区分。 |
| C | `DataSource` | 数据源基类。 |
| f | `DataSource.supports(collection)` |  |
| f | `DataSource.fetch(collection)` | 取回该集合的行（list[dict]，GolemQ 口径）。 |
| f | `DataSource.available()` | 依赖与凭证是否就绪。不得抛异常 —— 排障路径本身不该成为故障点。 |
| f | `DataSource.gate()` | 取数前调用：走限频。所有网络请求都应包在这里面。 |
| f | `DataSource.session()` | HTTP 源用的 requests 会话（含代理注入）。非 HTTP 源不应调用。 |

### `proxy`
代理注入点。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `ProxyConfig` | 从配置读出的代理设置。未配置即禁用。 |
| f | `ProxyConfig.from_settings(section)` |  |
| f | `ProxyConfig.enabled()` | 是否启用了代理。空串、纯空白、None 一律视为未启用。 `[dt]` |
| f | `ProxyConfig.apply(session)` | 把代理挂到 requests.Session。未配置则原样返回（直连）。 |
| f | `ProxyConfig.apply_env()` | 给不暴露 session 的库（如 akshare）用。仅在启用时设置环境变量。 |

### `throttle`
按源的最小请求间隔节流器。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `SourceThrottle` | 进程内、按源名维护 last-request 时间。 |
| f | `SourceThrottle.interval(name)` | 某源的最小间隔：按源覆盖优先，否则用默认（30s）。 `[dt]` |
| f | `SourceThrottle.wait(name)` | 睡到距上次请求满 interval(name) 秒。返回实际睡了多少秒。 |
| f | `SourceThrottle.reset(name)` | 清掉节流状态（测试与手动触发用）。 |
| f | `from_settings(section)` | 从配置构造节流器。 |

### `writer`
参考集合的通用落库器。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `ensure_indexes(coll, unique_keys)` | 建唯一索引。已存在同键索引时不报错。 |
| f | `save_collection(coll, rows, unique_keys, delete_delta_key, batch, verbose)` | 把 rows upsert 进 coll，可选删 delta。 |
| f | `save_block_collection(coll, rows, batch, verbose)` | stock_block 专用：键是 (blockname, code)，删除分两层。 |

## GolemQ.features

### `empirical`
Feature empirical data stubs (to be populated).

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `load_massive_reviews(symbol, start, end, compact, collections)` | Load massive review features (stub). |
| f | `load_massive_model(revision, eval_range, start, end, collections)` | Load massive reality model (stub). |
| f | `save_stock_metadata(data, collections)` | Save stock metadata (stub). |
| f | `GQ_fetch_stock_metadata_major(code, verbose, start, end, collections)` | Fetch major stock metadata (stub). |

### `reviews`
Feature review stubs (to be populated).

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `attach_reality_features(features_dummy, annual, collections)` | Attach reality features to a baseline dataframe (stub). |

## GolemQ.fetch

### `concept`
Concept kline fetching stubs (to be populated).

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `get_stock_concept_kline(symbol, start, end, freq)` | Fetch stock concept kline data (stub). |

### `kline`
K 线获取的门面 —— 不含任何市场知识，一律调度到市场实现。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `resolve_market(market)` | 把 market 参数解析成市场实例。 `[dt]` |
| f | `get_kline_price_min(symbol, start, end, verbose, realtime, market)` | 分钟线。market 省略则用当前激活市场。 |
| f | `get_kline_price_v3(symbol, start, end, verbose, realtime, market)` | 日线。market 省略则用当前激活市场。 |

## GolemQ.gateway.xtquant

### `config`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `check_xtquant_config()` | 检查XTQuant配置是否存在 |
| f | `setup_xtquant_config()` | 交互式设置XTQuant配置 |
| f | `get_xtquant_config()` | 获取XTQuant配置 |
| f | `setup_xtquant_config_interactive()` | 交互式设置XTQuant配置 |

### `helper`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `get_recent_xtquant_order_symbols(days, collection)` | 获取最近一段时间内（默认30天）XTQuant挂单的股票代码列表 |

### `realtime`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `sub_l1_from_xtquant(database_realtime, stock_list_codes)` | 从讯投获取L1数据，大约3秒钟更新一次 |

### `trader`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `xtQmtTrader` |  |
| f | `xtQmtTrader.random_session_id()` | 随机id |
| f | `xtQmtTrader.connect()` | 连接 |
| f | `xtQmtTrader.get_position()` | 查询账户所有的持仓 |
| f | `xtQmtTrader.get_balance()` | 返回当前证券账号的资产数据 |
| f | `xtQmtTrader.today_trades()` | 当日成交 |
| f | `xtQmtTrader.today_entrusts()` | 当日委托 |
| f | `xtQmtTrader.check_stock_is_av_buy(stock, price, amount, hold_limit)` | 检查是否可以买入 |
| f | `xtQmtTrader.check_stock_is_av_sell(stock, amount)` | 检查是否可以卖出 |
| f | `xtQmtTrader.make_buy(security, amount, price, strategy_name, order_remark)` | 单独独立股票买入函数 |
| f | `xtQmtTrader.make_sell(security, amount, price, strategy_name, order_remark)` | 单独独立股票卖出函数 |

### `xtquant_tools`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `get_xtquant_account_asset(xt_trader, acc)` | 获取XTQuant账户资金信息 |
| f | `get_xtquant_positions(xt_trader, acc)` | 获取XTQuant账户持仓数据 |
| f | `positions_to_dataframe(positions_data, account_id)` | 将持仓数据转换为DataFrame |
| f | `calculate_position_stats(positions_data)` | 计算持仓统计信息 |
| f | `save_positions_to_mongodb(df, collection, archive_collection)` | 将持仓数据保存到MongoDB，并将不在当前持仓中的XTQuant记录移动到归档库 |
| f | `save_sync_summary_to_mongodb(positions_data, asset_info, collection)` | 保存同步汇总信息到MongoDB |
| f | `export_xtquant_positions_to_mongodb(xt_trader, account, acc)` | 导出XTQuant持仓数据到MongoDB的主函数 |
| f | `watchdog_xtquant_positions_checkpoint(verbose)` |  |
| f | `get_xtquant_orders(xt_trader, acc)` | 获取XTQuant账户委托订单（挂单）数据 |
| f | `save_orders_to_database(orders_data, collection)` | 将挂单数据保存到数据库 |
| f | `get_pushed_order_ids(collection)` | 获取已推送的订单时间戳集合 |
| f | `mark_order_as_pushed(time_stamp, collection)` | 标记订单为已推送 |
| f | `check_new_orders_and_alert(xt_trader, acc, known_order_ids, sync_count)` | 检查新增挂单并发送提醒 |
| f | `xtquant_sync_during_trading_hours()` | 在交易时间内循环执行XTQuant同步，包含心跳签到机制 |
| f | `export_xtquant_positions_to_mongodb_v2(trader, account)` | 导出XTQuant持仓数据到MongoDB的主函数（使用xtQmtTrader类） |
| f | `xtquant_sync_during_trading_hours_v2()` | 在交易时间内循环执行XTQuant同步，包含心跳签到机制（使用xtQmtTrader类） |

## GolemQ.markets

### `base_market`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `BaseMarket` | 金融市场抽象基类 |
| f | `BaseMarket.init_db(uri, db_name)` | 初始化MongoDB连接 |
| f | `BaseMarket.purge_historical_collections()` | 清理历史数据集合 |
| f | `BaseMarket.get_stock_codes()` | 获取该市场全部股票代码 |
| f | `BaseMarket.get_kline_quotes(code, start, end, fq)` | 获取单只股票日线历史行情 |
| f | `BaseMarket.get_kline_quotes_min(code, start, end, frequency, fq)` | 获取单只股票分钟线历史行情 |
| f | `BaseMarket.get_kline_price_min(codelist, start, end, verbose, realtime)` | 分钟线。返回 (结果对象, codename)。 |
| f | `BaseMarket.get_kline_price_v3(codelist, start, end, verbose, realtime)` | 日线。返回 (结果对象 | None, codename)。 |
| f | `BaseMarket.get_stock_concept_kline(symbol, start, end, freq)` | 概念 K 线。未实现的市场应抛 NotImplementedError —— |
| f | `BaseMarket.name()` | 返回市场名称 |

## GolemQ.markets.StockCN

### `align`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `ckpo_align_stock_turnover_rate(code, data_day, start, end, verbose)` |  |
| f | `stock_min_aligned(verbose)` |  |

### `base`
StockCN base utilities.

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `resample_features_frequency(features, freq)` | Resample features to target frequency. |

### `constants`
StockCN specific constants

*(无公开成员)*

### `crawler`
爬虫模块, 爬取A股指数成分, 财报, 研报和资金流向等财务信息

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_SU_crawl_stock_valuation(code, start, end, collections, verbose)` |  |
| f | `GQ_featch_stock_valuation_from_baostock(code, start, end)` |  |
| f | `GQ_remove_stock_valuation(stock_valuation_detail, collections)` | 保存股价估值信息等数据 |

### `date_utils`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_util_if_trade(day)` | 得到前 n 个交易日 (不包含当前交易日) |
| f | `GQ_util_get_last_day(ts, n)` | 获取最后一个交易日(含当天) |
| f | `GQ_util_if_tradetime(_time, market, code)` | explanation: |
| f | `GQ_util_date_valid(date)` | explanation: |
| f | `get_15min_aligned_timestamp(current_time)` |  |
| f | `get_60min_aligned_timestamp(current_time)` |  |
| f | `GQ_util_date_stamp(date)` | explanation: |
| f | `GQ_util_time_stamp(time_)` | explanation: |

### `fetch`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_fetch_stock_list_day(start, end, code, collections)` | '获取股票日线清单' |
| f | `prepare_symbol_range(eval_range, verbose)` | 返回预设的标的合集 |
| f | `GQ_fetch_stock_min(code, start, end, format, frequence, collections)` | 获取股票分钟线 |
| f | `GQ_fetch_stock_min_adv(code, start, end, frequence, if_drop_index, verbose)` | '获取股票分钟线' |
| f | `get_kline_price_min(codelist, start, market_type, frequency, verbose, end, realtime)` | 写这个函数的目的就是不用去考虑乱七八糟币种和市场种类，直接怼一个或者几个代码就能读取到合适的数据 |
| f | `get_kline_price_v3(codelist, start, market_type, verbose, end, realtime)` | 写这个函数的目的就是不用去考虑乱七八糟币种和市场种类，直接怼一个或者几个代码就能读取到合适的数据 |

### `kline83`
A 股 K 线读取器 —— 走 MongoDB 8.3 时序库（golemq_stock_cn）。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `KlineResult` | 与 QUANTAXIS QA_DataStruct_* 对齐的最小接口 —— 调用方只取 .data。 |
| f | `get_kline_price_min(codelist, start, market_type, frequency, verbose, end, realtime)` | 分钟线读取（8.3 时序）。返回 (KlineResult, codename)。 |
| f | `get_kline_price_v3(codelist, start, market_type, verbose, end, realtime)` | 日线读取（8.3 时序，尚未就绪）。返回 (KlineResult | None, codename)。 |

### `quotes`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `StockCNQuotes` | A股市场行情数据获取 |
| f | `StockCNQuotes.get_kline_quotes(code, start, end, fq)` | 获取A股单只股票日线历史行情（前复权） |
| f | `StockCNQuotes.get_kline_quotes_min(code, start, end, frequency, fq)` | 获取A股单只股票分钟线历史行情（前复权） |

### `realtime`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `formater_l1_tick(code, l1_tick)` | 处理分发 Tick 数据，新浪和tdx l1 tick差异字段格式化处理 |
| f | `collections_of_today(database)` |  |
| f | `formater_l1_ticks(l1_ticks, codelist, stacks, symbol_list)` | 处理 l1 ticks 数据 |
| f | `sub_l1_from_tencent(database_realtime)` | 从腾讯获取L1数据，大约1分钟更新一次 |
| f | `GQ_fetch_stock_realtime_adv(code, num, collections, verbose, suffix)` | 返回当日的上下五档, code可以是股票可以是list, num是每个股票获取的数量 |
| f | `GQ_data_tick_resample_1min(tick, type_, if_drop, stack_vol)` | tick 采样为 分钟数据 |
| f | `GQ_fetch_stock_day_realtime_adv(codelist, data_day, market_type, verbose)` | 查询日线实盘数据，支持多股查询 |
| f | `GQ_fetch_stock_min_realtime_adv(codelist, data_min, frequency, verbose)` | 查询A股的指定小时/分钟线线实盘数据 |
| f | `GQ_fetch_index_min_realtime_adv(codelist, data_min, frequency, verbose)` | 查询指数和ETF的分钟线实盘数据 |
| f | `stock_individual_fund_flow_push(stock, market)` | 东方财富网-数据中心-实时资金流向 |
| f | `get_moneyflow_from_eastmoney_push(code, date_epoch)` | 从东方财富抓取股票资金流向 |

### `refdata`
A 股参考数据的读取层 —— 一律走 MongoDB 8.3 的 golemq_stock_cn。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_ref_collection(name, database)` | 取集合句柄。name 必须是 5 个规范名之一。 |
| f | `GQ_fetch_stock_list(codes, database)` | 股票列表。code 为 6 位，另有 name/pre_close/sse/sec |
| f | `GQ_fetch_stock_info(codes, database)` | 股本与上市信息。真实被消费的是 liutongguben 与 ipo_date/IPODate。 |
| f | `GQ_fetch_etf_list(codes, database)` | ETF 列表。sec 恒为 'etf_cn'。 |
| f | `GQ_fetch_stock_block(blocknames, codes, database)` | 板块成分。 |
| f | `GQ_fetch_financial(codes, report_date, database)` | 季频财务。一行 = 一个 (code, report_date)。 |

### `refdata_save`
A 股参考集合的取数与落库编排 —— 对应 CLI 的 --save-x / --save-qmt。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `save_refdata(collections, source, codelist, verbose)` | 把参考集合取回并落库到 8.3 的 golemq_stock_cn。 |
| f | `refdata_status()` | 各集合当前的源可用性与库存量 —— 排障用。 |
| f | `format_status(report)` | 把 save_refdata 的结果或 refdata_status() 渲染成可读文本。 |

### `scribe`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `QA_fetch_trade_date()` | 获取交易日期 |
| f | `QA_fetch_stock_list(collections)` | 获取股票列表 |
| f | `QA_fetch_index_list(collections)` | 获取指数列表 |
| f | `QA_fetch_stock_terminated(collections)` | 获取股票基本信息 , 已经退市的股票列表 |
| f | `GQ_save_metadata_reality(features, collections)` | 保存同花顺的牛股诊股 |
| f | `GQ_save_daily_metadata_reality(features, collections)` | 保存同花顺的牛股诊股 |
| f | `GQ_fetch_daily_metadata_reality(code, start, end, collections)` | '获取 A股 复盘数据'， |
| f | `GQ_fetch_hourly_metadata_reality(code, start, end, verbose, collections)` | '获取 A股 复盘数据'， |
| f | `GQ_get_etf_list(verbose)` | 获取A股全部ETF列表 |
| f | `GQ_stock_a_spot_em(market_type, collections, verbose)` | 获取A股实时行情数据并保存到不同时间周期的数据集合中 |
| f | `GQ_etf_a_spot_em(collections)` |  |
| f | `GQ_fetch_stock_moneyflow(code, start, end, offset, format, collections, verbose)` | '获取股票资金流向' |

### `symbol`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `EXCHANGE` |  |
| f | `normalize_code(symbol, pre_close, market_type)` | 归一化证券代码 |
| f | `is_stock_cn(code)` | 判断 股票代码，市场来源，板块 |
| f | `is_future_cn(code)` |  |
| f | `is_cryptocurrency(code)` |  |
| f | `GQ_fetch_stock_info(code, collections)` |  |
| f | `GQ_fetch_stock_name(code, collections)` | 获取股票名称 |
| f | `GQ_fetch_index_name(code, collections)` | 获取股票名称 |
| f | `GQ_fetch_etf_name(code, collections)` | 获取ETF名称 |
| f | `get_codelist(codepool)` | 将各种‘随意’写法的A股股票代码列表，转换为6位数字list标准规格， |
| f | `stock_cn_blacklist()` | 股票黑名单，这些票没有当前行情数据，容错会影响策略计算速度 |
| f | `GQ_util_firstDayTrading(codelist)` | explanation: |
| f | `GQ_fetch_stock_list()` |  |
| f | `GQ_fetch_etf_list()` |  |

### `tools`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `purge_historical_collections(client)` | 清理历史数据集合 |

### `utils`

*(无公开成员)*

## GolemQ.markets.StockCN.datasource

### `akshare_source`
akshare 数据源适配器。供 etf_list 与 financial。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `AkshareSource` |  |
| f | `AkshareSource.available()` |  |
| f | `AkshareSource.fetch(collection)` |  |
| f | `AkshareSource.fetch_etf_list(verbose)` | 新浪 ETF 分类快照 → GolemQ 口径的行。不落库。 |
| f | `AkshareSource.fetch_financial(codelist, start_year, verbose)` | 季频财务指标（stock_financial_analysis_indicator）→ 宽表行。 |

### `baostock_source`
baostock 数据源适配器 —— 骨架，当前不可用。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `BaostockSource` |  |
| f | `BaostockSource.available()` | 当前恒 False。 |
| f | `BaostockSource.unavailable_reason()` |  |
| f | `BaostockSource.fetch(collection)` |  |

### `eastmoney_source`
东方财富数据源适配器 —— 骨架，尚未实现。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `EastmoneySource` |  |
| f | `EastmoneySource.available()` | 网络可达性与接口实现是两回事。 |
| f | `EastmoneySource.unavailable_reason()` |  |
| f | `EastmoneySource.fetch(collection)` |  |

### `pytdx_source`
pytdx 数据源适配器 —— 通达信协议的社区实现。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TdxSource` |  |
| f | `TdxSource.available()` | pytdx 可导入且有候选服务器即算可用。不在此处做网络探测 —— |
| f | `TdxSource.host()` |  |
| f | `TdxSource.close()` |  |
| f | `TdxSource.fetch(collection)` |  |
| f | `TdxSource.fetch_stock_list(markets, verbose)` | 全市场股票列表，已过滤为真实股票（排除基金/债券/指数）。 |
| f | `TdxSource.fetch_stock_info(codelist, verbose)` | 股本与上市信息。字段名与目标 schema 逐字对齐。 |
| f | `TdxSource.fetch_stock_block(verbose)` | 板块成分。四个文件分别对应概念/行业/指数/风格。 |

### `qmt_source`
MiniQMT（xtquant）数据源适配器。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_qmt_resolve_xt_code(code, prefer_index)` | '600000' → '600000.SH'；'sh.600000' / 已带后缀则归一。 |
| f | `GQ_qmt_to_qa_code(xt_code)` | '600000.SH' → '600000'（GolemQ 的 6 位 code）。 |
| f | `GQ_qmt_sse(code)` | 小写交易所后缀 'sh'/'sz'/'bj'。 |
| f | `GQ_qmt_decimal_point(detail)` | 由 PriceTick 推小数位（A 股 = 2）。 |
| f | `GQ_qmt_sector_members(sector_names)` | 板块成分并集 → 6 位 code 列表（需已 download_sector_data）。 |
| f | `GQ_qmt_stock_codes()` | 沪深A股+京市A股 里确实是 A 股的 code。 |
| C | `QmtSource` |  |
| f | `QmtSource.available()` | 只判断包是否可导入。不做在线探测 —— 那要连客户端，成本高且 |
| f | `QmtSource.fetch(collection)` |  |
| f | `QmtSource.fetch_stock_list(verbose)` | 全市场 A 股列表。字段与 4.4 quantaxis.stock_list 逐一对齐。 |
| f | `QmtSource.fetch_stock_info(verbose)` | 股本与上市信息。liutongguben/zongguben 是唯一被真实消费的字段， |
| f | `QmtSource.fetch_stock_block(include_market_sectors, verbose)` | QMT 板块成分。容器板块按 SECTOR_SKIP 排除。 |

### `tdxaidata_source`
tdxaidata（通达系官方数据源）适配器 —— 已打通，可提供全部 5 个集合。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TdxAiDataSource` |  |
| f | `TdxAiDataSource.available()` | 包在 + 配了 token。不在此处做镜像或联网 —— 那是 fetch 的职责。 |
| f | `TdxAiDataSource.unavailable_reason()` |  |
| f | `TdxAiDataSource.fetch(collection)` |  |
| f | `TdxAiDataSource.fetch_stock_list(verbose)` | 全市场 A 股。沪深A股(5226) + 北交所(348)，实测合计 5574。 |
| f | `TdxAiDataSource.fetch_stock_info(codelist, verbose)` | 股本与基础信息。get_gb_info 给 Ltgb(流通)/Zgb(总股本)。 |
| f | `TdxAiDataSource.fetch_stock_block(verbose)` | 板块成分。get_sector_list 给 560 个板块，逐个取成分。 |
| f | `TdxAiDataSource.fetch_financial(codelist, start_time, end_time, verbose)` | 季频财务。必须传 field_list，否则库报 [错误] field_list 不能为空。 |
| f | `TdxAiDataSource.fetch_etf_list(verbose)` | 跟踪指数的 ETF 列表（get_trackzs_etf_info）。 |

### `tencent_source`
腾讯行情（easyquotation）适配器。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TencentSource` |  |
| f | `TencentSource.available()` |  |
| f | `TencentSource.fetch(collection)` |  |
| f | `TencentSource.fetch_stock_list(codelist, verbose)` | 降级形态的 stock_list。codelist 省略则用 pytdx 的全市场名单做输入。 |

### `tushare_source`
tushare 数据源适配器 —— 骨架，缺 token 未启用。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TushareSource` |  |
| f | `TushareSource.token()` |  |
| f | `TushareSource.available()` | 包在 + 有 token 才算可用。缺任一项返回 False，不抛异常。 |
| f | `TushareSource.unavailable_reason()` |  |
| f | `TushareSource.fetch(collection)` |  |

## GolemQ.models

### `alias`
Model alias constants (stub — to be populated).

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `LTT` | LTT alias constants (stub). |

### `massive`
Massive model constants (stub — to be populated).

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `MAS` | Massive model feature constants (stub). |

### `poolcoef`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `calc_4Quad_push_credit(features)` | Calculate quadrant push credit scores. |

### `risk`
Risk model constants (stub — to be populated).

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `RSK` | Risk model feature constants (stub). |

## GolemQ.pipeline

### `base`
Benchmark 基类

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `BaseBenchmark` | Benchmark 基类 |
| f | `BaseBenchmark.calculate(code)` | 计算核心指标（抽象方法，子类必须实现） |
| f | `BaseBenchmark.calc_workload_agent(codelist, portfolio_batch, verbose, offset, massive_trend, massive_model_major, eval_range, pool_size, markup, ret_meta)` | 计算负载代理，统计计算成功率，统计计算时间等数据 |
| f | `BaseBenchmark.get_shared_memory_buffer(size)` | 获取共享内存缓冲区 |

### `compact_benchmark`
Compact Benchmark 子类

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `CompactBenchmark` | Compact Benchmark 实现 |
| f | `CompactBenchmark.calculate(code)` | 计算compact指标 |
| f | `calc_workload_agent()` | 兼容性函数，保持与原有API一致 |

### `mainstream_benchmark`
Mainstream Benchmark 子类

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `MainstreamBenchmark` | Mainstream Benchmark 实现 |
| f | `MainstreamBenchmark.calculate(code)` | 计算mainstream指标 |
| f | `calc_workload_agent()` | 兼容性函数，保持与原有API一致 |

### `poolcoef_benchmark`
Poolcoef Benchmark 子类

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `PoolcoefBenchmark` | Poolcoef Benchmark 实现 |
| f | `PoolcoefBenchmark.calculate(code)` | 计算poolcoef指标 |
| f | `calc_workload_agent()` | 兼容性函数，保持与原有API一致 |

## GolemQ.portfolio

### `costs`
交易成本模型 —— 中立的参数容器，不预设任何市场的费率。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `CostModel` | 一次回测的成本假设。 |
| f | `CostModel.buy_fee(turnover)` | 买入侧费用（佣金有下限、无印花税）。 `[dt]` |
| f | `CostModel.sell_fee(turnover)` | 卖出侧费用（佣金有下限 + 印花税 + 过户费）。 `[dt]` |
| f | `CostModel.fill_price(price, side)` | 滑点后的成交价。买入抬价、卖出压价 —— 方向不能反， `[dt]` |

### `engine`
回测引擎 —— 接口与输出契约已定，撮合实现待移植。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `BacktestResult` | 一次回测的产物。字段与落盘的两张 CSV 一一对应。 |
| f | `BacktestResult.to_csv(util_path, trades_path, encoding)` | 落盘。调用方负责保证路径不冲突（见模块文档的覆盖事故）。 |
| C | `BacktestEngine` | 回测引擎。 |
| f | `BacktestEngine.run(features_dummy)` | 跑一次回测。 |
| f | `make_ashare_engine(strategy, sizer, principal, benchmark)` | 便捷构造：用 A 股口径的成本与规则。 |

### `rules`
交易规则 —— 中立的参数容器，不预设任何市场的规则。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TradeRules` | 一次回测的交易规则假设。 |
| f | `TradeRules.can_sell(buy_date, current_date)` | 按 T+N 判断能否卖出。 `[dt]` |

### `sizing`
仓位分配 —— 决定「同时持几只」与「每只多少钱」。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `PositionSizer` | 仓位分配契约。 |
| f | `PositionSizer.target_slots(equity)` | 当前权益下最多同时持有几只。 |
| f | `PositionSizer.slot_notional(equity, n_slots)` | 单个槽位的目标金额。n_slots 个槽位合计即为目标总仓位。 |
| f | `PositionSizer.entry_fractions()` | 建仓批次。(1.0,) = 一次性；(0.5, 0.5) = 两批各半（梯形建仓）。 |
| C | `NotionalSlotSizer` | 每 notional 元权益 1 个槽位，下限 base。 参照实现的默认口径。 |
| f | `NotionalSlotSizer.target_slots(equity)` | 当前权益下的槽位数。 `[dt]` |
| f | `NotionalSlotSizer.slot_notional(equity, n_slots)` | 单槽金额 = 权益 / 槽位数。 `[dt]` |
| f | `NotionalSlotSizer.entry_fractions()` |  |
| C | `EvenSizer` | 固定持仓只数，等权分配。 用于「不按权益缩放」的对照实验。 |
| f | `EvenSizer.target_slots(equity)` |  |
| f | `EvenSizer.slot_notional(equity, n_slots)` |  |
| f | `EvenSizer.entry_fractions()` |  |

### `strategy`
策略接口 —— 仓位优化工具与具体策略之间的唯一契约。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `StrategyError` | 策略实现不满足契约时抛出（如信号形状与 features_dummy 不符）。 |
| C | `Strategy` | 仓位优化的策略侧契约。 |
| f | `Strategy.hold_signal(features_dummy)` | 返回「该 bar 该标的是否应持有」的布尔序列。 |
| f | `Strategy.priority(features_dummy)` | 返回同 bar 候选之间的抢占资金优先级，值大者优先。 |
| f | `Strategy.signals_for(features_dummy)` | 校验并返回 (hold, priority)。引擎只应经此调用策略。 `[dt]` |

## GolemQ.services

### `align`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `save_symbol_checkpoint_log(log_context, catalog, collections)` | 保存当天股票走势分组聚类 |
| f | `symbol_checkpoint_log(logs, symbol, FrozenExpired, length, missing_kline_index, frequency, catalog, collections)` |  |
| f | `GQ_fetch_checkpoint_symbols(FrozenExpired, collections)` | 保存当天股票走势分组聚类 |
| f | `calc_stock_hourly_kline_align(code, freq, market_type, collections)` |  |
| f | `calc_stock_metadata_missing_queries(features, verbose, collections)` |  |
| f | `kline_missing_checkpoints(features, ohlc_data, checkpoint_remark, verbose)` | 检查 kline 是否缺失，CLOSE 字段 是否为np.nan |

### `features`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_save_metadata_reality(features, collections)` | 保存同花顺的牛股诊股 |
| f | `GQ_save_daily_metadata_reality(features, collections)` | 保存同花顺的牛股诊股 |
| f | `GQ_fetch_daily_metadata_reality(code, start, end, collections)` | '获取 A股 复盘数据'， |
| f | `GQ_fetch_hourly_metadata_reality(code, start, end, verbose, collections)` | '获取 A股 复盘数据'， |
| f | `GQ_fix_daily_metadata(code, start, end, collections)` | 获取并修正date字段错误。 |
| f | `GQ_update_daily_metadata(features, collections)` |  |
| f | `GQ_remove_daily_metadata(features, collections)` |  |
| f | `GQ_update_hourly_metadata(features, collections)` |  |
| f | `GQ_remove_hourly_metadata(features, collections)` |  |
| f | `GQ_move_hourly_metadata(features, verbose, collections, collections_to)` | save current day's stock_min data |
| f | `GQ_save_stock_valuation(stock_valuation_detail, collections, verbose)` | 保存股价估值信息等数据 |

### `iwencai`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `iwencai` |  |

## GolemQ.services.features

### `_daily_crud`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_fix_daily_metadata(code, start, end, collections)` | 获取并修正date字段错误。 |
| f | `GQ_update_daily_metadata(features, collections)` |  |
| f | `GQ_remove_daily_metadata(features, collections)` |  |

### `_daily_fetch`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_fetch_daily_metadata_reality(code, start, end, collections)` | '获取 A股 复盘数据'， |

### `_daily_save`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_save_daily_metadata_reality(features, collections)` | 保存同花顺的牛股诊股 |

### `_hourly_fetch`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_fetch_hourly_metadata_reality(code, start, end, verbose, collections)` | '获取 A股 复盘数据'， |

### `_reality_save`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_save_metadata_reality(features, collections)` | 保存同花顺的牛股诊股 |

### `_valuation`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_save_stock_valuation(stock_valuation_detail, collections, verbose)` | 保存股价估值信息等数据 |

## GolemQ.services.persistence

### `_concept`
Concept and massive data persistence checks.

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `dataloader_concept_check(symbol, offset, market_type, verbose, peek_column, debug)` | 持久化指标数据加载和缺漏检查 |
| f | `dataloader_massive_check(revision, eval_range, offset, market_type, peek_column, verbose, collections)` | 持久化指标数据加载和缺漏检查 |

### `_daily`
Daily persistence check for stock reality features.

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `dataloader_persistence_daily_check(symbol, offset, market_type, verbose, peek_column, debug, collections)` | 持久化指标数据加载和缺漏检查 |

### `_review`
Stock review data persistence check.

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `dataloader_review_check_reflush(persistence_ratio, persistence_features_hourly, hour_baseline)` |  |
| f | `dataloader_review_check(symbol, features, offset, market_type, verbose, peek_column, collections)` | 持久化指标数据加载和缺漏检查 |

### `_schema`
Persistence column schemas and helper functions.

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `concept_review_columns_of_persistence()` |  |
| f | `stock_review_columns_of_persistence()` |  |
| f | `massive_review_columns_of_persistence()` |  |
| f | `reality_columns_of_persistence()` |  |
| f | `daily_columns_of_persistence()` |  |
| f | `calc_masked_tail_missing_index(unmasked_missing_index, baseline, persistence_ratio)` | 计算标的K线近端缺失的特征索引 |
| f | `features_reasonableness_checks(features)` |  |

### `_stock`
Stock persistence check for hourly/daily feature completeness.

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `dataloader_persistence_check(symbol, offset, market_type, verbose, peek_column, debug)` | 持久化指标数据加载和缺漏检查 |

## GolemQ.supervisor

### `function_checkin`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `get_gateway()` | 获取默认网关地址 |
| f | `get_optimized_caller_ip()` | 优化的IP获取方案，优先零延迟方法，备用快速国内服务 |
| C | `FunctionCheckinManager` | 函数调用频率控制管理器 |
| f | `FunctionCheckinManager.expired_time(expired_time)` | 计算过期时间戳 |
| f | `FunctionCheckinManager.checkin_function(function_name, expired_time)` | 函数调用签到 |
| f | `FunctionCheckinManager.complete_function_call(function_name, caller_ip)` | 标记函数调用完成（减少并发计数） |
| f | `FunctionCheckinManager.get_function_stats(function_name, caller_ip)` | 获取函数的调用统计信息 |
| f | `FunctionCheckinManager.get_all_function_stats()` | 获取所有函数的调用统计信息 |
| f | `FunctionCheckinManager.reset_function_stats(function_name, caller_ip)` | 重置函数的调用统计 |
| f | `FunctionCheckinManager.archive_old_records(days)` | 归档旧的调用记录 |
| f | `FunctionCheckinManager.cleanup_archive(days)` | 清理旧的归档记录 |
| f | `FunctionCheckinManager.get_concurrent_count(function_name, caller_ip)` | 获取当前的并发调用数 |
| f | `FunctionCheckinManager.get_call_frequency(function_name, caller_ip)` | 计算当前的调用频率（次/分钟） |
| f | `FunctionCheckinManager.is_function_expired(function_name, caller_ip)` | 检查函数调用是否已过期 |
| f | `checkin_function(function_name, expired_time)` | 使用全局函数管理器进行调用签到 |
| f | `complete_function_call(function_name, caller_ip)` | 使用全局函数管理器标记调用完成 |

### `heartbeat`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `HeartbeatMonitor` | 功能模块心跳监控器 |
| f | `HeartbeatMonitor.start_module(module_name, instance_id, timeout_seconds, initial_message)` | 开始一个模块的执行记录 |
| f | `HeartbeatMonitor.checkin(module_name, instance_id, message)` | 模块签到（心跳） |
| f | `HeartbeatMonitor.complete_module(module_name, instance_id, exit_code, completion_message)` | 标记模块执行完成 |
| f | `HeartbeatMonitor.start_monitoring()` | 启动监控线程 |
| f | `HeartbeatMonitor.stop_monitoring()` | 停止监控线程 |
| f | `HeartbeatMonitor.get_running_modules()` | 获取所有运行中的模块 |
| f | `HeartbeatMonitor.get_all_modules()` | 获取所有模块状态（包括运行中、已完成、超时、错误的模块） |
| f | `HeartbeatMonitor.get_module_history(module_name, limit)` | 获取模块执行历史 |
| f | `HeartbeatMonitor.get_module_status(module_name, instance_id)` | 获取特定模块的状态 |
| f | `HeartbeatMonitor.cleanup_old_records(days)` | 清理旧的归档记录 |
| f | `HeartbeatMonitor.update_module_status(module_name, instance_id, status, message)` | 更新模块状态 |
| f | `HeartbeatMonitor.stop_all_monitoring()` | 停止所有监控并清理资源，将所有需要归档的模块（包括运行中和已停止的）转移到归档库 |
| C | `HeartbeatModule` | 功能模块心跳监控器 |
| f | `HeartbeatModule.mutex(verbose)` |  |
| f | `HeartbeatModule.start(initial_message)` | 开始一个模块的执行记录 |
| f | `HeartbeatModule.complete(completion_message, exit_code)` | 标记模块执行完成 |
| f | `HeartbeatModule.checkin(message)` | 模块签到（心跳） |

### `messenger`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `MessageLevel` | 消息级别 |
| C | `Messenger` | 消息推送器 |
| f | `Messenger.configure_dingtalk(webhook_url)` | 配置钉钉机器人webhook |
| f | `Messenger.configure_serverchan(sendkey)` | 配置Server酱sendkey |
| f | `Messenger.configure_custom_webhook(name, webhook_url)` | 配置自定义webhook |
| f | `Messenger.send_alert(title, message, level)` | 发送告警消息 |
| f | `Messenger.send_test_message()` | 发送测试消息到所有配置的渠道 |
| f | `configure_global_messenger(dingtalk_webhook, serverchan_key)` | 配置全局消息推送器 |
| f | `send_alert(title, message, level)` | 使用全局消息推送器发送告警 |

### `scheduler`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TradingTimeChecker` | 交易时间检查器 |
| f | `TradingTimeChecker.is_trading_time()` | 检查当前是否为交易时间 |
| C | `XtquantSyncScheduler` | XTQuant 同步调度器 |
| f | `XtquantSyncScheduler.start_scheduling()` | 启动定时调度 |
| f | `XtquantSyncScheduler.stop_scheduling()` | 停止定时调度 |
| f | `start_xtquant_sync_scheduler()` | 启动XTQuant同步调度器 |
| f | `stop_xtquant_sync_scheduler()` | 停止XTQuant同步调度器 |

## GolemQ.test_cases

### `check_heartbeat`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `check_heartbeat_records()` | 检查心跳记录 |

### `check_running_modules`
检查当前运行中的模块

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `check_running_modules()` | 检查当前运行中的模块 |

### `cleanup_timeout_instance`
清理超时的xtquant_sync_loop实例

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `cleanup_timeout_instance()` | 清理超时的xtquant_sync_loop实例 |

### `debug_instance_id`
调试实例ID问题

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `debug_instance_id()` | 调试实例ID问题 |

### `direct_cleanup`
直接清理超时的xtquant_sync_loop实例

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `direct_cleanup()` | 直接清理超时的xtquant_sync_loop实例 |

### `final_cleanup`
最终清理超时的xtquant_sync_loop实例

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `final_cleanup()` | 最终清理超时的xtquant_sync_loop实例 |

### `run_tests`
GolemQ 测试运行脚本

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `run_tests()` | 运行所有测试 |

### `test_available_volume`
测试可用数量显示功能

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `test_available_volume_display()` | 测试可用数量显示功能 |

### `test_chip_distribution`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestChipDistributionPerformance` | 性能基准（原为命令行脚本，已纳入 unittest 以便记录依赖缺口） |
| f | `TestChipDistributionPerformance.test_performance()` | 测试优化前后的性能对比 |
| f | `test_performance()` | 命令行直接运行入口（保持原脚本用法） |

### `test_cli_tools`
测试 CLI 工具功能

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestCLITools` | 测试 CLI 工具功能 |
| f | `TestCLITools.setUp()` | 测试前备份 GQMARKETS 状态 |
| f | `TestCLITools.tearDown()` | 测试后恢复 GQMARKETS 状态 |
| f | `TestCLITools.test_auto_register_markets_skips_registered()` | 测试自动注册跳过已注册的市场 |
| f | `TestCLITools.test_auto_register_markets_skips_abstract_classes()` | 测试自动注册跳过抽象类 |
| f | `TestCLITools.test_purge_mongodb_database_verbose(mock_print)` | 测试数据库清理功能（详细模式） |
| f | `TestCLITools.test_purge_mongodb_database_silent(mock_print)` | 测试数据库清理功能（静默模式） |
| f | `TestCLITools.test_auto_register_markets_with_mock_module()` | 测试自动注册处理异常情况 |
| f | `TestCLITools.test_auto_register_markets_with_invalid_module()` | 测试自动注册处理无效模块 |

### `test_doctests`
把 docstring 里的 doctest 纳入 unittest 发现。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `load_tests(loader, tests, ignore)` | unittest 的收集协议：把各模块的 doctest 挂进本次发现。 |

### `test_export_positions`
测试导出持仓数据功能

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `test_data_conversion()` | 测试数据转换功能 |
| f | `test_cli_help()` | 测试CLI帮助信息 |

### `test_heartbeat_fix`
测试HeartbeatMonitor超时实例清理修复

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `test_timeout_instance_cleanup()` | 测试超时实例清理功能 |

### `test_market_align`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestMarketAlign` |  |
| f | `TestMarketAlign.setUp()` |  |
| f | `TestMarketAlign.test_check_etf_data_freshness(mock_get_client, mock_datetime)` | 测试ETF数据新鲜度检查功能 |
| f | `TestMarketAlign.test_check_etf_data_freshness_no_data(mock_get_client, mock_datetime)` | 测试没有ETF数据时的新鲜度检查 |

### `test_market_crawler`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestMarketCrawler` |  |
| f | `TestMarketCrawler.setUp()` |  |
| f | `TestMarketCrawler.test_get_all_etf_list_with_existing_files(mock_read_excel, mock_listdir, mock_exists)` | 测试获取ETF列表功能（有现有文件的情况） |
| f | `TestMarketCrawler.test_get_all_etf_list_no_directory(mock_exists)` | 测试获取ETF列表功能（目录不存在的情况） |
| f | `TestMarketCrawler.test_get_all_etf_list_empty_directory(mock_listdir, mock_exists)` | 测试获取ETF列表功能（空目录的情况） |

### `test_market_quotes`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestMarketQuotes` |  |
| f | `TestMarketQuotes.setUp()` |  |
| f | `TestMarketQuotes.test_get_kline_quotes_returns_dataframe(mock_fetch)` | get_kline_quotes should return a pd.DataFrame with expected columns |
| f | `TestMarketQuotes.test_get_kline_quotes_calls_fetch_with_correct_params(mock_fetch)` | get_kline_quotes should normalize code and call QA_fetch_stock_day_adv |
| f | `TestMarketQuotes.test_get_kline_quotes_applies_qfq_when_fq_set(mock_fetch)` | When fq=1, to_qfq() should be called |
| f | `TestMarketQuotes.test_get_kline_quotes_skips_qfq_when_fq_zero(mock_fetch)` | When fq=0, to_qfq() should NOT be called |
| f | `TestMarketQuotes.test_get_kline_quotes_returns_empty_on_none_data(mock_fetch)` | If QA_fetch_stock_day_adv returns None, return empty DataFrame |
| f | `TestMarketQuotes.test_get_kline_quotes_min_returns_dataframe(mock_fetch)` | get_kline_quotes_min should return a pd.DataFrame |
| f | `TestMarketQuotes.test_get_kline_quotes_min_normalizes_frequency(mock_fetch)` | Frequency aliases like '60m' should be normalized to '60min' |
| f | `TestMarketQuotes.test_get_kline_quotes_min_returns_empty_on_none(mock_fetch)` | If GQ_fetch_stock_min_adv returns None, return empty DataFrame |
| f | `TestMarketQuotes.test_get_kline_quotes_min_applies_qfq_when_fq_set(mock_fetch)` | When fq=1, to_qfq() should be called for minute klines |
| f | `TestMarketQuotes.test_get_kline_quotes_min_skips_qfq_when_fq_zero(mock_fetch)` | When fq=0, to_qfq() should NOT be called for minute klines |
| f | `TestMarketQuotes.test_get_kline_quotes_default_params()` | Default start/end dates and fq should work without error when data is available |
| f | `TestMarketQuotes.test_get_kline_quotes_min_default_params()` | Default params for minute kline should be correct |

### `test_market_tools`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestMarketTools` |  |
| f | `TestMarketTools.test_purge_historical_collections(mock_get_client)` | 测试清理历史数据集合功能 |
| f | `TestMarketTools.test_purge_historical_collections_no_collections(mock_get_client)` | 测试没有历史数据集合时的清理功能 |

### `test_messenger`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestDingtalkConfig` |  |
| f | `TestDingtalkConfig.test_check_config_success(mock_get)` |  |
| f | `TestDingtalkConfig.test_check_config_failure(mock_get)` |  |
| f | `TestDingtalkConfig.test_setup_config(mock_input, mock_set)` |  |
| f | `TestDingtalkConfig.test_get_config(mock_get, mock_setup, mock_check)` |  |
| C | `TestDingtalkAccessToken` |  |
| f | `TestDingtalkAccessToken.test_get_access_token(mock_config, mock_client)` |  |
| f | `TestDingtalkAccessToken.test_get_access_token_failure(mock_config, mock_client)` |  |
| C | `TestDingReminder` |  |
| f | `TestDingReminder.test_send_message(mock_config, mock_client, mock_token)` |  |
| f | `TestDingReminder.test_send_message_markdown_content(mock_config, mock_token)` |  |
| f | `TestDingReminder.test_send_message_no_token(mock_config, mock_token)` |  |

### `test_stock_cn`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestStockCN` |  |
| f | `TestStockCN.setUpClass()` | 测试类初始化 |
| f | `TestStockCN.test_get_all_etf_list()` | 测试get_all_etf_list方法 |
| f | `TestStockCN.test_realtime_data_fetch()` | 测试实时数据获取功能 |

### `test_stockcn_singleton`
测试 StockCN 单例自动注册功能

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestStockCNSingleton` | 测试 StockCN 单例自动注册功能 |
| f | `TestStockCNSingleton.setUp()` | 测试前备份 GQMARKETS 状态 |
| f | `TestStockCNSingleton.tearDown()` | 测试后恢复 GQMARKETS 状态 |
| f | `TestStockCNSingleton.test_singleton_auto_registration()` | 测试 StockCN 模块导入时自动注册 |
| f | `TestStockCNSingleton.test_singleton_property()` | 测试 StockCN 单例特性 |
| f | `TestStockCNSingleton.test_registry_consistency()` | 测试注册表与实例的一致性 |
| f | `TestStockCNSingleton.test_cli_auto_register_coordination()` | 测试 CLI 自动注册与模块自动注册的协调工作 |
| f | `TestStockCNSingleton.test_market_name_property()` | 测试市场名称属性 |
| f | `TestStockCNSingleton.test_exchange_codes_property()` | 测试交易所代码属性 |

### `test_timestamp_format`
测试时间戳格式

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `test_timestamp_format()` | 测试时间戳格式转换 |

### `test_xtquant_sync`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `test_trading_time_checker()` | 测试交易时间检查器 |
| f | `test_scheduler_initialization()` | 测试调度器初始化 |
| f | `test_heartbeat_integration()` | 测试心跳监控集成 |

### `test_xtquant_sync_simple`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `test_trading_time_checker()` | 测试交易时间检查器 |
| f | `test_scheduler_initialization()` | 测试调度器初始化 |
| f | `test_heartbeat_integration()` | 测试心跳监控集成 |
| f | `test_messenger_integration()` | 测试消息通知集成 |

### `verify_heartbeat`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `test_heartbeat()` |  |

## GolemQ.test_cases.xtquant

### `xtquant_01_hello_quant`

*(无公开成员)*

### `xtquant_02_自动逆回购`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `A` |  |
| f | `price_compare(list2)` | 参数为两个股票代码，如'204001.SH'， |
| f | `get_last_price(stock_code)` |  |
| f | `reverse_repos(xt_trader, acc, symbol)` |  |
| C | `MyXtQuantTraderCallback` |  |
| f | `MyXtQuantTraderCallback.on_disconnected()` | 连接断开 |
| f | `MyXtQuantTraderCallback.on_stock_order(order)` | 委托回报推送 |
| f | `MyXtQuantTraderCallback.on_stock_trade(trade)` | 成交变动推送 |
| f | `MyXtQuantTraderCallback.on_order_error(order_error)` | 委托失败推送 |
| f | `MyXtQuantTraderCallback.on_cancel_error(cancel_error)` | 撤单失败推送 |
| f | `MyXtQuantTraderCallback.on_order_stock_async_response(response)` | 异步下单回报推送 |
| f | `MyXtQuantTraderCallback.on_cancel_order_stock_async_response(response)` | :param response: XtCancelOrderResponse 对象 |
| f | `MyXtQuantTraderCallback.on_account_status(status)` | :param response: XtAccountStatus 对象 |

### `xtquant_03_趋势网格策略`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TrendGridStrategy` | 趋势网格做多策略 |
| f | `TrendGridStrategy.is_trend_up(stock_code)` | 判断股票是否处于多头趋势 |
| f | `TrendGridStrategy.get_15min_ma30(stock_code)` | 获取15分钟线的MA30 |
| f | `TrendGridStrategy.get_current_price(stock_code)` | 获取当前价格 |
| f | `TrendGridStrategy.initialize_grid(stock_code, current_price)` | 初始化网格 |
| f | `TrendGridStrategy.calculate_grid_level(stock_code, current_price)` | 计算当前网格层级 |
| f | `TrendGridStrategy.should_buy(stock_code)` | 判断是否应该加仓 |
| f | `TrendGridStrategy.should_sell(stock_code)` | 判断是否应该减仓 |
| f | `TrendGridStrategy.get_position_quantity(stock_code)` | 获取持仓数量 |
| f | `TrendGridStrategy.execute_buy(stock_code)` | 执行买入操作 |
| f | `TrendGridStrategy.execute_sell(stock_code)` | 执行卖出操作（减仓但不清仓） |
| f | `TrendGridStrategy.run_strategy(stock_codes, check_interval)` | 运行策略主循环 |
| C | `MyXtQuantTraderCallback` | 交易回调类 |
| f | `MyXtQuantTraderCallback.on_disconnected()` | 连接断开回调 |
| f | `MyXtQuantTraderCallback.on_stock_order(order)` | 委托回报推送 |
| f | `MyXtQuantTraderCallback.on_stock_trade(trade)` | 成交变动推送 |
| f | `MyXtQuantTraderCallback.on_order_error(order_error)` | 委托失败推送 |
| f | `MyXtQuantTraderCallback.on_cancel_error(cancel_error)` | 撤单失败推送 |
| f | `MyXtQuantTraderCallback.on_order_stock_async_response(response)` | 异步下单回报推送 |
| f | `MyXtQuantTraderCallback.on_cancel_order_stock_async_response(response)` | 异步撤单回报推送 |
| f | `MyXtQuantTraderCallback.on_account_status(status)` | 账户状态回调 |

### `xtquant_03_趋势网格策略_test`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `MockTrendGridStrategy` | 模拟趋势网格策略用于测试 |
| f | `MockTrendGridStrategy.mock_is_trend_up(ma_values)` | 模拟趋势判断 |
| f | `MockTrendGridStrategy.initialize_grid(stock_code, current_price)` | 初始化网格 |
| f | `MockTrendGridStrategy.calculate_grid_level(stock_code, current_price)` | 计算网格层级 |
| f | `MockTrendGridStrategy.test_trend_conditions()` | 测试趋势判断条件 |
| f | `MockTrendGridStrategy.test_grid_logic()` | 测试网格逻辑 |
| f | `MockTrendGridStrategy.test_trading_conditions()` | 测试交易条件 |
