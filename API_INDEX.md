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

### `ChipDistribution_jit`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `ChipDistribution` | ChipDistribution class with Numba JIT optimized core functions |
| f | `ChipDistribution.get_data(code, data, offset, verbose)` |  |
| f | `ChipDistribution.calcuChip(flag, AC)` |  |
| f | `ChipDistribution.winner(p)` | 计算获利盘比例 - JIT优化版本 |
| f | `ChipDistribution.lwinner(N, p)` | 滑动窗口获利盘计算 - JIT优化版本 |
| f | `ChipDistribution.cost(N)` | 返回百分比的筹码价位 - JIT优化版本 |
| f | `ChipDistribution.calc_cost5_bootstrap(features, verbose)` | 计算筹码变化起终点 |
| f | `calc_stock_chip_distribution_jit(features, ohlc_data, annual, verbose)` | 使用 Numba JIT 优化的筹码分布计算函数 |

### `_zs`
缠论中枢识别的算法核心（find_zs）。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `find_zs(points)` | 输入笔或线段标记点，输出中枢识别结果。 `[dt]` |

### `peak`
鲁棒性极值点识别（PEAK_POINT）—— 基于方差统计与 ZSCORE 排序。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `peak_status()` | numba 在不在、jit 开没开（DISABLE_JIT 会关掉）。 |
| f | `thresholding_algo_py(y, lag, threshold, influence)` | 纯 numpy 参照实现（jit 版必须与它逐值一致）。 |
| f | `thresholding_algo(y, lag, threshold, influence)` | 鲁棒极值点识别（z-score）。返回 3×len(y) 的数组。 `[dt]` |
| f | `calc_peak_point_v8(ohlc_data, features)` | 鲁棒性极值点识别（基于方差统计与 ZSCORE 排序）—— 写 ST.PEAK_POINT。 `[dt]` |
| f | `calc_peak_points(closep, lineareg_price)` | 两个输入的极值合成：收盘价算一遍、拟合价算一遍，加权合成。 `[dt]` |

### `pivot`
缠中说禅 走势中枢（pivot，盘整箱体）识别与绘制

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `bi_confirm_map(bars, max_bi_count, verbose)` | 逐 bar 回放 CZSC，得到每根笔端点（bi.fx_b.dt）首次成为末笔的时间。 |
| f | `bi_list_to_points(bi_list, conf_map)` | 把 czsc 的笔列表（BI 对象）转成 find_zs 所需的标记点序列。 |
| f | `calc_pivots(bi_list, points, conf_map, strict)` | 计算缠论走势中枢（盘整中枢 / pivot）。 |
| f | `classify_pivots_per(pivots)` | 逐中枢判断走势类型——走到每个中枢结束时重新判一次（因果）。 |
| f | `classify_pivots(pivots)` | 判断中枢序列构成的走势类型：盘整 / 上涨趋势 / 下跌趋势。 `[dt]` |
| f | `pivots_to_df(pivots, kind)` | 把中枢列表转成 DataFrame，便于检查与导出。 `[dt]` |
| f | `causal_pivot_series(pivots, dts, strict)` | 把中枢字段展开为与 dts 对齐的因果（无未来函数）特征列。 `[dt]` |
| f | `plot_pivots(ax, pivots, x_mapper, x_last, box_color, box_alpha, edge_color, line_width, label, fontsize, strict)` | 在 K 线图上绘制中枢箱体（因果口径，无未来函数）。 |
| f | `plot_pivots_plotly(fig, pivots, x_last, row, col, box_color, edge_color, line_width, label, show_ggdd, strict)` | 在 Plotly K 线图上绘制中枢箱体（矩形 + GG/DD 因果台阶 + ZG/ZD 标注）。 |
| f | `attach_pivot_features(features, symbol, max_bi_count, strict)` | 把中枢字段（ZD/ZG/GG/DD/direction）作为列附加到特征 DataFrame 上。 |

### `regtree`
回归树（CART）与 regtree 拟合线 —— 从旧树 analysis/regtree.py 搬运。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `choose_best_split_branch(seq_data, rate, dur)` | 判断所有样本是否为同一分类 |
| f | `split_data_into_binary(seq_data, feature, value)` | 用feature把seq_data按value分成两个子集 |
| f | `solve_lineareg_func(seq_data)` | 求给定数据集的线性方程 |
| f | `fit_lineareg_slope_of_the_model_leaf(seq_data)` | 求线性方程的参数 |
| f | `calc_err_of_the_model(seq_data)` | 预测值和y的方差 |
| f | `tree_model_evaluation_func(model, branch_data)` | 预测评估函数,数据乘模型,模型是斜率和截距的矩阵 |
| f | `is_tree_py(obj)` | 用字典保存的二叉树结构 |
| f | `tree_model_forecast_decision_func_np(tree, branch_data, level)` | 预测/遍历整颗树的二元函数 |
| f | `create_whole_forecast_tree(tree, seq_data, directions)` | 对测试数据集预测一系列结果, 用于输出， |
| f | `predict_regression_tree_branch(seq_data, rate, dur, level, code)` | 生成回归树, seq_data是数据, rate是误差下降, dur是叶节点的最小样本数 |
| f | `fit_regtree_trend(tree, seq_data)` | 输出回归树预测 |
| f | `calc_regtree_fractal_func(data)` | 快速计算regtree拟合线，因为超过500bar计算速度会变得很慢（超过5秒），所以 bar_limit 默认限制为 300 |
| f | `calc_regtree_renko_fractal_func(data)` | 计算 regtree_renko 延长形态 |
| f | `calc_regtree_fractal_vXI(data)` | 快速计算regtree拟合线，因为超过500bar计算速度会变得很慢（超过5秒），所以 bar_limit 默认限制为 300 |
| f | `calc_regtree_fractal_vXIs(data)` | 快速计算regtree拟合线，因为超过500bar计算速度会变得很慢（超过5秒），所以 bar_limit 默认限制为 300 |

### `regtree_jit`
regtree 的 numba 加速版（analysis/regtree.py 的 jit 对照实现）。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `available()` | numba 在不在。 |
| f | `regtree_jit_status()` | 排障用：numba 版本 + jit 是否真的开了（DISABLE_JIT 环境变量会关掉它）。 |
| f | `choose_best_split_branch_jit(seq_data, rate, dur)` | regtree.choose_best_split_branch 的 jit 版；返回形状保持一致 |
| f | `predict_regression_tree_branch_jit(seq_data, rate, dur, level, code)` | 逐行照搬 regtree.predict_regression_tree_branch，只把切分搜索换成 jit 版。 |
| f | `tree_forecast_jit(tree, seq_data, directions)` | :func:regtree.create_whole_forecast_tree 的 jit 遍历版。 |
| f | `calc_regtree_fractal_jit(data)` | :func:regtree.calc_regtree_fractal_func 的 jit 建树版。 |

### `timeseries`
时间序列工具 —— 多频重采样与时间轴对齐。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_data_min_resample(min_data, type_)` | 分钟线 → 更大周期的分钟线（5min / 15min / 30min / 60min / 1D）。 |
| f | `GQ_data_min_to_day(min_data, type_)` | 分钟线 → 日线。QUANTAXIS QA_data_min_to_day 的忠实回迁。 |

### `timing`
时域累积器与金叉/死叉时间间隔。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `Timeline_Integral(Tm)` | 时域金叉/死叉信号的累积和（死叉 1→0 时清零）。 `[dt]` |
| f | `Timeline_duration(Tm)` | 时域累积和（金叉 0→1 时清零，与 :func:Timeline_Integral 相反）。 `[dt]` |
| f | `calc_event_timing_lag(vhma_directions)` | 事件的时间间隔：金叉取正、死叉取负。 `[dt]` |
| f | `calc_feature_event_timing_lag(features, column)` | 技术指标特征的金叉/死叉时序。 `[dt]` |
| f | `calc_energy_f8(signal)` | calc_energy 的 float64 内核：同号累加、异号重启。 `[dt]` |
| f | `calc_energy_f4(signal)` | calc_energy 的 float32 内核（逻辑同 :func:calc_energy_f8，只差 dtype）。 |
| f | `calc_energy(signal)` | 信号的绝对能量（同号连续累加、异号重启），按 dtype 选内核。 `[dt]` |
| f | `resample_multi_frequency_indices_func(data)` | 把另一个频率算好的指标对齐到 data 的时间轴上（单标的）。 |
| f | `rolling_sum(a, n)` | pandas.Series.rolling(n).sum() 的 numpy 版（前 n-1 个是 NaN）。 `[dt]` |
| f | `lineareg_intercept(slope1, y1, slope2, y2)` | 两条直线的交点横坐标：y = slope*x + y 两式相等解 x。 `[dt]` |

## GolemQ.cli

### `__main__`
GolemQ 命令行入口 —— 只剩「建 parser → 分发 → help」。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `build_parser()` | 建 parser：全局开关 + 每条命令自己的参数。 |
| f | `main()` | 主函数：处理命令行参数 |

### `bootstrap`
CLI 环境自检 —— 版权页 + 机器 / 配置 / 服务三道闸。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `check_python()` | (ok, 说明)。sys.version_info 前两位与本项目声明的最低版本比。 |
| f | `check_packages()` | 逐包检查，返回 [(ok, 说明), ...]。没装的也算不满足并说明。 |
| f | `check_tty()` | (是色终端吗, 说明)。 |
| f | `check_blas()` | (ok, 说明)：numpy 的 BLAS/OpenMP 线程数上限有没有压到 1。 |
| f | `check_os()` | (状态, 说明)：操作系统 + 位数。 |
| f | `check_cpu()` | (状态, 说明)：厂商 · 物理核/逻辑线程 · 有无超线程 · 是否混合大小核。 |
| f | `check_cuda()` | (状态, 说明)：有 N 卡就报卡型 + 驱动 + 驱动支持的 CUDA 版本。 |
| f | `check_source(name)` | (状态, 说明)：某个可选数据源能不能用（配了 token / 包在不在）。 |
| f | `check_xtquant()` | (状态, 说明)：[XTQUANT] 那一段配置齐不齐。 |
| f | `check_serverchan()` | (状态, 说明)：Server酱（推送告警渠道）配了没。 |
| f | `check_iwencai()` | (状态, 说明)：东方财富问财的配置。 |
| f | `check_calendar(calendar, today)` | (状态, 说明)：交易日历（TRADE_DATE_SSE）够不够用。 |
| f | `check_tz()` | (状态, 说明)：本机时区是不是北京时间（UTC+08:00）。 |
| f | `check_config()` | (ok, 说明)。查 ~/.GolemQ/settings/config.ini 在不在、能不能解析、 |
| f | `check_mongodb(uri)` | (ok, 说明)。连上并取 server_info() 判版本 —— 本项目只用 8.3。 |
| f | `print_copyright()` | 只打版权那一行（兼容保留；新版用 :func:print_identity）。 |
| f | `print_identity(when)` | 身份块 —— 产品名 + 本次运行的时刻戳 + 版权行 + 一个空行。 |
| f | `run_checks(packages)` | 跑完七个节点，只返回结果、不打印。 |
| f | `check_environment(verbose, strict)` | 本地段自检：画 banner + 报异常。返回「硬拦项是否全过」。 |
| f | `require_mongodb(verbose)` | 服务段自检：MongoDB 连接 + 8.3 版本。不过就退出 :data:EXIT_FAILURE。 |

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

## GolemQ.cli.commands

### `_registry`
CLI 命令注册表 —— 机制层，不认识任何具体命令。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `bind_parser(parser)` | build_parser 调一次。 |
| f | `usage_error(message, hint)` | 按 argparse 的形状报一个用法错误并退出 :data:EXIT_USAGE。 |
| C | `Command` | 一个 CLI 命令。字段含义见模块 docstring。 |
| f | `Command.matches(args)` |  |
| f | `register(commands)` | 按给定顺序登记。不要在别处 append COMMANDS。 |
| f | `no_arguments(parser)` | 给「不拥有任何参数」的命令用（参数归别的模块注册，见 __init__.py）。 |
| f | `pick(args)` | 按登记顺序找第一个命中的命令；都不命中 → None。 |
| f | `dispatch(args)` | pick + run 的便捷版。 |

### `heartbeat`
心跳监控：--heartbeat-watchdog（查看）与 --stop-heartbeat-monitor（停止）。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `add_watchdog_arguments(parser)` |  |
| f | `add_stop_arguments(parser)` |  |
| f | `run_watchdog(args)` |  |
| f | `run_stop(args)` |  |

### `migrate`
一次性搬运入口：从 4.4 取材落进 8.3。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `add_turnover_arguments(parser)` |  |
| f | `run_migrate_turnover(args)` | --migrate-turnover 的处理体：只做参数校验，干活在 metadata_save。 |

### `purge`
--purge-l1 / --purge：清理 MongoDB 数据库。破坏性，故有确认闸。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `add_purge_arguments(parser)` |  |
| f | `run_purge(args)` |  |

### `save`
--save <SOURCE>：参考数据 + K 线 + xdxr/adj 的主流水线（DECISIONS.md D11/D14/D15/D16）。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `add_save_arguments(parser)` |  |
| f | `run_save(args)` |  |

### `save_report`
--save 的只读报告：--save-coverage（K 线覆盖缺口）与 --save-status（库存量/源可用性）。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `add_coverage_arguments(parser)` |  |
| f | `add_status_arguments(parser)` |  |
| f | `run_coverage(args)` |  |
| f | `run_status(args)` |  |

### `setup`
--setup / --init 与三个单项初始化开关。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `add_setup_arguments(parser)` |  |
| f | `add_mongodb_arguments(parser)` |  |
| f | `add_dingtalk_arguments(parser)` |  |
| f | `add_serverchan_arguments(parser)` |  |
| f | `run_setup(args)` |  |
| f | `run_mongodb_init(args)` |  |
| f | `run_dingtalk_init(args)` |  |
| f | `run_serverchan_init(args)` |  |

### `subscribe`
--sub <KEY>：跑一个第三方行情订阅器。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `add_subscribe_arguments(parser)` |  |
| f | `run_subscribe(args)` |  |

### `tdx_hosts`
--update-tdx-hosts：每周一次的通达信服务器池刷新。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `add_update_arguments(parser)` |  |
| f | `run_update(args)` |  |

### `watchlist`
关注列表：--eneloop-add / --eneloop-remove / --eneloop-list（+ --symbols）。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `add_add_arguments(parser)` | --eneloop-add 与 --symbols 的注册点。 |
| f | `add_remove_arguments(parser)` |  |
| f | `add_list_arguments(parser)` |  |
| f | `run_add(args)` |  |
| f | `run_remove(args)` |  |
| f | `run_list(args)` |  |

## GolemQ.core

### `base`
根层的基础工具（与市场无关的那些）。

| | 名称 | 摘要 |
|:--|:--|:--|
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

### `gq_logging`
日志接口 —— 替代 QUANTAXIS 的 QA_util_log_info。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_util_log_info(logs, ui_log, ui_progress, ui_progress_int_value)` | INFO 级日志接口 —— 行为对齐 QUANTAXIS.QAUtil.QA_util_log_info。 `[dt]` |

### `market_registry`
市场注册表、「默认 / 当前激活市场」与市场类型的解析。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `register_market(name, instance, replace)` | 把市场实例登记进注册表。 `[dt]` |
| f | `register_subscriber(key, func, replace)` | 登记订阅者。replace 语义同 :func:register_market。 |
| f | `active_market_name()` | 当前激活的市场名。 |
| f | `set_active_market(name)` | 切换激活市场。 `[dt]` |
| f | `get_active_market()` | 返回当前激活的市场实例。 |
| f | `get_market(name)` | 按名取市场实例（不影响激活状态）。 |
| f | `get_default_market()` | 返回系统默认市场实例（:data:DEFAULT_MARKET 指的那个）。 `[dt]` |
| f | `resolve_market(market)` | 把 market 参数解析成市场实例。 `[dt]` |

### `migrate44`
一次性搬运用的 4.4 只读通道。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `mongo44_uri(uri, port)` | 8.3 的 uri → 4.4 的 uri（同主机，端口换成 port）。 `[dt]` |
| f | `client44(uri, port)` | 4.4 的 MongoClient。kwargs 直通 pymongo.MongoClient。 |
| f | `db44(name, uri, port)` | 4.4 上名为 name 的库句柄（如 'golemq' / 'quantaxis'）。 |

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
| f | `ansi_enabled(stream)` | 颜色/光标控制开关：只对真 TTY 且真的支持 ANSI 时打开。 `[dt]` |
| f | `display_width(text)` | 终端显示宽度 —— 中日韩字符占两列。只为把阶段名 / 组名对齐。 `[dt]` |
| f | `aligned_row(label, body, width)` | 阶段名 + 值 的一行，与 :func:render_pipeline_banner 的阶段列同宽。 `[dt]` |
| f | `stamp_text(when)` | 裸的时刻戳：[2026-10-09 15:34:43]。 `[dt]` |
| f | `stamp(text, when)` | 给静态头行盖时刻戳：[2026-10-09 15:34:43]: source: pytdx。 `[dt]` |
| f | `dim(text, color)` | 压暗一行（color=True 时）。用 :data:_ANSI_GRAY，非 TTY 原样返回。 |
| f | `stamp_done(caption, when, color)` | 收尾行：[2026-10-10 00:33:22]: bootstrap done.（color=True 时压暗）。 `[dt]` |
| f | `identity(app, contact, when, color)` | 身份块 —— 每次运行开头那一小块，末尾带一个空行。 |
| f | `render_pipeline_banner(rows, states, color)` | 把「各阶段」渲染成多行表头（纯函数）。 `[dt]` |
| C | `Banner` | 整条流水线的固定表头 + 表头之下的一块滚动状态窗，自己管光标行数原地重画。 |
| f | `Banner.states()` | 当前状态快照（排障/测试用）。 |
| f | `Banner.log()` | 攒下的全部状态行（close 后仍可读；测试与排障用）。 |
| f | `Banner.render()` | 首次落盘：先打一次静态头行，再画表头（全部未获取）。 |
| f | `Banner.mark(name, state)` | 把节点置为某个状态并重画。取值见 :data:_MARKS —— |
| f | `Banner.echo(text)` | 在状态窗里加一行 —— banner 活跃时，下方的一切输出都走这里。 |
| f | `Banner.close()` | 收尾：TTY 下换成「表头 + 全部状态行」，把滚动窗里被挤掉的补回来。 |

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
落库的机制层：三种写策略，按集合类型与文档语义选用。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `ensure_indexes(coll, unique_keys)` | 建唯一索引。已存在同键索引时不报错。 |
| f | `save_collection(coll, rows, unique_keys, delete_delta_key, batch, verbose)` | 把 rows upsert 进 coll，可选删 delta。 |
| f | `upsert_fields(coll, rows, keys, payload_keys, batch, verbose)` | 只覆盖指定字段的 upsert（$set）—— 给「多列共用一个文档」的表用。 `[dt]` |
| f | `save_block_collection(coll, rows, batch, verbose)` | stock_block 专用：键是 (blockname, code)，删除分两层。 |
| f | `save_bar_chunk(coll, docs)` | 时间序列集合专用写口：只 drop 掉本次要写的那几根，再插。 |
| f | `replace_code_rows(coll, docs)` | 按 code 整体替换：delete_many({'code': code}) → 插。 |

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
| f | `BaseMarket.get_kline_price_min(codelist, start, end, verbose, realtime, frequency)` | 分钟线。返回 (结果对象, codename)。 |
| f | `BaseMarket.get_kline_price_v3(codelist, start, end, verbose, realtime)` | 日线。返回 (结果对象 | None, codename)。 |
| f | `BaseMarket.get_stock_concept_kline(symbol, start, end, freq)` | 概念 K 线。未实现的市场应抛 NotImplementedError —— |
| f | `BaseMarket.name()` | 返回市场名称 |

## GolemQ.markets.StockCN

### `base`
StockCN base utilities.

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `resample_features_frequency(features, freq)` | Resample features to target frequency. |

### `constants`
StockCN specific constants

*(无公开成员)*

### `datastruct`
QUANTAXIS QA_DataStruct_* 的替身 —— 只覆盖实际被调用的接口。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `apply_qfq(data, verbose)` | 就地把 OHLC 乘上前复权因子，返回新的 DataFrame（不改入参）。 |
| C | `GQ_DataStruct` | K 线容器。data 是以 (时间, code) 为 MultiIndex 的 DataFrame。 |
| f | `GQ_DataStruct.new(data, type, if_fq)` |  |
| f | `GQ_DataStruct.index()` | 透出 data.index。 |
| f | `GQ_DataStruct.code()` |  |
| f | `GQ_DataStruct.date()` |  |
| f | `GQ_DataStruct.select_code(code)` | 取单只标的，返回同类型对象；标的不存在抛 ValueError。 |
| C | `GQ_DataStruct_Stock_day` | 个股日线。 |
| C | `GQ_DataStruct_Stock_min` | 个股分钟线。 |
| C | `GQ_DataStruct_ETF_day` | ETF 日线。有 to_qfq（走 etf_adj），与个股同接口。 |
| C | `GQ_DataStruct_ETF_min` | ETF 分钟线。有 to_qfq（走 etf_adj）。 |
| C | `GQ_DataStruct_Index_day` | 真指数日线。没有 to_qfq —— 与 QUANTAXIS 一致。 |
| C | `GQ_DataStruct_Index_min` | 真指数分钟线。没有 to_qfq —— 与 QUANTAXIS 一致。 |
| C | `GQ_DataStruct_Stock_block` | 板块成分容器 —— QUANTAXIS QA_DataStruct_Stock_block 的替身。 |
| f | `GQ_DataStruct_Stock_block.new(data)` |  |
| f | `GQ_DataStruct_Stock_block.block_name()` | 全部板块名，排序后返回（QUANTAXIS 的 index.levels[0] 本就是有序的）。 |
| f | `GQ_DataStruct_Stock_block.code()` | 成分代码去重后排序返回。 |
| f | `GQ_DataStruct_Stock_block.get_block(blockname)` | 取若干板块的全部成分。blockname 是单个名字或名字的可迭代对象。 |
| f | `GQ_DataStruct_Stock_block.get_blocklist(blockname)` | get_block 的别名 —— QUANTAXIS 无此方法，见类 docstring 分歧 ②。 |
| f | `frame_to_datastruct(df, market, frequency)` | 8.3 读取器产出的扁平表 → 对应的容器类型。 |

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
| f | `GQ_util_timestamp_to_str(ts_epoch, local_tz)` | 时间戳 → '%Y-%m-%d %H:%M:%S' 字符串（默认 UTC+8）。 `[dt]` |
| f | `GQ_util_get_pre_trade_date(cursor_date, n)` | 前 n 个交易日。行为对齐 QUANTAXIS.QAUtil.QADate_trade.QA_util_get_pre_trade_date。 `[dt]` |

### `etf_fq`
ETF 复权（除权）读取侧支持。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_is_etf(code)` | 判断 6 位代码是否为场内 ETF。 |
| f | `GQ_fetch_etf_adj(codelist, start, end, adj_collection)` | 读取逐日复权系数，返回 {code: {date: adj}}。 |
| f | `GQ_apply_etf_qfq(data_day, codelist, verbose)` | 把 ETF 的 OHLC 换成前复权价，并原样返回同一个对象。 |

### `fetch`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_fetch_stock_min(code, start, end, format, frequence, market_type)` | A 股分钟线（MongoDB 8.3 时序库）。返回以 datetime 为索引的扁平帧。 |
| f | `GQ_fetch_stock_min_adv(code, start, end, frequence, if_drop_index, verbose)` | '获取股票分钟线' |
| f | `GQ_fetch_index_min_adv(code, start, end, frequence, if_drop_index, verbose)` | 指数 / ETF 分钟线。与 GQ_fetch_stock_min_adv 是同一条路径。 |

### `fq`
复权的纯函数核心 —— 股票与 ETF 共用。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `index_field(data, name)` | 从 (Multi)Index 里取某个 level 作为 Series；没有则返回 None。 `[dt]` |
| f | `row_dates(data)` | 每行的所属日期（'YYYY-MM-DD' 字符串），找不到返回 None。 `[dt]` |
| f | `row_codes(data)` | 每行的 6 位代码；索引的 code 层优先，其次 code 列，都没有则 None。 `[dt]` |
| f | `flatten_factor_map(factor_map)` | {code: {date: adj}} → {code|date: adj}，供 :func:align_factors。 `[dt]` |
| f | `factor_dict_from_frame(df, code_col, date_col, value_col)` | 因子 DataFrame → {code|date: adj}。 `[dt]` |
| f | `align_factors(data, flat)` | 把因子对齐到 data 的每一行，返回与 data.index 同索引的 Series。 `[dt]` |
| f | `multiply_ohlc(data, factor)` | 把 OHLC 乘上因子，返回新帧（不改入参）。 `[dt]` |
| f | `xdxr_to_adj(dates, close, xdxr)` | 日线 + 除权除息事件 → 前复权因子（stock_adj 的 adj 列）。 `[dt]` |

### `kline83`
A 股 K 线读取器 —— 走 MongoDB 8.3 时序库（golemq_stock_cn）。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `KlineResult` | 与 QUANTAXIS QA_DataStruct_* 对齐的最小接口 —— 调用方只取 .data。 |
| f | `bj_date(x)` | 北京时间的裸值 → UTC-aware datetime（给 ts 字段用）。 |
| f | `normalize_frequency(frequence)` | 频率别名 → 规范频率；不认识的值抛 ValueError。 |
| f | `market_prefix(codelist, market_type)` | 决定读哪一族集合 —— 'stock' / 'index' / 'etf'。 `[dt]` |
| f | `get_kline_price_min(codelist, start, market_type, frequency, verbose, end, realtime)` | 分钟线读取（8.3 时序）。返回 (KlineResult, codename)。 |
| f | `get_kline_price_v3(codelist, start, market_type, verbose, end, realtime)` | 日线读取（8.3 时序）。返回 (KlineResult | None, codename)。 |
| f | `read_min_frame(codelist, start, end, frequence, market_type)` | 分钟线扁平帧（8.3 时序）。列含 datetime / code / volume。 |
| f | `GQ_fetch_stock_day_adv(codelist, start, end, market_type, verbose)` | 日线 → GQ_DataStruct_*。替代 QUANTAXIS 的 QA_fetch_stock_day_adv。 |

### `kline_doc`
pytdx bar / xdxr → 8.3 时序文档的纯函数核心（写侧契约）。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `bar_date(bar)` | pytdx bar 的 'YYYY-MM-DD'。 `[dt]` |
| f | `bar_stamp(bar, frequency)` | pytdx bar → 北京口径的 unix 秒（8.3 的 time_stamp）。 `[dt]` |
| f | `bar_datetime(stamp)` | unix 秒 → 北京时间的 '%Y-%m-%d %H:%M:%S'（分钟集合的 datetime 列）。 `[dt]` |
| f | `normalize_vol(vol)` | pytdx 的成交量 → 8.3 存量的单位（实测表见模块 docstring 第 ② 条）。 `[dt]` |
| f | `normalize_amount(amount)` | pytdx 的成交额 → 8.3 的口径：哨兵归零，其余直传。 `[dt]` |
| f | `bar_doc(bar, code)` | pytdx bar → 8.3 时序文档（逐字段对齐存量，见模块 docstring 与 MONGODB83.md §五）。 `[dt]` |
| f | `xdxr_doc(row, code, market)` | pytdx get_xdxr_info 的一行 → 8.3 stock_xdxr 文档。 `[dt]` |
| f | `dedup_docs(docs)` | 按 (code, ts) 去重（保留先出现的一行）。 `[dt]` |
| f | `trade_days_between(start, end, calendar)` | [start, end] 内的交易日（含端点，闭区间）。 `[dt]` |
| f | `alive_threshold(frequency, now)` | 「这个集合该有的最早一根 bar 的 ts」—— 集合里存在 ts >= 它 的 bar， |
| f | `last_closed_session_bar(frequency, now)` | 最近一次收盘那一批 bar 里最早那根的 ts（UTC-aware）；答不出来 → None。 |
| f | `bars_needed(first_date, today, bars_per_day, calendar, extra_days)` | 「从 first_date 到 today 大约有多少根 bar」的上界（用来估首次翻页次数，不用于截断）。 `[dt]` |
| f | `page_offsets(total, page, extra)` | 升序的 start 偏移列表（0 = 最新一根）。 `[dt]` |
| f | `xdxr_adj_events_changed(old_docs, new_docs)` | 两组 xdxr 文档里会影响价格的事件集合是否不同（决定要不要重写/重算 adj）。 `[dt]` |
| f | `bar_docs(bars, code)` | 一批 pytdx bar → 8.3 文档（列式，polars）—— 与逐行 :func:bar_doc 等价。 `[dt]` |

### `kline_save`
--save tdx 的 K 线主体：增量水位 → pytdx 取数 → 落 8.3 时序集合。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `progress_extra(last, stats, elapsed, done, total, today)` | 进度行里「除标签与百分比之外」的那段：当前标的+窗口、累计计数、用时与预计。 |
| f | `progress_summary(stats, elapsed, done, total)` | 汇总段（不带"当前标的"）：写/删/空/跳过/重连/错 + 用时。 |
| f | `progress_line(name, done, total, last, stats, elapsed, every, today)` | 单行进度：集合 / 进度 / 当前 code 与窗口 / 累计行数 / 异常计数 / 已用与预计。 `[dt]` |
| f | `target_collection_name(target, frequency)` | ('stock', '1min') → 'stock_1min'；('index', 'day') → 'index_day'。 `[dt]` |
| f | `universe(target, db, codes, verbose)` | 该族要处理的 code 列表（裸 6 位）。 |
| f | `last_bar(coll, code)` | 该 code 在该集合里的最后一根 bar：(ts, date)；没有则 (None, None)。 |
| f | `collection_frontier(coll, lo, hi)` | 二分找集合里最后一个有数据的时刻 → unix 秒；定不了 → None。 |
| f | `kline_checkin_name(collection)` |  |
| f | `kline_ttl_hours(collection, now)` | 该集合此刻该用的强制全查间隔（小时）。盘中判定复用 GQ_util_if_tradetime |
| f | `kline_sweep_age_hours(collection, now)` | 该集合距上次「完整扫过一遍」多少小时；从没扫过 → None（调用方按「必须全查」）。 |
| f | `mark_kline_sweep(collection)` | 记下「该集合刚刚完整扫完」。失败不抛（见 mark_checkin 的说明）。 |
| f | `allow_shortcircuit(collection, now)` | 该集合本次允不允许短路 —— TTL 没过期才允许。 |
| f | `hours_since_last_close(now)` | 距最近一次收盘（交易日 15:00）过去了多少小时；日历答不出来 → None。 |
| f | `fast_skip_reason(coll, collection, frequency, now)` | 收盘后的快路径：整个集合已整批落库 ⇒ 返回一句说明；否则 None。 |
| f | `hours_since_last_open(now)` | 距最近一次开盘（交易日 09:30）过去了多少小时；日历答不出来 → None。 |
| f | `allow_xdxr_shortcircuit(collection, now)` | 复权（*_xdxr）的闸 —— 比 K 线多一条判据（DECISIONS.md D26）。 |
| f | `intraday_blocks_shortcircuit(frequency, now)` | 盘中 + 分钟频率 ⇒ 禁止短路（判据④）。返回 True 表示「不许跳」。 |
| f | `floor_date(last_date, margin_days, today)` | 窗口起点：水位日往前 margin_days 个交易日。 `[dt]` |
| f | `floor_ts(last_ts, last_date, margin_days, start_date)` | 增量窗口的起点 —— 上一次写进去的那根 bar 本身，不是它所在那天的 00:00。 |
| f | `save_kline_tdx(targets, frequencies, codes, start_min, jobs, margin_days, dry_run, progress_every, verbose, echo, on_progress, force_refresh)` | --save tdx 的 K 线主体：增量取数并落 8.3（可用零参调用 —— PITFALLS P10）。 |
| f | `save_xdxr_tdx(codes, jobs, recompute_adj, verbose, echo, target)` | {target}_xdxr：逐 code 取全历史事件，整 code 替换（见 replace_code_rows）。 |
| f | `adj_docs(code, dates, adj)` | 因子序列 → stock_adj 文档（形态与日线一致：time_stamp == date_stamp = 当日零点）。 |
| f | `save_adj(codes, verbose, echo, target)` | 对给定 code 整条重算前复权因子并落 {target}_adj。 |
| f | `verify_adj(codes, tol, verbose)` | 把重算结果与存量 stock_adj 逐值比 —— 移植保真度的一次性核对。 |

### `kline_status`
只读覆盖核对：8.3 的 K 线有没有洞（--save tdx --save-coverage）。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `expected_dates(first, last, suspended)` | [first, last] 内应有 bar 的交易日（含端点，扣掉 suspended）。 `[dt]` |
| f | `kline_status(targets, frequencies, sample, window_days, with_adj, verbose)` | 逐个集合核对覆盖率。 |
| f | `roots_check(name, day, expected, verbose)` | 「存在但缺根」：只看某个交易日，逐 code 数行数，列出不等于应有根数的 code。 |
| f | `format_kline_status(report)` | 把 kline_status() 的结果渲染成可读文本。 |

### `maintenance`
8.3 库存量数据的维护操作（清理类）。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `GQ_suspension_dates(daily_collection, verbose)` | {(code, date)} —— 当日日线 vol < 1，即停牌日。 |
| f | `GQ_purge_suspended(dry_run, limit, targets, verbose)` | 把停牌日的 bar 移出主集合到 <集合名>_removed。默认只统计不移动。 |
| f | `GQ_restore_suspended(targets, removed_by, verbose)` | 回迁：把 <集合名>_removed 里由清理移走的内容搬回主集合，幂等。 |

### `metadata_save`
stock_metadata_day —— 日频元数据的统一落点。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `day_stamp(day)` | 'YYYY-MM-DD'（或带时分秒）→ 本表 date_stamp（秒，int）。 `[dt]` |
| f | `day_date(date_str)` | 源端日期 → '%Y-%m-%d'。 `[dt]` |
| f | `metadata_day_doc(code, day, field, rate, created_at)` | 一行元数据 → stock_metadata_day 文档（只带一列载荷）。 `[dt]` |
| f | `turnover_rows(rows, src_field, dst_field, created_at)` | 某个源的源行 → stock_metadata_day 文档列表。 `[dt]` |
| f | `save_turnover(since, until, batch, verbose, echo, created_at, dry_run)` | 把 4.4 两源的换手率搬进 8.3 stock_metadata_day（两列，各写各的）。 |
| f | `refresh_turnover(days, verbose, echo)` | 增量刷新最近 days 个自然日的换手率（供 --save 调用）。 |

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
| f | `realtime_collection_name(day)` | 当日实时集合名 —— realtime_YYYY-MM-DD。 |
| f | `collections_of_today(database)` | 当天实时集合 —— 普通集合，QMT / 老 L1 路径的写法。 |
| f | `formater_l1_ticks(l1_ticks, codelist, stacks, symbol_list)` | 处理 l1 ticks 数据 |
| f | `sub_l1_from_tencent(database_realtime)` | 从腾讯获取 L1 数据（成交快照，含五档）。 |
| f | `realtime_ts_collection(database, name)` | 取（必要时创建）实时用的时间序列集合。 |
| f | `sub_l2_from_tencent(database_realtime, sleep_time, etf_codelist, verbose)` | L2（五档盘口）订阅：全市场走腾讯；ETF 那条留给 MiniQMT（已停服）。 |
| f | `GQ_fetch_stock_realtime_adv(code, num, collections, verbose, suffix, day, source)` | 返回当日的上下五档, code可以是股票可以是list, num是每个股票获取的数量 |
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
| f | `GQ_fetch_stock_metadata_day(code, start, end, columns, database)` | stock_metadata_day（日频元数据，一行 = 一标的 × 一交易日）的读取口。 |

### `refdata_save`
A 股参考集合的取数与落库编排 —— 对应 CLI 的 --save <SOURCE> 的参考数据段。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `refdata_ttl_hours(collection, now)` | 该集合此刻该用的刷新间隔（小时）—— 查 :data:TTL_HOURS 的 (盘中, 盘后)。 `[dt]` |
| f | `mark_refdata_success(collection, echo)` | 把「该集合刚刚成功完成」记进 supervisor 的签到表。只该在真写完之后调。 |
| f | `refdata_age_hours(collection, now)` | 该集合距上次成功完成过去了多少小时；从没成功过 → None（调用方按「该取」处理）。 |
| f | `save_refdata(collections, source, codelist, exclude_sources, verbose, on_progress, echo, ttl_hours)` | 把参考集合取回并落库到 8.3 的 golemq_stock_cn。 |
| f | `refdata_status()` | 各集合当前的源可用性与库存量 —— 排障用。 |
| f | `format_status(report)` | 把 save_refdata 的结果或 refdata_status() 渲染成可读文本。 |

### `symbol`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `EXCHANGE` |  |
| f | `normalize_code(symbol, pre_close, market_type)` | 归一化证券代码 |
| f | `is_stock_cn(code)` | 判断 股票代码，市场来源，板块。 `[dt]` |

### `tools`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `purge_historical_collections(client)` | 清理过期的实时集合 —— 纯逻辑，不打印。 |

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

### `pytdx_kline`
pytdx 的 K 线取数层：只管取 bar / xdxr，不碰数据库、不管文档形态。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `bars_page(api, market, code, category, offset, count, is_index)` | 取一页 bar。 |
| f | `bars_paged(api, market, code, category, offsets)` | 按 offsets（升序）翻页取 bar，短页即止。 |
| f | `xdxr_rows(api, market, code)` | get_xdxr_info：该 code 的全部除权除息事件。 |
| f | `probe_market_bars(api, code, market, category)` | 探「这个 market 号能不能取到 bar」（北交所要探，见 kline_save 的说明）。 |

### `pytdx_source`
pytdx 数据源适配器 —— 通达信协议的社区实现。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `tdx_market_of(code, target)` | 6 位代码 + 标的族 → pytdx market 号（0=深 1=沪 2=北）；不认识返回 None。 `[dt]` |
| C | `TdxSource` |  |
| f | `TdxSource.available()` | pytdx 可导入且有候选服务器即算可用。不在此处做网络探测 —— |
| f | `TdxSource.new_api()` | 新建并连上一个可用服务器（逐个试 hosts）。调用方负责 disconnect()。 |
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
| f | `QmtSource.available()` | 本机 QMT 源是否可用。恒为 False（见 :data:QMT_SOURCE_ENABLED）。 |
| f | `QmtSource.unavailable_reason()` | 给 :func:refdata_save._pick_source 的报错用 —— 否则报「原因未知」。 |
| f | `QmtSource.fetch(collection)` |  |
| f | `QmtSource.fetch_stock_list(verbose)` | 全市场 A 股列表。字段与 4.4 quantaxis.stock_list 逐一对齐。 |
| f | `QmtSource.fetch_stock_info(verbose)` | 股本与上市信息。liutongguben/zongguben 是唯一被真实消费的字段， |
| f | `QmtSource.fetch_stock_block(include_market_sectors, verbose)` | QMT 板块成分。容器板块按 SECTOR_SKIP 排除。 |

### `tdx_hosts`
通达信行情服务器池：候选收集 → 真协议探活 → 排序 → 缓存。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `cache_path()` | 缓存文件全路径（~/.GolemQ/settings/tdx_hosts.json）。 |
| f | `is_loopback(ip)` | 127.0.0.0/8 / ::1 / 0.0.0.0 —— 行情服务器不可能是本机。 `[dt]` |
| f | `probe(ip, port, attempts)` | 对一台测 attempts 次，返回 {'ip','port','ok','runs','median','fail_rate'}。 |
| f | `candidates(include_builtin_pool)` | 候选服务器：pytdx 包内池（64 台）∪ 我们的 DEFAULT_HOSTS，去重。 |
| f | `probe_pool(pool, attempts, workers)` | 并行探活整个池子，返回按中位延迟升序的可用列表。 |
| f | `load(now)` | 读缓存；文件不存在 / 过期 / 坏了都返回 None（调用方退回 DEFAULT_HOSTS）。 |
| f | `save(alive, now)` | 写缓存。alive 是 :func:probe_pool 的返回。 |
| f | `refresh(force, verbose, pool)` | 每周一次的刷新：不到期就直接用缓存；到期才真探。 |

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
| C | `ZEN` | 走势中枢（盘整箱体 / pivot）的字段名。 |

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

### `returns`
持仓收益的计算 —— 纯函数，为 JIT / Cython 而写成显式循环。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `calc_onhold_returns_np(closep, daily_position, long)` | 当前持仓的浮动收益；一次持仓结束时（仓位归 0）下一根起重新计。 `[dt]` |

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

### `iwencai`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `iwencai` |  |

## GolemQ.supervisor

### `function_checkin`

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `get_gateway()` | 获取默认网关地址 |
| f | `get_optimized_caller_ip()` | 优化的IP获取方案，优先零延迟方法，备用快速国内服务 |
| f | `resolve_caller_ip()` | 本机在签到表里的 caller_ip 标识。 |
| C | `FunctionCheckinManager` | 函数调用频率控制管理器 |
| f | `FunctionCheckinManager.expired_time(expired_time)` | 计算过期时间戳 |
| f | `FunctionCheckinManager.checkin_function(function_name, expired_time, caller_ip)` | 函数调用签到 |
| f | `FunctionCheckinManager.complete_function_call(function_name, caller_ip)` | 标记函数调用完成（减少并发计数） |
| f | `FunctionCheckinManager.get_function_stats(function_name, caller_ip)` |  |
| f | `FunctionCheckinManager.get_all_function_stats()` | 获取所有函数的调用统计信息 |
| f | `FunctionCheckinManager.reset_function_stats(function_name, caller_ip)` | 重置函数的调用统计 |
| f | `FunctionCheckinManager.archive_old_records(days)` | 归档旧的调用记录 |
| f | `FunctionCheckinManager.cleanup_archive(days)` | 清理旧的归档记录 |
| f | `FunctionCheckinManager.get_concurrent_count(function_name, caller_ip)` | 获取当前的并发调用数 |
| f | `FunctionCheckinManager.get_call_frequency(function_name, caller_ip)` | 计算当前的调用频率（次/分钟） |
| f | `FunctionCheckinManager.is_function_expired(function_name, caller_ip)` | 检查函数调用是否已过期 |
| f | `stable_caller_key()` | 做计时 / 限流的调用方该用的 caller_ip —— 钉死成主机名，不走 resolve_caller_ip()。 |
| f | `checkin_function(function_name, expired_time, caller_ip)` | 使用全局函数管理器进行调用签到。 |
| f | `checkin_age_hours(function_name, caller_ip, now)` | 只读：function_name 上次签到距今多少小时；从没签过 → None（= 该做）。 |
| f | `mark_checkin(function_name, hours, caller_ip)` | 记账：把「刚刚成功完成」写进签到表。返回是否记上。 |
| f | `complete_function_call(function_name, caller_ip)` | 使用全局函数管理器标记调用完成 |
| f | `last_checkin(function_name, caller_ip)` | 只读：本机最近一次签到记录（不调用、不写库）；从没签过 → None。 |

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

### `test_blas_guard`
BLAS / OpenMP 线程数守卫（GolemQ/__init__.py 顶部）。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestBlasThreadGuard` |  |
| f | `TestBlasThreadGuard.test_all_variables_are_present()` |  |
| f | `TestBlasThreadGuard.test_guard_sets_one_in_a_clean_interpreter()` | 真回归：起一个剥掉这些变量的新解释器，import GolemQ 之后必须变成 '1'。 |
| f | `TestBlasThreadGuard.test_guard_does_not_override_explicit_setting()` | setdefault 语义：调用方自己 export 的都该原样保留。 |

### `test_chip_distribution`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestChipDistributionPerformance` | 性能基准（原为命令行脚本，已纳入 unittest 以便记录依赖缺口） |
| f | `TestChipDistributionPerformance.test_performance()` | 测试优化前后的性能对比 |
| f | `test_performance()` | 命令行直接运行入口（保持原脚本用法） |

### `test_cli_bootstrap`
cli/bootstrap.py —— 环境自检的口径。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestVersionGate` | 版本比较的边界 —— doctest 覆盖了主干，这里钉几个容易错的。 |
| f | `TestVersionGate.test_meets_pads_shorter_found()` |  |
| f | `TestVersionGate.test_meets_unequal_lengths()` |  |
| f | `TestVersionGate.test_empty_found_is_not_meeting()` | 取不到版本（空元组）算不满足 —— 别把「不知道」当成「没问题」。 |
| f | `TestVersionGate.test_parse_version_stops_at_non_numeric()` |  |
| C | `TestEnvironmentChecks` |  |
| f | `TestEnvironmentChecks.test_every_required_package_gets_a_line()` |  |
| f | `TestEnvironmentChecks.test_pandas_floor_is_the_last_2x_not_an_imaginary_2_5()` | pandas 的门槛必须是 2.3（3.0 之前最后一个 2.x 是 2.3.3，2025-09-29）。 |
| f | `TestEnvironmentChecks.test_pandas_floor_accepts_3x()` |  |
| f | `TestEnvironmentChecks.test_checks_return_ok_and_detail()` |  |
| f | `TestEnvironmentChecks.test_tty_does_not_affect_pass_fail()` | 非 TTY 是正常用法（管道/重定向），不该让自检失败、也不该报 ⚠️。 |
| f | `TestEnvironmentChecks.test_non_verbose_prints_only_problems()` | 非 verbose 的明细只有出问题才打 —— 一切正常时不该往外铺。 |
| f | `TestEnvironmentChecks.test_verbose_prints_passing_lines_too()` | -v 时每个节点的明细行都要出来（用户口径：banner 只报结果，-v 报文字）。 |
| C | `TestNeedsDb` | 哪些命令要求先连上库。 |
| f | `TestNeedsDb.test_config_commands_do_not_need_db()` |  |
| f | `TestNeedsDb.test_data_commands_need_db()` |  |
| C | `TestPick` |  |
| f | `TestPick.test_pick_returns_none_when_nothing_matches()` |  |
| f | `TestPick.test_pick_returns_the_matching_command()` |  |

### `test_cli_bootstrap_banner`
环境自检的 banner（cli/bootstrap.py + core/presentation.py 的四态）。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestStamp` |  |
| f | `TestStamp.test_exact_format()` |  |
| f | `TestStamp.test_only_the_header_gets_stamped()` | caption 覆盖头行全文，且只有头行带戳 —— 状态行不带。 |
| f | `TestStamp.test_save_banner_keeps_source_prefix()` | 取数 banner 不给 caption → 头行仍是 source: {名}，只是前面多了戳。 |
| C | `TestFourStates` | 四态在两种渲染模式下都要分得出来。 |
| f | `TestFourStates.test_colored_uses_three_colours_and_one_gray()` |  |
| f | `TestFourStates.test_plain_mode_distinguishes_by_symbol()` | 非 TTY 的命门：没有颜色时 警告/失败 必须换成不同符号。 |
| f | `TestFourStates.test_pending_is_not_a_failure()` | 灰 = 未检查 / 不适用，不是失败 —— 别把它算进红。 |
| f | `TestFourStates.test_new_states_do_not_change_the_old_three()` | 扩四态不得动 --save 那三态的渲染（那是 2026-10-09 刚定的契约）。 |
| C | `TestCheckResults` |  |
| f | `TestCheckResults.test_run_checks_returns_one_entry_per_node_in_order()` |  |
| f | `TestCheckResults.test_every_node_is_in_the_banner_header()` | 节点名与表头键必须是同一批 —— 不然 mark 会静默忽略（不报错）。 |
| f | `TestCheckResults.test_details_are_always_a_list_of_lines()` | 明细一律是 list —— 裸字符串会被下游逐字迭代（实测踩过）。 |
| f | `TestCheckResults.test_cuda_is_never_a_failure()` | 新树零 GPU 依赖 → 探不到只算「未检查」。 |
| f | `TestCheckResults.test_blas_warns_on_a_non_one_value()` | OPENBLAS_NUM_THREADS=4 与「没设」后果几乎一样，旧版却报绿。 |
| f | `TestCheckResults.test_pytdx_has_no_version_and_is_not_penalised()` | want='' = 只查 import —— pytdx 没有 __version__，不能因此报红。 |
| C | `TestHardGate` | 硬拦项（python / 依赖包）才拦人，且修配置那四条命令必须放行。 |
| f | `TestHardGate.test_failing_python_exits_when_strict()` |  |
| f | `TestHardGate.test_same_failure_does_not_exit_when_not_strict()` |  |
| f | `TestHardGate.test_a_warning_never_exits()` | 黄点（线程环境 / 时区 / CUDA）不拦 —— 它们可能是显式选择或不适用。 |
| f | `TestHardGate.test_repair_commands_are_exactly_the_config_ones()` | 放行集合必须就是那四条修配置的命令 —— 多一条就绕过闸，少一条就修不回来。 |
| C | `TestNothingPrintsWhileTheBannerIsAlive` | PITFALLS.md P22：banner 活跃期间一个 print 都不许有。 |
| f | `TestNothingPrintsWhileTheBannerIsAlive.test_no_print_when_everything_is_fine()` |  |
| f | `TestNothingPrintsWhileTheBannerIsAlive.test_no_print_even_when_there_is_a_problem()` | 有问题的路径也要走 「关掉 banner → 再 print」，不然屏上会花。 |
| C | `TestTradingCalendarCheck` | 交易日历节点（用户 2026-10-10 定）。 |
| f | `TestTradingCalendarCheck.test_expired_is_fail()` | 末端 < 今天 ⇒ 红（后面的日期全被判成非交易日）。 |
| f | `TestTradingCalendarCheck.test_fresh_before_nov10_is_ok()` |  |
| f | `TestTradingCalendarCheck.test_after_nov10_should_renew_is_warn()` |  |
| f | `TestTradingCalendarCheck.test_nov10_boundary_itself_is_still_ok()` | 边界取含 11-10（用户口径「11月10日以前」）。 |
| f | `TestTradingCalendarCheck.test_short_coverage_is_warn()` | ⚠️ 用户没给的第四档：末端在未来、但没到今年年底 ⇒ 还能跑、覆盖不够长。 |
| f | `TestTradingCalendarCheck.test_empty_calendar_is_fail()` |  |
| f | `TestTradingCalendarCheck.test_malformed_tail_is_fail()` |  |
| f | `TestTradingCalendarCheck.test_year_rollover_expires()` | 跨年：到了次年 1 月而日历没续 ⇒ 红（这是这条检查最该抓的情形）。 |
| f | `TestTradingCalendarCheck.test_it_is_a_node_but_not_a_hard_gate()` | 在 banner 上有节点，但不进硬拦（红点也不拦启动）。 |
| f | `TestTradingCalendarCheck.test_real_calendar_is_current()` | 真日历此刻应该是绿（覆盖到今年年底）。 |
| C | `TestBannerTwoColumns` | 自检 banner 分两栏（用户 2026-10-10）。 |
| f | `TestBannerTwoColumns.test_two_rows()` |  |
| f | `TestBannerTwoColumns.test_rows_flatten_to_the_node_list_in_order()` | 两栏的顺序拼起来必须等于 SELF_CHECK_NODES —— 顺序是语义。 |
| f | `TestBannerTwoColumns.test_column_split_is_semantic()` |  |
| f | `TestBannerTwoColumns.test_second_row_shares_the_phase_name()` | 两行的阶段名相同 ⇒ 第二行留白对齐（渲染成一块，不是两块）。 |
| C | `TestOptionalSourceChecks` | tdxidata / tushare / iwencai 三个节点。 |
| f | `TestOptionalSourceChecks.test_configured_is_ok()` |  |
| f | `TestOptionalSourceChecks.test_not_configured_is_pending_not_warn()` | 未配置 ⇒ 灰，不是红也不是黄：可选源没配不影响任何命令跑得通。 |
| f | `TestOptionalSourceChecks.test_probe_failure_does_not_raise()` | 探测抛错也要给状态（灰 + 原因），不许把启动自检搞崩。 |
| f | `TestOptionalSourceChecks.test_get_source_failure_does_not_raise()` |  |
| f | `TestOptionalSourceChecks.test_iwencai_is_pending_and_says_unimplemented()` | ⚠️ 问财在新树尚未实现 —— 节点如实说，且不发明没人读的配置键。 |
| f | `TestOptionalSourceChecks.test_three_nodes_are_present_but_not_hard_gates()` |  |
| f | `TestOptionalSourceChecks.test_optional_source_map_covers_the_two_adapters()` | 节点名 → 适配器名的映射必须与 datasource/ 的注册键一致。 |
| C | `TestServerchanAndQmtChecks` | serverchan 与 讯投QMT 两个节点。 |
| f | `TestServerchanAndQmtChecks.test_serverchan_configured_is_ok()` |  |
| f | `TestServerchanAndQmtChecks.test_serverchan_unconfigured_is_pending()` | 未配 ⇒ 灰：推送是可选渠道，不配不影响任何命令。 |
| f | `TestServerchanAndQmtChecks.test_serverchan_probe_failure_does_not_raise()` |  |
| f | `TestServerchanAndQmtChecks.test_qmt_checks_the_xtquant_config_section()` | 用户 2026-10-10 明确：「讯投QMT 检查的是这一段 [XTQUANT] account/min_path」。 |
| f | `TestServerchanAndQmtChecks.test_qmt_missing_keys_is_warn()` |  |
| f | `TestServerchanAndQmtChecks.test_qmt_unreadable_config_is_warn()` |  |
| f | `TestServerchanAndQmtChecks.test_qmt_green_still_reports_the_shutdown()` | ⚠️ 绿点不表示这条路可用 —— 「已停服」必须留在 detail 里。 |
| f | `TestServerchanAndQmtChecks.test_qmt_is_not_an_adapter_source()` | 它查的是配置段，不该混进"适配器可用吗"那张表。 |
| C | `TestBootstrapStartDoneLines` | 阶段起止各一行（用户 2026-10-10）： |
| f | `TestBootstrapStartDoneLines.test_start_line_and_done_line_both_appear_in_order()` |  |
| f | `TestBootstrapStartDoneLines.test_no_escape_codes_when_not_a_tty()` | 非 TTY 里一个转义码都不许有（颜色规则第一条）—— 收尾行也一样。 |
| f | `TestBootstrapStartDoneLines.test_done_line_is_printed_only_after_close()` | ⚠️ 收尾行必须在 Banner.close() 之后 —— banner 活着时 print 会 |
| f | `TestBootstrapStartDoneLines.test_stamp_done_format_and_gray()` |  |

### `test_cli_commands`
cli/commands/ 的注册表契约。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestCommandRegistry` |  |
| f | `TestCommandRegistry.test_build_parser_succeeds()` | 每个命令的 add_arguments 各注册一次 —— 有重复 argparse 会直接抛。 |
| f | `TestCommandRegistry.test_every_flag_is_a_real_argparse_dest()` | FLAGS 里的名字必须是 parser 真会产出的属性 —— 打错一个字母 |
| f | `TestCommandRegistry.test_command_names_are_unique()` |  |
| f | `TestCommandRegistry.test_save_coverage_is_dispatched_before_save()` | ⚠️ 顺序即语义：两个开关同时给时，老链选 coverage。 |
| C | `TestMatchSemantics` | matches 的判据是真值 —— 等价老链的 elif args.X:。 |
| f | `TestMatchSemantics.test_default_is_truthiness()` |  |
| f | `TestMatchSemantics.test_none_does_not_match()` |  |
| C | `TestSaveSourceValidation` | --save 的取值校验只有一处：argparse 的 choices=。 |
| f | `TestSaveSourceValidation.test_invalid_source_rejected_by_argparse()` |  |
| f | `TestSaveSourceValidation.test_empty_source_rejected_by_argparse()` |  |
| f | `TestSaveSourceValidation.test_bare_save_means_tdx()` |  |
| f | `TestSaveSourceValidation.test_uppercase_is_lowered_before_the_choices_check()` | 老代码是 str(args.save).lower() → --save TDX 能用。 |
| f | `TestSaveSourceValidation.test_absent_save_is_none()` |  |
| C | `TestExitCodeConvention` | 退出码口径：用法/参数错 = 2、运行期失败 = 1。 |
| f | `TestExitCodeConvention.setUp()` |  |
| f | `TestExitCodeConvention.test_constants()` |  |
| f | `TestExitCodeConvention.test_usage_error_exits_2_in_argparse_shape()` |  |
| f | `TestExitCodeConvention.test_usage_error_without_bound_parser_still_exits_2()` | 没 bind_parser 过也不能炸 —— 退化成不打 usage 行，码不变。 |
| C | `TestBlockingCallsLightUpTheBanner` | 阻塞调用之前必须先点 RUNNING —— 否则那一段屏上是个灰点。 |
| f | `TestBlockingCallsLightUpTheBanner.test_every_blocking_call_is_preceded_by_running()` |  |
| C | `TestXdxrRefreshGate` | 复权段的 TTL 闸（用户 2026-10-09 定，推翻 D25 原先的「只记账不开闸」）。 |
| f | `TestXdxrRefreshGate.test_gate_message_only_under_verbose()` | ⚠️ 用户 2026-10-09：刷新闸那句话只在 -v 下打。 |
| f | `TestXdxrRefreshGate.test_gate_open_skips_the_whole_pass()` |  |
| f | `TestXdxrRefreshGate.test_gate_closed_runs_both_targets()` |  |
| f | `TestXdxrRefreshGate.test_save_refresh_forces_it()` | 一个旗子 = 别信缓存、全查一遍（与参考数据闸、K 线短路同一口径）。 |
| f | `TestXdxrRefreshGate.test_limited_codes_do_not_open_the_gate()` | ⚠️ 这条是静默丢数据的防线：试跑不许记账。 |
| f | `TestXdxrRefreshGate.test_full_universe_marks_the_sweep()` |  |
| C | `TestAdjNodeMarking` | stock_adj / etf_adj 两个节点什么时候点白。 |
| f | `TestAdjNodeMarking.test_adj_lights_white_when_the_gate_skips_xdxr()` | ⚠️ 用户 2026-10-10 实报的那条：xdxr 被闸跳过时，_adj 也必须点白。 |
| f | `TestAdjNodeMarking.test_adj_lights_white_when_there_is_nothing_to_do()` |  |
| f | `TestAdjNodeMarking.test_adj_stays_gray_when_events_changed_but_recompute_skipped()` | ⚠️ 反向那条：事件变了却 --save-no-adj ⇒ _adj 是过期的。 |
| f | `TestAdjNodeMarking.test_adj_lights_white_after_recomputing()` |  |
| C | `TestSaveStageStartDoneLines` | --save 阶段的起止两行（用户 2026-10-10，与 bootstrap 同构）： |
| f | `TestSaveStageStartDoneLines.test_start_line_is_before_the_source_row()` |  |
| f | `TestSaveStageStartDoneLines.test_done_line_comes_last()` |  |
| f | `TestSaveStageStartDoneLines.test_caption_matches_what_actually_runs()` | ⚠️ --save qmt 不取 K 线（只做参考数据）—— 给它打 "klines" 就是假话。 |
| f | `TestSaveStageStartDoneLines.test_no_done_line_when_it_crashes()` | ⚠️ 中途炸了不许打 done. —— 那一段没 done。 |
| f | `TestSaveStageStartDoneLines.test_no_escape_codes_when_not_a_tty()` |  |

### `test_cli_tools`
测试 CLI 工具功能

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestCLITools` | 测试 CLI 工具功能 |
| f | `TestCLITools.setUp()` | 测试前备份 GQMARKETS 状态 |
| f | `TestCLITools.tearDown()` | 测试后恢复 GQMARKETS 状态 |
| f | `TestCLITools.test_auto_register_markets_skips_registered()` | 已注册的市场不被顶掉 —— 钉的是实例同一性，不是总数。 |
| f | `TestCLITools.test_auto_register_markets_skips_abstract_classes()` | 抽象类被跳过、具体类被注册 —— 一对正反断言。 |
| f | `TestCLITools.test_purge_mongodb_database_delegates_to_market(mock_print)` | purge_mongodb_database 只委托给市场实例，自己不碰库。 |
| f | `TestCLITools.test_auto_register_markets_with_mock_module()` | 测试自动注册处理异常情况 |
| f | `TestCLITools.test_auto_register_markets_with_invalid_module()` | 测试自动注册处理无效模块 |

### `test_doctests`
把 docstring 里的 doctest 纳入 unittest 发现。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `load_tests(loader, tests, ignore)` | unittest 的收集协议：把各模块的 doctest 挂进本次发现。 |

### `test_etf_routing`
ETF 独立成 ETF_CN / etf_* 之后的路由测试。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestMarketPrefix` | market_prefix 的三值契约 —— 它决定读哪张表。 |
| f | `TestMarketPrefix.test_three_way()` |  |
| f | `TestMarketPrefix.test_explicit_market_type_wins()` |  |
| f | `TestMarketPrefix.test_list_uses_first()` |  |
| C | `TestFrameToDatastruct` | frame_to_datastruct 的分派表 —— 少了 ('etf', …) 会 KeyError。 |
| f | `TestFrameToDatastruct.test_dispatch()` |  |
| f | `TestFrameToDatastruct.test_empty_returns_none()` |  |
| C | `TestQfqCapability` | to_qfq 的层级契约：股票与 ETF 有，真指数没有。 |
| f | `TestQfqCapability.test_capability()` |  |
| f | `TestQfqCapability.test_etf_and_stock_use_different_factor_tables()` | ETF 与股票必须是两条路 —— 走同一张表会静默算错价。 |
| f | `TestQfqCapability.test_type_attribute()` |  |
| C | `TestGQIsEtfAgreesWithClassifier` | GQ_is_etf 必须与 is_stock_cn 的 ETF_CN 完全一致。 |
| f | `TestGQIsEtfAgreesWithClassifier.test_agreement()` |  |
| f | `TestGQIsEtfAgreesWithClassifier.test_tolerates_tagged_forms()` | 带交易所标记的写法也要能判 —— is_stock_cn 负责归一。 |
| C | `TestMinContainer` | fetch._min_container 的三分支 —— 少了 etf 会静默拿错容器。 |
| f | `TestMinContainer.test_container_choice()` |  |
| C | `TestAdjFactorQueryUsesTheTimefield` | 复权因子表（stock_adj / etf_adj）必须按 ts（timeField）过滤。 |
| f | `TestAdjFactorQueryUsesTheTimefield.test_stock_adj_frame_filters_on_ts()` |  |
| f | `TestAdjFactorQueryUsesTheTimefield.test_etf_adj_filters_on_ts()` |  |
| f | `TestAdjFactorQueryUsesTheTimefield.test_bounds_cover_the_whole_day()` | 边界必须是北京口径的当日两端 —— 只给日期时不能塌成零点。 |
| f | `TestAdjFactorQueryUsesTheTimefield.test_projection_keeps_the_date_join_key()` | 投影里保留 date —— 它与 K 线帧的 join 键仍是日期字符串。 |

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

### `test_kline_save`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestSaveBarChunk` | 时序集合的写口：先删后插（不能 upsert，见 PITFALLS P14）。 |
| f | `TestSaveBarChunk.setUp()` |  |
| f | `TestSaveBarChunk.test_delete_filter_covers_exactly_the_dedup_keys()` | 删除条件 = 本次要写的那些 (code, ts) —— 不多删一行（冻结日不碰）。 |
| f | `TestSaveBarChunk.test_replace_code_rows_deletes_whole_code()` | replace_code_rows 的语义：整 code 替换（xdxr / adj 用）。 |
| f | `TestSaveBarChunk.test_empty_docs_never_deletes()` | 承重守卫：取到空绝不删 —— 否则一次上游抽风就清空集合。 |
| f | `TestSaveBarChunk.test_batches_at_bar_batch()` |  |
| C | `TestBarDoc` | 文档字段集必须与 8.3 存量逐字段一致 —— 多一个键就多一列读出。 |
| f | `TestBarDoc.test_day_doc_has_exactly_the_stored_key_set()` |  |
| f | `TestBarDoc.test_day_time_stamp_is_midnight()` | 存量实测 time_stamp == date_stamp（日线归零）；不归零会与存量差 15 小时。 |
| f | `TestBarDoc.test_minute_doc_adds_datetime_with_same_label_as_source()` | 分钟标签不做偏移 —— 实测 pytdx 与存量同标签（09:31 … 15:00）。 |
| f | `TestBarDoc.test_index_only_gets_updown_counts_when_given()` |  |
| f | `TestBarDoc.test_code_is_truncated_to_six_digits()` |  |
| f | `TestBarDoc.test_vol_unit_conversion_and_amount_passthrough()` | vol 按市场/频率换单位（实测）；amount 直传，哨兵值原样。 |
| C | `TestBarDocsMatchesRowWise` | 列式构造（bar_docs，polars）必须与逐行 bar_doc 逐字段相等。 |
| f | `TestBarDocsMatchesRowWise.test_day_and_minute_and_index()` |  |
| f | `TestBarDocsMatchesRowWise.test_index_carries_updown_and_x100_vol()` |  |
| f | `TestBarDocsMatchesRowWise.test_empty_bars()` |  |
| C | `TestXdxrDoc` |  |
| f | `TestXdxrDoc.test_field_mapping()` |  |
| C | `TestPagingMath` | 翻页与根数：估多估少都不该丢数据（短页即止才是停的条件）。 |
| f | `TestPagingMath.test_page_offsets_ascending()` |  |
| f | `TestPagingMath.test_bars_needed_is_an_upper_bound()` |  |
| f | `TestPagingMath.test_every_frequency_has_a_tdx_category_and_daily_bar_count()` |  |
| f | `TestPagingMath.test_dedup_keeps_first_of_same_code_and_ts()` |  |
| C | `TestNothingPrintsWhileTheBarIsAlive` | ⚠️ tqdm 进度条活着的时候，谁都不许往 stdout 写。 |
| f | `TestNothingPrintsWhileTheBarIsAlive.test_retry_page_reports_through_note_not_stdout()` |  |
| f | `TestNothingPrintsWhileTheBarIsAlive.test_retry_page_is_silent_without_note()` | 没给 note 时必须一个字都不打（默认是静默，不是退回 print）。 |
| f | `TestNothingPrintsWhileTheBarIsAlive.test_per_code_write_is_silent()` | 每 code 的落库不打印 —— _process_code 调 save_bar_chunk 不传 verbose。 |
| C | `FakeApi` | 假 pytdx 连接：按 pages 依次吐页面；元素为 None 表示「连接已废」（P3b）。 |
| f | `FakeApi.get_security_bars(category, market, code, start, count)` |  |
| f | `FakeApi.get_index_bars(category, market, code, start, count)` |  |
| f | `FakeApi.disconnect()` |  |
| C | `TestBarsPaged` | 翻页三条路径：正常到头 / 短页即止 / 连接废掉后重连（P3b）。 |
| f | `TestBarsPaged.test_stops_on_short_page()` |  |
| f | `TestBarsPaged.test_empty_first_page_is_empty_not_aborted()` |  |
| f | `TestBarsPaged.test_no_offsets_means_no_request()` |  |
| f | `TestBarsPaged.test_none_means_dead_connection_and_reconnects()` | None 不是「没有数据」，是「连接已废」—— 必须换新连接重试同一页。 |
| f | `TestBarsPaged.test_retry_exhausted_aborts_whole_code()` | 重试耗尽 → aborted：调用方必须整只跳过，不能拿半截数据写库。 |
| f | `TestBarsPaged.test_index_flag_uses_get_index_bars()` |  |
| f | `TestBarsPaged.test_page_returns_none_only_on_dead_connection()` |  |
| C | `TestTargetRoutedCollections` | save_adj / save_xdxr_tdx 的 target 决定读写哪三个集合。 |
| f | `TestTargetRoutedCollections.test_save_adj_routes_to_target_collections()` |  |
| f | `TestTargetRoutedCollections.test_save_adj_default_target_is_stock()` |  |
| f | `TestTargetRoutedCollections.test_save_xdxr_routes_to_target_collection()` |  |
| C | `TestFastPathIntegration` | 快路径穿过 save_kline_tdx 的那一段 —— 谓词单测不够。 |
| f | `TestFastPathIntegration.test_fast_path_completes_without_touching_tdx()` |  |
| f | `TestFastPathIntegration.test_force_refresh_disables_the_fast_path()` | --save-refresh 必须把快路径也关掉（一旗两用）。 |
| f | `TestFastPathIntegration.test_dry_run_disables_the_fast_path()` |  |
| C | `TestColdStartVerifiesInsteadOfFetching` | 冷启动：集合从没有过签到记录 ⇒ 核水位（零连接），不是全量真取。 |
| f | `TestColdStartVerifiesInsteadOfFetching.test_cold_start_verifies_and_records_without_connecting()` |  |
| f | `TestColdStartVerifiesInsteadOfFetching.test_stalled_collection_still_fetches_for_real()` | ⚠️ 探针探不到前沿（集合停在水位之前 / 空集合）⇒ 必须全量真取。 |
| f | `TestColdStartVerifiesInsteadOfFetching.test_code_without_data_is_never_skipped_on_cold_start()` | ⚠️ 最要紧的那条：没数据的 code 永远不跳 —— 否则全新集合会被判成"已齐"。 |
| f | `TestColdStartVerifiesInsteadOfFetching.test_warm_ttl_pass_does_not_renew_the_record()` | 回归闸：TTL 新鲜 + 按水位跳那一次不许记账 —— |

### `test_kline_shortcircuit`
K 线保存的本地水位短路（DECISIONS.md D25）。

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `bj(value)` |  |
| C | `TestAliveThreshold` | 「这个集合该有的最早一根」—— 日历只在日级出场。 |
| f | `TestAliveThreshold.test_intraday_never_falls_back_to_yesterday()` | ⚠️ 盘中绝不退到昨天。 退了的话：上午还没人取过 ⇒ 探针说「集合是活的」 |
| f | `TestAliveThreshold.test_before_open_falls_back_to_previous_session()` |  |
| f | `TestAliveThreshold.test_day_bars_use_midnight()` | 日线 bar 的 ts 口径是该日北京零点（实测）。 |
| f | `TestAliveThreshold.test_weekend_uses_previous_trading_day()` |  |
| f | `TestAliveThreshold.test_holiday_uses_previous_trading_day()` |  |
| f | `TestAliveThreshold.test_outside_calendar_is_none()` | 日历覆盖之外必须返回 None —— 那时「该有」无从谈起，宁可每次都取。 |
| f | `TestAliveThreshold.test_open_boundary_is_inclusive()` |  |
| f | `TestAliveThreshold.test_day_does_not_count_today_before_the_close()` | ⚠️ 盘中日线不认今天（2026-10-09 修）：当天日线盘中不该存在 |
| f | `TestAliveThreshold.test_minute_still_counts_today_intraday()` | 分钟与日线相反：开市就该有今天的数据，退到昨天会死锁。 |
| C | `TestIntradayBlocksMinuteShortcircuit` | 判据④：盘中 + 分钟 ⇒ 禁止短路。 |
| f | `TestIntradayBlocksMinuteShortcircuit.test_minute_is_blocked_all_session()` |  |
| f | `TestIntradayBlocksMinuteShortcircuit.test_lunch_break_allows_it()` | 11:31–12:59 午休：上午的分钟 bar 已定稿 ⇒ 短路是对的。 |
| f | `TestIntradayBlocksMinuteShortcircuit.test_after_close_allows_it()` | 15:00 之后分钟 bar 全部定稿 ⇒ 照常短路（批量跑的主场）。 |
| f | `TestIntradayBlocksMinuteShortcircuit.test_non_trading_day_allows_it()` |  |
| f | `TestIntradayBlocksMinuteShortcircuit.test_day_is_never_blocked()` | 盘中日线反而该跳：盘中不写当天日线，所有 code 都停在前一交易日那根上 |
| C | `FakeColl` | 假集合：只知道「最新一根在 newest」这一件事，并记下每次查询。 |
| f | `FakeColl.find_one(filt, projection, sort, limit)` |  |
| C | `TestCollectionFrontier` |  |
| f | `TestCollectionFrontier.test_finds_the_exact_newest_bar()` | 精确命中，不是"收敛到区间"。低估一分钟就会让所有票都被跳（见模块 docstring）。 |
| f | `TestCollectionFrontier.test_probe_count_is_logarithmic()` | 一个会话 ≤ 330 分钟 ⇒ log2(330) ≈ 9 次。别退化成线性扫。 |
| f | `TestCollectionFrontier.test_empty_collection_is_none()` |  |
| f | `TestCollectionFrontier.test_probe_error_is_none_not_raise()` | 探针炸了 ⇒ 判不了 ⇒ 不跳（保守），不能把异常抛给取数主流程。 |
| f | `TestCollectionFrontier.test_probe_uses_the_timefield_not_the_int_stamp()` | ⚠️ 钉住 8800 倍那个坑：查询键必须是 ts。 |
| C | `TestShortCircuit` | _process_code 的三道判据：TTL 放行 + 前沿探到 + 该 code 水位到前沿。 |
| f | `TestShortCircuit.test_skips_and_never_connects()` |  |
| f | `TestShortCircuit.test_ttl_gate_closed_means_fetch()` | TTL 过期 ⇒ 强制全查，哪怕水位已经到前沿。 |
| f | `TestShortCircuit.test_unknown_frontier_means_fetch()` | 前沿探不到（集合空/停住/探针失败）⇒ 不跳。 |
| f | `TestShortCircuit.test_code_behind_the_frontier_is_fetched()` |  |
| f | `TestShortCircuit.test_code_with_no_data_is_never_skipped()` | ⚠️ 从没数据的 code 要全量回填 —— 跳了就是永久空洞。 |
| f | `TestShortCircuit.test_naive_ts_is_read_as_utc()` | pymongo 默认返回裸 datetime（实值是 UTC）—— 当成本地时间就差 8 小时， |
| C | `TestMarketResolvedBeforeConnecting` | tdx_market_of 是纯查表，必须排在建连之前。 |
| f | `TestMarketResolvedBeforeConnecting.test_unmapped_code_never_connects()` |  |
| C | `TestTtlGate` |  |
| f | `TestTtlGate.test_checkin_name_is_per_collection()` | 颗粒度 = 每个集合一条记录（26 个节点各自独立）。 |
| f | `TestTtlGate.test_never_swept_means_no_shortcircuit()` |  |
| f | `TestTtlGate.test_fresh_sweep_allows_shortcircuit()` |  |
| f | `TestTtlGate.test_expired_sweep_blocks_shortcircuit()` |  |
| f | `TestTtlGate.test_sweep_is_marked_only_after_a_full_pass()` | ⚠️ 短路过的那一遍不许记账 —— 记了 TTL 就被无限推后，兜底网永远不触发。 |
| f | `TestTtlGate.test_ttl_is_open_closed_pair()` | 同参考数据的口径：盘中 5h / 盘后 24h。 |
| C | `TestXdxrBoardOpenGate` | 复权的闸比 K 线多一条：跨没过一个开盘（DECISIONS.md D26）。 |
| f | `TestXdxrBoardOpenGate.test_swept_after_the_open_is_up_to_date()` | 今天开盘后扫过（age 1h < 距开盘 8h）⇒ 本交易日已拿到 ⇒ 可以跳。 |
| f | `TestXdxrBoardOpenGate.test_swept_before_the_open_must_run()` | ⚠️ 核心那条：昨晚扫的（age 15h），今早已开盘（距开盘 0.5h） |
| f | `TestXdxrBoardOpenGate.test_ttl_gate_still_applies()` | TTL 那道仍然在 —— 两条是与关系。 |
| f | `TestXdxrBoardOpenGate.test_calendar_unavailable_must_run()` | 日历答不出来（覆盖之外）⇒ 全查（保守）。 |
| f | `TestXdxrBoardOpenGate.test_hours_since_open_uses_the_calendar()` | 真调交易日历（不打桩）—— 复用 alive_threshold('1min')， |
| f | `TestXdxrBoardOpenGate.test_hours_since_open_before_the_bell_counts_from_previous_session()` | 开盘前：距今最近的"开盘"是上一交易日 09:30（10-08 09:30 → 10-09 08:30 = 23h）。 |
| C | `TestFastSkipAfterClose` | 收盘后的快路径：整个集合已整批落库 ⇒ 连逐只读都省掉。 |
| f | `TestFastSkipAfterClose.test_swept_after_the_close_skips_the_whole_collection()` |  |
| f | `TestFastSkipAfterClose.test_swept_before_the_close_goes_the_normal_path()` | 昨晚扫的（age 20h）> 距收盘（3h）⇒ 今天这批还没查 ⇒ 走常规路径。 |
| f | `TestFastSkipAfterClose.test_never_swept()` |  |
| f | `TestFastSkipAfterClose.test_probe_miss_means_no_skip()` | ⚠️ 记录说扫过 ≠ 数据还在（误删、或那一轮有 code 失败）⇒ 探针说不 ⇒ 不跳。 |
| f | `TestFastSkipAfterClose.test_intraday_minute_is_blocked()` | ⚠️ 承重：盘中「最近一次收盘」是昨天 15:00，不加这条 ①， |
| f | `TestFastSkipAfterClose.test_calendar_unavailable()` |  |
| f | `TestFastSkipAfterClose.test_probe_uses_the_timefield()` | 同 collection_frontier：只有 ts（timeField）能吃到时序索引。 |
| f | `TestFastSkipAfterClose.test_last_closed_session_bar_real_calendar()` | 真调交易日历：收盘口径与开盘口径共用 _session_base。 |
| C | `TestFastPathDoesNotSurviveTheNextSession` | ⏭ 模拟时间推进到下一个交易日：快路径必须自己失效。 |
| f | `TestFastPathDoesNotSurviveTheNextSession.test_minute_fast_path_expires_when_the_next_session_opens()` | ✅ 分钟：周一盘中不再短路 —— 判据①（盘中×分钟禁跳）拦住快路径， |
| f | `TestFastPathDoesNotSurviveTheNextSession.test_day_fast_path_still_skips_intraday()` | ❗日线不一样：盘中仍会跳，而且这是对的 —— 盘中不写当天日线 |
| f | `TestFastPathDoesNotSurviveTheNextSession.test_everything_expires_after_the_next_close()` | 周一收盘后：since_close 归零 ⇒ 四个组合全都不许跳 ⇒ 去取周一那批。 |
| f | `TestFastPathDoesNotSurviveTheNextSession.test_since_close_resets_at_each_close()` | 直接钉那个机械原因：新收盘让 since_close 归零。 |

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
| C | `TestPurgeHistoricalCollections` | purge_historical_collections 的契约有两半：集合名格式 + 返回删掉的名单。 |
| f | `TestPurgeHistoricalCollections.test_drops_collection_from_14_days_ago()` | 14 天前的那一个被 drop，14 天内的不动。 |
| f | `TestPurgeHistoricalCollections.test_walks_back_until_14_consecutive_misses()` | 命中就继续往回走：14/15/16 天前三个都该删。 |
| f | `TestPurgeHistoricalCollections.test_no_match_means_no_drop()` |  |
| f | `TestPurgeHistoricalCollections.test_name_without_hyphens_is_not_matched()` | realtime_20261008（无连字符）不认 —— 格式是契约的一部分。 |
| f | `TestPurgeHistoricalCollections.test_name_containing_time_is_not_matched()` | realtime_2026-10-08 01:00:27.222907 不认。 |
| f | `TestPurgeHistoricalCollections.test_collection_list_is_fetched_once()` | 集合清单只取一次 —— 原来在循环里逐日问，一轮跑 28 次。 |

### `test_messenger`
钉钉 / Server酱 推送的测试 —— 全部是 mock 的。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestDingtalkConfig` |  |
| f | `TestDingtalkConfig.test_check_config_success(mock_get)` |  |
| f | `TestDingtalkConfig.test_check_config_failure(mock_get)` | 配置缺失/为空 → False。 |
| f | `TestDingtalkConfig.test_setup_config(mock_input, mock_set)` |  |
| f | `TestDingtalkConfig.test_get_config(mock_get, mock_setup, mock_check)` |  |
| C | `TestDingtalkAccessToken` |  |
| f | `TestDingtalkAccessToken.test_get_access_token(mock_config, mock_client)` |  |
| f | `TestDingtalkAccessToken.test_get_access_token_failure(mock_config, mock_client)` |  |
| C | `TestDingReminder` |  |
| f | `TestDingReminder.test_send_message(mock_config, mock_client, mock_token)` |  |
| f | `TestDingReminder.test_send_message_markdown_content(mock_config, mock_token)` |  |
| f | `TestDingReminder.test_send_message_no_token(mock_config, mock_token)` |  |

### `test_metadata_save`
stock_metadata_day 的落库契约。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestDayStampConvention` | date_stamp 必须与 8.3 的邻居一致（差 8h 就静默 join 错）。 |
| f | `TestDayStampConvention.test_matches_stock_day()` | 锚点取自 8.3 实测：stock_day/stock_adj 里 date='2026-10-08' |
| f | `TestDayStampConvention.test_is_not_the_wall_clock_convention()` | 反证：不是「墙上时间当 UTC」那套（4.4 stock_ranking 用的是它）。 |
| f | `TestDayStampConvention.test_accepts_both_source_date_formats()` | 源端两种格式都收（stock_ranking 带时分秒、stock_valuation 裸日期）。 |
| f | `TestDayStampConvention.test_ts_is_the_real_instant()` | ts 是 UTC-aware 的真实时刻；与同为真实时刻的 date_stamp 差 8h |
| C | `TestDateRange` | 范围过滤按源端 date 格式造 —— 不能一刀切补 ' 00:00:00'。 |
| f | `TestDateRange.test_datetime_format_gets_time_part()` |  |
| f | `TestDateRange.test_bare_date_format_gets_none()` | ⚠️ 裸日期源不能补时分秒：字符串比较下 '2026-09-30' 小于 |
| f | `TestDateRange.test_no_bounds_is_empty()` |  |
| f | `TestDateRange.test_each_source_declares_its_date_format()` |  |
| C | `TestMetadataDayDoc` |  |
| f | `TestMetadataDayDoc.test_field_set_and_types()` |  |
| f | `TestMetadataDayDoc.test_payload_column_is_parameterised()` | 同一行可以带东财列、也可以带 baostock 列 —— 载荷字段名由调用方给。 |
| C | `TestTurnoverRows` |  |
| f | `TestTurnoverRows.test_missing_values_are_skipped_not_zeroed()` | 缺值丢掉，不写 0 —— 0 是「当天真的零换手」的意思。 |
| f | `TestTurnoverRows.test_row_without_date_is_skipped()` | 没有 date 就推不出 date_stamp ⇒ 丢（源端 stamp 不再被信任）。 |
| f | `TestTurnoverRows.test_above_max_ratio_is_rejected()` | > 1.08 判为「百分比没换算就落库」，丢并计数（用户 2026-10-10 的判据）。 |
| f | `TestTurnoverRows.test_boundary_is_inclusive()` |  |
| f | `TestTurnoverRows.test_negative_and_non_numeric_are_skipped()` |  |
| f | `TestTurnoverRows.test_writes_only_the_named_source_field()` | 两个源各自只认自己的字段 —— 免得 stock_valuation 的行被当东财的写进去。 |
| C | `TestSourceTable` | 源的声明本身是契约：集合名 / 源字段 / 目标列 / date 格式。 |
| f | `TestSourceTable.test_two_sources_two_columns()` |  |
| f | `TestSourceTable.test_columns_keep_the_source_names()` | 列名保留源端原名（用户 2026-10-10）—— 好让 calcuChip 读 |
| f | `TestSourceTable.test_unique_key_has_no_revision()` | revision 是旧库的历史包袱，明确丢弃（用户 2026-10-10）。 |
| f | `TestSourceTable.test_collection_name()` |  |
| C | `TestUpsertFieldsDoesNotClobber` | 多列共存：这是「用 $set 而不是 ReplaceOne」的唯一证明。 |
| f | `TestUpsertFieldsDoesNotClobber.setUp()` |  |
| f | `TestUpsertFieldsDoesNotClobber.test_second_column_does_not_wipe_the_first()` |  |
| f | `TestUpsertFieldsDoesNotClobber.test_unique_index_is_created()` |  |
| f | `TestUpsertFieldsDoesNotClobber.test_save_collection_would_wipe_it()` | 反证：证明「用 $set」不是洁癖 —— ReplaceOne 真会抹掉别列。 |

### `test_no_quantaxis`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestNoQuantaxis` | 把「QUANTAXIS 已从全树剔除」从人工 grep 升级成会红的测试。 |
| f | `TestNoQuantaxis.test_no_source_import_of_quantaxis()` | ① 源码守卫：GolemQ//*.py 里不许出现 import QUANTAXIS。 |
| f | `TestNoQuantaxis.test_quantaxis_not_loaded_at_runtime()` | ② 运行时守卫：把入口与市场包导入一遍，sys.modules 里不许有它。 |
| f | `TestNoQuantaxis.test_dead_handles_stay_dead()` | ③ 符号守卫：那 5 个句柄不许在 settings / core / GolemQ 上复活。 |

### `test_peak`
analysis/peak.py（PEAK_POINT）—— jit 与纯实现的对拍，以及加权合成的口径。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestJitMatchesPython` | 两个实现同口径 —— 这条比速度更重要。 |
| f | `TestJitMatchesPython.test_random_walk_identical()` |  |
| f | `TestJitMatchesPython.test_spike_shape()` | 恒定序列插一个尖峰：只有尖峰那一点是 ±1。 |
| f | `TestJitMatchesPython.test_returns_three_rows()` |  |
| f | `TestJitMatchesPython.test_short_input_raises_indexerror_like_the_old_one()` | 比 lag 还短 —— 循环体一次都不进（只填第 lag-1 个 avg/std 之前就越界）。 |
| C | `TestCalcPeakPointV8` |  |
| f | `TestCalcPeakPointV8.test_writes_the_peak_point_column()` |  |
| f | `TestCalcPeakPointV8.test_accepts_an_existing_features_frame()` | 传 features 时就地写列（旧树就这么用的）。 |
| f | `TestCalcPeakPointV8.test_values_are_in_plus_minus_one_and_zero()` |  |
| f | `TestCalcPeakPointV8.test_first_lag_points_are_zero()` | 照抄旧实现：前 lag 个点恒为 0。 |
| C | `TestCalcPeakPoints` | 两个输入加权合成（9 - i）。 |
| f | `TestCalcPeakPoints.test_second_input_speaks_only_where_first_is_silent()` | 第一个输入有信号 → 权重 9；第二个只在第一个为 0 的位置上生效（权重 8）。 |
| f | `TestCalcPeakPoints.test_first_input_wins_on_overlap()` | 同一位置两个输入都有信号 ⇒ 第一个（权 9）胜出，因为后写的只在原值为 0 时才覆盖。 |
| f | `TestCalcPeakPoints.test_no_signal_is_zero()` |  |

### `test_pivot`
缠论中枢（盘整箱体）的回归用例。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestPivotConsolidation` |  |
| f | `TestPivotConsolidation.test_single_consolidation_is_found()` |  |
| f | `TestPivotConsolidation.test_empty_and_tiny_input_do_not_raise()` | 笔数不足成枢时返回空列表 —— 空结果不是错误。 |
| f | `TestPivotConsolidation.test_requires_one_of_the_two_inputs()` |  |
| C | `TestPivotClassify` |  |
| f | `TestPivotClassify.test_kind_per_case()` |  |
| f | `TestPivotClassify.test_per_pivot_kinds_are_one_way()` | 逐中枢标签只能 盘整 → 趋势，不来回翻转。 |
| f | `TestPivotClassify.test_pivots_to_df_takes_per_row_labels()` |  |
| C | `TestPivotInterface` |  |
| f | `TestPivotInterface.setUp()` |  |
| f | `TestPivotInterface.test_points_count_is_bi_count_plus_one()` |  |
| f | `TestPivotInterface.test_bi_list_and_points_agree()` |  |
| f | `TestPivotInterface.test_mark_normalisation_covers_czsc_enum()` | czsc 给的是 Mark.G / Mark.D，归一化后必须是 'g' / 'd'。 |
| C | `TestPivotNoLookahead` | 截断历史重算，已走完的中枢不得改变。 |
| f | `TestPivotNoLookahead.setUpClass()` |  |
| f | `TestPivotNoLookahead.test_closed_pivots_survive_truncation()` |  |
| C | `TestPivotRealData` |  |
| f | `TestPivotRealData.test_000711_60min_end_to_end()` |  |

### `test_presentation_banner`
core/presentation.py 的状态 banner —— 纯渲染、显示宽度、光标行数记账、三态。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestAnsiEnabled` |  |
| f | `TestAnsiEnabled.test_non_tty_is_off()` |  |
| f | `TestAnsiEnabled.test_stream_without_isatty_is_off()` |  |
| f | `TestAnsiEnabled.test_tty_needs_vt_support_too()` | isatty() 为真还不够 —— Windows conhost 默认不解释 ANSI， |
| C | `TestDisplayWidth` |  |
| f | `TestDisplayWidth.test_cjk_counts_two()` |  |
| f | `TestDisplayWidth.test_ascii_counts_one()` |  |
| C | `TestRenderPipelineBanner` |  |
| f | `TestRenderPipelineBanner.test_plain_text_layout()` |  |
| f | `TestRenderPipelineBanner.test_repeated_phase_name_is_printed_once()` | 连续同阶段名的行只在第一行打阶段名（K线 两行不重复打）。 |
| f | `TestRenderPipelineBanner.test_group_prefix_stripped_from_node_label()` |  |
| f | `TestRenderPipelineBanner.test_no_state_words_anywhere()` | 状态只由那个点表达 —— 不许出现「已获取 / 未获取」。 |
| f | `TestRenderPipelineBanner.test_three_states_use_two_symbols_and_three_colors()` | ·=队列中(灰) / ●=正在读取(绿) / ●=读取完成(白)。 |
| f | `TestRenderPipelineBanner.test_only_the_dot_is_coloured()` | 只给点上色 —— 节点名保持默认色（用户口径：「白●点」）。 |
| f | `TestRenderPipelineBanner.test_missing_key_counts_as_pending()` |  |
| C | `TestBannerNonTty` | 非 TTY：不上色。表头只打一次（= 作业单），之后每次变化补一行   名字 符号。 |
| f | `TestBannerNonTty.setUp()` |  |
| f | `TestBannerNonTty.test_no_escape_codes_at_all()` |  |
| f | `TestBannerNonTty.test_header_printed_at_render()` |  |
| f | `TestBannerNonTty.test_mark_appends_one_line_with_symbol()` | 表头只打一次，之后每次变化补一行 —— 不整块重打（26 节点会变 182 行噪声）。 |
| f | `TestBannerNonTty.test_header_printed_exactly_once()` |  |
| f | `TestBannerNonTty.test_unchanged_state_is_not_reprinted()` |  |
| f | `TestBannerNonTty.test_state_can_go_backwards()` | RUNNING → DONE → PENDING 都算变化 —— 别把它们当「无变化」早退。 |
| f | `TestBannerNonTty.test_unknown_name_is_ignored()` |  |
| f | `TestBannerNonTty.test_unknown_state_raises()` |  |
| f | `TestBannerNonTty.test_echo_prints_in_order()` |  |
| C | `TestBannerTty` | TTY：原地重画，source: 只写一次，回退量 = 上一次画的总行数。 |
| f | `TestBannerTty.setUp()` |  |
| f | `TestBannerTty.test_first_render_does_not_move_cursor()` |  |
| f | `TestBannerTty.test_redraw_rewinds_by_block_height()` |  |
| f | `TestBannerTty.test_rewind_tracks_growth_from_echo_window()` | 状态窗从 0 行长到 1 行 → 下一次回退量必须跟着变大，否则错位。 |
| f | `TestBannerTty.test_echo_window_limits_visible_lines()` |  |
| f | `TestBannerTty.test_close_dumps_full_log()` |  |
| f | `TestBannerTty.test_running_is_green_and_done_is_white()` |  |
| C | `TestBannerSurvivesAnActiveBar` | banner 与 tqdm 进度条共存的不变量。 |
| f | `TestBannerSurvivesAnActiveBar.setUp()` |  |
| f | `TestBannerSurvivesAnActiveBar.test_rewind_is_unchanged_across_a_closed_bar()` |  |
| C | `TestIdentityBlockAndColumn` | 开头的身份块 + 跨 banner 的阶段列（2026-10-09 改版）。 |
| f | `TestIdentityBlockAndColumn.test_identity_block_shape()` |  |
| f | `TestIdentityBlockAndColumn.test_contact_line_is_dimmed_only_on_tty()` | 版权行压暗（brew/npm 的老做法）—— 但只在真 TTY 上。 |
| f | `TestIdentityBlockAndColumn.test_identity_without_contact_is_one_line()` |  |
| f | `TestIdentityBlockAndColumn.test_header_false_prints_no_static_header()` | header=False ⇒ 一个静态头行都不打 —— 时刻戳交给身份块去打。 |
| f | `TestIdentityBlockAndColumn.test_header_true_still_prints_it()` | 默认 header=True 保持旧行为 —— 直接调用方与旧用例不受影响。 |
| f | `TestIdentityBlockAndColumn.test_phase_column_is_shared()` | ⚠️ 跨 banner 的那一列必须同宽 —— 环境自检（自检块）与 |
| f | `TestIdentityBlockAndColumn.test_aligned_row_matches_the_phase_column()` |  |
| C | `TestDimming` | 压暗（用户 2026-10-10 一次点了三处）：身份块两行、banner 头行、阶段起行。 |
| f | `TestDimming.test_dim_wraps_only_when_color()` |  |
| f | `TestDimming.test_identity_both_lines_are_dimmed()` |  |
| f | `TestDimming.test_banner_head_line_is_dimmed_on_tty_only()` |  |
| f | `TestDimming.test_banner_head_line_has_no_escape_on_a_pipe()` | 非 TTY 的头行是日志 —— 一个转义码都不许有。 |

### `test_realtime_read`
读路径的「带 REALTIME」—— kline83._merge_realtime 的两道门。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestRealtimeReadGuard` |  |
| f | `TestRealtimeReadGuard.test_missing_collection_skips_and_never_touches_it()` | ⚠️ 集合不存在 ⇒ 直接返回：既省一次 tick 读，也不建集合。 |
| f | `TestRealtimeReadGuard.test_existing_collection_calls_the_minute_merger()` |  |
| f | `TestRealtimeReadGuard.test_existing_collection_calls_the_day_merger()` |  |
| f | `TestRealtimeReadGuard.test_merge_failure_returns_history_not_none()` | 合并炸了 ⇒ 返回纯历史 —— 不能把已经读到手的数据也丢掉。 |
| f | `TestRealtimeReadGuard.test_realtime_false_never_touches_the_realtime_db()` | realtime=False ⇒ 一次都不碰实时库（这是「不带 REALTIME」该有的代价：0）。 |
| f | `TestRealtimeReadGuard.test_both_readers_default_to_realtime_true()` | ⚠️ 两族的默认都必须是 True，且必须与 base_market 的抽象声明一致。 |
| f | `TestRealtimeReadGuard.test_default_call_does_merge()` | 默认（不给 realtime）就走合并 —— 用户口径是「带 REALTIME」。 |

### `test_realtime_store`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestRealtimeCollectionName` | 集合名是跨模块契约（写入端 ↔ 读取端 ↔ purge），故逐字钉住。 |
| f | `TestRealtimeCollectionName.test_none_means_today()` |  |
| f | `TestRealtimeCollectionName.test_accepts_four_input_types()` |  |
| f | `TestRealtimeCollectionName.test_never_contains_time_of_day()` | 'realtime_{}'.format(dt.today()) 会拼出带时分秒的名字。 |
| f | `TestRealtimeCollectionName.test_name_matches_what_purge_looks_for()` | 写入端产出的名字，purge 必须认（否则保留策略静默失效）。 |
| C | `TestWriteTsRows` | _write_ts_rows 是唯一的写口，幂等性全靠它（时间序列不能 upsert）。 |
| f | `TestWriteTsRows.setUp()` |  |
| f | `TestWriteTsRows.test_delete_carries_source_and_runs_before_insert()` |  |
| f | `TestWriteTsRows.test_first_seen_covers_every_ts_of_that_code_in_the_batch()` | 重启后第一批：同一 code 的每个 ts 都要进删除条件。 |
| f | `TestWriteTsRows.test_delete_only_for_first_seen_codes()` | 稳态下不删 —— 每轮 4900 个值的 delete 实测约 3.7 秒，跑不动。 |
| f | `TestWriteTsRows.test_row_not_newer_than_last_ts_is_skipped()` |  |
| f | `TestWriteTsRows.test_last_ts_is_keyed_by_source_too()` | 同一个 code、同一个 ts，两条流互不遮挡。 |
| f | `TestWriteTsRows.test_rows_missing_code_or_ts_are_dropped()` |  |

### `test_refdata_guard`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestPartialScopeNeverDeletes` | 部分取数 × 差量删除 = 静默删掉其余全部（PITFALLS.md P19）。 |
| f | `TestPartialScopeNeverDeletes.test_partial_scope_skips_delta_delete()` |  |
| f | `TestPartialScopeNeverDeletes.test_full_scope_still_deletes_delta()` | 不给 codelist 时语义仍是「全量」，差量删除照旧（别把护栏扩大成不删）。 |
| C | `TestOnProgressCallback` | --save 的 banner 靠 on_progress 点亮节点，每条出口都得调到。 |
| f | `TestOnProgressCallback.test_fires_once_per_requested_collection()` |  |
| f | `TestOnProgressCallback.test_fires_on_every_exit_including_failure()` | 源不可用（--save qmt 的常态）时也要回调，否则节点永远不亮。 |
| f | `TestOnProgressCallback.test_absent_callback_is_harmless()` | 默认 on_progress=None：除 --save 外的调用方不受任何影响。 |
| C | `TestEtfListReachableFromSaveTdx` | --save tdx 现在也做 etf_list。⚠️ 它只能由 akshare 供。 |
| f | `TestEtfListReachableFromSaveTdx.test_tdx_source_can_supply_etf_list()` |  |
| f | `TestEtfListReachableFromSaveTdx.test_qmt_source_does_not_claim_etf_list()` | qmt 适配器没声明 etf_list —— 列进去只会每次报一遍 UnsupportedCollection。 |
| f | `TestEtfListReachableFromSaveTdx.test_akshare_is_first_for_etf_list()` |  |
| f | `TestEtfListReachableFromSaveTdx.test_neither_tdx_side_source_supplies_etf_list()` | pytdx / qmt 都不提供 etf_list → akshare 是唯一可能，不能排除它。 |
| C | `TestRefdataTtlGate` | 刷新闸：距上次成功完成不足 ttl_hours 就跳过、连源都不选。 |
| f | `TestRefdataTtlGate.test_fresh_is_skipped_and_source_untouched()` |  |
| f | `TestRefdataTtlGate.test_stale_is_fetched()` |  |
| f | `TestRefdataTtlGate.test_never_succeeded_is_fetched()` | 从没成功过（age=None）按「该取」处理 —— 别把新库冻住。 |
| f | `TestRefdataTtlGate.test_no_ttl_means_no_gate()` | 默认 ttl_hours=None = 不做闸，行为与从前逐字相同。 |

### `test_regtree_jit`
analysis/regtree_jit.py —— jit 版与 regtree.py 的对拍。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestSplitSearchMatches` | 切分搜索必须逐值一致 —— 它是整棵树的骨架。 |
| f | `TestSplitSearchMatches.test_same_split_on_random_data()` |  |
| f | `TestSplitSearchMatches.test_no_split_returns_none_feature()` | 所有 y 相同 ⇒ 不切分（旧版走 set(...) == 1 那条早退）。 |
| C | `TestEndToEndMatches` | 端到端：离散列逐值相同，连续列在浮点容差内。 |
| f | `TestEndToEndMatches.setUpClass()` |  |
| f | `TestEndToEndMatches.test_discrete_columns_are_bit_identical()` |  |
| f | `TestEndToEndMatches.test_continuous_columns_within_tolerance()` |  |
| C | `TestSpeedup` | 加速要有数字（用户口径）。这里只做松断言：CI 机器抖动不该让用例红。 |
| f | `TestSpeedup.test_jit_is_at_least_three_times_faster()` |  |
| C | `TestStatusWithoutNumba` | regtree_jit_status / available 不管 numba 在不在都要能答。 |
| f | `TestStatusWithoutNumba.test_status_shape()` |  |

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
| f | `TestStockCNSingleton.test_cli_auto_register_coordination()` | CLI 自动注册与模块自动注册的协调：已注册的不被顶掉。 |
| f | `TestStockCNSingleton.test_market_name_property()` | 测试市场名称属性 |
| f | `TestStockCNSingleton.test_exchange_codes_property()` | 测试交易所代码属性 |

### `test_subscribers`

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestSubscriberRegistry` | GQSUBSCRIBER 的两条契约（cli/__main__.py 的 --sub 依赖它们）。 |
| f | `TestSubscriberRegistry.test_legacy_tencent_key_points_to_l1()` | 老树的 --sub tencent 必须仍可用，且与 l1_tencent 是同一个函数。 |
| f | `TestSubscriberRegistry.test_every_subscriber_is_callable_without_args()` | 表里每个订阅函数都必须能零参调用 —— 否则 CLI 一跑就 TypeError。 |
| f | `TestSubscriberRegistry.test_expected_keys_are_present()` | 当前应支持的三条实时订阅键（新增/改名时这个用例会红 —— 那是有意的）。 |

### `test_symbol_classify`
symbol.is_stock_cn 的号段分类回归测试。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestIsStockCnSegments` | 逐号段的 (code, 期望 market_type, 期望交易所)。 |
| f | `TestIsStockCnSegments.test_segments()` |  |
| f | `TestIsStockCnSegments.test_etf_is_not_index()` | 本次改动的核心断言：ETF 与真指数必须是两个类型。 |
| C | `TestExchangeTaggedForms` | 长度 >6 的带交易所标记写法。 |
| f | `TestExchangeTaggedForms.test_exchange_tokens()` |  |
| f | `TestExchangeTaggedForms.test_owner_named_forms()` | 所有者点名的四种写法 —— 点号/紧贴 × 前缀/后缀，四种都要认。 |
| f | `TestExchangeTaggedForms.test_dotted_matrix()` | 点号写法的完整矩阵 —— 前后缀 × 三个交易所，大小写都收。 |
| f | `TestExchangeTaggedForms.test_bj_prefix_no_longer_false()` | bj430489 曾被判成"不是 A 股"（返回 False），并连带丢掉交易所。 |
| f | `TestExchangeTaggedForms.test_never_returns_none()` | 绝不能返回 None —— 全树按位置解包，None[1] 会 TypeError。 |
| f | `TestExchangeTaggedForms.test_unknown_token_is_not_guessed()` | 不认识的 token 不猜。 |
| f | `TestExchangeTaggedForms.test_explicit_token_beats_segment()` | 显式交易所优先 —— 即使号段暗示另一个交易所。 |
| C | `TestIsStockCnContract` | 返回契约与输入容忍度 —— 全树十余处按位置解包，形状不能变。 |
| f | `TestIsStockCnContract.test_returns_four_tuple()` |  |
| f | `TestIsStockCnContract.test_list_takes_first_element()` | 列表输入只看第一个 —— 沿袭原行为，勿改（调用方依赖它）。 |
| f | `TestIsStockCnContract.test_unknown_code()` |  |
| f | `TestIsStockCnContract.test_empty_code()` |  |
| f | `TestIsStockCnContract.test_suffix_forms()` | 带交易所后缀的写法要归一化到同一结论。 |
| f | `TestIsStockCnContract.test_longest_prefix_wins()` | 包含关系必须由最长前缀解决，而不是靠书写顺序。 |

### `test_tdx_hosts`
markets/StockCN/datasource/tdx_hosts.py —— 服务器池的候选 / 探活 / 缓存。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestCandidates` |  |
| f | `TestCandidates.test_returns_deduped_ip_port_pairs()` |  |
| f | `TestCandidates.test_never_empty_without_builtin_pool()` | 拿不到 pytdx 包内池时，也要有东西可试 —— 否则源直接不可用。 |
| C | `TestCacheRoundTrip` |  |
| f | `TestCacheRoundTrip.setUp()` |  |
| f | `TestCacheRoundTrip.test_save_then_load_keeps_order()` |  |
| f | `TestCacheRoundTrip.test_saved_file_records_when_and_why()` | 缓存要能自证「什么时候探的、当时多快」—— 排障全靠它。 |
| f | `TestCacheRoundTrip.test_missing_cache_is_none()` |  |
| f | `TestCacheRoundTrip.test_stale_cache_is_none()` | 超过 REFRESH_DAYS 就当没有 —— 这是「每周一次」的实现方式（靠 mtime）。 |
| f | `TestCacheRoundTrip.test_corrupt_cache_is_none_not_crash()` | 缓存坏了必须静默退回，不能让取数链跑不起来。 |
| f | `TestCacheRoundTrip.test_refresh_skips_probing_when_cache_is_fresh()` |  |
| f | `TestCacheRoundTrip.test_refresh_keeps_old_cache_when_nothing_probed()` | 探测全灭（网络抖动）时保留旧缓存，别把可用列表清空 —— 那是自毁。 |
| C | `TestLoopbackIsNeverATdxServer` | ⚠️ 127.0.0.1 绝不能被判成一台 TDX 服务器（用户 2026-10-09 明确）。 |
| f | `TestLoopbackIsNeverATdxServer.test_loopback_is_rejected_without_probing()` | 连探都不探 —— 判据是地址本身，不是"连不上"。 |
| f | `TestLoopbackIsNeverATdxServer.test_all_loopback_and_unspecified_forms()` |  |
| f | `TestLoopbackIsNeverATdxServer.test_pool_never_returns_loopback()` | 整池探活也不许把回环混进结果 —— 它会被写进缓存、被 TdxSource 选中。 |
| f | `TestLoopbackIsNeverATdxServer.test_domain_name_is_not_loopback()` | is_loopback 只认 IP 字面量；域名（shtdx.gtjas.com）不是回环。 |
| C | `TestProbe` |  |
| f | `TestProbe.test_unreachable_host_is_not_ok()` | 连不上 → ok=False。不能只看"没抛异常" —— pytdx 连不上时返回 None。 |
| f | `TestProbe.test_none_return_from_the_wire_counts_as_failure()` | 真连一个连不上但会返回 None 的地址 —— 端口 1，pytdx 约 7s 后返回 None。 |
| f | `TestProbe.test_probe_pool_sorts_by_median()` |  |
| C | `TestHostOrderPerThread` | 服务器选择的顺序：每个 worker 线程各粘各的，首次用随机起点。 |
| f | `TestHostOrderPerThread.test_order_is_a_permutation_of_hosts()` |  |
| f | `TestHostOrderPerThread.test_sticks_to_this_thread_s_last_good_host()` |  |
| f | `TestHostOrderPerThread.test_two_threads_are_independent()` | 两个线程互不影响 —— 这正是「分布式」：各粘各的，不会挤一起。 |
| f | `TestHostOrderPerThread.test_fresh_thread_shuffles()` | 新线程没有粘性 → 起点是整个列表的一个排列（随机打散）。 |

### `test_timestamp_format`
测试时间戳格式

| | 名称 | 摘要 |
|:--|:--|:--|
| f | `test_timestamp_format()` | 测试时间戳格式转换 |

### `test_timing`
analysis/timing.py —— 从旧树搬回的时序累积器与金叉/死叉间隔。

| | 名称 | 摘要 |
|:--|:--|:--|
| C | `TestAccumulators` | 两个累积器的清零条件相反 —— 这是最容易抄错的一处。 |
| f | `TestAccumulators.test_integral_clears_on_zero()` | Timeline_Integral：Tm[i]==0 清零（死叉 1→0）。 |
| f | `TestAccumulators.test_duration_clears_on_one()` | Timeline_duration：Tm[i]==1 清零（金叉 0→1）。 |
| f | `TestAccumulators.test_first_element_uses_numpy_negative_index()` | i=0 时 T[-1] 取到数组最后一个元素（新数组全 0）—— 依赖 numpy 负索引。 |
| f | `TestAccumulators.test_integral_only_meaningful_for_binary_input()` | ⚠️ Timeline_Integral 只对 0/1 输入有意义 —— 它是给二值信号的。 |
| C | `TestEventTimingLag` |  |
| f | `TestEventTimingLag.test_sign_convention()` | 金叉取正、死叉取负；<=0 一律走负分支。 |
| f | `TestEventTimingLag.test_feature_lag_first_element_is_minus_one()` | ⚠️ 第一个元素是 -1，不是 +1 —— shift(1) 给 NaN ⇒ 两个比较都是 False |
| f | `TestEventTimingLag.test_returns_int32()` |  |
| C | `TestEnergy` |  |
| f | `TestEnergy.test_same_sign_accumulates_and_flips_restart()` | 同号累加、异号重启。 |
| f | `TestEnergy.test_dtype_dispatch()` | float64 → f8 内核；float32 → f4 内核（返回值 dtype 随之不同）。 |
| f | `TestEnergy.test_accepts_series_and_ndarray()` |  |
| f | `TestEnergy.test_other_dtype_falls_back_to_float32()` | int 输入走 astype(float32) 那条兜底路（不是 f8）。 |
| f | `TestEnergy.test_all_zeros_is_all_zeros()` |  |
| f | `TestEnergy.test_length_one_and_empty()` |  |
| C | `TestResampleIsImportable` | 对齐函数要能用（细节由筹码分布那条链覆盖；这里只钉「在且可调」）。 |
| f | `TestResampleIsImportable.test_importable()` |  |
| C | `TestMatchesOldTree` | 对拍：同一批随机输入，与旧树的同名实现逐值比。 |
| f | `TestMatchesOldTree.setUpClass()` |  |
| f | `TestMatchesOldTree.test_all_six_match()` |  |

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
