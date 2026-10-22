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

"""
GolemQ Constants Module

This module provides centralized constant definitions for the GolemQ project,
implemented as a singleton pattern (AKA class) for easy access and management.
"""


class _StubMeta(type):
    """未定义的常量一律 **抛 AttributeError**。

    此前它返回 ``f'{类名小写}_{属性名小写}'`` —— 一个「看起来合理」的假名。
    后果见 ``PITFALLS.md`` P1：假名在数据里从不存在，取值静默失败，
    回测「跑通了」却零成交，**没有任何报错**。

    ⚠️ **不要改回返回值。** 若某常量抛 AttributeError，正解是从
    ``GolemQ_old/utils/parameter.py`` 取它的**真实值**补上 ——
    那才是该字段在 MongoDB 里的实际存储名（如 ``'ZPLagMajR'``）。
    """
    def __getattr__(cls, name):
        raise AttributeError(
            f'{cls.__name__}.{name} 未定义。本类不再伪造常量名'
            f'（见 PITFALLS.md P1）。请从 GolemQ_old/utils/parameter.py '
            f'取真实值补上 —— 那是数据在 MongoDB 里的实际字段名。')


class MARKET_TYPE():
    """市场种类

    日线 尾数01
    分钟线 尾数02
    tick 尾数03

    市场:
    股票 0
    指数/基金 1
    期货 2
    港股 3
    美股 4
    比特币/加密货币市场 5
    """
    STOCK_CN = 'stock_cn'  # 中国A股
    STOCK_CN_B = 'stock_cn_b'  # 中国B股
    STOCK_CN_D = 'stock_cn_d'  # 中国D股 沪伦通
    STOCK_HK = 'stock_hk'  # 港股
    STOCK_US = 'stock_us'  # 美股
    FUTURE_CN = 'future_cn'  # 国内期货
    OPTION_CN = 'option_cn'  # 国内期权
    STOCKOPTION_CN = 'stockoption_cn'  # 个股期权
    # BITCOIN = 'bitcoin'  # 比特币
    CRYPTOCURRENCY = 'cryptocurrency'  # 加密货币(衍生货币)
    INDEX_CN = 'index_cn'  # 中国指数
    ETF_CN = 'etf_cn'     # 中国ETF场内基金
    FUND_CN = 'fund_cn'   # 中国基金
    BOND_CN = 'bond_cn'  # 中国债券


class BROKER_TYPE():
    """执行环境

    回测
    模拟
    实盘
    随机(按算法/分布随机生成行情)/仅用于训练测试
    """

    BACKTEST = 'backtest'
    SIMULATION = 'simulation'
    REAL = 'real'
    RANDOM = 'random'
    SHIPANE = 'shipane'
    TTS = 'tts'


class EVENT_TYPE():
    """[summary]
    """

    BROKER_EVENT = 'broker_event'
    ACCOUNT_EVENT = 'account_event'
    MARKET_EVENT = 'market_event'
    TRADE_EVENT = 'trade_event'
    ENGINE_EVENT = 'engine_event'
    ORDER_EVENT = 'order_event'


class MARKET_EVENT():
    """交易前置事件
    query_order 查询订单
    query_assets 查询账户资产
    query_account 查询账户
    query_cash 查询账户现金
    query_data 请求数据
    query_deal 查询成交记录
    query_position 查询持仓
    """

    QUERY_ORDER = 'query_order'
    QUERY_ASSETS = 'query_assets'
    QUERY_ACCOUNT = 'query_account'
    QUERY_CASH = 'query_cash'
    QUERY_DATA = 'query_data'
    QUERY_DEAL = 'query_deal'
    QUERY_POSITION = 'query_position'


class ENGINE_EVENT():
    """引擎事件"""
    MARKET_INIT = 'market_init'
    UPCOMING_DATA = 'upcoming_data'
    UPCOMING_TICK = 'upcoming_tick'
    UPCOMING_BAR = 'upcoming_bar'
    BAR_SETTLE = 'bar_settle'
    DAILY_SETTLE = 'daily_settle'
    UPDATE = 'update'
    TRANSACTION = 'transaction'
    ORDER = 'order'


class ACCOUNT_EVENT():
    """账户事件"""
    UPDATE = 'account_update'
    SETTLE = 'account_settle'
    MAKE_ORDER = 'account_make_order'


class BROKER_EVENT():
    """BROKER事件
    BROKER
    有加载数据的任务 load data
    撮合成交的任务 broker_trade

    轮询是否有成交记录 query_deal

    """
    LOAD_DATA = 'load_data'
    TRADE = 'broker_trade'
    SETTLE = 'broker_settle'
    DAILY_SETTLE = 'broker_dailysettle'
    RECEIVE_ORDER = 'receive_order'
    QUERY_DEAL = 'query_deal'
    NEXT_TRADEDAY = 'next_tradeday'


class ORDER_EVENT():
    """订单事件

    创建订单 create
    交易 trade
    撤单 cancel

    """
    CREATE = 'create'
    TRADE = 'trade'
    CANCEL = 'cancel'
    FAIL = 'fail'


class FREQUENCE():
    """查询的级别

    YEAR = 'year'  # 年bar
    QUARTER = 'quarter'  # 季度bar
    MONTH = 'month'  # 月bar
    WEEK = 'week'  # 周bar
    DAY = 'day'  # 日bar
    ONE_MIN = '1min'  # 1min bar
    FIVE_MIN = '5min'  # 5min bar
    FIFTEEN_MIN = '15min'  # 15min bar
    THIRTY_MIN = '30min'  # 30min bar
    HOUR = '60min'  # 60min bar
    SIXTY_MIN = '60min'  # 60min bar
    TICK = 'tick'  # transaction
    ASKBID = 'askbid'  # 上下五档/一档
    REALTIME_MIN = 'realtime_min' # 实时分钟线
    LATEST = 'latest'  # 当前bar/latest

    2019/08/06 @yutiansut
    """

    YEAR = 'year'  # 年bar
    QUARTER = 'quarter'  # 季度bar
    MONTH = 'month'  # 月bar
    WEEK = 'week'  # 周bar
    DAY = 'day'  # 日bar
    ONE_MIN = '1min'  # 1min bar
    FIVE_MIN = '5min'  # 5min bar
    FIFTEEN_MIN = '15min'  # 15min bar
    THIRTY_MIN = '30min'  # 30min bar
    HOUR = '60min'  # 60min bar
    SIXTY_MIN = '60min'  # 60min bar
    TICK = 'tick'  # transaction
    ASKBID = 'askbid'  # 上下五档/一档
    REALTIME_MIN = 'realtime_min'  # 实时分钟线
    LATEST = 'latest'  # 当前bar/latest


class CURRENCY_TYPE():
    """货币种类"""
    RMB = 'rmb'  # 人民币
    USD = 'usd'  # 美元
    EUR = 'eur'  # 欧元
    HKD = 'hkd'  # 港币
    GBP = 'GBP'  # 英镑
    BTC = 'btc'  # 比特币
    JPY = 'jpy'  # 日元
    AUD = 'aud'  # 澳元
    CAD = 'cad'  # 加拿大元


class DATASOURCE():
    """数据来源
    """

    WIND = 'wind'  # wind金融终端
    QMT = 'xtquant'  # 讯投 MiniQMT
    TDB = 'tdb'  # wind tdb
    THS = 'ths'  # 同花顺网页
    TUSHARE = 'tushare'  # tushare
    TDX = 'tdx'  # 通达信
    MONGO = 'mongo'  # 本地/远程Mongodb
    EASTMONEY = 'eastmoney'  # 东方财富网
    CHOICE = 'choice'  # choice 金融终端
    CCXT = 'ccxt'  # github/ccxt 虚拟货币
    LOCALFILE = 'localfile'  # 本地文件
    AUTO = 'auto'  # 优先从Mongodb中读取数据，不足的数据从tdx下载


class AKA(metaclass=_StubMeta):
    """
    A singleton class to manage all global constants in GolemQ.
    """
    _instance = None

    # 基础价格类型
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'
    CLOSE_RAW = 'closeRaw'
    UNCLOSED = 'unclosed'
    CLOSE_MAJOR = 'closeMaj'
    LAST_CLOSE = 'last_close'
    OPEN_LASTDAY = 'openLastday'
    PRICE = 'price'

    # 成交量与金额
    VOLUME = 'volume'
    VOL = 'vol'
    VOLUME_S1 = 'volumeS1'
    AMOUNT = 'amount'
    AMOUNT_MAJOR = 'amountMaj'
    AMOUNT_S1 = 'amountS1'
    AMOUNT_SUM = 'amountSum'
    TURNOVER = 'turnover'
    TURNOVER_RATE = 'TurnoverRate'

    # 标识与时间
    CODE = 'code'
    ISOCODE = 'isocode'
    FULL_SYMBOL = 'fullSymbol'
    MARKET_TYPE = 'marketType'
    NAME = 'name'
    DATE = 'date'
    DATETIME = 'datetime'
    STATISTICAL_DATE = 'StatDate'
    IPO_DATE = 'IPODate'
    EXPIRED_TIME = 'ExpireTimestamp'

    # 时间周期
    DAILY = '1d'
    WEEKLY = '1w'
    MONTHLY = '1M'

    # 技术分析
    DELTA = 'delta'
    SUPPORT = 'support'
    OVERALL = 'overall'
    METAPHASE = 'metaphase'
    COMPREHENSIVE = 'comprehensive'
    TPS_SUPPORT = 'TpsSupport'
    TPS_PRESSURE = 'TpsPress'
    BUY_SIGNAL = 'BuySignal'
    TECHNICAL_PATTERN = 'TechPatt'
    SHORT = 'short'

    # 百分比与排名
    PCT_CHANGE = 'pctChg'
    PERCENT_CHANGE = 'PctChg'
    PERCENT_CHANGE_COMBO = 'PctChgCmb'
    PCT_RANKING = 'pctRank'
    PCT_RANK_CHECKPOINT = 'PctRnkChkPOI'

    # 熵值分析
    SAMPLE_ENTROPY = 'SE'
    SAMPLE_ENTROPY_DUMMY = 'SED'

    # 评分系统
    SCORES = 'scores'
    NEWS_SCORE = 'NewsScr'
    BASELINE_SCORE = 'BslScr'
    TECH_SCORE = 'TechScr'
    TREND_SCORE = 'TrdScr'

    # 热度排名
    HOT_RANK = 'HotRnk'
    HOT_SCORE = 'HotScr'
    HOT_POSITION = 'HotPos'
    HOT_RANK_CHECKPOINT = 'HotRnkChkPOI'

    # 股东与市值
    CAPITALIZATION = 'capitalization'
    MARKET_CAPITALIZATION = 'MarketCap'
    STOCKHOLDER_COUNT = 'StoHldrCnt'
    STOCKHOLDER_COUNT_MEAN = 'StoHldrCntMean'
    STOCKHOLDER_PREVIOUS = 'StoHldrPrev'
    STOCKHOLDER_CHANGE = 'StoHldrChg'
    STOCKHOLDER_CHANGE_COMBO = 'StoHldrChgCmb'
    STOCKHOLDER_CHANGE_PERCENT = 'StoHldrChgPct'
    STOCKHOLDER_CHANGE_PERCENT_COMBO = 'StoHldrChgPctCmb'
    STOCKHOLDER_VALUE = 'StoHldrVal'
    STOCKHOLDER_CAPITALIZATION = 'StoHldrCap'
    TOTAL_EQUITY = 'TotalEqu'
    CHANGES_IN_EQUITY = 'ChgInEqu'
    CHANGES_IN_EQUITY_REASON = 'ChgInEquR'

    # DDE数据
    DDE_NUMBER_OF_INDIVIDUAL_INVESTORS = 'DDENoOfIndInvs'
    DDE_NUMBER_POSITION = 'DDEPos'
    DDE_NUMBER_RANK = 'DDERnk'

    # 系统
    SYSTEM_NAME = 'GolemQuant'
    TRADE_CAL = 'tradeCalendar'
    COST5_PRICE = 'cost5Prz'
    RETURN_COST95 = 'RetCost95'
    COST15_PRICE = 'cost15Prz'
    CATALOG = 'catalog'
    LENGTH = 'length'
    FROZEN = 'FrozenExpired'
    CHECKPOINT = 'ChkPoi'
    STAGE = 'STAGE'

    # ── 2026-10-21 补：ChipDistribution 用到的列名（真值逐字取自老树
    # `utils/parameter.py::AKA`；MongoDB 实际存储名，别改写）──
    CHIP_DENSITY = 'ChipDensity'
    COST5_BOOTSTRAP_BEFORE = 'cost5BstBf'
    COST5_BOOTSTRAP_CANDIDATE = 'cost5BstCand'
    COST5_BOOTSTRAP_CANDIDATE_BEFORE = 'cost5BstCandBf'
    COST85_PRICE = 'cost85Prz'
    COST90 = 'cost9'
    COST90_CHANNEL = 'cost9W'
    COST95_ENDPOINT_CANDIDATE = 'cost95EpCand'
    COST95_ENDPOINT_CANDIDATE_BEFORE = 'cost95EpCandBf'
    COST95_PRICE = 'cost95Prz'
    FINAL_COST90 = 'FIN_COST9'
    FINAL_PROFIT = 'FIN_PROF'
    PROFIT = 'profit'
    RETURN_COST5 = 'RetCost5'

    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(AKA, cls).__new__(cls)
        return cls._instance


class FIELD(metaclass=_StubMeta):
    """
    A singleton class to manage all global constants in GolemQ.
    """
    _instance = None

    # OHLC price types
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'
    TURNOVER_RATE = 'TurnoverRate'
    PCT_CHANGE = 'pctChg'

    DATE = 'date'
    DATETIME = 'datetime'
    CODE = 'code'

    # Common intervals
    DAILY = '1d'
    WEEKLY = '1w'
    MONTHLY = '1M'

    PE_RATION = 'PERation'

    # --- 千股千评（东财 `ak.stock_comment_em`）其余几列 -------------------------
    # ⚠️ 这批**不是**从旧树 `parameter.py` 取的 —— 那里根本没有它们。旧爬虫
    # （`GolemQ_old/fetch/crawler.py:1626-1646`）用的是**字面量**做的 `rename`，
    # 所以**真值就是那些字面量本身**（它决定了当年写进 Mongo 的字段名）。
    # 搬过来时按本项目「字段名一律进 `FIELD`」的约定收拢到一处。
    ZLCB = 'ZLCB'                     # 主力成本
    JGCYD = 'JGCYD'                   # 机构参与度
    TOTAL_SCORE = 'TotalScore'        # 综合得分
    RANKING_UP = 'RankingUp'          # 排名上升
    RANKING = 'Ranking'               # 目前排名
    FOCUS = 'Focus'                   # 关注指数
    CHANGE_PERCENT = 'ChangePercent'  # 涨跌幅（÷100 后的小数）

    # --- 同花顺资金流（`ak.stock_fund_flow_individual`）----------------------
    # ⚠️ 这四个是**全新**的列：旧树没有同花顺资金流（`--moneyflow ths` 是假分支，
    # 与 eastmoney 并进同一段东财代码）。命名由本轮定，前缀 `THS_` 标明来路。
    THS_INFLOW = 'thsInflow'
    THS_OUTFLOW = 'thsOutflow'
    THS_NET = 'thsNet'
    THS_AMOUNT = 'thsAmount'

    # 资金流
    MONEYFLOW_PERCENT = 'moneyflowPct'
    MONEYFLOW_VOLUME = 'moneyflowVolume'
    MONEYFLOW_IN = 'moneyflowIn'
    MONEYFLOW_OUT = 'moneyflowOut'
    MONEYFLOW_SCORE = 'MoneyflowScr'
    MONEYFLOW_VOLUME_MINOR = 'moneyflwVolMin'
    MONEYFLOW_PERCENT_MINOR = 'moneyflwPctMinor'

    # Trading calendar
    TRADE_CAL = 'tradeCalendar'  # 交易日历常量
    PCT_CHANGE_MAJOR = 'pctChgMaj'
    DEA_ZERO_TIMING_LAG_MAJOR = 'deaOLagMaj'
    CHARTSCAN_BUY_BEFORE_DUMMY = 'ChrScBuyBfD'
    CHARTSCAN_BUY_BEFORE_MINOR_DUMMY = 'ChrScBuyBfMinD'
    MACD_CROSS_SX_BEFORE_MAJOR = 'macdSxBfMaj'
    #: 60min 的 MACD 死叉计数（真值取自老树 `utils/parameter.py`）。
    MACD_CROSS_SX_BEFORE = 'MACD_SX_BF'
    #: 日线 DEA（老树同名字段，`DEA_MAJ`）。
    DEA_MAJOR = 'DEA_MAJ'
    CHARTSCAN_BUY_BEFORE_MAJOR_DUMMY = 'ChrScBuyBfMajD'
    ZEN_DASH_TIMING_LAG_DUMMY = 'zDLagD'
    #: 笔/线段的 ZEN 状态原始列（老树 `utils/parameter.py` 真值）。
    ZEN_DASH_TIMING_LAG = 'ZEN_D_LAG'
    ZEN_PEAK_TIMING_LAG = 'ZEN_P_LAG'
    RENKO_BOOST_S_TIMING_LAG = 'renkosBstLag'
    MACD_MAJOR = 'macdMaj'
    STOCK_SCORE_M15 = 'scoreM15'
    MAXFACTOR = 'MAXFACTOR'
    MAXFACTOR_MAJOR = 'mftMaj'
    MACD_ZERO_TIMING_LAG = 'macdOLag'
    BOLL_LB_HMA5_TIMING_LAG = 'bollLHma5Lag'
    DEA_NORM = 'deaNor'
    DIF_NORM = 'difNorm'
    MTM_NORM = 'mtmNorm'
    MAPOWER30 = 'MPWR3'
    MACD_ZERO_TIMING_LAG_MAJOR = 'macdOLagMaj'
    BOLL_LB_HMA5_TIMING_LAG_MAJOR = 'bollLHma5LagMaj'
    DEA_NORM_MAJOR = 'deaNormMaj'
    DIF_NORM_MAJOR = 'difNormMaj'
    MTM_NORM_MAJOR = 'mtmNormMaj'
    MAPOWER30_MAJOR = 'mpwr3Maj'
    MA5_MAJOR = 'maVMaj'
    MA10_MAJOR = 'maXMaj'
    #: Prophet 预测区间（**存的是相对同一 bar close 的比值**，不是绝对价 ——
    #: 与 `GAR_P` 同一条纪律：绝对价一落库就跟着复权过期。见 `D53`。
    #: `1.0` = 与 close 持平；还原价 = 比值 × 当日 close。
    FBPROPHET_LB_MAJOR = 'prophetLbMaj'
    FBPROPHET_UB_MAJOR = 'prophetUbMaj'
    FBPROPHET_LB = 'prophetLb'
    FBPROPHET_UB = 'prophetUb'
    #: Prophet 区间净空（D53：新树落**比值**，旧树是绝对价；模型层迁移时换算）。
    FBPROPHET_CLEARANCE = 'PROPHET_CLR'
    FBPROPHET_CLEARANCE_MAJOR = 'PROPHET_CLR_MAJ'
    FBPROPHET_CLEARANCE_NORM = 'PROPHET_CLR_NORM'
    FBPROPHET_CLEARANCE_NORM_MAJOR = 'PROPHET_CLR_NORM_MAJ'
    GARIDENT_PRICE = 'garP'
    GARIDENT_PRICE_COUNT = 'garPCount'
    GARIDENT_PRICE_CLEARANCE = 'garPClr'
    #: `GAR_P` 那组带是**哪个 checkpoint** 算出来的（日期字符串 `'YYYY-MM-DD'`）。
    #: 8.3 起 `GAR_P` 落的是**相对锚 bar close 的比值**（见 `docs/decisions/D49.md`：
    #: 绝对价位只存在 K 线里，指标一律存比值）—— 还原必须知道乘哪一个日期的
    #: （已复权的）close，所以锚日期得跟着落库。**老树没有这一列**（老树存绝对价）。
    GARIDENT_PRICE_ANCHOR = 'garPAnchor'
    GARIDENT_STAGE_TIMING_LAG = 'garSLag'
    # ⚠️ 老树那四列 `STOCK_TREND_CHECKOUT_CLOSE` / `STOCK_TREND_RETURNS` /
    # `STOCK_TREND_BEFORE` / `STOCK_TREND_RETURNS_GARIDENT` **没有搬进来**
    # （2026-10-16，用户定）：它们**老树从没落过库**、新树零消费，整条链在
    # `analysis/stages.py` 里连计算一起删了。老树真值是 `TRD_CHK_CLOSE` / `TRD_RET` /
    # `TRD_BEFORE` / `TRD_RET_GAR`（`GolemQ_old/utils/parameter.py` 的 INDICATOR_FIELD）
    # —— **别照着老树加回来**，先找出消费方。见 `docs/decisions/D49.md`。
    BOOTSTRAP_STAGE_BEFORE = 'bstStgBf'
    ENDPOINT_STAGE_BEFORE = 'epStgBf'
    STOCK_SCORE_M15_NORM = 'ssm15Norm'
    MAGIC_NINE_TURNS = 'm9t'
    MAGIC_NINE_TURNS_TIMING_LAG = 'm9tLag'
    MAGIC_NINE_TURNS_MAJOR = 'm9tMaj'
    MAGIC_NINE_TURNS_BASELINE = 'm9tBsl'
    MAGIC_NINE_TURNS_BASELINE_TIMING_LAG = 'm9tBslLag'
    MAGIC_NINE_TURNS_BEFORE = 'm9tBf'
    MAGIC_NINE_TURNS_SX_BEFORE = 'm9tSxBf'
    # Magic Nine Turns 的两个 dummy 列。老值 `m9tDmy` / `M9TWekDmy`。
    # ⚠️ 后者老树里是**大小写不统一**的（同族其余全是 `m9t` 小写前缀），
    # 这里统一到 `m9tWekDmy` —— 同属 D41 的「未入库、纯计算」，三道查已过。
    # ⚠️ **这两个常量此前缺失，而 `ChipDistribution_jit.py:1120` 正在用它们** ⇒
    # `_StubMeta` 早已不再伪造（`MIGRATION_STATUS` #3），所以那次调用会直接抛
    # `AttributeError`。补上即修好（见 `docs/decisions/D49.md`）。
    MAGIC_NINE_TURNS_DUMMY = 'm9tDmy'
    MAGIC_NINE_TURNS_WEEKLY_DUMMY = 'm9tWekDmy'
    # —— Poly9 家族的中间列 ——
    # `POLYNOMIAL9_NORM` 老值 `poly9pwr3` **无下划线** ⇒ 按 D41 第 1 条**原样不动**。
    POLYNOMIAL9_NORM = 'poly9pwr3'
    POLYNOMIAL9_NORM_MAJOR = 'poly9pwr3Maj'
    POLYNOMIAL9_DELTA = 'poly9Delta'
    POLYNOMIAL9_MAJOR = 'poly9Maj'
    POLYNOMIAL9_WEEKLY = 'poly9Wek'
    POLYNOMIAL9_NORM_WEEKLY = 'poly9NorWek'
    RENKO_TREND_S_TIMING_LAG = 'renkosBarLag'
    RENKO_TREND_S_LB = 'renkosBarLb'
    RENKO_TREND_S_UB = 'renkosBarUb'
    RENKO_TREND = 'renkoBar'
    RENKO_TREND_S = 'renkoSbar'
    RENKO_PRICE_S = 'renkoSprice'
    RENKO_TREND_S_BEFORE = 'renkosBarBf'
    RENKO_S_JX_BEFORE = 'renkoSbarJxBf'
    RENKO_TREND_L = 'renkoLbar'
    RENKO_PRICE_L = 'renkoLprice'
    RENKO_TREND_L_LB = 'renkolBarLb'
    RENKO_TREND_L_UB = 'renkolBarUb'
    RENKO_TREND_L_BEFORE = 'renkoLbarBf'
    RENKO_TREND_L_JX_BEFORE = 'renkoLbarJxBf'
    RENKO_TREND_L_TIMING_LAG = 'renkoLbarLag'
    RENKO_BOOST_L_TIMING_LAG = 'renkoLboostLag'
    RENKO_OPTIMAL = 'renkoOpt'
    # ⚠️ 上面这三个值是**对旧树的刻意偏离**（用户 2026-10-11 定）：
    # 旧树 `parameter.py` L1850/L1855/L1856 的真值是 `RENKO_BAR` /
    # `RENKO_lBAR_LB` / `RENKO_lBAR_UB`（**全大写**），而 S 侧同族是
    # `renkosBarLb` / `renkosBarUb`（**驼峰**）—— 旧树自己 L/S 两侧不一致。
    # 这次把 L 侧这三个**对齐到 S 侧的驼峰风格**，理由：
    #   1. **零兼容成本**：实测全库扫描 + 全仓 grep，这三个字段名
    #      **从来没有被写进任何集合**，也没有第二处引用（用户 2026-10-11 确认）。
    #   2. 留着两套风格，下一个搬 RENKO 的人**一定会问哪边是对的**，
    #      而「哪边都对、只是历史」是个只增不减的解释负担。
    # ⚠️ 其余 11 个 RENKO_* 值**原样保留**（`RENKO_lBAR` / `RENKO_lPRICE` …）——
    # 本轮只动用户点名的这三个。**别"顺手"把整族都改了**：
    # 那几个要与旧树对拍/对读，改了就得重新论证。
    STOCK_SCORE_M15_TIMING_LAG = 'scoreM15Lag'
    STOCK_SCORE_M15_NORM_TIMING_LAG = 'ssm15NormLag'
    POLYNOMIAL9 = 'poly9'
    ATR_Stopline_TIMING_LAG = 'atrStoplineLag'
    ATR_SuperTrend_TIMING_LAG = 'atrSuptrdLag'
    ATR_Stopline_TIMING_LAG_MAJOR = 'atrStopLagMaj'
    ATR_SuperTrend_TIMING_LAG_MAJOR = 'atrSuptrdLagMaj'
    ATR_Stopline_PRICE_MAJOR = 'ATRStplnPrcMaj'
    ATR_Stopline_PRICE_WEEKLY = 'ATRStplnPrcWek'

    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(FIELD, cls).__new__(cls)
        return cls._instance



    # ------------------------------------------------------------------ #
    # regtree 一族（2026-10-10 随 `analysis/regtree.py` 搬回时补）
    #
    # 值与旧树 `GolemQ_old/utils/parameter.py` 的 `INDICATOR_FIELD` **逐字相同** ——
    # 它们是数据在 MongoDB 里的实际存储名（`PITFALLS.md` P1：缺常量不许编）。
    # 补齐前实测：本文件用到的 29 个名字在新树**全部抛 AttributeError**。
    # ------------------------------------------------------------------ #
    ATR_LB = 'atrLb'
    COMBINE_DENSITY = 'combDensity'
    DEA_ZERO_TIMING_LAG = 'deaOLag'
    HMA10 = 'HMA10'
    HMA5 = 'HMA5'
    HMAPOWER120 = 'HMPWR12'
    HMAPOWER120_QUARTER = 'hmpwr12Qtr'
    HMAPOWER120_TIMING_LAG = 'hmpwr12Lag'
    MA30 = 'MA30'
    MA90 = 'MA90'
    MA90_SLOPE = 'ma90Slope'
    # ⚠️ 下面这些是 2026-10-11 起随「四量图 / RSRS / 筹码分布」搬运补的
    # （`analysis/{quadrant,rsrs,ChipDistribution_jit}.py` 要用）。
    # **标识符**照搬老树 `GolemQ_old/utils/parameter.py` 的 `FIELD` 类；
    # **值**按 `DECISIONS.md` **D41** 的规则改写 —— 老树那些值**全是 snake**
    # （`MA250_MAJ` / `ATR_UB_WEK` / `KDJ_K` …），而这里落的是驼峰且 ≤16 字符。
    # ⚠️ 2026-10-13 更正：本注释原先写「真值逐字取自老树、未改一个字符」——**那是错的**，
    # 会让人以为可以直接拿老树的值去对库。
    MA60 = 'MA60'
    MA120 = 'MA120'
    MA250 = 'MA250'
    MA250_MAJOR = 'ma250Maj'
    #: MA250 的**时序滞回**（旧树 `'MA250_LAG_MAJ'` → 驼峰）。
    MA250_TIMING_LAG_MAJOR = 'ma250LagMaj'
    ATR_UB_WEEKLY = 'atrUbWek'
    KDJ_K = 'kdjK'
    KDJ_D = 'kdjD'
    KDJ_J = 'kdjJ'
    KDJ_TREND_TIMING_LAG = 'kdjTrdLag'
    #: MACD 三件套（`talib.MACD` 的输出）。`DIF` / `DEA` 是**单词**，无 `_`，值照旧。
    DIF = 'DIF'
    DEA = 'DEA'
    MACD = 'MACD'
    MACD_DELTA_MAJOR = 'macdDiff1tMaj'
    RSRS_BETA = 'rsrsBeta'
    RSRS_BETA_R2 = 'rsrsBetaR2'
    RSRS_ZSCORE = 'rsrsZscore'
    RSRS_ZSCORE_R2 = 'rsrsZsR2'
    RSRS_RIGHT = 'rsrsRight'
    MACD_DELTA = 'macdDiff1t'
    MAPOWER120 = 'MPWR12'
    MAPOWER120_QUARTER = 'mpwr12Qtr'
    MAPOWER120_TIMING_LAG = 'mpwr12Lag'
    MAPOWER30_QUARTER = 'mpwr3Qtr'
    MAPOWER30_TIMING_LAG = 'mpwr3Lag'
    POLYNOMIAL9_TIMING_LAG = 'poly9Lag'
    REGTREE_CORRCOEF = 'regtreeCorrcoef'
    REGTREE_DENY_LINE = 'regtreeDenyLine'
    REGTREE_MA90_INTERCEPT = 'regtreeMa90Int'
    REGTREE_PRICE = 'regtreePrice'
    REGTREE_SLOPE = 'regtreeSlp'
    REGTREE_TIMING_LAG = 'regtreeLag'
    REGTREE_TREND = 'regtreeTrd'
    REGTREE_TREND_RETURNS = 'regtreeTrdRet'

    # ── 2026-10-21 随 compact builder 补（真值逐字取自老树
    # `utils/parameter.py::INDICATOR_FIELD`；MongoDB 实际存储名，别改写）──
    CCI = 'CCI'
    RSI = 'RSI'
    CCI_NORM = 'CCI_NORM'
    BOLL_UB = 'BOLL_UB'
    BOLL = 'BOLL_MA'
    BOLL_LB = 'BOLL_LB'
    BOLL_CHANNEL = 'BOLL_WIDTH'
    BOLL_DELTA = 'BBW_diff_1T'
    MTM = 'MTM'
    MTMMA = 'MTMMA'
    MA90_CLEARANCE = 'MA90_CLR'
    MA90_CLEARANCE_TIMING_LAG = 'MA90_CLR_LAG'
    MA90_CLR_RETURNS_GARIDENT = 'MA90_CLR_RET_GAR'
    DRAWDOWN_RATIO = 'DRDN_RATIO'
    VOLUME_NORM = 'VOL_NORM'
    VOLUME_NORM_TIMING_LAG = 'VOL_NORM_LAG'
    MAGIC_NINE_TURNS_BEFORE_MAJOR = 'm9tBfMaj'
    MAGIC_NINE_TURNS_SX_BEFORE_MAJOR = 'm9tSxBfMaj'
    MAPOWER30_TIMING_LAG_MAJOR = 'MPWR3_LAG_MAJ'
    MAPOWER30_MEDIAN = 'MPWR3_MED'
    MAPOWER30_MAJOR_MEDIAN = 'MPWR3_MAJ_MED'
    MTM_NORM_WEEKLY = 'MTM_NORM_WEK'
    MTMMA_NORM = 'MTMMA_NORM'
    MTMMA_NORM_WEEKLY = 'MTMMA_NORM_WEK'
    MTMMA_NORM_MAJOR = 'MTMMA_NORM_MAJ'
    POLYNOMIAL9_TIMING_LAG_MAJOR = 'poly9_LAG_MAJ'
    PEAK_LOW_TIMING_LAG = 'PeakLowLag'
    PEAK_HIGH_BACKWARD = 'PEAK_HI_BKWRD'
    PEAK_LOW_BACKWARD = 'PEAK_LO_BKWRD'
    MAPOWER30_PEAK_HIGH_BEFORE = 'MPWR3_PEAK_HI_BF'
    MAPOWER30_PEAK_LOW_BEFORE = 'MPWR3_PEAK_LO_BF'
    #: ATR Stopline / SuperTrend（2026-10-21 随 markup 工具补；真值取自老树）。
    ATR_Stopline = 'ATR_Stopline'
    ATR_Stopline_PRICE = 'ATRStplnPrc'
    ATR_SuperTrend = 'ATR_SUPTRD'
    ATR_Stopline_TIMING_LAG_WEEKLY = 'ATR_Stopline_LAG_WEK'
    ATR_SuperTrend_TIMING_LAG_WEEKLY = 'ATR_SUPTRD_LAG_WEK'
    #: Linear-regression band / peak 极值点（2026-10-21 随 `calc_peak_point_vX` 补）。
    LINEAREG_PRICE = 'LRC_PRICE'
    ATR_UB = 'ATR_UB'
    LINEAREG_TREND_TIMING_LAG = 'LRG_TRD_LAG'
    LINEAREG_BAND_TIMING_LAG = 'LRG_BAND_LAG'
    PEAK_TOWARDS_TIMING_LAG = 'PEAK_TOWARDS_LAG'
    PEAK_HIGH_TIMING_LAG = 'PEAK_HIGH_LAG'
    PEAK_TOWARDS_BEFORE = 'PEAK_TOWARDS_BF'

class FEATURES(metaclass=_StubMeta):
    """
    Feature column name constants (stub — to be populated).

    These are used by services/persistence/ modules for data completeness checks.
    """
    _instance = None

    # Zen/Dash timing lag features
    ZEN_DASH_TIMING_LAG_MINOR_REAL = 'zDLagMinR'
    ZEN_DASH_TIMING_LAG_MAJOR_REAL = 'zDLagMajR'
    ZEN_DASH_TIMING_LAG_WEEKLY_REAL = 'zDLagWekR'
    ZEN_PEAK_TIMING_LAG_MAJOR_REAL = 'ZPLagMajR'
    ZEN_PEAK_TIMING_LAG_REAL = 'zPLagR'
    ZEN_PEAK_TIMING_LAG_MINOR_REAL = 'zPLagMinR'

    # CVaR risk features
    CVaR_risk90 = 'EsRisk90'
    CVaR_risk95 = 'EsRisk95'
    CVaR_risk90_MAJOR = 'EsRisk90Maj'
    CVaR_risk95_MAJOR = 'EsRisk95Maj'

    # Polynomial features
    POLYNOMIAL9_WEEKLY_REAL = 'poly9WekR'
    POLYNOMIAL9_NORM_WEEKLY_REAL = 'poly9NorWekR'
    POLYNOMIAL9_MAJOR_REAL = 'poly9MajR'
    POLYNOMIAL9_NORM_MAJOR_REAL = 'poly9NorMajR'

    # Magic Nine Turns features
    MAGIC_NINE_TURNS_MAJOR_REAL = 'm9tMajR'
    MAGIC_NINE_TURNS_WEEKLY_REAL = 'm9tWekR'
    MAGIC_NINE_TURNS_TIMING_LAG_WEEKLY_REAL = 'm9tLagWekR'
    MAGIC_NINE_TURNS_TIMING_LAG_MAJOR_REAL = 'm9tLagMajR'

    # Regression tree features
    REGTREE_PRICE_MAJOR_REAL = 'rTrePrcMajR'
    REGTREE_TIMING_LAG_MAJOR_REAL = 'rTreLagMajR'
    REGTREE_SLOPE_MAJOR_REAL = 'rTreSlopMajR'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FEATURES, cls).__new__(cls)
        return cls._instance


class TREND_STATUS(metaclass=_StubMeta):
    """
    Trend status constants (stub — to be populated).

    Used by services/persistence/ for cluster group checkpoint tracking.
    """
    _instance = None
    CLUSTER_GROUP_CHECKPOINTS = 'CLT_GRP_CHK'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TREND_STATUS, cls).__new__(cls)
        return cls._instance



    # 鲁棒性极值点识别（`analysis/peak.py` 写这一列）。
    # 值与旧树 `GolemQ_old/utils/parameter.py` 的 `TREND_STATUS.PEAK_POINT` **逐字相同** ——
    # 它是数据在 MongoDB 里的实际存储名（`PITFALLS.md` P1：缺常量不许编）。
    PEAK_POINT = 'PEAK_POINT'
    PEAK_POINT_MAX_FACTOR = 'PEAK_POINT_MAX_FACTOR'
    #: 2026-10-21 随 `calc_peak_point_vX` 补（老树 `TREND_STATUS` 真值）。
    CLUSTER_GROUP_TOWARDS = 'CLT_GRP_TOWARDS'
    PEAK_TOWARDS = 'PEAK_TOWARDS'

class STATE(metaclass=_StubMeta):
    """
    State constants (stub — to be populated).

    Used by services/persistence/ for data quality checks.
    """
    _instance = None
    MACD_COMPOUDED_BAND_RATIO = 'sMacdCpdBandRto'
    #: 2026-10-11 随「四量图」搬运补；真值取自老树 `utils/parameter.py` 的 `STATE` 类。
    CVaR_BOWL_CANDIDATE = 'esBowCan'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(STATE, cls).__new__(cls)
        return cls._instance


# Module-level singleton instance
# aka = AKA()


# ---------------------------------------------------------------------------
# `datetime64` → 秒级 epoch 的**换算系数**（全树唯一出处）
# ---------------------------------------------------------------------------
# 背景（`DECISIONS.md` D52）：pandas 3.0 把 `pd.to_datetime` 的默认精度从
# `datetime64[ns]` 改成 `datetime64[us]`（本机 3.0.6 实测）。于是老写法
#
#     series.astype(np.int64) // 10 ** 9
#
# 在 pandas 3 上**少 1000 倍**，而且**不报错** —— 只是把 `date_stamp` /
# `time_stamp` 这类**键**写成另一个数量级。
#
# 这里**不是**看版本号字符串，也**不是**一个全局标量 —— 系数**按 dtype 查表**。
# 原因是 pandas 3 会按**输入**推不同的精度，同一次运行里几种精度是并存的
# （2026-10-17 本机 3.0.6 实测）：
#
#     ==============================  ==================
#     构造方式                         得到的 dtype
#     ==============================  ==================
#     `pd.to_datetime(date 对象)`       `datetime64[s]`
#     `pd.to_datetime(字符串)`          `datetime64[us]`
#     `pd.to_datetime(Series)`          `datetime64[us]`
#     pandas 2 的默认                    `datetime64[ns]`
#     ==============================  ==================
#
# ⚠️ **单个标量系数会算错**：`reality_save` 的 `day_dt` 走的正是 `date` 对象那一路
# （秒），拿 µs 的 1e6 去除 ⇒ `date_stamp` 变成 `1767`（实测，当场红了 doctest）。
#
#     秒级 epoch = series.astype(np.int64) // epoch_divisor(series.dtype)
#
# ⚠️ 只用它换**秒**。`analysis/pivot.py` 那处是**纳秒域**（与 `Timestamp.value`
# 比 `searchsorted`，而 `Timestamp.value` 恒为 ns），**不能**换成秒 —— 见该处注释。

#: `datetime64` 单位 → 1 秒对应的整数单位数。**固态常量**（单位名是 numpy 的稳定 API）。
EPOCH_UNIT_TO_SECONDS = {'s': 1, 'ms': 10 ** 3, 'us': 10 ** 6, 'ns': 10 ** 9}


def epoch_divisor(dtype) -> int:
    """`datetime64` 的 dtype（naive 或 tz-aware）→ **秒**系数。

    >>> epoch_divisor('datetime64[s]')
    1
    >>> epoch_divisor('datetime64[us]')
    1000000
    >>> import pandas as pd
    >>> epoch_divisor(pd.DatetimeIndex(['2026-01-05']).tz_localize('Asia/Shanghai').dtype)
    1000000
    """
    unit = getattr(dtype, 'unit', None)          # tz-aware（pandas 的 DatetimeTZDtype）
    if unit is None:
        import numpy as _np
        unit = _np.datetime_data(_np.dtype(dtype))[0]
    try:
        return EPOCH_UNIT_TO_SECONDS[unit]
    except KeyError:                      # 未来的 pandas/numpy 又换单位时**响亮地**失败
        raise RuntimeError(
            'datetime64 精度是 {!r}，EPOCH_UNIT_TO_SECONDS 表里没有 —— '
            '先在 core.constants 里补一行，再跑全量测试'.format(unit))


def _default_epoch_divisor() -> int:
    """pandas **当前默认**构造出来的系数。**只用于诊断/文档，别拿它去算值**。"""
    import pandas as _pd
    return epoch_divisor(_pd.to_datetime(['2000-01-01']).dtype)


#: pandas 当前默认精度下的系数（pandas 2 ⇒ 1e9 / pandas 3 ⇒ 1e6）。
#: ⚠️ 算值请用 `epoch_divisor(series.dtype)` —— 不同来源的 Series 精度可能不同。
EPOCH_DIVISOR = _default_epoch_divisor()
