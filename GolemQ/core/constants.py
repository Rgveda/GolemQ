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
    """Metaclass that returns stub values for undefined attributes."""
    def __getattr__(cls, name):
        return f'{cls.__name__.lower()}_{name.lower()}'


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
    CLOSE_RAW = 'close_raw'
    UNCLOSED = 'unclosed'
    CLOSE_MAJOR = 'close_maj'
    LAST_CLOSE = 'last_close'
    OPEN_LASTDAY = 'open_lastday'
    PRICE = 'price'

    # 成交量与金额
    VOLUME = 'volume'
    VOL = 'vol'
    VOLUME_S1 = 'volume_s1'
    AMOUNT = 'amount'
    AMOUNT_MAJOR = 'amount_maj'
    AMOUNT_S1 = 'amount_s1'
    AMOUNT_SUM = 'AMOUNT_SUM'
    TURNOVER = 'turnover'
    TURNOVER_RATE = 'TurnoverRate'

    # 标识与时间
    CODE = 'code'
    ISOCODE = 'isocode'
    FULL_SYMBOL = 'FULL_SYMBOL'
    MARKET_TYPE = 'MARKET_TYPE'
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
    PCT_RANKING = 'pct_rank'
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
    TRADE_CAL = 'trade_calendar'
    COST5_PRICE = 'cost5Prz'
    RETURN_COST95 = 'RetCost95'
    COST15_PRICE = 'cost15Prz'
    CATALOG = 'catalog'
    LENGTH = 'length'
    FROZEN = 'FrozenExpired'
    CHECKPOINT = 'ChkPoi'
    STAGE = 'STAGE'

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
    PCT_CHANGE = 'PCT_CHG'

    DATE = 'date'
    DATETIME = 'datetime'
    CODE = 'code'

    # Common intervals
    DAILY = '1d'
    WEEKLY = '1w'
    MONTHLY = '1M'

    PE_RATION = 'PERation'

    # 资金流
    MONEYFLOW_PERCENT = 'MONEYFLOW_PCT'
    MONEYFLOW_VOLUME = 'moneyflow_volume'
    MONEYFLOW_IN = 'MONEYFLOW_IN'
    MONEYFLOW_OUT = 'MONEYFLOW_OUT'
    MONEYFLOW_SCORE = 'MoneyflowScr'
    MONEYFLOW_VOLUME_MINOR = 'moneyflw_vol_min'
    MONEYFLOW_PERCENT_MINOR = 'MONEYFLW_PCT_MINOR'

    # Trading calendar
    TRADE_CAL = 'trade_calendar'  # 交易日历常量
    PCT_CHANGE_MAJOR = 'PCT_CHG_MAJ'
    DEA_ZERO_TIMING_LAG_MAJOR = 'DEA_O_LAG_MAJ'
    CHARTSCAN_BUY_BEFORE_DUMMY = 'ChrScBuyBfD'
    CHARTSCAN_BUY_BEFORE_MINOR_DUMMY = 'ChrScBuyBfMinD'
    MACD_CROSS_SX_BEFORE_MAJOR = 'MACD_SX_BF_MAJ'
    CHARTSCAN_BUY_BEFORE_MAJOR_DUMMY = 'ChrScBuyBfMajD'
    ZEN_DASH_TIMING_LAG_DUMMY = 'zDLagD'
    RENKO_BOOST_S_TIMING_LAG = 'renkosBstLag'
    MACD_MAJOR = 'MACD_MAJ'
    STOCK_SCORE_M15 = 'Score_M15'
    MAXFACTOR = 'MAXFACTOR'
    MAXFACTOR_MAJOR = 'MFT_MAJ'
    MACD_ZERO_TIMING_LAG = 'MACD_O_LAG'
    BOLL_LB_HMA5_TIMING_LAG = 'BOLL_L_HMA5_LAG'
    DEA_NORM = 'DEA_NOR'
    DIF_NORM = 'DIF_NORM'
    MTM_NORM = 'MTM_NORM'
    MAPOWER30 = 'MPWR3'
    MACD_ZERO_TIMING_LAG_MAJOR = 'MACD_O_LAG_MAJ'
    BOLL_LB_HMA5_TIMING_LAG_MAJOR = 'BOLL_L_HMA5_LAG_MAJ'
    DEA_NORM_MAJOR = 'DEA_NORM_MAJ'
    DIF_NORM_MAJOR = 'DIF_NORM_MAJ'
    MTM_NORM_MAJOR = 'MTM_NORM_MAJ'
    MAPOWER30_MAJOR = 'MPWR3_MAJ'
    MA5_MAJOR = 'maVMaj'
    MA10_MAJOR = 'maXMaj'
    FBPROPHET_LB_MAJOR = 'PROPHET_LB_MAJ'
    FBPROPHET_LB = 'PROPHET_LB'
    GARIDENT_PRICE = 'GAR_P'
    GARIDENT_PRICE_COUNT = 'GAR_P_COUNT'
    GARIDENT_PRICE_CLEARANCE = 'GAR_P_CLR'
    GARIDENT_STAGE_TIMING_LAG = 'GAR_S_LAG'
    BOOTSTRAP_STAGE_BEFORE = 'BST_STG_BF'
    ENDPOINT_STAGE_BEFORE = 'EP_STG_BF'
    STOCK_SCORE_M15_NORM = 'SSM15_NORM'
    MAGIC_NINE_TURNS = 'm9t'
    MAGIC_NINE_TURNS_MAJOR = 'm9tMaj'
    MAGIC_NINE_TURNS_BASELINE = 'm9tBsl'
    MAGIC_NINE_TURNS_BASELINE_TIMING_LAG = 'm9tBslLag'
    MAGIC_NINE_TURNS_BEFORE = 'm9tBf'
    MAGIC_NINE_TURNS_SX_BEFORE = 'm9tSxBf'
    POLYNOMIAL9_NORM_MAJOR = 'poly9pwr3_MAJ'
    RENKO_TREND_S_TIMING_LAG = 'renkosBarLag'
    RENKO_TREND_S_LB = 'renkosBarLb'
    RENKO_TREND_S_UB = 'renkosBarUb'
    STOCK_SCORE_M15_TIMING_LAG = 'Score_M15_LAG'
    STOCK_SCORE_M15_NORM_TIMING_LAG = 'SSM15_NORM_LAG'
    POLYNOMIAL9 = 'poly9'
    ATR_Stopline_TIMING_LAG = 'ATR_Stopline_LAG'
    ATR_SuperTrend_TIMING_LAG = 'ATR_SUPTRD_LAG'
    ATR_Stopline_TIMING_LAG_MAJOR = 'ATR_Stopline_LAG_MAJ'
    ATR_SuperTrend_TIMING_LAG_MAJOR = 'ATR_SUPTRD_LAG_MAJ'
    ATR_Stopline_PRICE_MAJOR = 'ATRStplnPrcMaj'
    ATR_Stopline_PRICE_WEEKLY = 'ATRStplnPrcWek'

    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(FIELD, cls).__new__(cls)
        return cls._instance


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


class STATE(metaclass=_StubMeta):
    """
    State constants (stub — to be populated).

    Used by services/persistence/ for data quality checks.
    """
    _instance = None
    MACD_COMPOUDED_BAND_RATIO = 'sMacdCpdBandRto'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(STATE, cls).__new__(cls)
        return cls._instance


# Module-level singleton instance
# aka = AKA()
