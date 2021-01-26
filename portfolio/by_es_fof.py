#
# The MIT License (MIT)
#
# Copyright (c) 2018-2020 azai/Rgveda/GolemQuant
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# 克隆自聚宽文章：https://www.joinquant.com/post/24542
# 标题：FOF养老成长基金-v2.0
# 原作者：富在知足
import numpy as np
import talib
import pandas
import scipy as sp
import scipy.optimize
import datetime as dt
from scipy import linalg as sla
from scipy import spatial

from collections import namedtuple
#Context = namedtuple("_Context",['tradeRatio',
#                                 'positionDict',
#                                 'position',
#                                 'initPriceDict',
#                                 'transactionRecord',
#                                 'pool',]) #Point为返回的类，_Point为返回类的类名
context = dict(['doneTrade'])

def initialize(context):
    """
    初始化 context = indices
    """
    #set_benchmark('000300.XSHG')
    # 设置买卖手续费，万三，最小 5 元
    #set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0003, min_cost=5))
    #set_slippage(FixedSlippage(0.002))
    #set_option('use_real_price', True)
    # enable_profile()
    # 关闭部分
    #log.set_level('order', 'error')

    # 交易系数
    context['tradeRatio'] = 0

    # 交易收益率
    context['positionDict'] = np.nan

    # 仓位(未使用)
    context['position'] = np.nan

    # 需要进行盘整巩固
    context['initPriceDict'] = np.nan

    # 上次交易日所获取的平均值
    context['transactionRecord'] = np.nan

    return context


def handle_data(context, data):
    """
    新Bar处理
    """

	# 初始化交易数据
    initializeStockDict(context)
    # 进行仓位调整
    rebalance(context)
    # 获取买卖决策
    message = ""
    for stock in context.pool:
        if context.tradeRatio[stock] > 0:
            message += buyOrSellCheck(context,stock, 3, 1.4)
	# 此段代码仅用于发微信，可以跳过
    if message != "":
        send_message(message)
    # 每周进行一次调仓操作
    if getLastestTransactTime(context):
        tradeStockDict(context, context.tradeRatio)
        if context.doneTrade:
            print(("\n" + context.ratioMessage))


# initialize parameters
def initializeStockDict(context):
    ## 每天注入400元资金
    #inout_cash(400, pindex=0)

    # 是否进行了交易操作
    context.doneTrade = False

    # 需要进行过渡交易
    context.Transitional = True

    # 定义调仓周期
    context.weekDay = { 4 }

    # 置信区间
    context.confidenceLevel = 0.02

    # 定义参考周期
    context.referenceCycle = 252

    # 涨跌幅度
    context.maxRange = 30

    # 巩固幅度
    context.consolidation = 5

    # 持仓数量
    context.stockCount = 5

    # 入选参考
    context.referStock = '000002.XSHG'

    # 入选条件
    context.minPosition = 0

    # 持仓位数
    context.Multiple = 1

    # 最小持仓比例
    context.minRatio = 0.10 * context.Multiple

    # 最大持仓比例
    context.maxRatio = 0.30 * context.Multiple

    # 最小交易金额
    context.minAmount = 2000

    # 网格交易率(%)
    context.netRate = 3

    # 定义交易标的
    context.stockDict = {}

    # 分析标的
    context.trackStock = ''

    # 其余仓位
    context.bondStock = '163210' # 诺安纯债定开债A

    # 权重系数：1~10
    context.stockDict['515520.XSHG'] = 10 # 价值100
    context.stockDict['161907.XSHE'] = 10 # 中证红利
    context.stockDict['512890.XSHG'] = 10 # 红利低波
    context.stockDict['515300.XSHG'] = 10 # 红利300
    context.stockDict['510050.XSHG'] = 10 # 上证50
    context.stockDict['159916.XSHE'] = 10 # 深证60
    context.stockDict['512910.XSHG'] = 10 # 中证100
    context.stockDict['510310.XSHG'] = 10 # 沪深300
    context.stockDict['512260.XSHG'] = 10 # 中证500低波
    context.stockDict['512090.XSHG'] = 10 # MSCI
    context.stockDict['515000.XSHG'] = 10 # 中证科技
    context.stockDict['512040.XSHG'] = 10 # 中证价值
    context.stockDict['510900.XSHG'] = 10 # 恒生国企
    context.stockDict['501021.XSHG'] = 10 # 香港中小
    context.stockDict['513050.XSHG'] = 10 # 中概互联
    context.stockDict['518880.XSHG'] = 4  # 黄金

    # 上市不足参考周期的剔除掉
    context.stockDict = delNewStock(context, context.stockDict)

    # 整理能交易的标的
    context.pool = list(context.stockDict.keys())

    # 持仓数量
    if (len(context.pool) < context.stockCount):
        context.stockCount = len(context.pool)

    # 统计交易资料
    for stock in context.pool:
        if stock not in context.initPriceDict:
            context.initPriceDict[stock] = 0
        if stock not in context.positionDict:
            context.positionDict[stock] = 0
        if stock not in context.transactionRecord:
            context.transactionRecord[stock] = 0
    # 初始化交易记录
    context.tradeRecord = ""
    # 初始化权重记录
    context.ratioMessage = ""
        
def getStockRSIRatio(stock):
    try:
        # 根据周来计算出RSI系数
        if ratio > 0:
            nRSI = 60
            nRSIAvg = 5
            his = attribute_history(stock, 300, '1d', 'close', skip_paused=True, df=False, fq='pre')
            closeArray = his['close']
            # 计算RSI #
            rsiArray = talib.RSI(closeArray,nRSI)
            # RSI均值 #
            rsiAvgArrayS = talib.MA(rsiArray,nRSIAvg)
            rsiAvgArrayL = talib.MA(rsiArray,nRSI)
            rsiRatio = 100 - np.round(math.tan(math.radians(rsiAvgArrayS[-1] - (50.0 if math.isnan(rsiAvgArrayL[-1]) else rsiAvgArrayL[-1]))) * 50,0)
            if rsiRatio < 0:
                rsiRatio = 0
            return rsiRatio
        else:
            return 0
    except:
        return 50

def getStockRSI(stock):
    # 根据周来计算出RSI系数
    # 计算RSI所用天数
    try:
        nRSI = 7
        his = attribute_history(stock, 30, "1d", ("close","high","low"), skip_paused=True, df=False, fq="pre")
        closeArray = his["close"]
        rsiArray = talib.RSI(closeArray, nRSI)
        return np.round(rsiArray[-1],2)
    except Exception as err:
        return 100

def getLastestTransactTime(context):

    # 定义调仓周期
    # 只会有月初进行操作
    lastestDate = datetime.datetime(2000, 1, 1)
    # 获取最后一次交易日期
    for stock in list(context.portfolio.positions.keys()):
        if (context.portfolio.positions[stock].transact_time > lastestDate):
            lastestDate = context.portfolio.positions[stock].transact_time
    if (context.current_dt - lastestDate).days >= 30 and context.current_dt.isoweekday() in context.weekDay:
        return True

    return False

def getStockName(stock):
    return get_security_info(stock).display_name

def drawCloseValue(stock):
    his = attribute_history(stock, 1, '1d','close',df=False, skip_paused=False)
    record(C=his['close'][0])

def variance(stock):
    # 计算平均涨跌幅
    his = attribute_history(stock, 120, '1d','close',df=False, skip_paused=False)
    trList = []
    for i in range(len(his['close'])):
        if i > 0:
            trList.append(abs(his['close'][i - 1] - his['close'][i]) / his['close'][i - 1] * 100)
    trArray = np.array(trList)
    trMean = trArray.mean()
    return np.round(trMean,1) if not isnan(trMean) else 0

def getAvgMoney(stock):
    # 计算平均交易额
    his = attribute_history(stock, 120, '1d','volume',df=False, skip_paused=False)
    trMean = his['money'].mean()
    return np.round(trMean,1) if not isnan(trMean) else 0


def delNewStock(context, stockDict):
    """
	剔除上市时间较短和平均交易额过低的产品。
    """
    for stock in list(stockDict.keys()):
        stockData = context.ohlc.loc[(), :]
        if ():
            avgMoney = getAvgMoney(stock)
            if avgMoney >= 2000000:
                tmpDict[stock] = stockDict[stock]
    return tmpDict

# 每天开盘前用于判断某一只etf今日该买还是该卖的函数 #
# 此函数输入为一个股票代码，应卖出时输出-1，应买进时输出1 #
def buyOrSellCheck(context, stock, nATRValue, nstdValue):
    message = ""
    try:
        # 计算RSI所用天数
        nRSI = 7
        nRSIAvg = 14
        nATR = 21
        # 取得近90天的历史行情数据
        deltaDate = context.current_dt.date() - datetime.timedelta(90)
        if get_security_info(stock).start_date > deltaDate:
            return message
        his = attribute_history(stock, 120, '1d', ('close','high','low'), skip_paused=True, df=False, fq='pre')
        closeArray = his['close']
        # 计算长线是60天（月）
        emaArray = talib.MA(closeArray,60)
        # 计算RSI #
        rsiArray = talib.RSI(closeArray,nRSI)
        # RSI均值 #
        rsiAvgArray = talib.MA(rsiArray,nRSIAvg)
        # RSI标准差 #
        rsiStdArray = talib.STDDEV(rsiArray,nRSIAvg)
        # ATR #
        trList = []
        for i in range(len(closeArray)):
            if i > 0:
                trList.append(max([(his['high'][i] - his['low'][i]), 
                                   abs(his['close'][i - 1] - his['high'][i]),
                                   abs(his['close'][i - 1] - his['low'][i])]))
        trArray = np.array(trList)
        atrAvgArray = talib.MA(trArray,nATR)
        ATR = nATRValue * atrAvgArray[-1]
        # 买入的阈值 #
        buyThreshold = rsiAvgArray[-1] - nstdValue * rsiStdArray[-1]
        if buyThreshold > 30:
            buyThreshold = 30
        # 卖出的阈值 #
        sellThreshold = rsiAvgArray[-1] + nstdValue * rsiStdArray[-1]
        if sellThreshold < 70:
            sellThreshold = 70
        # 获取溢价率 #
        premiumRate = getPremiumRate(context, stock)
        #record(RA=sellThreshold,RB=rsiArray[-1],RC=buyThreshold)
        #record(TA=closeArray[-2]+ATR,TB=closeArray[-1],TC=closeArray[-2]-ATR)
        # 当天出现超过3%的跌幅时，禁止任何操盘
        if stopLoss(stock):
            message = getStockName(stock) + " : " + "禁止操作！\n"
        elif premiumRate >= 0.5:
            message = getStockName(stock) + " : " + "溢价[" + str(premiumRate) + "]！\n"
        elif premiumRate <= -1.0:
            message = getStockName(stock) + " : " + "折价[" + str(premiumRate) + "]！\n"
        # 如果RSI高于卖出阈值，则卖出股票
        elif rsiArray[-1] > sellThreshold:
            message = getStockName(stock) + " : " + "RSI[" + str(np.round(rsiArray[-1],2)) + ">" + str(np.round(sellThreshold,2)) + "]卖出！\n"
        # 如果RSI低于买入阈值，则买入股票
        elif rsiArray[-1] < buyThreshold:
            message = getStockName(stock) + " : " + "RSI[" + str(np.round(rsiArray[-1],2)) + "<" + str(np.round(buyThreshold,2)) + "]买入！\n"
        # 如果ATR高于卖出阈值，则卖出股票
        elif closeArray[-1] > closeArray[-2] + ATR:
            message = getStockName(stock) + " : " + "ATR[" + str(np.round(closeArray[-1],2)) + ">" + str(np.round(emaArray[-1],2)) + "]卖出！\n"
        # 如果ATR低于买入阈值，则买入股票
        elif closeArray[-1] < closeArray[-2] - ATR:
            message = getStockName(stock) + " : " + "ATR[" + str(np.round(closeArray[-1],2)) + "<" + str(np.round(emaArray[-1],2)) + "]买入！\n"
    except:
        message = ""
    return message

def stopLoss(stock, lag=2, loss=2, more=4):
    # 当跌幅大于2%时禁止当天交易，以观望的方式等待下一个交易日，防止股灾出现时仍连接交易 #
    hisArray = attribute_history(stock,lag,'1d',('open', 'close', 'high', 'low', 'volume'),skip_paused=True)
    closeArray = hisArray['close'].values
    rate = abs((closeArray[-1] - closeArray[-2]) / closeArray[-2] * 100)
    if (rate > loss and rate < more):
        return True
    else:
        return False

def getPremiumRate(context, stock):
    # 计算基金当前的溢价情况 #
    try:
        now = context.current_dt
        start_date = now + datetime.timedelta(days=-10)
        end_date = now + datetime.timedelta(days=-1)
        unitPriceArray = get_extras('unit_net_value', stock, start_date=start_date, end_date=end_date, df=False)
        unitPrice = unitPriceArray[stock][-1]
        his = attribute_history(stock, 5, '1d', 'close', skip_paused=True, df=False, fq='pre')
        closePrice = his['close'][-1]
        return np.round((closePrice - unitPrice) / unitPrice * 100, 2)
    except:
        return 0

def rebalance(context):
    # 重新调整仓位
    tradeRatio = caltradeStockRatio(context)
    context.tradeRatio = tradeRatio

def caltradeStockRatio(context):
    
    def getGrowthRate(stock, n=21):
        """
        获取股票n日以来涨幅，根据当前价计算 / 这个东西按理说是ROC
        """
        lc = attribute_history(stock, n, '1d', ('close'), True)['close'][0]
        c = attribute_history(stock, 1, '1d', ('close'), True)['close'][0]
        if not isnan(lc) and not isnan(c) and lc != 0:
            return (c - lc) / lc * 100
        else:
            return 0
    
    def calstockRiskVaR(stock):
		# 风险价值(VaR)
        portfolio_VaR = 0.0000001
        dailyReturns = fun_getdailyreturn(stock, '1d', context.referenceCycle)
        portfolio_VaR = 1 * context.confidenceLevel * np.std(dailyReturns) * 100
        if isnan(portfolio_VaR):
            portfolio_VaR = 0.0000001
        return 1 #portfolio_VaR
		
    def calstockRiskES(stock):
        # 期望损失(ES)
        portfolio_ES = 0
        dailyReturns = fun_getdailyreturn(stock, '1d', context.referenceCycle)
        dailyReturns_sort = sorted(dailyReturns)    
        count = 0
        sum_value = 0
        for i in range(len(dailyReturns_sort)):
            if i < (context.referenceCycle * context.confidenceLevel):
                sum_value += dailyReturns_sort[i]
                count += 1
        if count == 0:
            portfolio_ES = 0
        else:
            portfolio_ES = -(sum_value / (context.referenceCycle * context.confidenceLevel))
        if isnan(portfolio_ES):
            portfolio_ES = 0
        return portfolio_ES

    def fun_getdailyreturn(stock, freq, lag):
        stockHis = history(lag, freq, 'close', stock, df=True)
        #dailyReturns =
        #stockHis.resample('D',how='last').pct_change().fillna(value=0,
        #method=None, axis=0).values
        dailyReturns = stockHis.resample('D').last().pct_change().fillna(value=0, method=None, axis=0).values
        return dailyReturns

    def fun_caltraderatio(tradeRatio, stock, position, total_position):
        if stock in tradeRatio:
            tradeRatio[stock] += np.round((position / total_position), 3)
        else:
            tradeRatio[stock] = np.round((position / total_position), 3)
        tradeRatio[stock] = tradeRatio[stock] * 100 // context.Multiple * context.Multiple / 100
        return tradeRatio

    # 计算所有标的的仓位比例
    max_ES = -1
    max_VaR = -1
    ESDict = {}
    VaRDict = {}
    grDict = {}
    for stock in context.pool:
        ES = calstockRiskES(stock)
        if ES > max_ES:
            max_ES = ES
        ESDict[stock] = ES
        VaR = calstockRiskVaR(stock)
        if VaR > max_VaR:
            max_VaR = VaR
        VaRDict[stock] = VaR
        grDict[stock] = getGrowthRate(stock)

    # 计算入选条件
    referES = calstockRiskES(context.referStock)
    referVar = calstockRiskVaR(context.referStock)
    referGR = getGrowthRate(context.referStock)
    referPosition = np.round((max_ES / referES) * (max_VaR / referVar) * power(1.02, referGR) * 100, 3)
    if context.minPosition == 0:
        context.minPosition = referPosition
    else:
        context.minPosition = (referPosition + context.minPosition) / 2
    # 计算总仓位
    positionDict = {}
    for stock in context.pool:
        if ESDict[stock] == 0:
            positionDict[stock] = 0
        else:
            stockRatio = context.stockDict[stock]
            positionDict[stock] = np.round((max_ES / ESDict[stock]) * (max_VaR / VaRDict[stock]) * power(1.02, grDict[stock]) * stockRatio * 10,3)
        # 与上次交易的平均值再进行一次平均值计算
        if context.transactionRecord[stock] == 0:
            context.transactionRecord[stock] = positionDict[stock]
        else:
            positionDict[stock] = positionDict[stock] * 0.8 + context.transactionRecord[stock] * 0.2
            context.transactionRecord[stock] = positionDict[stock]
    positionDictSorted = sorted(list(positionDict.items()), key=lambda d: d[1], reverse = True)
    # 针对边际标的进行如果持仓则继续持仓
    stockIn = ""
    stockOut = ""
    doChange = False
    if len(positionDictSorted) > context.stockCount and context.Transitional:
        stockIn = positionDictSorted[(context.stockCount - 1)][0]
        stockOut = positionDictSorted[context.stockCount][0]
        doChange = stockIn not in context.portfolio.positions and stockOut in context.portfolio.positions
        # 如果需要进行边际交换操作时，仍需要判断交易的标的是否满足过滤条件
        if doChange:
            if np.round(positionDict[stockIn] - context.minPosition, 2) < 0 or np.round(positionDict[stockOut] - context.minPosition, 2) < 0:
                doChange = False
    # 把排前列的标的选择出来
    positionDict.clear()
    index = 1
    for (key,value) in positionDictSorted:
        if context.initPriceDict[key] == 0 and abs(grDict[key]) >= context.maxRange:
            # 近期涨跌幅度过大需要盘整巩固
            context.initPriceDict[key] = 1
        elif context.initPriceDict[key] == 1 and abs(grDict[key]) < context.consolidation:
            # 盘整巩固解除封闭
            context.initPriceDict[key] = 0
        # 涨跌幅过大需要等待一个周期后再解解封闭
        if context.initPriceDict[key] == 1:
            positionDict[key] = 0
        else:
            # 针对边际标的进行如果持仓则继续持仓
            if doChange and (key == stockIn or key == stockOut):
                if key == stockIn:
                    positionDict[key] = 0
                elif key == stockOut:
                    positionDict[key] = value
            elif index <= context.stockCount:
                positionDict[key] = value
            else:
                positionDict[key] = 0
        index += 1
    total_position = 0
    for stock in context.pool:
        total_position += positionDict[stock]
    if total_position == 0:
        total_position = 1
    # 计算所有标的的系数
    ratio = {}
    profitDict = {}
    for stock in context.pool:
        # 未超过入选条件时不进行交易
        stockPosition = positionDict[stock]
        # 如果RSI出现超卖情况时，不进行卖出操作
        RSI = getStockRSI(stock)
        if RSI > 30 and np.round(stockPosition - context.minPosition, 2) < 0:
            stockPosition = 0
        ratio = fun_caltraderatio(ratio, stock, stockPosition, total_position)
        # 计算持仓收益率
        if stock in context.portfolio.positions:
            profitDict[stock] = (context.portfolio.positions[stock].price - context.portfolio.positions[stock].avg_cost) / context.portfolio.positions[stock].avg_cost * 100
        else:
            profitDict[stock] = 0
    # 踢去持仓比例低于要求的标换
    if context.trackStock != "":
        record(T=1)
        drawCloseValue(context.trackStock)
    for stock in context.pool:
        if ratio[stock] < context.minRatio and ratio[stock] > 0:
            ratio[stock] = 0
        elif ratio[stock] > context.maxRatio:
            ratio[stock] = context.maxRatio
        if context.trackStock == stock:
            if ratio[stock] > 0:
                record(T=1.5)
            else:
                record(T=1)
    sumRatio = 0
    index = 1
    adjustment = np.round(context.stockCount * 1.0 / 2 / 100, 2)
    for (key,value) in positionDictSorted:
        if ratio[key] > 0:
            ratio[key] = adjustment + ratio[key]
        adjustment -= 0.01
        try:
            context.ratioMessage += "%2d.%s:%3d/%3d(%2d%%/%1.1f%%/%1.1f%%/%1.1f%%/%1.2f%%) %s\n" % (index,
                                                                                                   key,
                                                                                                   positionDict[key],
                                                                                                   value,
                                                                                                   ratio[key] * 100,
                                                                                                   variance(key),
                                                                                                   grDict[key],
                                                                                                   context.positionDict[key] + profitDict[key],
                                                                                                   getPremiumRate(context, key),
                                                                                                   getStockName(key))
        except:
            context.ratioMessage += "%2d.%s" % (index,key)
        sumRatio += ratio[key]
        index += 1
        if index > (context.stockCount + 2):
            break
    # 增加债券
    ratio[context.bondStock] = 0
    context.positionDict[context.bondStock] = 1
    try:
        context.ratioMessage += "合计：%3d%%，%3d，" % (sumRatio * 100,context.minPosition)
        if (1 - sumRatio) > 0.2:
            ratio[context.bondStock] = (1 - sumRatio)
    except:
        context.ratioMessage += "合计：%3d%%，%3d，" % (0,context.minPosition)
    context.ratioMessage += "累计金额：%d，可用资金：%d，总资产：%d\n" % (context.portfolio.inout_cash,context.portfolio.available_cash,context.portfolio.total_value)
    # 计算当前仓位
    if context.trackStock == '':
        record(P=(sumRatio * 100))
        record(B=(ratio[context.bondStock] * 100))
    return ratio

def tradeStockDict(context, buyDict):

    def tradeStock(context, stock, ratio):
        total_value = context.portfolio.total_value
        curPrice = history(1,'1d', 'close', stock, df=False)[stock][-1]
        curValue = 0
        if stock in context.portfolio.positions:
            curValue = context.portfolio.positions[stock].total_amount * curPrice
        quota = total_value * ratio
        # 平仓后记录实际收益率
        if quota == 0 and curValue > 0:
            if stock in context.portfolio.positions:
                context.positionDict[stock] += np.round((curPrice - context.portfolio.positions[stock].avg_cost) / context.portfolio.positions[stock].avg_cost * 100, 2)
            else:
                context.positionDict[stock] = curPrice
        deltaValue = np.round(abs(quota - curValue) / 1000, 0) * 1000
        if deltaValue >= context.minAmount or (quota == 0 and curValue > 0):
            order_target_value(stock, quota)
            context.doneTrade = True

    buylist = list(buyDict.keys())
    hStocks = history(1, '1d', 'close', buylist, df=False)
    myholdstock = list(context.portfolio.positions.keys())
    portfolioValue = context.portfolio.portfolio_value

    # 已有仓位
    holdDict = {}
    hholdstocks = history(1, '1d', 'close', myholdstock, df=False)
    for stock in myholdstock:
        tmpW = np.round((context.portfolio.positions[stock].total_amount * hholdstocks[stock]) / portfolioValue, 2)
        holdDict[stock] = float(tmpW)

    # 对已有仓位做排序
    tmpDict = {}
    for stock in holdDict:
        if stock in buyDict:
            tmpDict[stock] = np.round((buyDict[stock] - holdDict[stock]), 2)
        else:
            tmpDict[stock] = -999999
    tradeOrder = sorted(list(tmpDict.items()), key=lambda d:d[1], reverse=False)

    # 先卖掉持仓减少的标的
    tmpList = []
    for idx in tradeOrder:
        stock = idx[0]
        if stock in buyDict:
            tradeStock(context, stock, buyDict[stock])
        else:
            tradeStock(context, stock, 0)
        tmpList.append(stock)

    # 交易其他股票
    for i in range(len(buylist)):
        stock = buylist[i]
        if len(tmpList) != 0 :
            if stock not in tmpList:
                tradeStock(context, stock, buyDict[stock])
        else:
            tradeStock(context, stock, buyDict[stock])

if __name__ == '__main__':
    initialize(context)
    data = []
    handle_data(context, data)