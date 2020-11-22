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
"""
这里定义输出gfx图片
"""
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates
import numpy as np
from PIL import Image

from GolemQ.GQUtil.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST,
    FEATURES as FTR,
    )

def ohlc_plot_protype(ohlc_data, features, code=None, codename=None, title=None):

    # 暗色主题
    plt.style.use('Solarize_Light2')

    # 正常显示中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    fig = plt.figure(figsize = (22,9))
    plt.subplots_adjust(left=0.04, right=0.96)
    if (title is None):
        fig.suptitle(u'阿财的 {:s}（{:s}）机器学习买入点判断'.format(codename, code), fontsize=16)
    else:
        fig.suptitle(title, fontsize=16)
    ax1 = plt.subplot2grid((4,3),(0,0), rowspan=3, colspan=3)
    ax2 = plt.subplot2grid((4,3),(3,0), rowspan=1, colspan=3, sharex=ax1)
    #ax1 = fig.add_subplot(111)
    ax3 = ax1.twinx()

    # 绘制K线
    ohlc_data = ohlc_data.reset_index([1], drop=False)
    mc_stock_cn = mpf.make_marketcolors(up='r',down='g')
    s_stock_cn = mpf.make_mpf_style(marketcolors=mc_stock_cn)
    mpf.plot(data=ohlc_data, ax=ax1, type='candle', style=s_stock_cn)

    # 设定最标轴时间
    datetime_index = ohlc_data.index.get_level_values(level=0).to_series()
    DATETIME_LABEL = datetime_index.apply(lambda x: 
                                          x.strftime("%Y-%m-%d %H:%M")[2:16])

    ax1.plot(DATETIME_LABEL, features[FLD.BOLL_UB], lw=0.75, color='cyan', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.BOLL_LB], lw=0.75, color='fuchsia', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.BOLL], lw=0.75, color='purple', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.MA30], lw=0.75, color='green', alpha=0.6)
    ax1.fill_between(DATETIME_LABEL,
                     features[FLD.BOLL_UB],
                     features[FLD.BOLL_LB],
                     color='lightskyblue', alpha=0.15)

    ax1.plot(DATETIME_LABEL,
             features[FLD.MA90], lw=1, color='crimson',
             alpha=0.5)
    ax1.plot(DATETIME_LABEL, features[FLD.MA120], lw=1,
             color='limegreen', alpha=0.5)

    ax1.plot(DATETIME_LABEL,
             features[FLD.ATR_UB],
             lw=1, color='crimson', alpha=0.5)
    ax1.plot(DATETIME_LABEL,
             features[FLD.LINEAREG_PRICE],
             lw=1, color='crimson', alpha=0.8, linestyle='--')
    ax1.plot(DATETIME_LABEL,
             features[FLD.HMA5],
             lw=1, color='olive', alpha=0.8, linestyle='--')
    ax1.plot(DATETIME_LABEL,
             features[FLD.HMA10],
             lw=1, color='green', alpha=0.8, linestyle='-.')   
    ax1.plot(DATETIME_LABEL,
             features[FLD.ATR_LB],
             lw=1, color='crimson', alpha=0.5)

    ax1.set_xticks(range(0, len(DATETIME_LABEL), 
                         round(len(DATETIME_LABEL) / 12)))
    ax1.set_xticklabels(DATETIME_LABEL[::round(len(DATETIME_LABEL) / 12)])
    ax1.grid(False)

    ax2.plot(DATETIME_LABEL, features[FLD.DIF], 
             color='green', lw=1, label=FLD.DIF)
    ax2.plot(DATETIME_LABEL, features[FLD.DEA], 
             color = 'purple', lw = 1, label = FLD.DEA)

    barlist = ax2.bar(DATETIME_LABEL, features[FLD.MACD], width = 0.6, label = FLD.MACD)
    for i in range(len(DATETIME_LABEL.index)):
        if features[FLD.MACD][i] <= 0:
            barlist[i].set_color('g')
        else:
            barlist[i].set_color('r')
    ax2.set(ylabel='MACD(26,12,9)')

    ax2.set_xticks(range(0, len(DATETIME_LABEL), round(len(DATETIME_LABEL) / 12)))
    ax2.set_xticklabels(DATETIME_LABEL[::round(len(DATETIME_LABEL) / 12)], rotation=15)
    ax2.grid(False)
    ax3.grid(False)

    
    #plt.show()
    return ax1, ax2, ax3, DATETIME_LABEL


def ohlc_plot_protype_imf(ohlc_data, features, code, codename):

    # 暗色主题
    plt.style.use('Solarize_Light2')

    # 正常显示中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    fig = plt.figure(figsize = (22,9))

    fig.suptitle(u'阿财的 {:s}（{:s}）EEMD经验模态分解买入点判断'.format(codename, code), fontsize=16)
    ax1 = plt.subplot2grid((4,3),(0,0), rowspan=3, colspan=3)
    ax2 = plt.subplot2grid((4,3),(3,0), rowspan=1, colspan=3, sharex=ax1)
    #ax1 = fig.add_subplot(111)
    ax3 = ax1.twinx()

    ohlc_data = ohlc_data.reset_index([1], drop=False)

    mc_stock_cn = mpf.make_marketcolors(up='r',down='g')
    s_stock_cn = mpf.make_mpf_style(marketcolors=mc_stock_cn)
    mpf.plot(data=ohlc_data, ax=ax1, type='candle', style=s_stock_cn)
    DATETIME_LABEL = ohlc_data.index.get_level_values(level=0).to_series().apply(lambda x: 
                                                                                 x.strftime("%Y-%m-%d %H:%M")[2:16])

    ax1.plot(DATETIME_LABEL, features[FLD.BOLL_UB] , lw=0.75, color='cyan', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.BOLL_LB] , lw=0.75, color='fuchsia', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.BOLL] , lw=0.75, color='purple', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.MA30] , lw=0.75, color='green', alpha=0.6)
    ax1.fill_between(DATETIME_LABEL, 
                     features[FLD.BOLL_UB], 
                     features[FLD.BOLL_LB], 
                     color='lightskyblue', alpha=0.15)

    ax1.plot(DATETIME_LABEL,
             features[FLD.MA90], lw=1, color='crimson',
             alpha=0.5)
    ax1.plot(DATETIME_LABEL, features[FLD.MA120], lw=1,
             color='limegreen', alpha=0.5)

    ax1.plot(DATETIME_LABEL, 
             features[FLD.ATR_UB], 
             lw=1, color='crimson', alpha=0.5)
    ax1.plot(DATETIME_LABEL,
             features[FLD.LINEAREG_PRICE],
             lw=1, color='crimson', alpha=0.8, linestyle='--')
    ax1.plot(DATETIME_LABEL, 
             features[FLD.ATR_LB], 
             lw=1, color='crimson', alpha=0.5)

    ax1.set_xticks(range(0, len(DATETIME_LABEL), 
                         round(len(DATETIME_LABEL) / 12)))
    ax1.set_xticklabels(DATETIME_LABEL[::round(len(DATETIME_LABEL) / 12)])
    ax1.grid(False)

    ax2.plot(DATETIME_LABEL, features[FLD.DIF], 
             color='green', lw=1, label=FLD.DIF)
    ax2.plot(DATETIME_LABEL, features[FLD.DEA], 
             color='purple', lw=1, label=FLD.DEA)
    ax3.plot(DATETIME_LABEL, np.where((features[FLD.MAINFEST_UPRISING_TIMING_LAG] < 0), 
                                      (min(features[FTR.BEST_IMF3].min(),
                                           features[FTR.BEST_IMF4].min()) - 400 * 0.0168), 
                                      np.nan),
             color='green', lw=1.5, label=FLD.RENKO_BOOST_L_TIMING_LAG, alpha=0.8)
    ax3.plot(DATETIME_LABEL, np.where((features[FLD.MAINFEST_UPRISING_TIMING_LAG] > 0),
                                      (min(features[FTR.BEST_IMF3].min(),
                                           features[FTR.BEST_IMF4].min()) - 400 * 0.0168),
                                      np.nan),
             color='red', lw=1.5, label=FLD.RENKO_BOOST_L_TIMING_LAG, alpha=0.8)
    ax3.plot(DATETIME_LABEL, np.where((features[FTR.BEST_IMF4] < features[FTR.BEST_IMF4].shift(1)),
                                      features[FTR.BEST_IMF4],
                                      np.nan),  
             color='lime', lw=1, label=FLD.COMBINE_DENSITY, alpha=0.5)
    ax3.plot(DATETIME_LABEL, np.where((features[FTR.BEST_IMF4] > features[FTR.BEST_IMF4].shift(1)),
                                      features[FTR.BEST_IMF4],
                                      np.nan), 
             color='salmon', lw=1, label=FLD.COMBINE_DENSITY, alpha=0.5)
    barlist = ax2.bar(DATETIME_LABEL, features[FLD.MACD], width = 0.6, label = FLD.MACD)
    for i in range(len(DATETIME_LABEL.index)):
        if features[FLD.MACD][i] <= 0:
            barlist[i].set_color('g')
        else:
            barlist[i].set_color('r')
    ax2.set(ylabel='MACD(26,12,9)')

    ax2.set_xticks(range(0, len(DATETIME_LABEL), round(len(DATETIME_LABEL) / 12)))
    ax2.set_xticklabels(DATETIME_LABEL[::round(len(DATETIME_LABEL) / 12)], rotation=15)
    ax2.grid(False)
    ax3.grid(False)

    ax3.plot(DATETIME_LABEL,
             np.where((features[ST.BOOTSTRAP_GROUND_ZERO] > 0),
                      (min(features[FTR.BEST_IMF3].min(),
                           features[FTR.BEST_IMF4].min()) - 400 * 0.0382), np.nan),
             'y^', alpha = 0.33)
    #plt.show()
    return ax1, ax2, ax3, DATETIME_LABEL


def ohlc_plot_mapower(ohlc_data, features, code=None, codename=None, title=None):

    # 暗色主题
    plt.style.use('Solarize_Light2')

    # 正常显示中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    fig = plt.figure(figsize = (22,9))
    plt.subplots_adjust(left=0.04, right=0.96)
    if (title is None):
        fig.suptitle(u'阿财的 {:s}（{:s}）MA多头趋势买入点判断'.format(codename, code), fontsize=16)
    else:
        fig.suptitle(title, fontsize=16)
    ax1 = plt.subplot2grid((4,3),(0,0), rowspan=3, colspan=3)
    ax2 = plt.subplot2grid((4,3),(3,0), rowspan=1, colspan=3, sharex=ax1)
    #ax1 = fig.add_subplot(111)
    ax3 = ax1.twinx()

    ohlc_data = ohlc_data.reset_index([1], drop=False)

    ohlc_data = ohlc_data.rename(columns={AKA.OPEN: "Open", 
                                          AKA.HIGH: "High", 
                                          AKA.LOW: "Low",
                                          AKA.CLOSE: "Close"})
    mc_stock_cn = mpf.make_marketcolors(up='r',down='g')
    s_stock_cn = mpf.make_mpf_style(marketcolors=mc_stock_cn)
    mpf.plot(data=ohlc_data, ax=ax1, type='candle', style=s_stock_cn)
    DATETIME_LABEL = ohlc_data.index.get_level_values(level=0).to_series().apply(lambda x: 
                                                                                 x.strftime("%Y-%m-%d %H:%M")[2:16])

    ax1.plot(DATETIME_LABEL, features[FLD.BOLL_UB], lw=0.75, color='cyan', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.BOLL_LB], lw=0.75, color='fuchsia', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.BOLL], lw=0.75, color='purple', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.MA30], lw=0.75, color='green', alpha=0.6)
    ax1.fill_between(DATETIME_LABEL,
                     features[FLD.BOLL_UB],
                     features[FLD.BOLL_LB],
                     color='lightskyblue', alpha=0.15)

    ax1.plot(DATETIME_LABEL,
             features[FLD.MA90], lw=1, color='crimson',
             alpha=0.5)
    ax1.plot(DATETIME_LABEL, features[FLD.MA120], lw=1,
             color='limegreen', alpha=0.5)

    ax1.plot(DATETIME_LABEL,
             features[FLD.ATR_UB],
             lw=1, color='crimson', alpha=0.5)
    ax1.plot(DATETIME_LABEL,
             features[FLD.LINEAREG_PRICE],
             lw=1, color='crimson', alpha=0.8, linestyle='--')
    ax1.plot(DATETIME_LABEL,
             features[FLD.HMA5],
             lw=1, color='olive', alpha=0.8, linestyle='--')
    ax1.plot(DATETIME_LABEL,
             features[FLD.HMA10],
             lw=1, color='green', alpha=0.8, linestyle='-.')   
    ax1.plot(DATETIME_LABEL,
             features[FLD.ATR_LB],
             lw=1, color='crimson', alpha=0.5)

    ax1.set_xticks(range(0, len(DATETIME_LABEL), 
                         round(len(DATETIME_LABEL) / 12)))
    ax1.set_xticklabels(DATETIME_LABEL[::round(len(DATETIME_LABEL) / 12)])
    ax1.grid(False)

    if (FLD.COMBINE_DENSITY in features.columns):
        ax3.plot(DATETIME_LABEL, 
                 features[FLD.COMBINE_DENSITY], 
                 lw=0.75, color ='lightskyblue', alpha=0.33)
        #ax3.plot(DATETIME_LABEL,
        #         np.where(features[FLD.ATR_SuperTrend_TIMING_LAG] > 0,
        #                  features[FLD.COMBINE_DENSITY], np.nan),
        #         lw=0.75, color ='crimson', alpha=0.33)
    #if (FLD.CCI in features.columns):
    #    #ax3.plot(DATETIME_LABEL,
    #    #         features[FLD.RSI] / 100,
    #    #         lw=0.75, color ='green', alpha=0.33)
    #    ax3.plot(DATETIME_LABEL, 
    #             features[FLD.CCI_NORM], 
    #             lw=0.75, color ='coral', alpha=0.25)
    #if (FLD.HMAPOWER30 in features.columns):
    #    ax3.plot(DATETIME_LABEL, 
    #             np.where((features[FLD.HMAPOWER30_TIMING_LAG] > -1),
    #             features[FLD.HMAPOWER30], np.nan), 
    #             lw=1.25, color ='pink', alpha=0.8)
    #    ax3.plot(DATETIME_LABEL, 
    #             np.where((features[FLD.HMAPOWER30_TIMING_LAG] < 1),
    #             features[FLD.HMAPOWER30], np.nan), 
    #             lw=1.25, color ='skyblue', alpha=0.8)
    if (FLD.MAPOWER30 in features.columns):
        ax3.plot(DATETIME_LABEL, 
                 np.where((features[FLD.MAPOWER30_TIMING_LAG] < 1),
                 features[FLD.MAPOWER30], np.nan), 
                 lw=0.75, color ='lime', alpha=0.8)
        ax3.plot(DATETIME_LABEL,
                 np.where((features[FLD.MAPOWER30_TIMING_LAG] > -1),
                          features[FLD.MAPOWER30], np.nan),
                 lw=0.75, color ='gray', alpha=0.5)
    if (FLD.MAPOWER120 in features.columns):
        ax3.plot(DATETIME_LABEL, 
                 np.where(features[FLD.MAPOWER120_TIMING_LAG] < 1, 
                          features[FLD.MAPOWER120], np.nan),
                 lw=0.75, color ='olive', alpha=0.8)
        ax3.plot(DATETIME_LABEL, 
                 np.where(features[FLD.MAPOWER120_TIMING_LAG] > -1, 
                          features[FLD.MAPOWER120], np.nan),
                 lw=0.75, color ='orange', alpha=0.8)

    if (FLD.RENKO_BOOST_L_TIMING_LAG in features.columns):
        ax3.plot(DATETIME_LABEL, 
                 np.where((features[FLD.RENKO_BOOST_L_TIMING_LAG] > 0),
                          features[FLD.CCI_NORM].min() - 0.0168, np.nan),
                 lw=1, color ='red', alpha=0.8)
        ax3.plot(DATETIME_LABEL, 
                 np.where((features[FLD.RENKO_TREND_L_TIMING_LAG] > 0),
                         features[FLD.CCI_NORM].min() - 0.0168, np.nan),
                 lw=1, color ='coral', alpha=0.5)
    if (FTR.BOOTSTRAP_ENHANCED in features.columns):
        ax3.plot(DATETIME_LABEL, 
                 np.where((features[FTR.BOOTSTRAP_ENHANCED] > 0),
                          features[FLD.CCI_NORM].min() + 0.00382, np.nan),
                 lw=1.75, color ='olive', alpha=0.5)
        ax3.plot(DATETIME_LABEL, 
                 np.where((features[FTR.BOOTSTRAP_ENHANCED] > 0) & \
                     (features[FLD.HMAPOWER30_TIMING_LAG] > 0) & \
                     (features[FLD.MAXFACTOR_TREND_TIMING_LAG] > 0),
                          features[FLD.CCI_NORM].min() + 0.00382, np.nan),
                 lw=1.75, color ='purple', alpha=0.8)
        ax3.plot(DATETIME_LABEL, 
                 np.where((features[FTR.BOOTSTRAP_ENHANCED] > 0) & \
                     (features[FLD.HMAPOWER30_TIMING_LAG] > 0) & \
                     (features[FLD.ZEN_WAVELET_TIMING_LAG] > 0),
                          features[FLD.CCI_NORM].min() + 0.00382, np.nan),
                 lw=1.75, color ='orange', alpha=0.8)
    if (FLD.HMAPOWER120 in features.columns):
        #ax3.plot(DATETIME_LABEL,
        #         np.where((features[FLD.HMAPOWER30_MA_TIMING_LAG] > 0),
        #                  features[FLD.CCI_NORM].min() - 0.00512, np.nan),
        #         lw=1.75, color ='orange', alpha=0.8)
        ax3.plot(DATETIME_LABEL,
                 features[FLD.HMAPOWER120],
                 lw=0.75, color ='gray', alpha=0.22)
        ax3.plot(DATETIME_LABEL,
                 np.where(features[FLD.HMAPOWER120_TIMING_LAG] <= 1, 
                          features[FLD.HMAPOWER120], np.nan,),
                 lw=0.75, color ='darkgray', alpha=0.33)
        ax3.plot(DATETIME_LABEL,
                 np.where(features[FLD.HMAPOWER120_TIMING_LAG] >= -1, 
                          features[FLD.HMAPOWER120], np.nan,),
                 lw=1.25, color ='magenta', alpha=0.66)
        
    if (FLD.ZEN_BOOST_TIMING_LAG in features.columns) and \
        (FLD.MAPOWER120 in features.columns):
        ax1.plot(DATETIME_LABEL,
                 np.where((features[FLD.ZEN_BOOST_TIMING_LAG] == 1),
                          (features[FLD.BOLL_LB] + features[FLD.ATR_LB]) / 2.08, np.nan),
                 'r.', alpha = 0.33)
        ax3.plot(DATETIME_LABEL,
                 np.where((features[FLD.ZEN_BOOST_TIMING_LAG] == -1),
                          np.maximum(features[FLD.MAPOWER30],
                                     features[FLD.MAPOWER120]), np.nan),
                 'g.', alpha = 0.33)

        #ax3.plot(DATETIME_LABEL,
        #         np.where((features[FLD.RENKO_TREND_L_TIMING_LAG] > 0) & \
        #                  ((features[FLD.RENKO_TREND_L_TIMING_LAG] >
        #                  features[FLD.DEA_ZERO_TIMING_LAG]) | \
        #                  (features[FLD.REGTREE_TIMING_LAG] >
        #                  features[FLD.DEA_ZERO_TIMING_LAG]) | \
        #                  ((features[FLD.MA90_CLEARANCE_TIMING_LAG] >
        #                  features[FLD.DEA_ZERO_TIMING_LAG]) & \
        #                  (features[FLD.MA90] > features[FLD.MA120]) & \
        #                  ((features[FLD.BOLL] + features[FLD.MA90]) >
        #                  (features[FLD.MA120] + features[FLD.MA30])) & \
        #                  (features[FLD.MA90_CLEARANCE_TIMING_LAG] /
        #                  features[FLD.MA90_CROSS_JX_BEFORE] > 0.618)) | \
        #                  ((features[FLD.MA120_CLEARANCE_TIMING_LAG] >
        #                  features[FLD.DEA_ZERO_TIMING_LAG]) & \
        #                  (features[FLD.MA90] > features[FLD.MA120]) & \
        #                  ((features[FLD.BOLL] + features[FLD.MA90]) >
        #                  (features[FLD.MA120] + features[FLD.MA30])) & \
        #                  (features[FLD.MA120_CLEARANCE_TIMING_LAG] /
        #                  features[FLD.MA90_CROSS_JX_BEFORE] > 0.618))),
        #                  (features[FLD.CCI] / 600 + 0.5).min() - 0.00382,
        #                  np.nan),
        #         lw = 1.75, color = 'crimson', alpha = 0.5)
    ax2.plot(DATETIME_LABEL, features[FLD.DIF], 
             color='green', lw=1, label=FLD.DIF)
    ax2.plot(DATETIME_LABEL, features[FLD.DEA], 
             color = 'purple', lw = 1, label = FLD.DEA)

    barlist = ax2.bar(DATETIME_LABEL, features[FLD.MACD], width = 0.6, label = FLD.MACD)
    for i in range(len(DATETIME_LABEL.index)):
        if features[FLD.MACD][i] <= 0:
            barlist[i].set_color('g')
        else:
            barlist[i].set_color('r')
    ax2.set(ylabel='MACD(26,12,9)')

    ax2.set_xticks(range(0, len(DATETIME_LABEL), round(len(DATETIME_LABEL) / 12)))
    ax2.set_xticklabels(DATETIME_LABEL[::round(len(DATETIME_LABEL) / 12)], rotation=15)
    ax2.grid(False)
    ax3.grid(False)

    
    #plt.show()
    return ax1, ax2, ax3, DATETIME_LABEL


def save_png24_to_png8(png_filename):
    im = Image.open(png_filename)

    # PIL complains if you don't load explicitly
    im.load()

    # Get the alpha band
    alpha = im.split()[-1]

    im = im.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)

    # Set all pixel values below 128 to 255,
    # and the rest to 0
    mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)

    # Paste the color of index 255 and use alpha as a mask
    im.paste(255, mask)

    # The transparency index is 255
    im.save(png_filename, transparency=255)