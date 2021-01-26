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
import math
from PIL import Image

from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST,
    FEATURES as FTR,
    )

def ohlc_plot_protype(ohlc_data, features, code=None, 
                      codename=None, title=None):

    # 暗色主题
    plt.style.use('Solarize_Light2')

    # 正常显示中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    fig = plt.figure(figsize = (22,9))
    plt.subplots_adjust(left=0.04, right=0.96)
    if (title is None):
        fig.suptitle(u'阿财的 {:s}（{:s}）机器学习买入点判断'.format(codename, 
                                                       code), fontsize=16)
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

    # 设定x标轴时间
    datetime_index = ohlc_data.index.get_level_values(level=0).to_series()
    DATETIME_LABEL = datetime_index.apply(lambda x: 
                                          x.strftime("%Y-%m-%d %H:%M")[2:16])

    ax1.plot(DATETIME_LABEL, features[FLD.BOLL_UB], lw=0.75, 
             color='cyan', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.BOLL_LB], lw=0.75, 
             color='fuchsia', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.BOLL], lw=0.75, 
             color='purple', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.MA30], lw=0.75, 
             color='green', alpha=0.6)
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
    ax1.set_xticklabels(DATETIME_LABEL[::round(len(DATETIME_LABEL) / 12)], 
                        rotation=15)
    ax1.grid(False)

    ax2.plot(DATETIME_LABEL, features[FLD.DIF], 
             color='orange', lw=1, label=FLD.DIF)
    ax2.plot(DATETIME_LABEL, features[FLD.DEA], 
             color = 'purple', lw = 1, label = FLD.DEA)

    barlist = ax2.bar(DATETIME_LABEL, features[FLD.MACD], 
                      width=0.6, label=FLD.MACD)
    for i in range(len(DATETIME_LABEL.index)):
        if features[FLD.MACD][i] <= 0:
            barlist[i].set_color('g')
        else:
            barlist[i].set_color('r')
    ax2.set(ylabel='MACD(26,12,9)')

    ax2.set_xticks(range(0, len(DATETIME_LABEL), 
                         round(len(DATETIME_LABEL) / 12)))
    ax2.set_xticklabels(DATETIME_LABEL[::round(len(DATETIME_LABEL) / 12)], 
                        rotation=15)
    ax2.grid(False)
    ax3.set_xticks(range(0, len(DATETIME_LABEL), 
                         round(len(DATETIME_LABEL) / 12)))
    ax3.set_xticklabels(DATETIME_LABEL[::round(len(DATETIME_LABEL) / 12)], 
                        rotation=15)
    ax3.grid(False)
    
    #plt.show()
    return ax1, ax2, ax3, DATETIME_LABEL


def ohlc_plot_protype_imf(ohlc_data, features, code, codename):

    # 暗色主题
    plt.style.use('Solarize_Light2')

    # 正常显示中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    fig = plt.figure(figsize = (22,9))
    plt.subplots_adjust(left=0.04, right=0.96)
    if (title is None):
        fig.suptitle(u'阿财的 {:s}（{:s}）机器学习买入点判断'.format(codename, 
                                                       code), fontsize=16)
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

    # 设定x标轴时间
    datetime_index = ohlc_data.index.get_level_values(level=0).to_series()
    DATETIME_LABEL = datetime_index.apply(lambda x: 
                                          x.strftime("%Y-%m-%d %H:%M")[2:16])

    ax1.plot(DATETIME_LABEL, features[FLD.BOLL_UB], lw=0.75, 
             color='cyan', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.BOLL_LB], lw=0.75, 
             color='fuchsia', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.BOLL], lw=0.75, 
             color='purple', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.MA30], lw=0.75, 
             color='green', alpha=0.6)
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
    ax1.set_xticklabels(DATETIME_LABEL[::round(len(DATETIME_LABEL) / 12)], rotation=15)
    ax1.grid(False)

    ax2.plot(DATETIME_LABEL, features[FLD.DIF], 
             color='green', lw=1, label=FLD.DIF)
    ax2.plot(DATETIME_LABEL, features[FLD.DEA], 
             color='purple', lw=1, label=FLD.DEA)
    ax3.plot(DATETIME_LABEL, 
             np.where((features[FLD.MAINFEST_UPRISING_TIMING_LAG] < 0),
                      (min(features[FTR.BEST_IMF3].min(),
                           features[FTR.BEST_IMF4].min()) - 400 * 0.0168),
                      np.nan),
             color='green', lw=1.5, label=FLD.RENKO_BOOST_L_TIMING_LAG, alpha=0.8)
    ax3.plot(DATETIME_LABEL, 
             np.where((features[FLD.MAINFEST_UPRISING_TIMING_LAG] > 0),
                      (min(features[FTR.BEST_IMF3].min(),
                           features[FTR.BEST_IMF4].min()) - 400 * 0.0168),
                           np.nan),
             color='red', lw=1.5, label=FLD.RENKO_BOOST_L_TIMING_LAG, alpha=0.8)
    ax3.plot(DATETIME_LABEL, 
             np.where((features[FTR.BEST_IMF4] < features[FTR.BEST_IMF4].shift(1)),
                      features[FTR.BEST_IMF4], np.nan),  
             color='lime', lw=1, label=FLD.COMBINE_DENSITY, alpha=0.5)
    ax3.plot(DATETIME_LABEL, 
             np.where((features[FTR.BEST_IMF4] > features[FTR.BEST_IMF4].shift(1)),
                                      features[FTR.BEST_IMF4],
                                      np.nan), 
             color='salmon', lw=1, label=FLD.COMBINE_DENSITY, alpha=0.5)
    barlist = ax2.bar(DATETIME_LABEL, features[FLD.MACD], width=0.6, label=FLD.MACD)
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
    ax2 = plt.subplot2grid((4,3),(3,0), rowspan=1, colspan=3)
    ax1.zorder = 2.5
    ax2.zorder = 0
    #ax1 = fig.add_subplot(111)
    ax3 = ax1.twinx()
    ax3.zorder = 2.5

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

    ax1.plot(DATETIME_LABEL, features[FLD.BOLL_UB], lw=0.75, 
             color='cyan', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.BOLL_LB], lw=0.75, 
             color='fuchsia', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.BOLL], lw=0.75, 
             color='purple', alpha=0.6)
    ax1.plot(DATETIME_LABEL, features[FLD.MA30], lw=0.75, 
             color='green', alpha=0.6)
    ax1.fill_between(DATETIME_LABEL,
                     features[FLD.BOLL_UB],
                     features[FLD.BOLL_LB],
                     color='lightskyblue', alpha=0.15)

    ax1.fill_between(DATETIME_LABEL,
                     np.where((features[FLD.MAINFEST_UPRISING_TIMING_LAG] > 0),
        features[FLD.BOLL_UB], np.nan),
                     np.where((features[FLD.MAINFEST_UPRISING_TIMING_LAG] > 0),
        features[FLD.BOLL_LB], np.nan),
                     color='orangered', alpha=0.15)
    ax1.fill_between(DATETIME_LABEL,
                     np.where((features[FLD.MAINFEST_DOWNRISK_TIMING_LAG] < 0),
        features[FLD.BOLL_UB], np.nan),
                     np.where((features[FLD.MAINFEST_DOWNRISK_TIMING_LAG] < 0),
        features[FLD.BOLL_LB], np.nan),
                     color='darkseagreen', alpha=0.15) 
    ax3.plot(DATETIME_LABEL, 
             np.where((features[FLD.MAINFEST_DOWNRISK_TIMING_LAG] < 0),
                      features[FLD.CCI_NORM].min() - 0.0618, np.nan),
             'g.', alpha = 0.75)
    ax3.plot(DATETIME_LABEL, 
             np.where((features[FLD.MAINFEST_DOWNRISK_TIMING_LAG] > 0) & \
                      ((features[FLD.MAINFEST_DOWNRISK_TIMING_LAG] + features[FLD.ZEN_DASH_TIMING_LAG_MAJOR]) < 0) & \
                      (np.minimum(features[FLD.DEADPOOL_CANDIDATE_TIMING_LAG],
                                  features[FLD.DEADPOOL_REMIX_TIMING_LAG]) < 0),
                      features[FLD.CCI_NORM].min() - 0.0618, np.nan),
             'g.', alpha = 0.25)
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
    ax1.set_xticklabels(DATETIME_LABEL[::round(len(DATETIME_LABEL) / 12)], rotation=15)
    ax1.grid(False)

    ax2.plot(DATETIME_LABEL, features[FLD.DIF], 
             color='orange', lw=1, label=FLD.DIF)
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
    ax3.set_xticks(range(0, len(DATETIME_LABEL), 
                         round(len(DATETIME_LABEL) / 12)))
    ax3.set_xticklabels(DATETIME_LABEL[::round(len(DATETIME_LABEL) / 12)], rotation=15)
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


def mark_annotate_phase2(checkpoints, features, ax1, ax3, colorwarp='skyblue'):
    '''
    绘制标记_phase2
    '''
    if (ST.PREDICT_LONG in features.columns):
        markers = features.loc[checkpoints, [FLD.PEAK_LOW_TIMING_LAG,
                                             FLD.COMBINE_DENSITY,
                                             FLD.MAPOWER30_MAJOR,
                                             FLD.HMAPOWER120_MAJOR,
                                             FLD.MAPOWER30,
                                             FLD.HMAPOWER120,
                                             FLD.BOLL_LB,
                                             FLD.ATR_LB,
                                             FLD.MA90_CLEARANCE,
                                             FLD.MA120_CLEARANCE,
                                             FLD.BOLL_CHANNEL,
                                             FLD.MA_CHANNEL,
                                             FLD.MA120_CHANNEL,
                                             FTR.UPRISING_RAIL_TIMING_LAG,
                                             FLD.MAPOWER_BASELINE_TIMING_LAG,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR,
                                             FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                             FLD.DRAWDOWN_RATIO,
                                             FLD.DRAWDOWN_RATIO_MAJOR,
                                             FLD.ZEN_WAVELET_TIMING_LAG,
                                             FLD.RENKO_TREND_S_TIMING_LAG,
                                             FLD.RENKO_TREND_L_TIMING_LAG,
                                             FLD.DEADPOOL_CANDIDATE_TIMING_LAG,
                                             FLD.MAPOWER_HMAPOWER120_TIMING_LAG,
                                             ST.PREDICT_LONG,
                                             FLD.PREDICT_PROB_LONG,
                                             FLD.BOLL_JX_RSI,
                                             FLD.BOLL_JX_MAXFACTOR,]].copy()
    else:
        markers = features.loc[checkpoints, [FLD.PEAK_LOW_TIMING_LAG,
                                             FLD.COMBINE_DENSITY,
                                             FLD.MAPOWER30_MAJOR,
                                             FLD.HMAPOWER120_MAJOR,
                                             FLD.MAPOWER30,
                                             FLD.HMAPOWER120,
                                             FLD.BOLL_LB,
                                             FLD.ATR_LB,
                                             FLD.MA90_CLEARANCE,
                                             FLD.MA120_CLEARANCE,
                                             FLD.BOLL_CHANNEL,
                                             FLD.MA_CHANNEL,
                                             FLD.MA120_CHANNEL,
                                             FTR.UPRISING_RAIL_TIMING_LAG,
                                             FLD.MAPOWER_BASELINE_TIMING_LAG,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR,
                                             FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                             FLD.DRAWDOWN_RATIO,
                                             FLD.DRAWDOWN_RATIO_MAJOR,
                                             FLD.ZEN_WAVELET_TIMING_LAG,
                                             FLD.RENKO_TREND_S_TIMING_LAG,
                                             FLD.RENKO_TREND_L_TIMING_LAG,
                                             FLD.DEADPOOL_CANDIDATE_TIMING_LAG,
                                             FLD.MAPOWER_HMAPOWER120_TIMING_LAG,]].copy()

    #print(markers)
    for index, marker in markers.iterrows():
        mkr_x = marker.name[0].strftime("%Y-%m-%d %H:%M")[2:16]
        if (ST.PREDICT_LONG in features.columns) and (marker.at[ST.PREDICT_LONG] < 0.1):
            mkr_text_template = '*{}*\nxgboost_pred:{:02d}\npred_prob:{:.2f}\nRSI:{:.3f}/MFT:{:.3f}\nDRAWDOWN:{:.3f}/{:.3f}'
            mkr_text = mkr_text_template.format(mkr_x,
                                                int(marker.at[ST.PREDICT_LONG]),
                                                marker.at[FLD.PREDICT_PROB_LONG],
                                                marker.at[FLD.BOLL_JX_RSI],
                                                marker.at[FLD.BOLL_JX_MAXFACTOR],
                                                marker.at[FLD.DRAWDOWN_RATIO],
                                                marker.at[FLD.DRAWDOWN_RATIO_MAJOR],)
        else:
            mkr_text_template = '''*{}*
COMB:{:.3f}/BOL:{:.3f}
MAPWR3:{:.2f}/HMA12:{:.2f}
MACH:{:.2f}/MA12CH:{:.2f}
MA9CLR:{:.2f}/{:.2f}/{:.2f}
PEAK_LO:{:03d}/{:03d}/{:03d}
PEAK_HI:{:03d}/{:03d}/{:03d}
DRAWDOWN:{:.3f}/{:.3f}
RENKO:{:03d}/{:03d}/ZEN:{:03d}
MAPWR_BSL_HMAPWR12:{:02d}'''
            mkr_text = mkr_text_template.format(mkr_x,
                                                marker.at[FLD.COMBINE_DENSITY],
                                                marker.at[FLD.BOLL_CHANNEL],
                                                marker.at[FLD.MAPOWER30_MAJOR],
                                                marker.at[FLD.HMAPOWER120_MAJOR],
                                                marker.at[FLD.MA_CHANNEL],
                                                marker.at[FLD.MA120_CHANNEL],
                                                marker.at[FLD.MA90_CLEARANCE],
                                                marker.at[FLD.MA120_CLEARANCE],
                                                marker.at[FLD.MAPOWER30_MAJOR] + marker.at[FLD.HMAPOWER120_MAJOR] + marker.at[FLD.MAPOWER30],
                                                int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE]),
                                                int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR]),
                                                int(marker.at[FLD.DEADPOOL_CANDIDATE_TIMING_LAG]),
                                                int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE]),
                                                int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR]),
                                                int(marker.at[FLD.MAPOWER30_TIMING_LAG_MAJOR]),
                                                marker.at[FLD.DRAWDOWN_RATIO],
                                                marker.at[FLD.DRAWDOWN_RATIO_MAJOR],
                                                int(marker.at[FLD.RENKO_TREND_S_TIMING_LAG]),
                                                int(marker.at[FLD.RENKO_TREND_L_TIMING_LAG]),
                                                int(marker.at[FLD.ZEN_WAVELET_TIMING_LAG]),
                                                int(marker.at[FLD.MAPOWER_HMAPOWER120_TIMING_LAG]))
        if (marker.at[FTR.UPRISING_RAIL_TIMING_LAG] == 1) or \
            (marker.at[FLD.MAPOWER_BASELINE_TIMING_LAG] == 1):
            mkr_y = (marker.at[FLD.MAPOWER30_MAJOR] + marker.at[FLD.HMAPOWER120_MAJOR]) / 2
            ax3.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data', 
                         xytext=(20, -60), textcoords='offset points', 
                         fontsize=6, va="center", ha="center", zorder=4, 
                         alpha=0.66,
                         bbox=dict(boxstyle="round4", fc="w", 
                                   color=colorwarp),
                         arrowprops=dict(arrowstyle="->", 
                                         connectionstyle="arc3,rad=.2", 
                                         color=colorwarp))
        else:
            mkr_y = (marker.at[FLD.BOLL_LB] + marker.at[FLD.ATR_LB]) / 2.09
            ax1.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data',
                         xytext=(20, -60), textcoords='offset points', 
                         fontsize=6, va="center", ha="center", zorder=4,
                         alpha=0.66,
                         bbox=dict(boxstyle="round4", fc="w", 
                                   color=colorwarp),
                         arrowprops=dict(arrowstyle="->", 
                                         connectionstyle="arc3,rad=.2", 
                                         color=colorwarp))


def mark_annotate_div(checkpoints, features, ax1, colorwarp='olive', codex=0):
    '''
    绘制标记
    '''
    markers = features.loc[checkpoints, [FLD.ATR_LB,
                                         FLD.ATR_UB,
                                         FLD.HMA10,
                                         FLD.POLY9_REGTREE_DIVERGENCE,
                                         ST.BOOTSTRAP_GROUND_ZERO,
                                        FLD.DRAWDOWN_RATIO,
                                        FLD.DRAWDOWN_RATIO_MAJOR,
                                        FLD.BOLL_JX_RSI,
                                        FLD.BOLL_JX_MAXFACTOR,
                                        FLD.POLYNOMIAL9,
                                        ST.PREDICT_LONG,
                                        FLD.PREDICT_PROB_LONG,
                                        FLD.ZEN_BOOST_TIMING_LAG,
                                        FLD.ZEN_WAVELET_TIMING_LAG,
                                        FLD.RENKO_TREND_S_TIMING_LAG,
                                        FTR.POLYNOMIAL9_EMPIRICAL,
                                        FLD.POLY9_MA30_DIVERGENCE,
                                        FLD.POLY9_MA90_DIVERGENCE,
                                        FLD.POLYNOMIAL9_TIMING_GAP,
                                        FLD.POLYNOMIAL9_TIMING_LAG,
                                        FLD.MAPOWER30_TIMING_GAP,
                                        FLD.HMAPOWER120_TIMING_GAP,
                                        FLD.MAXFACTOR_TREND_TIMING_LAG,
                                        FLD.LRC_HMA10_TIMING_LAG,
                                        FLD.MA30_HMA5_TIMING_LAG,
                                        FLD.BOLL_RAISED_TIMING_LAG,
                                        FLD.ZEN_PEAK_TIMING_LAG,
                                        FLD.ZEN_DASH_TIMING_LAG,
                                        FLD.ZEN_PEAK_TIMING_LAG_MAJOR,
                                        FLD.ZEN_DASH_TIMING_LAG_MAJOR,
                                        FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                        FLD.MA90_CLEARANCE_TIMING_LAG,
                                        FLD.MA90_TREND_TIMING_LAG,
                                        FLD.DEA_ZERO_TIMING_LAG,
                                        FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG,
                                        FLD.MA120_CHANNEL,
                                        FLD.BOLL_CHANNEL,
                                        FLD.BOLL_JX_RANK,
                                        FLD.BOOTSTRAP_GROUND_ZERO_RANK,
                                        FLD.POLYNOMIAL9_CHECKOUT_TIMING_LAG,
                                        FLD.ATR_SuperTrend_TIMING_LAG,
                                        FLD.ATR_Stopline_TIMING_LAG,
                                        FLD.BOOTSTRAP_COMBO_TIMING_LAG,
                                        ST.BOOTSTRAP_ENDPOINTS,
                                        FLD.MAINFEST_UPRISING_COEFFICIENT,
                                        FTR.BOOTSTRAP_ENHANCED_TIMING_LAG,
                                        FLD.BOLL_JX_RANK_REMARK,
                                        ST.FULLSTACK_COEFFICIENT,
                                        ST.HALFSTACK_COEFFICIENT,
                                        FLD.ZEN_DASH_RETURNS,
                                        FLD.ZEN_PEAK_RETURNS,
                                        FLD.ZEN_PEAK_RETURNS_MAJOR,]].copy()

    #print(markers)
    codex_list = [int(math.pow(2, i)) for i in range(0, 31)]
    for index, marker in markers.iterrows():
        mkr_x = marker.name[0].strftime("%Y-%m-%d %H:%M")[2:16]
        mkr_text_template = """*{}*
BOLL:{:.3f}\MA12CH:{:.3f}
MA90CLR:{:03d}/{:03d}/DEA:{:03d}
POLY9:{:02d}\RT_DIV:{:.3f}
DIVERGENCE:{:.3f}/{:.3f}
RSI:{:.3f}/MFT:{:.3f}
DRAWDOWN:{:.3f}/{:.3f}
GAP:{:02d}\{:02d}\{:02d}\{:02d}\MFT:{:02d}
LRC:{:02d}\MA30:{:02d}:BOLL:{:02d}
ZEN:{:03d}\{:03d}\{:03d}\{:03d}\{:02d}\{:02d}
ATR:{:03d}\BST:{:03d}\COEF:{:03d}
RNK:{:02d}\{:02d}\ATR:{:03d}\CB:{:02d}
FS:{}/RET:{:.02%}
HS:{}/RET:{:.02%}"""
        mkr_text = mkr_text_template.format(mkr_x if (codex == 0) else '{} C:{:02x}'.format(mkr_x, int(math.log2(codex) + 1)),
                                            marker.at[FLD.BOLL_CHANNEL],
                                            marker.at[FLD.MA120_CHANNEL],
                                            int(marker.at[FLD.MA90_CLEARANCE_TIMING_LAG]),
                                            int(marker.at[FLD.MA90_TREND_TIMING_LAG]),
                                            int(marker.at[FLD.DEA_ZERO_TIMING_LAG]),
                                            int(marker.at[FLD.POLYNOMIAL9_TIMING_LAG]),
                                            max(min((marker.at[FLD.POLY9_REGTREE_DIVERGENCE]), 255), -255),
                                            marker.at[FLD.POLY9_MA30_DIVERGENCE],
                                            marker.at[FLD.POLY9_MA90_DIVERGENCE],
                                            marker.at[FLD.BOLL_JX_RSI],
                                            marker.at[FLD.BOLL_JX_MAXFACTOR],
                                            marker.at[FLD.DRAWDOWN_RATIO],
                                            marker.at[FLD.DRAWDOWN_RATIO_MAJOR],
                                            int(marker.at[FLD.POLYNOMIAL9_TIMING_GAP] if not np.isnan(marker.at[FLD.POLYNOMIAL9_TIMING_GAP]) else 0),
                                            int(marker.at[FLD.MAPOWER30_TIMING_GAP] if not np.isnan(marker.at[FLD.MAPOWER30_TIMING_GAP]) else 0),
                                            int(marker.at[FLD.HMAPOWER120_TIMING_GAP] if not np.isnan(marker.at[FLD.HMAPOWER120_TIMING_GAP]) else 0),
                                            int(marker.at[FLD.POLYNOMIAL9_CHECKOUT_TIMING_LAG]),
                                            int(marker.at[FLD.MAXFACTOR_TREND_TIMING_LAG]),
                                            int(marker.at[FLD.LRC_HMA10_TIMING_LAG]),
                                            int(marker.at[FLD.MA30_HMA5_TIMING_LAG]),
                                            int(marker.at[FLD.BOLL_RAISED_TIMING_LAG]),
                                            int(marker.at[FLD.ZEN_PEAK_TIMING_LAG]),
                                            int(marker.at[FLD.ZEN_DASH_TIMING_LAG]),
                                            int(marker.at[FLD.ZEN_PEAK_TIMING_LAG_MAJOR]),
                                            int(marker.at[FLD.ZEN_DASH_TIMING_LAG_MAJOR]),
                                            int(marker.at[FLD.ZEN_BOOST_TIMING_LAG]),
                                            int(marker.at[FLD.ZEN_WAVELET_TIMING_LAG]),
                                            int(marker.at[FLD.ATR_SuperTrend_TIMING_LAG]),
                                            int(marker.at[FTR.BOOTSTRAP_ENHANCED_TIMING_LAG]),
                                            int(marker.at[FLD.MAINFEST_UPRISING_COEFFICIENT]),
                                            int(marker.at[FLD.BOLL_JX_RANK]),
                                            int(marker.at[FLD.BOOTSTRAP_GROUND_ZERO_RANK]),
                                            int(marker.at[FLD.ATR_Stopline_TIMING_LAG]),
                                            int(marker.at[FLD.BOOTSTRAP_COMBO_TIMING_LAG]),
                                            list(filter(None, [int(math.log2(codex_list[i]) + 1) if (int(marker.at[ST.FULLSTACK_COEFFICIENT]) & codex_list[i]) != 0 else None for i in range(31)])),
                                            marker.at[FLD.ZEN_DASH_RETURNS],
                                            list(filter(None, [int(math.log2(codex_list[i]) + 1) if (int(marker.at[ST.HALFSTACK_COEFFICIENT]) & codex_list[i]) != 0 else None for i in range(31)])),
                                            marker.at[FLD.ZEN_PEAK_RETURNS] + marker.at[FLD.ZEN_PEAK_RETURNS_MAJOR],)

        mkr_y = marker.at[FLD.POLYNOMIAL9]
        if (marker.at[FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG] < 0) or \
            (marker.at[ST.BOOTSTRAP_ENDPOINTS] > 0) or \
            (marker.at[ST.BOOTSTRAP_GROUND_ZERO] < 0) or \
            ((marker.at[FLD.ZEN_DASH_TIMING_LAG] < 0) and \
            (marker.at[FLD.ZEN_PEAK_TIMING_LAG] < 0)) or \
            ((marker.at[FLD.ZEN_DASH_TIMING_LAG_MAJOR] < 0) and \
            (marker.at[FLD.ZEN_DASH_TIMING_LAG] < 0)) or \
            ((marker.at[FLD.ZEN_PEAK_TIMING_LAG_MAJOR] < 0) and \
            (marker.at[FLD.ZEN_DASH_TIMING_LAG] < 0)):
            if (marker.at[FLD.POLYNOMIAL9] < marker.at[FLD.HMA10]):
                mkr_y = marker.at[FLD.ATR_UB] * 1.01
            ax1.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data', 
                         xytext=(20, 50), textcoords='offset points', 
                         fontsize=6, va="center", ha="center", zorder=4,
                         alpha=0.66,
                         bbox=dict(boxstyle="round4", fc="w", 
                                   color=colorwarp),
                         arrowprops=dict(arrowstyle="->", 
                                         connectionstyle="arc3,rad=.2", 
                                         color=colorwarp))
        elif (marker.at[FLD.BOLL_JX_RANK] < 0):
            mkr_text_template = """*{}*
ZEN:{:03d}\{:03d}\{:03d}\{:03d}
ZEN:{:03d}\{:03d}\POLY_CHK:{:03d}
RC:{}
FS:{}/RET:{:.02%}
HS:{}/RET:{:.02%}
EJECTED RNK:{:02d}\{:02d}"""
            mkr_text = mkr_text_template.format(mkr_x if (codex == 0) else '{} C:{:02x}'.format(mkr_x, int(math.log2(codex) + 1)),
                                                int(marker.at[FLD.ZEN_DASH_TIMING_LAG]),
                                                int(marker.at[FLD.ZEN_PEAK_TIMING_LAG]),
                                                int(marker.at[FLD.ZEN_DASH_TIMING_LAG_MAJOR]),
                                                int(marker.at[FLD.ZEN_PEAK_TIMING_LAG_MAJOR]),
                                                int(marker.at[FLD.ZEN_BOOST_TIMING_LAG]),
                                                int(marker.at[FLD.ZEN_WAVELET_TIMING_LAG]),
                                                int(marker.at[FLD.POLYNOMIAL9_CHECKOUT_TIMING_LAG]),
                                                list(filter(None, [int(math.log2(codex_list[i]) + 1) if (int(marker.at[FLD.BOLL_JX_RANK_REMARK]) & codex_list[i]) != 0 else None for i in range(31)])),
                                                list(filter(None, [int(math.log2(codex_list[i]) + 1) if (int(marker.at[ST.FULLSTACK_COEFFICIENT]) & codex_list[i]) != 0 else None for i in range(31)])),
                                                marker.at[FLD.ZEN_DASH_RETURNS],
                                                list(filter(None, [int(math.log2(codex_list[i]) + 1) if (int(marker.at[ST.HALFSTACK_COEFFICIENT]) & codex_list[i]) != 0 else None for i in range(31)])),
                                                marker.at[FLD.ZEN_PEAK_RETURNS] + marker.at[FLD.ZEN_PEAK_RETURNS_MAJOR],
                                                int(marker.at[FLD.BOLL_JX_RANK]),
                                                int(marker.at[FLD.BOOTSTRAP_GROUND_ZERO_RANK]),)
            ax1.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data', 
                         xytext=(20, -50), textcoords='offset points', 
                         fontsize=6, va="center", ha="center", zorder=4,
                         alpha=0.66,
                         bbox=dict(boxstyle="round4", fc="w", 
                                   color=colorwarp),
                         arrowprops=dict(arrowstyle="->", 
                                         connectionstyle="arc3,rad=.2", 
                                         color=colorwarp))
        else:
            if (marker.at[FLD.POLYNOMIAL9] > marker.at[FLD.HMA10]):
                mkr_y = marker.at[FLD.ATR_LB] * 0.99
            ax1.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data', 
                         xytext=(20, -50), textcoords='offset points', 
                         fontsize=6, va="center", ha="center", zorder=4,
                         alpha=0.66,
                         bbox=dict(boxstyle="round4", fc="w", 
                                   color=colorwarp),
                         arrowprops=dict(arrowstyle="->", 
                                         connectionstyle="arc3,rad=.2", 
                                         color=colorwarp))


def mark_annotate_roof(checkpoints, features, ax1, colorwarp='olive', codex=0):
    '''
    绘制标记
    '''
    markers = features.loc[checkpoints, [FLD.ZEN_PEAK_TIMING_LAG,
                                         FLD.ZEN_DASH_TIMING_LAG,
                                         FLD.ZEN_PEAK_TIMING_LAG_MAJOR,
                                         FLD.ZEN_DASH_TIMING_LAG_MAJOR,
                                         FLD.MAPOWER30_PEAK_HIGH_BEFORE,
                                         FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR,
                                         ST.BOOTSTRAP_ENDPOINTS,
                                        FLD.MAINFEST_UPRISING_COEFFICIENT,
                                        FTR.BOOTSTRAP_ENHANCED_TIMING_LAG,
                                        FLD.BOLL_JX_RANK_REMARK,
                                        ST.FULLSTACK_COEFFICIENT,
                                        ST.HALFSTACK_COEFFICIENT,
                                        FLD.ZEN_DASH_RETURNS,
                                        FLD.ZEN_PEAK_RETURNS,
                                        FLD.ZEN_PEAK_RETURNS_MAJOR,]].copy()

    #print(markers)
    codex_list = [int(math.pow(2, i)) for i in range(0, 31)]
    for index, marker in markers.iterrows():
        mkr_x = marker.name[0].strftime("%Y-%m-%d %H:%M")[2:16]
        mkr_text_template = """*{}*
BOLL:{:.3f}\MA12CH:{:.3f}
MA90CLR:{:03d}/{:03d}/DEA:{:03d}
POLY9:{:02d}\RT_DIV:{:.3f}
DIVERGENCE:{:.3f}/{:.3f}
RSI:{:.3f}/MFT:{:.3f}
DRAWDOWN:{:.3f}/{:.3f}
GAP:{:02d}\{:02d}\{:02d}\{:02d}\MFT:{:02d}
LRC:{:02d}\MA30:{:02d}:BOLL:{:02d}
ZEN:{:03d}\{:03d}\{:03d}\{:03d}\{:02d}\{:02d}
ATR:{:03d}\BST:{:03d}\COEF:{:03d}
RNK:{:02d}\{:02d}\ATR:{:03d}\CB:{:02d}
FS:{}/RET:{:.02%}
HS:{}/RET:{:.02%}"""
        mkr_text = mkr_text_template.format(mkr_x if (codex == 0) else '{} C:{:02x}'.format(mkr_x, int(math.log2(codex) + 1)),
                                            marker.at[FLD.BOLL_CHANNEL],
                                            marker.at[FLD.MA120_CHANNEL],
                                            int(marker.at[FLD.MA90_CLEARANCE_TIMING_LAG]),
                                            int(marker.at[FLD.MA90_TREND_TIMING_LAG]),
                                            int(marker.at[FLD.DEA_ZERO_TIMING_LAG]),
                                            int(marker.at[FLD.POLYNOMIAL9_TIMING_LAG]),
                                            max(min((marker.at[FLD.POLY9_REGTREE_DIVERGENCE]), 255), -255),
                                            marker.at[FLD.POLY9_MA30_DIVERGENCE],
                                            marker.at[FLD.POLY9_MA90_DIVERGENCE],
                                            marker.at[FLD.BOLL_JX_RSI],
                                            marker.at[FLD.BOLL_JX_MAXFACTOR],
                                            marker.at[FLD.DRAWDOWN_RATIO],
                                            marker.at[FLD.DRAWDOWN_RATIO_MAJOR],
                                            int(marker.at[FLD.POLYNOMIAL9_TIMING_GAP] if not np.isnan(marker.at[FLD.POLYNOMIAL9_TIMING_GAP]) else 0),
                                            int(marker.at[FLD.MAPOWER30_TIMING_GAP] if not np.isnan(marker.at[FLD.MAPOWER30_TIMING_GAP]) else 0),
                                            int(marker.at[FLD.HMAPOWER120_TIMING_GAP] if not np.isnan(marker.at[FLD.HMAPOWER120_TIMING_GAP]) else 0),
                                            int(marker.at[FLD.POLYNOMIAL9_CHECKOUT_TIMING_LAG]),
                                            int(marker.at[FLD.MAXFACTOR_TREND_TIMING_LAG]),
                                            int(marker.at[FLD.LRC_HMA10_TIMING_LAG]),
                                            int(marker.at[FLD.MA30_HMA5_TIMING_LAG]),
                                            int(marker.at[FLD.BOLL_RAISED_TIMING_LAG]),
                                            int(marker.at[FLD.ZEN_PEAK_TIMING_LAG]),
                                            int(marker.at[FLD.ZEN_DASH_TIMING_LAG]),
                                            int(marker.at[FLD.ZEN_PEAK_TIMING_LAG_MAJOR]),
                                            int(marker.at[FLD.ZEN_DASH_TIMING_LAG_MAJOR]),
                                            int(marker.at[FLD.ZEN_BOOST_TIMING_LAG]),
                                            int(marker.at[FLD.ZEN_WAVELET_TIMING_LAG]),
                                            int(marker.at[FLD.ATR_SuperTrend_TIMING_LAG]),
                                            int(marker.at[FTR.BOOTSTRAP_ENHANCED_TIMING_LAG]),
                                            int(marker.at[FLD.MAINFEST_UPRISING_COEFFICIENT]),
                                            int(marker.at[FLD.BOLL_JX_RANK]),
                                            int(marker.at[FLD.BOOTSTRAP_GROUND_ZERO_RANK]),
                                            int(marker.at[FLD.ATR_Stopline_TIMING_LAG]),
                                            int(marker.at[FLD.BOOTSTRAP_COMBO_TIMING_LAG]),
                                            list(filter(None, [int(math.log2(codex_list[i]) + 1) if (int(marker.at[ST.FULLSTACK_COEFFICIENT]) & codex_list[i]) != 0 else None for i in range(31)])),
                                            marker.at[FLD.ZEN_DASH_RETURNS],
                                            list(filter(None, [int(math.log2(codex_list[i]) + 1) if (int(marker.at[ST.HALFSTACK_COEFFICIENT]) & codex_list[i]) != 0 else None for i in range(31)])),
                                            marker.at[FLD.ZEN_PEAK_RETURNS] + marker.at[FLD.ZEN_PEAK_RETURNS_MAJOR],)

        mkr_y = marker.at[FLD.POLYNOMIAL9]
        if (marker.at[FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG] < 0) or \
            (marker.at[ST.BOOTSTRAP_ENDPOINTS] > 0) or \
            (marker.at[ST.BOOTSTRAP_GROUND_ZERO] < 0) or \
            ((marker.at[FLD.ZEN_DASH_TIMING_LAG] < 0) and \
            (marker.at[FLD.ZEN_PEAK_TIMING_LAG] < 0)) or \
            ((marker.at[FLD.ZEN_DASH_TIMING_LAG_MAJOR] < 0) and \
            (marker.at[FLD.ZEN_DASH_TIMING_LAG] < 0)) or \
            ((marker.at[FLD.ZEN_PEAK_TIMING_LAG_MAJOR] < 0) and \
            (marker.at[FLD.ZEN_DASH_TIMING_LAG] < 0)):
            if (marker.at[FLD.POLYNOMIAL9] < marker.at[FLD.HMA10]):
                mkr_y = marker.at[FLD.ATR_UB] * 1.01
            ax1.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data', 
                         xytext=(20, 50), textcoords='offset points', 
                         fontsize=6, va="center", ha="center", zorder=4,
                         alpha=0.66,
                         bbox=dict(boxstyle="round4", fc="w", 
                                   color=colorwarp),
                         arrowprops=dict(arrowstyle="->", 
                                         connectionstyle="arc3,rad=.2", 
                                         color=colorwarp))
        elif (marker.at[FLD.BOLL_JX_RANK] < 0):
            mkr_text_template = """*{}*
ZEN:{:03d}\{:03d}\{:03d}\{:03d}
ZEN:{:03d}\{:03d}\POLY_CHK:{:03d}
RC:{}
FS:{}/RET:{:.02%}
HS:{}/RET:{:.02%}
EJECTED RNK:{:02d}\{:02d}"""
            mkr_text = mkr_text_template.format(mkr_x if (codex == 0) else '{} C:{:02x}'.format(mkr_x, int(math.log2(codex) + 1)),
                                                int(marker.at[FLD.ZEN_DASH_TIMING_LAG]),
                                                int(marker.at[FLD.ZEN_PEAK_TIMING_LAG]),
                                                int(marker.at[FLD.ZEN_DASH_TIMING_LAG_MAJOR]),
                                                int(marker.at[FLD.ZEN_PEAK_TIMING_LAG_MAJOR]),
                                                int(marker.at[FLD.ZEN_BOOST_TIMING_LAG]),
                                                int(marker.at[FLD.ZEN_WAVELET_TIMING_LAG]),
                                                int(marker.at[FLD.POLYNOMIAL9_CHECKOUT_TIMING_LAG]),
                                                list(filter(None, [int(math.log2(codex_list[i]) + 1) if (int(marker.at[FLD.BOLL_JX_RANK_REMARK]) & codex_list[i]) != 0 else None for i in range(31)])),
                                                list(filter(None, [int(math.log2(codex_list[i]) + 1) if (int(marker.at[ST.FULLSTACK_COEFFICIENT]) & codex_list[i]) != 0 else None for i in range(31)])),
                                                marker.at[FLD.ZEN_DASH_RETURNS],
                                                list(filter(None, [int(math.log2(codex_list[i]) + 1) if (int(marker.at[ST.HALFSTACK_COEFFICIENT]) & codex_list[i]) != 0 else None for i in range(31)])),
                                                marker.at[FLD.ZEN_PEAK_RETURNS] + marker.at[FLD.ZEN_PEAK_RETURNS_MAJOR],
                                                int(marker.at[FLD.BOLL_JX_RANK]),
                                                int(marker.at[FLD.BOOTSTRAP_GROUND_ZERO_RANK]),)
            ax1.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data', 
                         xytext=(20, -50), textcoords='offset points', 
                         fontsize=6, va="center", ha="center", zorder=4,
                         alpha=0.66,
                         bbox=dict(boxstyle="round4", fc="w", 
                                   color=colorwarp),
                         arrowprops=dict(arrowstyle="->", 
                                         connectionstyle="arc3,rad=.2", 
                                         color=colorwarp))
        else:
            if (marker.at[FLD.POLYNOMIAL9] > marker.at[FLD.HMA10]):
                mkr_y = marker.at[FLD.ATR_LB] * 0.99
            ax1.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data', 
                         xytext=(20, -50), textcoords='offset points', 
                         fontsize=6, va="center", ha="center", zorder=4,
                         alpha=0.66,
                         bbox=dict(boxstyle="round4", fc="w", 
                                   color=colorwarp),
                         arrowprops=dict(arrowstyle="->", 
                                         connectionstyle="arc3,rad=.2", 
                                         color=colorwarp))


def mark_annotate(checkpoints, features, ax3, colorwarp='olive'):
    '''
    绘制标记
    '''
    if (ST.PREDICT_LONG in features.columns):
        markers = features.loc[checkpoints, [FLD.PEAK_LOW_TIMING_LAG,
                                             FLD.COMBINE_DENSITY,
                                             FLD.MAPOWER30_MAJOR,
                                             FLD.HMAPOWER120_MAJOR,
                                             FLD.MAPOWER30,
                                             FLD.HMAPOWER120,
                                             FLD.BOLL_LB,
                                             FLD.ATR_LB,
                                             FLD.MA90_CLEARANCE,
                                             FLD.MA120_CLEARANCE,
                                             FLD.BOLL_CHANNEL,
                                             FLD.MA_CHANNEL,
                                             FLD.MA120_CHANNEL,
                                             FTR.UPRISING_RAIL_TIMING_LAG,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR,
                                             FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                             FLD.DRAWDOWN_RATIO,
                                             FLD.DRAWDOWN_RATIO_MAJOR,
                                             FLD.DEADPOOL_CANDIDATE_TIMING_LAG,
                                             FLD.ZEN_WAVELET_TIMING_LAG,
                                             FLD.RENKO_TREND_S_TIMING_LAG,
                                             FLD.RENKO_TREND_L_TIMING_LAG,
                                             FLD.HMAPOWER30_MA_TIMING_LAG,
                                             FLD.MAPOWER_HMAPOWER120_TIMING_LAG,
                                             ST.PREDICT_LONG,
                                             FLD.PREDICT_PROB_LONG,
                                             FLD.BOLL_JX_RSI,
                                             FLD.BOLL_JX_MAXFACTOR,]].copy()
    else:
        markers = features.loc[checkpoints, [FLD.PEAK_LOW_TIMING_LAG,
                                             FLD.COMBINE_DENSITY,
                                             FLD.MAPOWER30_MAJOR,
                                             FLD.HMAPOWER120_MAJOR,
                                             FLD.MAPOWER30,
                                             FLD.HMAPOWER120,
                                             FLD.BOLL_LB,
                                             FLD.ATR_LB,
                                             FLD.MA90_CLEARANCE,
                                             FLD.MA120_CLEARANCE,
                                             FLD.BOLL_CHANNEL,
                                             FLD.MA_CHANNEL,
                                             FLD.MA120_CHANNEL,
                                             FTR.UPRISING_RAIL_TIMING_LAG,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR,
                                             FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                             FLD.DRAWDOWN_RATIO,
                                             FLD.DRAWDOWN_RATIO_MAJOR,
                                             FLD.DEADPOOL_CANDIDATE_TIMING_LAG,
                                             FLD.ZEN_WAVELET_TIMING_LAG,
                                             FLD.RENKO_TREND_S_TIMING_LAG,
                                             FLD.RENKO_TREND_L_TIMING_LAG,
                                             FLD.HMAPOWER30_MA_TIMING_LAG,
                                             FLD.MAPOWER_HMAPOWER120_TIMING_LAG,]].copy()

    #print(markers)
    for index, marker in markers.iterrows():
        mkr_x = marker.name[0].strftime("%Y-%m-%d %H:%M")[2:16]
        if (ST.PREDICT_LONG in features.columns) and (marker.at[ST.PREDICT_LONG] < 0.1):
            mkr_text_template = '''*{}*
xgboost_pred:{:02d}
pred_prob:{:.2f}
RSI:{:.3f}/MFT:{:.3f}
DRAWDOWN:{:.3f}/{:.3f}'''
            mkr_text = mkr_text_template.format(mkr_x,
                                                int(marker.at[ST.PREDICT_LONG]),
                                                marker.at[FLD.PREDICT_PROB_LONG],
                                                marker.at[FLD.BOLL_JX_RSI],
                                                marker.at[FLD.BOLL_JX_MAXFACTOR],
                                                marker.at[FLD.DRAWDOWN_RATIO],
                                                marker.at[FLD.DRAWDOWN_RATIO_MAJOR],)
        else:
            mkr_text_template = '''*{}*
COMB:{:.3f}/BOL:{:.3f}
MAPWR3:{:.2f}/HMA12:{:.2f}
MACH:{:.2f}/MA12CH:{:.2f}
MA9CLR:{:.2f}/{:.2f}/{:.2f}
PEAK_LO:{:03d}/{:03d}/{:03d}
PEAK_HI:{:03d}/{:03d}/{:03d}
DRAWDOWN:{:.3f}/{:.3f}
RENKO:{:03d}/{:03d}/ZEN:{:03d}
MAPWR_BSL_HMAPWR12:{:02d}'''
            mkr_text = mkr_text_template.format(mkr_x,
                                                marker.at[FLD.COMBINE_DENSITY],
                                                marker.at[FLD.BOLL_CHANNEL],
                                                marker.at[FLD.MAPOWER30],
                                                marker.at[FLD.HMAPOWER120],
                                                marker.at[FLD.MA_CHANNEL],
                                                marker.at[FLD.MA120_CHANNEL],
                                                marker.at[FLD.MA90_CLEARANCE],
                                                marker.at[FLD.MA120_CLEARANCE],
                                                marker.at[FLD.MAPOWER30_MAJOR] + marker.at[FLD.HMAPOWER120_MAJOR] + marker.at[FLD.MAPOWER30],
                                                int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE]),
                                                int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR]),
                                                int(marker.at[FLD.DEADPOOL_CANDIDATE_TIMING_LAG]),
                                                int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE]),
                                                int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR]),
                                                int(marker.at[FLD.MAPOWER30_TIMING_LAG_MAJOR]),
                                                marker.at[FLD.DRAWDOWN_RATIO],
                                                marker.at[FLD.DRAWDOWN_RATIO_MAJOR],
                                                int(marker.at[FLD.RENKO_TREND_S_TIMING_LAG]),
                                                int(marker.at[FLD.RENKO_TREND_L_TIMING_LAG]),
                                                int(marker.at[FLD.ZEN_WAVELET_TIMING_LAG]),
                                                int(marker.at[FLD.MAPOWER_HMAPOWER120_TIMING_LAG]),)
        mkr_y = min(features[FLD.CCI_NORM].min(), 0) + 0.0168
        ax3.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data', 
                     xytext=(20, -50), textcoords='offset points', 
                     fontsize=6, va="center", ha="center", zorder=4,
                     alpha=0.66,
                     bbox=dict(boxstyle="round4", fc="w", 
                               color=colorwarp),
                     arrowprops=dict(arrowstyle="->", 
                                     connectionstyle="arc3,rad=.2", 
                                     color=colorwarp))


def mark_annotate_punch(checkpoints, features, ax3, colorwarp='olive'):
    '''
    绘制标记
    '''
    if (ST.PREDICT_LONG in features.columns):
        markers = features.loc[checkpoints, [FLD.PEAK_LOW_TIMING_LAG,
                                             FLD.COMBINE_DENSITY,
                                             FLD.MAPOWER30_MAJOR,
                                             FLD.HMAPOWER120_MAJOR,
                                             FLD.MAPOWER30,
                                             FLD.HMAPOWER120,
                                             FLD.BOLL_LB,
                                             FLD.ATR_LB,
                                             FLD.MA90_CLEARANCE,
                                             FLD.MA120_CLEARANCE,
                                             FLD.BOLL_CHANNEL,
                                             FLD.MA_CHANNEL,
                                             FLD.MA120_CHANNEL,
                                             FTR.UPRISING_RAIL_TIMING_LAG,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR,
                                             FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                             FLD.DRAWDOWN_RATIO,
                                             FLD.DRAWDOWN_RATIO_MAJOR,
                                             FLD.DEADPOOL_CANDIDATE_TIMING_LAG,
                                             FLD.ZEN_WAVELET_TIMING_LAG,
                                             FLD.RENKO_TREND_S_TIMING_LAG,
                                             FLD.RENKO_TREND_L_TIMING_LAG,
                                             FLD.HMAPOWER30_MA_TIMING_LAG,
                                             FLD.MAPOWER_HMAPOWER120_TIMING_LAG,
                                             FLD.BOLL_JX_RSI,
                                             FLD.BOLL_JX_MAXFACTOR,
                                             FLD.BOLL_JX_MAPOWER30,
                                             FLD.BOLL_JX_HMAPOWER120,
                                             FLD.DEADPOOL_REMIX_TIMING_LAG,
                                             FLD.MAINFEST_UPRISING_COEFFICIENT,
                                             FLD.DRAWDOWN_CHANNEL,
                                             FLD.DRAWDOWN_CHANNEL_MAJOR,
                                             FLD.HMA10_CLEARANCE,
                                             FLD.HMA10_CLEARANCE_ZSCORE,
                                             ST.PREDICT_LONG,
                                             FLD.PREDICT_PROB_LONG,]].copy()
    else:
        markers = features.loc[checkpoints, [FLD.PEAK_LOW_TIMING_LAG,
                                             FLD.COMBINE_DENSITY,
                                             FLD.MAPOWER30_MAJOR,
                                             FLD.HMAPOWER120_MAJOR,
                                             FLD.MAPOWER30,
                                             FLD.HMAPOWER120,
                                             FLD.BOLL_LB,
                                             FLD.ATR_LB,
                                             FLD.MA90_CLEARANCE,
                                             FLD.MA120_CLEARANCE,
                                             FLD.BOLL_CHANNEL,
                                             FLD.MA_CHANNEL,
                                             FLD.MA120_CHANNEL,
                                             FTR.UPRISING_RAIL_TIMING_LAG,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR,
                                             FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                             FLD.DRAWDOWN_RATIO,
                                             FLD.DRAWDOWN_RATIO_MAJOR,
                                             FLD.DEADPOOL_CANDIDATE_TIMING_LAG,
                                             FLD.ZEN_WAVELET_TIMING_LAG,
                                             FLD.RENKO_TREND_S_TIMING_LAG,
                                             FLD.RENKO_TREND_L_TIMING_LAG,
                                             FLD.HMAPOWER30_MA_TIMING_LAG,
                                             FLD.MAPOWER_HMAPOWER120_TIMING_LAG,
                                             FLD.BOLL_JX_RSI,
                                             FLD.BOLL_JX_MAXFACTOR,
                                             FLD.BOLL_JX_MAPOWER30,
                                             FLD.BOLL_JX_HMAPOWER120,
                                             FLD.DEADPOOL_REMIX_TIMING_LAG,
                                             FLD.MAINFEST_UPRISING_COEFFICIENT,
                                             FLD.DRAWDOWN_CHANNEL,
                                             FLD.DRAWDOWN_CHANNEL_MAJOR,
                                             FLD.HMA10_CLEARANCE,
                                             FLD.HMA10_CLEARANCE_ZSCORE,]].copy()

    #print(markers)
    for index, marker in markers.iterrows():
        mkr_x = marker.name[0].strftime("%Y-%m-%d %H:%M")[2:16]
        if (ST.PREDICT_LONG in features.columns) and (marker.at[ST.PREDICT_LONG] < 0.1):
            mkr_text_template = '*{}*\nxgboost_pred:{:02d}\npred_prob:{:.2f}\nRSI:{:.3f}/MFT:{:.3f}\nDRAWDOWN:{:.3f}/{:.3f}'
            mkr_text = mkr_text_template.format(mkr_x,
                                                int(marker.at[ST.PREDICT_LONG]),
                                                marker.at[FLD.PREDICT_PROB_LONG],
                                                marker.at[FLD.BOLL_JX_RSI],
                                                marker.at[FLD.BOLL_JX_MAXFACTOR],
                                                marker.at[FLD.DRAWDOWN_RATIO],
                                                marker.at[FLD.DRAWDOWN_RATIO_MAJOR],)
        else:
            mkr_text_template = '*{}*\nCOMB:{:.3f}/BOL:{:.3f}\nMACH:{:.2f}/MA12CH:{:.2f}\nDRAWDOWN:{:.3f}/{:.3f}\nRENKO:{:03d}/{:03d}/ZEN:{:03d}\nMAPWR_BSL_HMAPWR12:{:02d}\nRSI:{:.3f}/MFT:{:.3f}\nBJMPWR:{:.3f}/HMA12:{:.3f}\nDead:{:03d}/COEF:{:03d}\nDRDN_CHN:{:.3f}/{:.3f}\nHMA_LRC:{:.3f}/{:.3f}'
            mkr_text = mkr_text_template.format(mkr_x,
                                                marker.at[FLD.COMBINE_DENSITY],
                                                marker.at[FLD.BOLL_CHANNEL],
                                                marker.at[FLD.MA_CHANNEL],
                                                marker.at[FLD.MA120_CHANNEL],
                                                marker.at[FLD.DRAWDOWN_RATIO],
                                                marker.at[FLD.DRAWDOWN_RATIO_MAJOR],
                                                int(marker.at[FLD.RENKO_TREND_S_TIMING_LAG]),
                                                int(marker.at[FLD.RENKO_TREND_L_TIMING_LAG]),
                                                int(marker.at[FLD.ZEN_WAVELET_TIMING_LAG]),
                                                int(marker.at[FLD.MAPOWER_HMAPOWER120_TIMING_LAG]),
                                                marker.at[FLD.BOLL_JX_RSI],
                                                marker.at[FLD.BOLL_JX_MAXFACTOR],
                                                marker.at[FLD.BOLL_JX_MAPOWER30],
                                                marker.at[FLD.BOLL_JX_HMAPOWER120],
                                                int(marker.at[FLD.DEADPOOL_REMIX_TIMING_LAG]),
                                                int(marker.at[FLD.MAINFEST_UPRISING_COEFFICIENT]),
                                                marker.at[FLD.DRAWDOWN_CHANNEL],
                                                marker.at[FLD.DRAWDOWN_CHANNEL_MAJOR],
                                                marker.at[FLD.HMA10_CLEARANCE],
                                                marker.at[FLD.HMA10_CLEARANCE_ZSCORE],)
        mkr_y = min(features[FLD.CCI_NORM].min(), 0) - 0.0168
        ax3.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data', 
                     xytext=(20, -50), textcoords='offset points', 
                     fontsize=6, va="center", ha="center", zorder=4,
                     alpha=0.66,
                     bbox=dict(boxstyle="round4", fc="w", 
                               color=colorwarp),
                     arrowprops=dict(arrowstyle="->", 
                                     connectionstyle="arc3,rad=.2", 
                                     color=colorwarp))


def mark_annotate_mainfest(checkpoints, features, ax3, colorwarp='olive'):
    '''
    绘制标记
    '''
    if (ST.PREDICT_LONG in features.columns):
        markers = features.loc[checkpoints, [FLD.PEAK_LOW_TIMING_LAG,
                                             FLD.COMBINE_DENSITY,
                                             FLD.MAPOWER30_MAJOR,
                                             FLD.HMAPOWER120_MAJOR,
                                             FLD.MAPOWER30,
                                             FLD.HMAPOWER120,
                                             FLD.BOLL_LB,
                                             FLD.ATR_LB,
                                             FLD.MA90_CLEARANCE,
                                             FLD.MA120_CLEARANCE,
                                             FLD.BOLL_CHANNEL,
                                             FLD.MA_CHANNEL,
                                             FLD.MA120_CHANNEL,
                                             FTR.UPRISING_RAIL_TIMING_LAG,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR,
                                             FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                             FLD.DRAWDOWN_RATIO,
                                             FLD.DRAWDOWN_RATIO_MAJOR,
                                             FLD.DEADPOOL_CANDIDATE_TIMING_LAG,
                                             FLD.ZEN_WAVELET_TIMING_LAG,
                                             FLD.RENKO_TREND_S_TIMING_LAG,
                                             FLD.RENKO_TREND_L_TIMING_LAG,
                                             FLD.HMAPOWER30_MA_TIMING_LAG,
                                             FLD.MAPOWER_HMAPOWER120_TIMING_LAG,
                                             FLD.BOLL_JX_RSI,
                                             FLD.BOLL_JX_MAXFACTOR,
                                             FLD.BOLL_JX_MAPOWER30,
                                             FLD.BOLL_JX_HMAPOWER120,
                                             FLD.DEADPOOL_REMIX_TIMING_LAG,
                                             FLD.MAINFEST_UPRISING_COEFFICIENT,
                                             FLD.DRAWDOWN_CHANNEL,
                                             FLD.DRAWDOWN_CHANNEL_MAJOR,
                                             FLD.HMA10_CLEARANCE,
                                             FLD.HMA10_CLEARANCE_ZSCORE,
                                             FLD.MA90_CLEARANCE_TIMING_LAG,
                                             FLD.MA90_TREND_TIMING_LAG,
                                             FLD.MAPOWER_CROSS_TIMING_LAG,
                                             FLD.POLYNOMIAL9_DELTA,
                                             FLD.BOLL_DIFF,
                                             FLD.MA30_DIFF,
                                             FLD.PREDICT_GROWTH,
                                             ST.PREDICT_LONG,
                                             FLD.PREDICT_PROB_LONG,
                                             FLD.POLY9_MA30_DIVERGENCE,
                                             FLD.POLY9_MA90_DIVERGENCE,]].copy()
    else:
        markers = features.loc[checkpoints, [FLD.PEAK_LOW_TIMING_LAG,
                                             FLD.COMBINE_DENSITY,
                                             FLD.MAPOWER30_MAJOR,
                                             FLD.HMAPOWER120_MAJOR,
                                             FLD.MAPOWER30,
                                             FLD.HMAPOWER120,
                                             FLD.BOLL_LB,
                                             FLD.ATR_LB,
                                             FLD.MA90_CLEARANCE,
                                             FLD.MA120_CLEARANCE,
                                             FLD.BOLL_CHANNEL,
                                             FLD.MA_CHANNEL,
                                             FLD.MA120_CHANNEL,
                                             FTR.UPRISING_RAIL_TIMING_LAG,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR,
                                             FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                             FLD.DRAWDOWN_RATIO,
                                             FLD.DRAWDOWN_RATIO_MAJOR,
                                             FLD.DEADPOOL_CANDIDATE_TIMING_LAG,
                                             FLD.ZEN_WAVELET_TIMING_LAG,
                                             FLD.RENKO_TREND_S_TIMING_LAG,
                                             FLD.RENKO_TREND_L_TIMING_LAG,
                                             FLD.HMAPOWER30_MA_TIMING_LAG,
                                             FLD.MAPOWER_HMAPOWER120_TIMING_LAG,
                                             FLD.BOLL_JX_RSI,
                                             FLD.BOLL_JX_MAXFACTOR,
                                             FLD.BOLL_JX_MAPOWER30,
                                             FLD.BOLL_JX_HMAPOWER120,
                                             FLD.DEADPOOL_REMIX_TIMING_LAG,
                                             FLD.MAINFEST_UPRISING_COEFFICIENT,
                                             FLD.DRAWDOWN_CHANNEL,
                                             FLD.DRAWDOWN_CHANNEL_MAJOR,
                                             FLD.HMA10_CLEARANCE,
                                             FLD.HMA10_CLEARANCE_ZSCORE,
                                             FLD.MA90_CLEARANCE_TIMING_LAG,
                                             FLD.MA90_TREND_TIMING_LAG,
                                             FLD.MAPOWER_CROSS_TIMING_LAG,
                                             FLD.POLYNOMIAL9_DELTA,
                                             FLD.BOLL_DIFF,
                                             FLD.MA30_DIFF,
                                             FLD.PREDICT_GROWTH,
                                             FLD.POLY9_MA30_DIVERGENCE,
                                             FLD.POLY9_MA90_DIVERGENCE]].copy()

    #print(markers)
    for index, marker in markers.iterrows():
        mkr_x = marker.name[0].strftime("%Y-%m-%d %H:%M")[2:16]
        if (ST.PREDICT_LONG in features.columns) and (marker.at[ST.PREDICT_LONG] < 0.1):
            mkr_text_template = '*{}*\nxgboost_pred:{:02d}\npred_prob:{:.2f}\nRSI:{:.3f}/MFT:{:.3f}\nDRAWDOWN:{:.3f}/{:.3f}\nDIVERGENCE:{:.3f}/{:.3f}'
            mkr_text = mkr_text_template.format(mkr_x,
                                                int(marker.at[ST.PREDICT_LONG]),
                                                marker.at[FLD.PREDICT_PROB_LONG],
                                                marker.at[FLD.BOLL_JX_RSI],
                                                marker.at[FLD.BOLL_JX_MAXFACTOR],
                                                marker.at[FLD.DRAWDOWN_RATIO],
                                                marker.at[FLD.DRAWDOWN_RATIO_MAJOR],
                                                marker.at[FLD.POLY9_MA30_DIVERGENCE],
                                                marker.at[FLD.POLY9_MA90_DIVERGENCE],)
        else:
            mkr_text_template = '*{}*\nCOMB:{:.3f}/BOL:{:.3f}\nMACH:{:.2f}/MA12CH:{:.2f}\nMA9CLR:{:.2f}/{:.2f}/{:.2f}\nPEAK_LO:{:03d}/{:03d}/{:03d}\nPEAK_HI:{:03d}/{:03d}/{:03d}\nDRAWDOWN:{:.3f}/{:.3f}\nRENKO:{:03d}/{:03d}/ZEN:{:03d}\nMAPWR_BSL_HMAPWR12:{:02d}\nRSI:{:.3f}/MFT:{:.3f}\nBJMPWR:{:.3f}/HMA12:{:.3f}\nDead:{:03d}/COEF:{:03d}/MAT:{:03d}\nMA90CLR:{:03d}/TRD:{:03d}\nDRDN_CHN:{:.3f}/{:.3f}\nHMA_LRC:{:.3f}/{:.3f}\nDIFF_POLY9:{:.3f}/GROW:{:.3f}\nMA20:{:.3f}/30:{:.3f}'
            mkr_text = mkr_text_template.format(mkr_x,
                                                marker.at[FLD.COMBINE_DENSITY],
                                                marker.at[FLD.BOLL_CHANNEL],
                                                marker.at[FLD.MA_CHANNEL],
                                                marker.at[FLD.MA120_CHANNEL],
                                                marker.at[FLD.MA90_CLEARANCE],
                                                marker.at[FLD.MA120_CLEARANCE],
                                                marker.at[FLD.MAPOWER30_MAJOR] + marker.at[FLD.HMAPOWER120_MAJOR] + marker.at[FLD.MAPOWER30],
                                                int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE]),
                                                int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR]),
                                                int(marker.at[FLD.DEADPOOL_CANDIDATE_TIMING_LAG]),
                                                int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE]),
                                                int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR]),
                                                int(marker.at[FLD.MAPOWER30_TIMING_LAG_MAJOR]),
                                                marker.at[FLD.DRAWDOWN_RATIO],
                                                marker.at[FLD.DRAWDOWN_RATIO_MAJOR],
                                                int(marker.at[FLD.RENKO_TREND_S_TIMING_LAG]),
                                                int(marker.at[FLD.RENKO_TREND_L_TIMING_LAG]),
                                                int(marker.at[FLD.ZEN_WAVELET_TIMING_LAG]),
                                                int(marker.at[FLD.MAPOWER_HMAPOWER120_TIMING_LAG]),
                                                marker.at[FLD.BOLL_JX_RSI],
                                                marker.at[FLD.BOLL_JX_MAXFACTOR],
                                                marker.at[FLD.BOLL_JX_MAPOWER30],
                                                marker.at[FLD.BOLL_JX_HMAPOWER120],
                                                int(marker.at[FLD.DEADPOOL_REMIX_TIMING_LAG]),
                                                int(marker.at[FLD.MAINFEST_UPRISING_COEFFICIENT]),
                                                int(marker.at[FLD.MAPOWER_CROSS_TIMING_LAG]),
                                                int(marker.at[FLD.MA90_CLEARANCE_TIMING_LAG]),
                                                int(marker.at[FLD.MA90_TREND_TIMING_LAG]),
                                                marker.at[FLD.DRAWDOWN_CHANNEL],
                                                marker.at[FLD.DRAWDOWN_CHANNEL_MAJOR],
                                                marker.at[FLD.HMA10_CLEARANCE],
                                                marker.at[FLD.HMA10_CLEARANCE_ZSCORE],
                                                marker.at[FLD.POLYNOMIAL9_DELTA],
                                                marker.at[FLD.PREDICT_GROWTH],
                                                marker.at[FLD.BOLL_DIFF],
                                                marker.at[FLD.MA30_DIFF],)
        mkr_y = min(features[FLD.CCI_NORM].min(), 0) - 0.0382
        ax3.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data', 
                     xytext=(20, -50), textcoords='offset points', 
                     fontsize=6, va="top", ha="center", zorder=4,
                     alpha=0.66,
                     bbox=dict(boxstyle="round4", fc="w", 
                               color=colorwarp),
                     arrowprops=dict(arrowstyle="->", 
                                     connectionstyle="arc3,rad=.2", 
                                     color=colorwarp))


def mark_annotate_subplot(checkpoints, features, ax2, 
                          colorwarp='violet', marker=None):
    '''
    绘制标记
    '''
    if (ST.PREDICT_LONG in features.columns):
        markers = features.loc[checkpoints, [FLD.PEAK_LOW_TIMING_LAG,
                                             FLD.COMBINE_DENSITY,
                                             FLD.MAPOWER30_MAJOR,
                                             FLD.HMAPOWER120_MAJOR,
                                             FLD.MAPOWER30,
                                             FLD.HMAPOWER120,
                                             FLD.BOLL_LB,
                                             FLD.ATR_LB,
                                             FLD.DIF,
                                             FLD.DEA,
                                             FLD.MA90_CLEARANCE,
                                             FLD.MA120_CLEARANCE,
                                             FLD.BOLL_CHANNEL,
                                             FLD.MA_CHANNEL,
                                             FLD.MA120_CHANNEL,
                                             FTR.UPRISING_RAIL_TIMING_LAG,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR,
                                             FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                             FLD.DRAWDOWN_RATIO,
                                             FLD.DRAWDOWN_RATIO_MAJOR,
                                             FLD.DEADPOOL_CANDIDATE_TIMING_LAG,
                                             FLD.ZEN_WAVELET_TIMING_LAG,
                                             FLD.RENKO_TREND_S_TIMING_LAG,
                                             FLD.RENKO_TREND_L_TIMING_LAG,
                                             FLD.MAPOWER_HMAPOWER120_TIMING_LAG,
                                             FLD.BOLL_SX_DRAWDOWN,
                                             ST.PREDICT_LONG,
                                             FLD.PREDICT_PROB_LONG,
                                             FLD.BOLL_JX_RSI,
                                             FLD.BOLL_JX_MAXFACTOR,]].copy()
    else:
        markers = features.loc[checkpoints, [FLD.PEAK_LOW_TIMING_LAG,
                                             FLD.COMBINE_DENSITY,
                                             FLD.MAPOWER30_MAJOR,
                                             FLD.HMAPOWER120_MAJOR,
                                             FLD.MAPOWER30,
                                             FLD.HMAPOWER120,
                                             FLD.BOLL_LB,
                                             FLD.ATR_LB,
                                             FLD.DIF,
                                             FLD.DEA,
                                             FLD.MA90_CLEARANCE,
                                             FLD.MA120_CLEARANCE,
                                             FLD.BOLL_CHANNEL,
                                             FLD.MA_CHANNEL,
                                             FLD.MA120_CHANNEL,
                                             FTR.UPRISING_RAIL_TIMING_LAG,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE,
                                             FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR,
                                             FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR,
                                             FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                             FLD.DRAWDOWN_RATIO,
                                             FLD.DRAWDOWN_RATIO_MAJOR,
                                             FLD.DEADPOOL_CANDIDATE_TIMING_LAG,
                                             FLD.ZEN_WAVELET_TIMING_LAG,
                                             FLD.RENKO_TREND_S_TIMING_LAG,
                                             FLD.RENKO_TREND_L_TIMING_LAG,
                                             FLD.MAPOWER_HMAPOWER120_TIMING_LAG,
                                             FLD.BOLL_SX_DRAWDOWN,]].copy()

    #print(markers)
    for index, marker in markers.iterrows():
        mkr_x = marker.name[0].strftime("%Y-%m-%d %H:%M")[2:16]
        if (ST.PREDICT_LONG in features.columns) and (marker.at[ST.PREDICT_LONG] < 0.1):
            mkr_text_template = '*{}*\nxgboost_pred:{:02d}\npred_prob:{:.2f}\nRSI:{:.3f}/MFT:{:.3f}\nDRAWDOWN:{:.3f}/{:.3f}'
            mkr_text = mkr_text_template.format(mkr_x,
                                                int(marker.at[ST.PREDICT_LONG]),
                                                marker.at[FLD.PREDICT_PROB_LONG],
                                                marker.at[FLD.BOLL_JX_RSI],
                                                marker.at[FLD.BOLL_JX_MAXFACTOR],
                                                marker.at[FLD.DRAWDOWN_RATIO],
                                                marker.at[FLD.DRAWDOWN_RATIO_MAJOR],)
        else:
            mkr_text_template = '*{}*\nCOMB:{:.3f}/BOL:{:.3f}\nMAPWR3:{:.2f}/HMA12:{:.2f}\nMACH:{:.2f}/MA12CH:{:.2f}\nMA9CLR:{:.2f}/{:.2f}/{:.2f}\nPEAK_LO:{:03d}/{:03d}/{:03d}\nPEAK_HI:{:03d}/{:03d}/{:03d}\nDRAWDOWN:{:.3f}/{:.3f}/{:.3f}\nRENKO:{:03d}/{:03d}/ZEN:{:03d}\nMAPWR_BSL_HMAPWR12:{:02d}'
            mkr_text = mkr_text_template.format(mkr_x,
                                                marker.at[FLD.COMBINE_DENSITY],
                                                marker.at[FLD.BOLL_CHANNEL],
                                                marker.at[FLD.MAPOWER30],
                                                marker.at[FLD.HMAPOWER120],
                                                marker.at[FLD.MA_CHANNEL],
                                                marker.at[FLD.MA120_CHANNEL],
                                                marker.at[FLD.MA90_CLEARANCE],
                                                marker.at[FLD.MA120_CLEARANCE],
                                                marker.at[FLD.MAPOWER30_MAJOR] + marker.at[FLD.HMAPOWER120_MAJOR] + marker.at[FLD.MAPOWER30],
                                                int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE]),
                                                int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR]),
                                                int(marker.at[FLD.DEADPOOL_CANDIDATE_TIMING_LAG]),
                                                int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE]),
                                                int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR]),
                                                int(marker.at[FLD.MAPOWER30_TIMING_LAG_MAJOR]),
                                                marker.at[FLD.DRAWDOWN_RATIO],
                                                marker.at[FLD.DRAWDOWN_RATIO_MAJOR],
                                                marker.at[FLD.BOLL_SX_DRAWDOWN],
                                                int(marker.at[FLD.RENKO_TREND_S_TIMING_LAG]),
                                                int(marker.at[FLD.RENKO_TREND_L_TIMING_LAG]),
                                                int(marker.at[FLD.ZEN_WAVELET_TIMING_LAG]),
                                                int(marker.at[FLD.MAPOWER_HMAPOWER120_TIMING_LAG]),)

        mkr_y = min(marker.at[FLD.DEA], marker.at[FLD.DIF]) * 1.10
        ax2.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data', 
                     xytext=(20, -60), textcoords='offset points', 
                     fontsize=6, va="top", ha="center", zorder=4,
                     alpha=0.66,
                     bbox=dict(boxstyle="round4", fc="w", 
                               color=colorwarp),
                     arrowprops=dict(arrowstyle="->", 
                                     connectionstyle="arc3,rad=.2", 
                                     color=colorwarp))


def back_plot():
    pass
    #if (FLD.COMBINE_DENSITY in features.columns):
    #    ax3.plot(DATETIME_LABEL, 
    #             features[FLD.COMBINE_DENSITY], 
    #             lw=0.75, color ='lightskyblue', alpha=0.33)
    #    ax3.plot(DATETIME_LABEL,
    #             np.where(features[FLD.ATR_SuperTrend_TIMING_LAG] > 0,
    #                      features[FLD.COMBINE_DENSITY], np.nan),
    #             lw=0.75, color ='crimson', alpha=0.33)
    #if (FLD.CCI in features.columns):
    #    ax3.plot(DATETIME_LABEL,
    #     features[FLD.RSI] / 100,
    #     lw=0.75, color ='green', alpha=0.33)
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
    #if (FLD.MAPOWER30 in features.columns):
    #    ax3.plot(DATETIME_LABEL, 
    #             np.where((features[FLD.MAPOWER30_TIMING_LAG] < 1),
    #             features[FLD.MAPOWER30], np.nan), 
    #             lw=0.75, color ='lime', alpha=0.8)
    #    ax3.plot(DATETIME_LABEL,
    #             np.where((features[FLD.MAPOWER30_TIMING_LAG] > -1),
    #                      features[FLD.MAPOWER30], np.nan),
    #             lw=0.75, color ='gray', alpha=0.5)
    #if (FLD.MAPOWER120 in features.columns):
    #    ax3.plot(DATETIME_LABEL, 
    #             np.where(features[FLD.MAPOWER120_TIMING_LAG] < 1, 
    #                      features[FLD.MAPOWER120], np.nan),
    #             lw=0.75, color ='olive', alpha=0.8)
    #    ax3.plot(DATETIME_LABEL, 
    #             np.where(features[FLD.MAPOWER120_TIMING_LAG] > -1, 
    #                      features[FLD.MAPOWER120], np.nan),
    #             lw=0.75, color ='orange', alpha=0.8)

    #if (FLD.RENKO_BOOST_L_TIMING_LAG in features.columns):
    #    ax3.plot(DATETIME_LABEL, 
    #             np.where((features[FLD.RENKO_BOOST_L_TIMING_LAG] > 0),
    #                      features[FLD.CCI_NORM].min() - 0.0168, np.nan),
    #             lw=1, color ='red', alpha=0.8)
    #    ax3.plot(DATETIME_LABEL, 
    #             np.where((features[FLD.RENKO_TREND_L_TIMING_LAG] > 0),
    #                     features[FLD.CCI_NORM].min() - 0.0168, np.nan),
    #             lw=1, color ='coral', alpha=0.5)

    #if (FLD.HMAPOWER120 in features.columns):
    #    ax3.plot(DATETIME_LABEL,
    #             np.where((features[FLD.HMAPOWER30_MA_TIMING_LAG] > 0),
    #                      features[FLD.CCI_NORM].min() - 0.00512, np.nan),
    #             lw=1.75, color ='orange', alpha=0.8)
    #    ax3.plot(DATETIME_LABEL,
    #             features[FLD.HMAPOWER120],
    #             lw=0.75, color ='gray', alpha=0.22)
    #    ax3.plot(DATETIME_LABEL,
    #             np.where(features[FLD.HMAPOWER120_TIMING_LAG] <= 1, 
    #                      features[FLD.HMAPOWER120], np.nan,),
    #             lw=0.75, color ='darkgray', alpha=0.33)
    #    ax3.plot(DATETIME_LABEL,
    #             np.where(features[FLD.HMAPOWER120_TIMING_LAG] >= -1, 
    #                      features[FLD.HMAPOWER120], np.nan,),
    #             lw=1.25, color ='magenta', alpha=0.66)
        
    #if (FLD.ZEN_BOOST_TIMING_LAG in features.columns) and \
    #    (FLD.MAPOWER120 in features.columns):
    #    ax1.plot(DATETIME_LABEL,
    #             np.where((features[FLD.ZEN_BOOST_TIMING_LAG] == 1),
    #                      (features[FLD.BOLL_LB] + features[FLD.ATR_LB]) / 2.08, np.nan),
    #             'r.', alpha = 0.33)
    #    ax1.plot(DATETIME_LABEL,
    #             np.where(((features[FLD.HMAPOWER120_TIMING_LAG] == 1) | \
    #                      (features[FLD.ZEN_BOOST_TIMING_LAG] == 1)) & \
    #                      (features[ST.CLUSTER_GROUP_TOWARDS] > 0),
    #                      (features[FLD.BOLL_LB] + features[FLD.ATR_LB]) / 2.11, np.nan),
    #             'k+', alpha = 0.25)
    #    ax1.plot(DATETIME_LABEL,
    #             np.where((features[FLD.ZEN_BOOST_TIMING_LAG] == 1) & \
    #                      (features[ST.CLUSTER_GROUP_TOWARDS] > 0) & \
    #                      (features[FLD.PEAK_LOW_TIMING_LAG] <= 5) & \
    #                      (features[FLD.DIF].shift(2) < features[FLD.DIF]) & \
    #                      (features[FLD.MACD].shift(2) < features[FLD.MACD]) & \
    #                      (features[AKA.CLOSE] > features[FLD.PEAK_ANCHOR_PREV_LOW_REF6]) & \
    #                      (((features[FLD.HMAPOWER30_MA_TIMING_GAP] > 0) & \
    #                      ((features[FLD.HMAPOWER30_MA_TIMING_GAP] > 36) | \
    #                      (features[FLD.HMAPOWER30_MA_TIMING_GAP] > features[FLD.DEA_ZERO_TIMING_LAG]) | \
    #                      (features[FLD.HMAPOWER30_MA_TIMING_GAP] > features[FLD.DIF_ZERO_TIMING_LAG]))) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                      (features[FLD.BOLL_JX_MAXFACTOR] < -200)) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 46.6) & \
    #                      ((features[FLD.BOLL_JX_MAXFACTOR] < -150) | \
    #                      ((features[FLD.BOLL_JX_MAXFACTOR] < 0) & \
    #                      (features[FLD.DEA_ZERO_TIMING_LAG] < 0) & \
    #                      (features[FLD.DIF_ZERO_TIMING_LAG] < 0) & \
    #                      (features[FLD.PEAK_LOW_TIMING_LAG] == 1) & \
    #                      (features[FLD.MACD_ZERO_TIMING_LAG] == 1) & \
    #                      (features[FLD.BOLL_JX_MAPOWER120] < 0.1954))) & \
    #                      (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                      (features[FLD.BOLL_JX_MAXFACTOR] < -50) & \
    #                      (features[FLD.BOLL_JX_MAPOWER120] < 0.168) & \
    #                      (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 45) & \
    #                      (features[FLD.BOLL_JX_MAXFACTOR] < -200) & \
    #                      (features[FLD.BOLL_JX_HMAPOWER120] < 0.0927))),
    #                      (features[FLD.BOLL_LB] + features[FLD.ATR_LB]) / 2.095, np.nan),
    #             'mP', alpha = 0.66)
    #    ax1.plot(DATETIME_LABEL,
    #             np.where((features[FLD.ZEN_BOOST_TIMING_LAG] == 1) & \
    #                      (features[ST.CLUSTER_GROUP_TOWARDS] > 0) & \
    #                      (features[FLD.PEAK_LOW_TIMING_LAG] <= 5) & \
    #                      (features[FLD.DIF].shift(2) > features[FLD.DIF]) & \
    #                      (features[FLD.MACD].shift(2) > features[FLD.MACD]) & \
    #                      (features[AKA.CLOSE] > features[FLD.PEAK_ANCHOR_PREV_LOW_REF6]) & \
    #                      (((features[FLD.HMAPOWER30_MA_TIMING_GAP] > 0) & \
    #                      ((features[FLD.HMAPOWER30_MA_TIMING_GAP] > 36) | \
    #                      (features[FLD.HMAPOWER30_MA_TIMING_GAP] > features[FLD.DEA_ZERO_TIMING_LAG]) | \
    #                      (features[FLD.HMAPOWER30_MA_TIMING_GAP] > features[FLD.DIF_ZERO_TIMING_LAG]))) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                      (features[FLD.BOLL_JX_MAXFACTOR] < -200)) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 46.6) & \
    #                      ((features[FLD.BOLL_JX_MAXFACTOR] < -150) | \
    #                      ((features[FLD.BOLL_JX_MAXFACTOR] < 0) & \
    #                      (features[FLD.DEA_ZERO_TIMING_LAG] < 0) & \
    #                      (features[FLD.DIF_ZERO_TIMING_LAG] < 0) & \
    #                      (features[FLD.PEAK_LOW_TIMING_LAG] == 1) & \
    #                      (features[FLD.MACD_ZERO_TIMING_LAG] == 1) & \
    #                      (features[FLD.BOLL_JX_MAPOWER120] < 0.1954))) & \
    #                      (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                      (features[FLD.BOLL_JX_MAXFACTOR] < -50) & \
    #                      (features[FLD.BOLL_JX_MAPOWER120] < 0.168) & \
    #                      (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 45) & \
    #                      (features[FLD.BOLL_JX_MAXFACTOR] < -200) & \
    #                      (features[FLD.BOLL_JX_HMAPOWER120] < 0.0927))),
    #                      (features[FLD.BOLL_LB] + features[FLD.ATR_LB]) / 2.11, np.nan),
    #             'kx', alpha = 0.66)
    #    ax1.plot(DATETIME_LABEL,
    #             np.where(((features[FLD.HMAPOWER120_TIMING_LAG] == 1) | \
    #                (features[FLD.ZEN_BOOST_TIMING_LAG] == 1)) & \
    #                (features[FLD.PEAK_LOW_TIMING_LAG] <= 5) & \
    #                (features[FLD.DIF].shift(1) < features[FLD.DIF]) & \
    #                (features[FLD.MACD_DELTA] > 0) & \
    #                (features[AKA.CLOSE] > features[FLD.PEAK_ANCHOR_PREV_LOW_REF6]) & \
    #                (((features[FLD.HMAPOWER30_MA_TIMING_GAP] > 0) & \
    #                      ((features[FLD.HMAPOWER30_MA_TIMING_GAP] > 36) | \
    #                      (features[FLD.HMAPOWER30_MA_TIMING_GAP] > features[FLD.DEA_ZERO_TIMING_LAG]) | \
    #                      (features[FLD.HMAPOWER30_MA_TIMING_GAP] > features[FLD.DIF_ZERO_TIMING_LAG]))) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                      (features[FLD.BOLL_JX_MAXFACTOR] < -200)) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 46.6) & \
    #                      ((features[FLD.BOLL_JX_MAXFACTOR] < -150) | \
    #                      ((features[FLD.BOLL_JX_MAXFACTOR] < 0) & \
    #                      (features[FLD.DEA_ZERO_TIMING_LAG] < 0) & \
    #                      (features[FLD.DIF_ZERO_TIMING_LAG] < 0) & \
    #                      (features[FLD.PEAK_LOW_TIMING_LAG] == 1) & \
    #                      (features[FLD.MACD_ZERO_TIMING_LAG] == 1) & \
    #                      (features[FLD.BOLL_JX_MAPOWER120] < 0.1954))) & \
    #                      (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                      (features[FLD.BOLL_JX_MAXFACTOR] < -50) & \
    #                      (features[FLD.BOLL_JX_MAPOWER120] < 0.168) & \
    #                      (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 45) & \
    #                      (features[FLD.BOLL_JX_MAXFACTOR] < -200) & \
    #                      (features[FLD.BOLL_JX_HMAPOWER120] < 0.0927))),
    #                      (features[FLD.BOLL_LB] + features[FLD.ATR_LB]) / 2.095, np.nan),
    #             'cP', alpha = 0.66)
    #    ax1.plot(DATETIME_LABEL,
    #             np.where(((features[FLD.HMAPOWER120_TIMING_LAG] == 1) | \
    #                (features[FLD.ZEN_BOOST_TIMING_LAG] == 1)) & \
    #                (features[FLD.PEAK_LOW_TIMING_LAG] <= 5) & \
    #                ((features[FLD.DIF].shift(1) > features[FLD.DIF]) | \
    #                (features[FLD.MACD_DELTA] < 0)) & \
    #                (features[AKA.CLOSE] > features[FLD.PEAK_ANCHOR_PREV_LOW_REF6]) & \
    #                (((features[FLD.HMAPOWER30_MA_TIMING_GAP] > 0) & \
    #                      ((features[FLD.HMAPOWER30_MA_TIMING_GAP] > 36) | \
    #                      (features[FLD.HMAPOWER30_MA_TIMING_GAP] > features[FLD.DEA_ZERO_TIMING_LAG]) | \
    #                      (features[FLD.HMAPOWER30_MA_TIMING_GAP] > features[FLD.DIF_ZERO_TIMING_LAG]))) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                      (features[FLD.BOLL_JX_MAXFACTOR] < -200)) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 46.6) & \
    #                      ((features[FLD.BOLL_JX_MAXFACTOR] < -150) | \
    #                      ((features[FLD.BOLL_JX_MAXFACTOR] < 0) & \
    #                      (features[FLD.DEA_ZERO_TIMING_LAG] < 0) & \
    #                      (features[FLD.DIF_ZERO_TIMING_LAG] < 0) & \
    #                      (features[FLD.PEAK_LOW_TIMING_LAG] == 1) & \
    #                      (features[FLD.MACD_ZERO_TIMING_LAG] == 1) & \
    #                      (features[FLD.BOLL_JX_MAPOWER120] < 0.1954))) & \
    #                      (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                      (features[FLD.BOLL_JX_MAXFACTOR] < -50) & \
    #                      (features[FLD.BOLL_JX_MAPOWER120] < 0.168) & \
    #                      (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                      ((features[FLD.BOLL_JX_RSI] < 45) & \
    #                      (features[FLD.BOLL_JX_MAXFACTOR] < -200) & \
    #                      (features[FLD.BOLL_JX_HMAPOWER120] < 0.0927))),
    #                      (features[FLD.BOLL_LB] + features[FLD.ATR_LB]) / 2.095, np.nan),
    #             'kP', alpha = 0.66)
    #    ax1.plot(DATETIME_LABEL,
    #             np.where((features[FLD.ZEN_BOOST_TIMING_LAG] == 1) & \
    #                      (features[FLD.HMAPOWER120] < features[FLD.COMBINE_DENSITY]) & \
    #                      ((features[FLD.HMAPOWER120_TIMING_LAG] == 1) | \
    #                      (features[FLD.HMAPOWER120_TIMING_LAG] == 2) | \
    #                      (features[FLD.HMAPOWER120_TIMING_LAG] == 3)) & \
    #                      ((features[FLD.PEAK_LOW_TIMING_LAG] == 1) | \
    #                      ((features[FLD.DEA_ZERO_TIMING_LAG] > 0) & \
    #                      (features[FLD.PEAK_LOW_TIMING_LAG] <= 1) & \
    #                      (features[FLD.PEAK_ANCHOR_PREV_LOW_REF6] < features[AKA.CLOSE]))) & \
    #                      ((features[FLD.BOLL_JX_RSI] > 66.6) | \
    #                      (features[FLD.BOLL_JX_RSI] < 43.3)),
    #                      (features[FLD.BOLL_LB] + features[FLD.ATR_LB]) / 2.08, np.nan),
    #             'ro', alpha = 0.33)
    #    ax1.plot(DATETIME_LABEL,
    #             np.where((features[FLD.ZEN_BOOST_TIMING_LAG] == 1) & \
    #                      (features[FLD.ZEN_WAVELET_TIMING_LAG] > features[FLD.ZEN_BOOST_TIMING_LAG]) & \
    #                      (features[FLD.HMAPOWER30_MA_TIMING_GAP] > 0) & \
    #                      (features[FLD.PEAK_ANCHOR_PREV_LOW_REF6] < features[AKA.CLOSE]) & \
    #                      (features[FLD.BOLL_JX_RSI] > 66.6) & \
    #                      (features[FLD.BOLL_JX_MAXFACTOR] > 200) & \
    #                      (features[FLD.MAPOWER120] > 0.918),
    #                      (features[FLD.BOLL_LB] + features[FLD.ATR_LB]) / 2.08, np.nan),
    #             'ro', alpha = 0.33)
    #    ax1.plot(DATETIME_LABEL,
    #             np.where((features[FLD.ZEN_BOOST_TIMING_LAG] == 1) & \
    #                      (features[FLD.MAPOWER30] < features[FLD.COMBINE_DENSITY]) & \
    #                      ((features[FLD.MAPOWER30_TIMING_LAG] == 1) | \
    #                      (features[FLD.MAPOWER30_TIMING_LAG] == 2)) & \
    #                      ((features[FLD.PEAK_LOW_TIMING_LAG] == 1) | \
    #                      ((features[FLD.DEA_ZERO_TIMING_LAG] > 0) & \
    #                      (features[FLD.PEAK_LOW_TIMING_LAG] <= 1))) & \
    #                      (((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                         (features[FLD.BOLL_JX_MAXFACTOR] < -200)) | \
    #                         ((features[FLD.BOLL_JX_RSI] < 46.6) & \
    #                         ((features[FLD.BOLL_JX_MAXFACTOR] < -150) | \
    #                         ((features[FLD.BOLL_JX_MAXFACTOR] < 0) & \
    #                         (features[FLD.DEA_ZERO_TIMING_LAG] < 0) & \
    #                         (features[FLD.DIF_ZERO_TIMING_LAG] < 0) & \
    #                         (features[FLD.PEAK_LOW_TIMING_LAG] == 1) & \
    #                         (features[FLD.MACD_ZERO_TIMING_LAG] == 1) & \
    #                         (features[FLD.BOLL_JX_MAPOWER120] < 0.1954))) & \
    #                         (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                         ((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                         (features[FLD.BOLL_JX_MAXFACTOR] < -50) & \
    #                         (features[FLD.BOLL_JX_MAPOWER120] < 0.168) & \
    #                         (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                         ((features[FLD.BOLL_JX_RSI] < 45) & \
    #                         (features[FLD.BOLL_JX_MAXFACTOR] < -200) & \
    #                         (features[FLD.BOLL_JX_HMAPOWER120] < 0.0927))),
    #                      (features[FLD.BOLL_LB] + features[FLD.ATR_LB]) / 2.08, np.nan),
    #             'go', alpha = 0.33)
    #    ax1.plot(DATETIME_LABEL,
    #             np.where((features[FLD.ZEN_BOOST_TIMING_LAG] == 1) & \
    #                      (features[FLD.HMAPOWER120] < features[FLD.COMBINE_DENSITY]) & \
    #                      (features[FLD.HMAPOWER120_TIMING_LAG] > 0) & \
    #                      ((features[FLD.PEAK_LOW_TIMING_LAG] == 1) | \
    #                      ((features[FLD.DEA_ZERO_TIMING_LAG] > 0) & \
    #                      (features[FLD.PEAK_LOW_TIMING_LAG] <= 1))) & \
    #                      (((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                         (features[FLD.BOLL_JX_MAXFACTOR] < -200)) | \
    #                         ((features[FLD.BOLL_JX_RSI] < 46.6) & \
    #                         ((features[FLD.BOLL_JX_MAXFACTOR] < -150) | \
    #                         ((features[FLD.BOLL_JX_MAXFACTOR] < 0) & \
    #                         (features[FLD.DEA_ZERO_TIMING_LAG] < 0) & \
    #                         (features[FLD.DIF_ZERO_TIMING_LAG] < 0) & \
    #                         (features[FLD.PEAK_LOW_TIMING_LAG] == 1) & \
    #                         (features[FLD.MACD_ZERO_TIMING_LAG] == 1) & \
    #                         (features[FLD.BOLL_JX_MAPOWER120] < 0.1954))) & \
    #                         (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                         ((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                         (features[FLD.BOLL_JX_MAXFACTOR] < -50) & \
    #                         (features[FLD.BOLL_JX_MAPOWER120] < 0.168) & \
    #                         (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                         ((features[FLD.BOLL_JX_RSI] < 45) & \
    #                         (features[FLD.BOLL_JX_MAXFACTOR] < -200) & \
    #                         (features[FLD.BOLL_JX_HMAPOWER120] < 0.0927))),
    #                      (features[FLD.BOLL_LB] + features[FLD.ATR_LB]) / 2.08, np.nan),
    #             'mo', alpha = 0.33)
    #    ax1.plot(DATETIME_LABEL,
    #             np.where((features[FLD.ZEN_BOOST_TIMING_LAG] == 1) & \
    #                      (features[FLD.HMAPOWER30] < features[FLD.COMBINE_DENSITY]) & \
    #                      (features[FLD.PEAK_ANCHOR_PREV_LOW_REF6] < features[AKA.CLOSE]) & \
    #                      (features[FLD.BOLL_CROSS_JX_BEFORE] < features[FLD.BOLL_CROSS_SX_BEFORE]) & \
    #                      (features[FLD.DEA_ZERO_TIMING_LAG] > 0) & \
    #                      (features[FLD.PEAK_LOW_TIMING_LAG] <= 1) & \
    #                      (((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                         (features[FLD.BOLL_JX_MAXFACTOR] < -200)) | \
    #                         ((features[FLD.BOLL_JX_RSI] < 46.6) & \
    #                         ((features[FLD.BOLL_JX_MAXFACTOR] < -150) | \
    #                         ((features[FLD.BOLL_JX_MAXFACTOR] < 0) & \
    #                         (features[FLD.DEA_ZERO_TIMING_LAG] < 0) & \
    #                         (features[FLD.DIF_ZERO_TIMING_LAG] < 0) & \
    #                         (features[FLD.PEAK_LOW_TIMING_LAG] == 1) & \
    #                         (features[FLD.MACD_ZERO_TIMING_LAG] == 1) & \
    #                         (features[FLD.BOLL_JX_MAPOWER120] < 0.1954))) & \
    #                         (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                         ((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                         (features[FLD.BOLL_JX_MAXFACTOR] < -50) & \
    #                         (features[FLD.BOLL_JX_MAPOWER120] < 0.168) & \
    #                         (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                         ((features[FLD.BOLL_JX_RSI] < 45) & \
    #                         (features[FLD.BOLL_JX_MAXFACTOR] < -200) & \
    #                         (features[FLD.BOLL_JX_HMAPOWER120] < 0.0927))),
    #                      (features[FLD.BOLL_LB] + features[FLD.ATR_LB]) / 2.08, np.nan),
    #             'co', alpha = 0.33)
    #    checkpoint = ((features[FLD.HMAPOWER120_TIMING_LAG] == 1) | \
    #                (features[FLD.ZEN_BOOST_TIMING_LAG] == 1)) & \
    #                (features[FLD.PEAK_LOW_TIMING_LAG] <= 5) & \
    #                (features[AKA.CLOSE] > features[FLD.PEAK_ANCHOR_PREV_LOW_REF6]) & \
    #                (((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                (features[FLD.BOLL_JX_MAXFACTOR] < -200)) | \
    #                ((features[FLD.BOLL_JX_RSI] < 46.6) & \
    #                ((features[FLD.BOLL_JX_MAXFACTOR] < -150) | \
    #                ((features[FLD.BOLL_JX_MAXFACTOR] < 0) & \
    #                (features[FLD.DEA_ZERO_TIMING_LAG] < 0) & \
    #                (features[FLD.DIF_ZERO_TIMING_LAG] < 0) & \
    #                (features[FLD.PEAK_LOW_TIMING_LAG] == 1) & \
    #                (features[FLD.MACD_ZERO_TIMING_LAG] == 1) & \
    #                (features[FLD.BOLL_JX_MAPOWER120] < 0.1954))) & \
    #                (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                ((features[FLD.BOLL_JX_RSI] < 43.3) & \
    #                (features[FLD.BOLL_JX_MAXFACTOR] < -50) & \
    #                (features[FLD.BOLL_JX_MAPOWER120] < 0.168) & \
    #                (features[FLD.BOLL_JX_HMAPOWER120] < 0.1236)) | \
    #                ((features[FLD.BOLL_JX_RSI] < 45) & \
    #                (features[FLD.BOLL_JX_MAXFACTOR] < -200) & \
    #                (features[FLD.BOLL_JX_HMAPOWER120] < 0.0927)))
    #    print(features.loc[checkpoint,
    #                      [FLD.PEAK_LOW_TIMING_LAG,
    #                       FLD.HMAPOWER120_TIMING_LAG,
    #                       FLD.MACD_ZERO_TIMING_LAG,
    #                       FLD.DIF_ZERO_TIMING_LAG,
    #                       FLD.DEA_ZERO_TIMING_LAG,
    #                       FLD.ZEN_BOOST_TIMING_LAG,
    #                       AKA.CLOSE,
    #                       FLD.PEAK_ANCHOR_PREV_LOW_REF6,
    #                       FLD.MA90_CLEARANCE_TIMING_LAG,
    #                       FLD.MA90_TREND_TIMING_LAG,
    #                       FTR.ATR_BAND_MA90_TIMING_LAG,
    #                       FLD.HMAPOWER120_DUAL_TIMING_LAG,
    #                       FLD.BOLL_JX_MAXFACTOR,
    #                       FLD.BOLL_JX_RSI,
    #                       FLD.BOLL_JX_MAPOWER120,
    #                       FLD.BOLL_JX_HMAPOWER120,]])
    #    ax3.plot(DATETIME_LABEL,
    #             np.where((features[FLD.ZEN_BOOST_TIMING_LAG] == -1),
    #                      np.maximum(features[FLD.MAPOWER30],
    #                                 features[FLD.MAPOWER120]), np.nan),
    #             'g.', alpha = 0.33)

    #    ax3.plot(DATETIME_LABEL,
    #             np.where((features[FLD.RENKO_TREND_L_TIMING_LAG] > 0) & \
    #                      ((features[FLD.RENKO_TREND_L_TIMING_LAG] >
    #                      features[FLD.DEA_ZERO_TIMING_LAG]) | \
    #                      (features[FLD.REGTREE_TIMING_LAG] >
    #                      features[FLD.DEA_ZERO_TIMING_LAG]) | \
    #                      ((features[FLD.MA90_CLEARANCE_TIMING_LAG] >
    #                      features[FLD.DEA_ZERO_TIMING_LAG]) & \
    #                      (features[FLD.MA90] > features[FLD.MA120]) & \
    #                      ((features[FLD.BOLL] + features[FLD.MA90]) >
    #                      (features[FLD.MA120] + features[FLD.MA30])) & \
    #                      (features[FLD.MA90_CLEARANCE_TIMING_LAG] /
    #                      features[FLD.MA90_CROSS_JX_BEFORE] > 0.618)) | \
    #                      ((features[FLD.MA120_CLEARANCE_TIMING_LAG] >
    #                      features[FLD.DEA_ZERO_TIMING_LAG]) & \
    #                      (features[FLD.MA90] > features[FLD.MA120]) & \
    #                      ((features[FLD.BOLL] + features[FLD.MA90]) >
    #                      (features[FLD.MA120] + features[FLD.MA30])) & \
    #                      (features[FLD.MA120_CLEARANCE_TIMING_LAG] /
    #                      features[FLD.MA90_CROSS_JX_BEFORE] > 0.618))),
    #                      (features[FLD.CCI] / 600 + 0.5).min() - 0.00382,
    #                      np.nan),
    #             lw = 1.75, color = 'crimson', alpha = 0.5)