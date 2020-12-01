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
             color='green', lw=1, label=FLD.DIF)
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
                                         FLD.DEADPOOL_CANDIDATE_TIMING_LAG,]].copy()

    #print(markers)
    for index, marker in markers.iterrows():
        mkr_x = marker.name[0].strftime("%Y-%m-%d %H:%M")[2:16]
        mkr_text_template = '*{}*\nCOMB:{:.3f}/BOL:{:.3f}\nMAPWR3:{:.2f}/HMA12:{:.2f}\nMACH:{:.2f}/MA12CH:{:.2f}\nMA9CLR:{:.2f}/{:.2f}\nPEAK_LO:{:03d}/{:03d}/{:03d}\nPEAK_HI:{:03d}/{:03d}/{:03d}\nDRAWDOWN:{:.3f}/{:.3f}'
        mkr_text = mkr_text_template.format(mkr_x,
                                            marker.at[FLD.COMBINE_DENSITY],
                                            marker.at[FLD.BOLL_CHANNEL],
                                            marker.at[FLD.MAPOWER30_MAJOR],
                                            marker.at[FLD.HMAPOWER120_MAJOR],
                                            marker.at[FLD.MA_CHANNEL],
                                            marker.at[FLD.MA120_CHANNEL],
                                            marker.at[FLD.MA90_CLEARANCE],
                                            marker.at[FLD.MA120_CLEARANCE],
                                            int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE]),
                                            int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR]),
                                            int(marker.at[FLD.DEADPOOL_CANDIDATE_TIMING_LAG]),
                                            int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE]),
                                            int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR]),
                                            int(marker.at[FLD.MAPOWER30_TIMING_LAG_MAJOR]),
                                            marker.at[FLD.DRAWDOWN_RATIO],
                                            marker.at[FLD.DRAWDOWN_RATIO_MAJOR],)
        if (marker.at[FTR.UPRISING_RAIL_TIMING_LAG] == 1):
            mkr_y = marker.at[FLD.HMAPOWER120_MAJOR]
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


def mark_annotate(checkpoints, features, ax3, colorwarp='olive'):
    '''
    绘制标记
    '''
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
                                         FLD.HMAPOWER30_MA_TIMING_LAG,]].copy()

    #print(markers)
    for index, marker in markers.iterrows():
        mkr_x = marker.name[0].strftime("%Y-%m-%d %H:%M")[2:16]
        mkr_text_template = '*{}*\nCOMB:{:.3f}/BOL:{:.3f}\nMAPWR3:{:.2f}/HMA12:{:.2f}\nMACH:{:.2f}/MA12CH:{:.2f}\nMA9CLR:{:.2f}/{:.2f}\nPEAK_LO:{:03d}/{:03d}/{:03d}\nPEAK_HI:{:03d}/{:03d}/{:03d}\nDRAWDOWN:{:.3f}/{:.3f}\nRENKO:{:03d}/ZEN:{:03d}'
        mkr_text = mkr_text_template.format(mkr_x,
                                            marker.at[FLD.COMBINE_DENSITY],
                                            marker.at[FLD.BOLL_CHANNEL],
                                            marker.at[FLD.MAPOWER30],
                                            marker.at[FLD.HMAPOWER120],
                                            marker.at[FLD.MA_CHANNEL],
                                            marker.at[FLD.MA120_CHANNEL],
                                            marker.at[FLD.MA90_CLEARANCE],
                                            marker.at[FLD.MA120_CLEARANCE],
                                            int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE]),
                                            int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR]),
                                            int(marker.at[FLD.DEADPOOL_CANDIDATE_TIMING_LAG]),
                                            int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE]),
                                            int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR]),
                                            int(marker.at[FLD.MAPOWER30_TIMING_LAG_MAJOR]),
                                            marker.at[FLD.DRAWDOWN_RATIO],
                                            marker.at[FLD.DRAWDOWN_RATIO_MAJOR],
                                            int(marker.at[FLD.ZEN_WAVELET_TIMING_LAG]),
                                            int(marker.at[FLD.RENKO_TREND_S_TIMING_LAG]),)
        mkr_y = marker.at[FLD.MAPOWER30]
        ax3.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data', 
                     xytext=(20, -40), textcoords='offset points', 
                     fontsize=6, va="center", ha="center", zorder=4,
                     alpha=0.66,
                     bbox=dict(boxstyle="round4", fc="w", 
                               color=colorwarp),
                     arrowprops=dict(arrowstyle="->", 
                                     connectionstyle="arc3,rad=.2", 
                                     color=colorwarp))


def mark_annotate_ax2(checkpoints, features, ax2, colorwarp='violet'):
    '''
    绘制标记
    '''
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
                                         FLD.RENKO_TREND_S_TIMING_LAG,]].copy()

    #print(markers)
    for index, marker in markers.iterrows():
        mkr_x = marker.name[0].strftime("%Y-%m-%d %H:%M")[2:16]
        mkr_text_template = '*{}*\nCOMB:{:.3f}/BOL:{:.3f}\nMAPWR3:{:.2f}/HMA12:{:.2f}\nMACH:{:.2f}/MA12CH:{:.2f}\nMA9CLR:{:.2f}/{:.2f}\nPEAK_LO:{:03d}/{:03d}/{:03d}\nPEAK_HI:{:03d}/{:03d}/{:03d}\nDRAWDOWN:{:.3f}/{:.3f}\nRENKO:{:03d}/ZEN:{:03d}'
        mkr_text = mkr_text_template.format(mkr_x,
                                            marker.at[FLD.COMBINE_DENSITY],
                                            marker.at[FLD.BOLL_CHANNEL],
                                            marker.at[FLD.MAPOWER30],
                                            marker.at[FLD.HMAPOWER120],
                                            marker.at[FLD.MA_CHANNEL],
                                            marker.at[FLD.MA120_CHANNEL],
                                            marker.at[FLD.MA90_CLEARANCE],
                                            marker.at[FLD.MA120_CLEARANCE],
                                            int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE]),
                                            int(marker.at[FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR]),
                                            int(marker.at[FLD.DEADPOOL_CANDIDATE_TIMING_LAG]),
                                            int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE]),
                                            int(marker.at[FLD.MAPOWER30_PEAK_HIGH_BEFORE_MAJOR]),
                                            int(marker.at[FLD.MAPOWER30_TIMING_LAG_MAJOR]),
                                            marker.at[FLD.DRAWDOWN_RATIO],
                                            marker.at[FLD.DRAWDOWN_RATIO_MAJOR],
                                            int(marker.at[FLD.ZEN_WAVELET_TIMING_LAG]),
                                            int(marker.at[FLD.RENKO_TREND_S_TIMING_LAG]),)
        mkr_y = min(marker.at[FLD.DEA], marker.at[FLD.DIF]) * 1.10
        ax2.annotate(mkr_text, xy=(mkr_x, mkr_y), xycoords='data', 
                     xytext=(20, -40), textcoords='offset points', 
                     fontsize=6, va="center", ha="center", zorder=4,
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