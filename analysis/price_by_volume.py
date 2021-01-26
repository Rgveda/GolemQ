# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2018 yutiansut/QUANTAXIS
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
"""
基于 QUANTAXIS 的 DataStruct.add_func 使用，也可以单独使用处理 Kline，
使用历史成交量分析筹码分布
"""


def custom_round(x, base=5):
    return int(base * round(float(x)/base))
 
def round_and_group(df, base=5):
    # https://stackoverflow.com/questions/40372030/pandas-round-to-the-nearest-n
    # Extract the data we want
    df = data[['Close', 'Volume']].copy()
    # Round to nearest X
    df['Close'] = df['Close'].apply(lambda x: custom_round(x, base=base))
    # Remove the date index
    df = df.set_index('Close')
    df = df.groupby(['Close']).sum()
    return df
 
def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.0fK' % (x*1e-3)
 
def create_plot(x_series, 
                x_label, 
                y_pos, 
                y_tick_labels, 
                colour, 
                title):
    # Excample horizontal bar chart code taken from:
    # https://matplotlib.org/gallery/lines_bars_and_markers/barh.html
 
    plt.rcdefaults()
    fig, ax = plt.subplots()
 
    ax.barh(y_pos, x_series, align='center',color=colour)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_tick_labels)
    formatter = FuncFormatter(thousands)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    plt.xticks(rotation=325)
    plt.show()
 
if __name__ == '__main__':
    # 命令行分析库
    import argparse

    #Get Args
    args = parse_args()
 
    print("VOLUME AT PRICE")
    print("QCODE: {}".format(args.qcode))
 
    # Get data
    data = get_quandl_data(args.qcode, args.start, args.end)
 
    # Massage the data
    data = round_and_group(data, base=args.round)
 
    # Prepare data for the chart
    y_pos = np.arange(len(data.index.values))
 
    # Get the chart
    plt = create_plot(
        data['Volume'], # x_series
        'Volume', # x_label
        y_pos, # Y positioning
        data.index.values, # y_tick_labels
        'Green', # Bar color
        'VOLUME AT PRICE: {}'.format(args.qcode) # Title
        )
