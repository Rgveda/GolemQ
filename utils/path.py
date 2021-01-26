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
这里定义的是一些本地目录
"""

import os
import datetime

try:
    import QUANTAXIS as QA
    from QUANTAXIS.QAUtil.QAParameter import ORDER_DIRECTION
    from QUANTAXIS.QAData.QADataStruct import (
        QA_DataStruct_Index_min, 
        QA_DataStruct_Index_day, 
        QA_DataStruct_Stock_day, 
        QA_DataStruct_Stock_min,
        QA_DataStruct_CryptoCurrency_day,
        QA_DataStruct_CryptoCurrency_min,
        )
    from QUANTAXIS.QAIndicator.talib_numpy import *
    from QUANTAXIS.QAUtil.QADate_Adv import (
        QA_util_timestamp_to_str,
        QA_util_datetime_to_Unix_timestamp,
        QA_util_print_timestamp
    )
    from QUANTAXIS.QAUtil.QALogs import (
        QA_util_log_info, 
        QA_util_log_debug,
        QA_util_log_expection)
    from QUANTAXIS.QAFetch.QAhuobi import (
        FIRST_PRIORITY,
    )
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.utils.path modules')
    pass


"""创建本地文件夹


1. setting_path ==> 用于存放配置文件 setting.cfg
2. cache_path ==> 用于存放临时文件
3. log_path ==> 用于存放储存的log
4. download_path ==> 下载的数据/财务文件
5. strategy_path ==> 存放策略模板
6. bin_path ==> 存放一些交易的sdk/bin文件等
"""

basepath = os.getcwd()
path = os.path.expanduser('~')
user_path = '{}{}{}'.format(path, os.sep, '.GolemQ')
#cache_path = os.path.join(user_path, 'datastore', 'cache')
def cache_path(dirname, portable=False):
    """
    返回本地用户目录下的'.GolemQ'为根目录的缓存临时文件目录，如果 portable 参数等于 True，
    则返回程序代码启动目录为根目录的缓存目录。
    """
    if (portable):
        ret_cache_path = os.path.join(basepath, 'datastore', 'cache', dirname)
    else:
        ret_cache_path = os.path.join(user_path, 'datastore', 'cache', dirname)
    if not (os.path.exists(ret_cache_path) and \
        os.path.isdir(ret_cache_path)):
        #print(u'文件夹',dirname,'不存在，重新建立')
        #os.mkdir(dirname)
        try:
            os.makedirs(ret_cache_path)
        except:
            # 如果目录已经存在，那么可能是并发冲突，当做什么事情都没发生
            if not (os.path.exists(ret_cache_path)):
                # 否则继续触发异常
                os.makedirs(os.path.join(ret_cache_path))
    return ret_cache_path


def mkdirs_user(dirname):
    if not (os.path.exists(os.path.join(user_path, dirname)) and \
        os.path.isdir(os.path.join(user_path, dirname))):
        #print(u'文件夹',dirname,'不存在，重新建立')
        #os.mkdir(dirname)
        try:
            os.makedirs(os.path.join(user_path, dirname))
        except:
            # 如果目录已经存在，那么可能是并发冲突，当做什么事情都没发生
            if not (os.path.join(user_path, dirname)):
                # 否则继续触发异常
                os.makedirs(os.path.join(user_path, dirname))
    return os.path.join(user_path, dirname)


def mkdirs(dirname):
    if not (os.path.exists(os.path.join(basepath, dirname)) and \
        os.path.isdir(os.path.join(basepath, dirname))):
        #print(u'文件夹',dirname,'不存在，重新建立')
        #os.mkdir(dirname)
        try:
            os.makedirs(os.path.join(basepath, dirname))
        except:
            # 如果目录已经存在，那么可能是并发冲突，当做什么事情都没发生
            if not (os.path.exists(os.path.join(basepath, dirname))):
                # 否则继续触发异常
                os.makedirs(os.path.join(basepath, dirname))
    return os.path.join(basepath, dirname)


def export_csv_min(code, market_type):
    """
    训练用隶属数据导出模块
    """
    if (isinstance(code, list)):
        code = code[0]

    frequence = '60min'
    if (market_type == QA.MARKET_TYPE.STOCK_CN):
        market_type_alis = 'A股'
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        market_type_alis = '指数'
    elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
        market_type_alis = '数字货币'


    #print(u'{} 开始读取{}历史数据'.format(QA_util_timestamp_to_str()[2:16],
    #                                    market_type_alis),
    #        code)
    if (market_type == QA.MARKET_TYPE.STOCK_CN):
        data_day = QA.QA_fetch_stock_min_adv(code,
                                            '1991-01-01',
                                            '{}'.format(datetime.date.today(),
                                            frequency=frequence))
 
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        #data_day = QA.QA_fetch_index_day_adv(code,
        #                                    '1991-01-01',
        #                                    '{}'.format(datetime.date.today(),))
        data_day = QA.QA_fetch_index_min_adv(code,
                                            '1991-01-01',
                                            '{}'.format(datetime.date.today(),
                                            frequency=frequence))
 
    elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
        frequence = '60min'
        data_hour = data_day = QA.QA_fetch_cryptocurrency_min_adv(code=code,
                start='2009-01-01',
                end=QA_util_timestamp_to_str(),
                frequence=frequence)

    if (data_day is None):
        #print('{}没有数据'.format(code))
        pass
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        mkdirs(os.path.join(export_path, 'index'))
        data_day.data.to_csv(os.path.join(export_path, 'index', '{}_{}_kline.csv'.format(code, frequence)))
    elif (market_type == QA.MARKET_TYPE.STOCK_CN):
        mkdirs(os.path.join(export_path, 'stock'))
        data_day.data.to_csv(os.path.join(export_path, 'stock', '{}_{}_kline.csv'.format(code, frequence)))
        
    return data_day.data


def save_hdf_min(code, market_type, export_path='export', features=None):
    """
    训练用隶属特征数据导出模块
    """
    if (isinstance(code, list)):
        code = code[0]

    frequence = '60min'
    if (features is None):
        #print('{}没有数据'.format(code))
        pass
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        mkdirs(os.path.join(export_path, 'index'))
        features.to_hdf(os.path.join(export_path, 'index', '{}_{}_features.hdf'.format(code, frequence)), key='df', mode='w')
    elif (market_type == QA.MARKET_TYPE.STOCK_CN):
        mkdirs(os.path.join(export_path, 'stock'))
        features.to_hdf(os.path.join(export_path, 'stock', '{}_{}_features.hdf'.format(code, frequence)), key='df', mode='w')

    return features


def export_hdf_min(code, market_type, export_path='export', features=None):
    """
    训练用隶属数据导出模块
    """
    if (isinstance(code, list)):
        code = code[0]

    frequence = '60min'
    if (market_type == QA.MARKET_TYPE.STOCK_CN):
        market_type_alis = 'A股'
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        market_type_alis = '指数'
    elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
        market_type_alis = '数字货币'


    #print(u'{} 开始读取{}历史数据'.format(QA_util_timestamp_to_str()[2:16],
    #                                    market_type_alis),
    #        code)
    if (market_type == QA.MARKET_TYPE.STOCK_CN):
        data_day = QA.QA_fetch_stock_min_adv(code,
                                            '1991-01-01',
                                            '{}'.format(datetime.date.today()),
                                            frequence=frequence)
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        #data_day = QA.QA_fetch_index_day_adv(code,
        #                                    '1991-01-01',
        #                                    '{}'.format(datetime.date.today(),))
        data_day = QA.QA_fetch_index_min_adv(code,
                                            '1991-01-01',
                                            '{}'.format(datetime.date.today()),
                                            frequence=frequence)
    elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
        frequence = '60min'
        data_hour = data_day = QA.QA_fetch_cryptocurrency_min_adv(code=code,
                                                                start='2009-01-01',
                                                                end='{}'.format(datetime.date.today()),
                                                                frequence=frequence)

    if (data_day is None):
        #print('{}没有数据'.format(code))
        pass
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        mkdirs(os.path.join(export_path, 'index'))
        data_day.data.to_hdf(os.path.join(export_path, 'index', '{}_{}_kline.hdf'.format(code, frequence)), key='df', mode='w')
    elif (market_type == QA.MARKET_TYPE.STOCK_CN):
        mkdirs(os.path.join(export_path, 'stock'))
        data_day.data.to_hdf(os.path.join(export_path, 'stock', '{}_{}_kline.hdf'.format(code, frequence)), key='df', mode='w')

    return data_day.data


def export_csv_day(code, market_type=None, export_path='export'):
    """
    训练用隶属数据导出模块
    """
    if (isinstance(code, list)):
        code = code[0]

    if (market_type == QA.MARKET_TYPE.STOCK_CN):
        market_type_alis = 'A股'
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        market_type_alis = '指数'
    elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
        market_type_alis = '数字货币'

    print(u'{} 开始读取{}历史数据'.format(QA_util_timestamp_to_str()[2:16], 
                                        market_type_alis), 
            code)
    #data_day = QA.QA_fetch_stock_min_adv(codelist,
    #                                      '2018-11-01',
    #                                      '{}'.format(datetime.date.today()),
    #                                      frequence=frequence)
    if (market_type == QA.MARKET_TYPE.STOCK_CN):
        data_day = QA.QA_fetch_stock_day_adv(code,
                                            '1991-01-01',
                                            '{}'.format(datetime.date.today(),)).to_qfq()

        if (np.isnan(data_day).any() == True):
            # 在下载数据的时候，有时候除权后莫名其妙丢数据了，我只能拿没除权的数据补
            predict_null = pd.isnull(data_day.data[AKA.CLOSE])
            data_null = data_day.data[predict_null == True]
            data_day.data.loc[data_null.index, :] = QA.QA_fetch_stock_day_adv(code,
                                                '{}'.format(data_null.index.get_level_values(level=0).values[0]),
                                                '{}'.format(datetime.date.today(),)).data
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        data_day = QA.QA_fetch_index_day_adv(code,
                                            '1991-01-01',
                                            '{}'.format(datetime.date.today(),))

    elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
        frequency = '60min'
        data_hour = data_day = QA.QA_fetch_cryptocurrency_min_adv(code=code,
                start='2009-01-01',
                end=QA_util_timestamp_to_str(),
                frequence=frequency)

    if (data_day is None):
        print('{}没有数据'.format(code))
        pass
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        mkdirs(os.path.join(export_path, 'index'))
        data_day.data.drop(['date_stamp','down_count','up_count'], axis=1).to_csv(os.path.join(export_path, 'index', '{}.csv'.format(code)))
    elif (market_type == QA.MARKET_TYPE.STOCK_CN):
        mkdirs(os.path.join(export_path, 'stock'))
        data_day.data.drop(['adj'], axis=1).to_csv(os.path.join(export_path, 'stock', '{}.csv'.format(code)))

    return data_day.data


def export_hdf_day(code, market_type=None, export_path='export', features=None):
    """
    训练用隶属数据导出模块
    """
    if (isinstance(code, list)):
        code = code[0]

    if (market_type == QA.MARKET_TYPE.STOCK_CN):
        market_type_alis = 'A股'
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        market_type_alis = '指数'
    elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
        market_type_alis = '数字货币'

    print(u'{} 开始读取{}历史数据'.format(QA_util_timestamp_to_str()[2:16], 
                                        market_type_alis), 
            code)
    #data_day = QA.QA_fetch_stock_min_adv(codelist,
    #                                      '2018-11-01',
    #                                      '{}'.format(datetime.date.today()),
    #                                      frequence=frequence)
    if (market_type == QA.MARKET_TYPE.STOCK_CN):
        data_day = QA.QA_fetch_stock_day_adv(code,
                                            '1991-01-01',
                                            '{}'.format(datetime.date.today(),)).to_qfq()

        if (np.isnan(data_day).any() == True):
            # 在下载数据的时候，有时候除权后莫名其妙丢数据了，我只能拿没除权的数据补
            predict_null = pd.isnull(data_day.data[AKA.CLOSE])
            data_null = data_day.data[predict_null == True]
            data_day.data.loc[data_null.index, :] = QA.QA_fetch_stock_day_adv(code,
                                                '{}'.format(data_null.index.get_level_values(level=0).values[0]),
                                                '{}'.format(datetime.date.today(),)).data
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        data_day = QA.QA_fetch_index_day_adv(code,
                                            '1991-01-01',
                                            '{}'.format(datetime.date.today(),))

    elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
        frequency = '60min'
        data_hour = data_day = QA.QA_fetch_cryptocurrency_min_adv(code=code,
                start='2009-01-01',
                end=QA_util_timestamp_to_str(),
                frequence=frequency)

    if (data_day is None):
        print('{}没有数据'.format(code))
        pass
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        mkdirs(os.path.join(export_path, 'index'))
        data_day.data.drop(['date_stamp','down_count','up_count'], axis=1).to_hdf(os.path.join(export_path, 'index', '{}.hdf'.format(code)), key='df', mode='w')
    elif (market_type == QA.MARKET_TYPE.STOCK_CN):
        mkdirs(os.path.join(export_path, 'stock'))
        data_day.data.drop(['adj'], axis=1).to_hdf(os.path.join(export_path, 'stock', '{}.hdf'.format(code)), key='df', mode='w')

    return data_day.data


def export_hdf_metadata(export_path, code, frequence='60min', metadata=None):
    """
    训练用隶属特征数据导出模块
    """
    if (isinstance(code, list)):
        code = code[0]

    if (metadata is None):
        #print('{}没有数据'.format(code))
        pass
    else:
        print(os.path.join(export_path, '{}_{}.hdf5'.format(code, frequence)),
              metadata.tail(10))
        #metadata.to_hdf(os.path.join(export_path, '{}_{}.hdf5'.format(code,
        #frequence)), key='df', mode='w')
        metadata.to_pickle(os.path.join(export_path, '{}_{}.hdf5'.format(code, frequence)))
    return metadata


def export_metadata_to_pickle(export_path, code, frequence='60min', metadata=None):
    """
    训练用隶属特征数据导出模块
    """
    if (isinstance(code, list)):
        code = code[0]

    if (metadata is None):
        #print('{}没有数据'.format(code))
        pass
    else:
        print(os.path.join(export_path, '{}_{}.pickle'.format(code, frequence)),
              metadata.tail(3))
        #metadata.to_hdf(os.path.join(export_path, '{}_{}.hdf5'.format(code,
        #frequence)), key='df', mode='w')
        metadata.to_pickle(os.path.join(export_path, '{}_{}.pickle'.format(code, frequence)))
    return metadata


def import_metadata_from_pickle(export_path, code, frequence='60min'):
    if (isinstance(code, list)):
        code = code[0]

    print(os.path.join(export_path, '{}_{}.pickle'.format(code, frequence)))
    metadata = pd.read_pickle(os.path.join(export_path, '{}_{}.pickle'.format(code, frequence)))
    return metadata


def save_hdf_min(code, market_type, export_path='export', features=None):
    """
    训练用隶属特征数据导出模块
    """
    if (isinstance(code, list)):
        code = code[0]

    frequence = '60min'
    if (features is None):
        #print('{}没有数据'.format(code))
        pass
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        mkdirs(os.path.join(export_path, 'index'))
        features.to_hdf(os.path.join(export_path, 'index', '{}_{}_features.hdf'.format(code, frequence)), key='df', mode='w')
    elif (market_type == QA.MARKET_TYPE.STOCK_CN):
        mkdirs(os.path.join(export_path, 'stock'))
        features.to_hdf(os.path.join(export_path, 'stock', '{}_{}_features.hdf'.format(code, frequence)), key='df', mode='w')

    return features


def export_hdf_min(code, market_type, export_path='export', features=None):
    """
    训练用隶属数据导出模块
    """
    if (isinstance(code, list)):
        code = code[0]

    frequence = '60min'
    if (market_type == QA.MARKET_TYPE.STOCK_CN):
        market_type_alis = 'A股'
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        market_type_alis = '指数'
    elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
        market_type_alis = '数字货币'


    #print(u'{} 开始读取{}历史数据'.format(QA_util_timestamp_to_str()[2:16],
    #                                    market_type_alis),
    #        code)
    if (market_type == QA.MARKET_TYPE.STOCK_CN):
        data_day = QA.QA_fetch_stock_min_adv(code,
                                            '1991-01-01',
                                            '{}'.format(datetime.date.today()),
                                            frequence=frequence)
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        #data_day = QA.QA_fetch_index_day_adv(code,
        #                                    '1991-01-01',
        #                                    '{}'.format(datetime.date.today(),))
        data_day = QA.QA_fetch_index_min_adv(code,
                                            '1991-01-01',
                                            '{}'.format(datetime.date.today()),
                                            frequence=frequence)
    elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
        frequence = '60min'
        data_hour = data_day = QA.QA_fetch_cryptocurrency_min_adv(code=code,
                                                                start='2009-01-01',
                                                                end='{}'.format(datetime.date.today()),
                                                                frequence=frequence)

    if (data_day is None):
        #print('{}没有数据'.format(code))
        pass
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        mkdirs(os.path.join(export_path, 'index'))
        data_day.data.to_hdf(os.path.join(export_path, 'index', '{}_{}_kline.hdf'.format(code, frequence)), key='df', mode='w')
    elif (market_type == QA.MARKET_TYPE.STOCK_CN):
        mkdirs(os.path.join(export_path, 'stock'))
        data_day.data.to_hdf(os.path.join(export_path, 'stock', '{}_{}_kline.hdf'.format(code, frequence)), key='df', mode='w')

    return data_day.data

def load_cache(filename='cache.pickle'):
    filename = filename.replace(' ', '_').replace(':', '_')
    metadata = pd.read_pickle(os.path.join(mkdirs(os.path.join('cache')), filename))
    return metadata

def save_cache(filename='cache.pickle', metadata=None):
    filename = filename.replace(' ', '_').replace(':', '_')
    metadata = metadata.to_pickle(os.path.join(mkdirs(os.path.join('cache')), filename))
    return filename


def load_snapshot_cache(dirpath, filename='cache.pickle'):
    filename = filename.replace(' ', '_').replace(':', '_')
    metadata = pd.read_pickle(os.path.join(mkdirs(dirpath), filename))
    return metadata

def save_snapshot_cache(dirpath, filename='cache.pickle', metadata=None):
    filename = filename.replace(' ', '_').replace(':', '_')
    metadata = metadata.to_pickle(os.path.join(mkdirs(dirpath), filename))
    return os.path.join(mkdirs(dirpath), filename)