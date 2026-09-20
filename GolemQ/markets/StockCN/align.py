# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ(uant)
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

from datetime import datetime as dt
import numpy as np
import pandas as pd
import baostock as bs
import traceback
from datetime import (
    timedelta,
)
import time
from .date_utils import GQ_util_get_last_day
from .scribe import GQ_stock_a_spot_em
from .fetch import (
    GQ_fetch_stock_list_day,
    get_kline_price_v3,
)
from GolemQ.core.constants import (
    AKA, FIELD as FLD,
)
from GolemQ.supervisor.heartbeat import HeartbeatModule
from GolemQ.core.settings import (
    DATABASE as DATABASE_GolemQ,
)
from GolemQ.services.features import (
    GQ_save_daily_metadata_reality,
    GQ_fetch_daily_metadata_reality,
)
from .crawler import (
    GQ_featch_stock_valuation_from_baostock,
    GQ_SU_crawl_stock_valuation,
)
from .symbol import (
    GQ_fetch_stock_info,
)
from .symbol import (
    GQ_fetch_stock_list,
)
from tqdm import tqdm  # 进度条工具，可选安装
import sys
try:
    import akshare as ak
except ImportError:
    # 获取Python主版本和次版本
    major, minor = sys.version_info[:2]
    if (major == 3 and minor > 11) or major > 3:
        print('PLEASE run "pip install akshare" before call GolemQ.cli modules')
    pass


def ckpo_align_stock_turnover_rate(
    code: str = None,
    data_day: pd.DataFrame = None,
    start: str = None,
    end: str = f'{dt.now():%Y-%m-%d}',
    verbose: bool = True,
):
    recall = False
    if (data_day is None):
        data_day, stock_name_faked = get_kline_price_v3(
            code, start=start, end=end, verbose=verbose, realtime=False)

    if (data_day is not None):
        ohlc_data = data_day.data
    else:
        ohlc_data = None

    stock_valuation_pd = GQ_fetch_daily_metadata_reality(
        code=code,
        start=start, end=end,
        collections=DATABASE_GolemQ.stock_valuation)

    column_list = [
        FLD.PE_RATION,
        FLD.TURNOVER_RATE,]
    if (stock_valuation_pd is not None) and (len(stock_valuation_pd) > 0):
        stock_valuation_pd = stock_valuation_pd.rename(
            columns={
                u'peTTM': FLD.PE_RATION,
                AKA.TURNOVER: FLD.TURNOVER_RATE
            })
        stock_valuation_idx = ohlc_data.index.intersection(stock_valuation_pd.index)
        ohlc_data.loc[
            stock_valuation_idx, 
            column_list] = stock_valuation_pd.loc[
                stock_valuation_idx,
                column_list]
        if (ohlc_data is not None):
            try:
                missing_stock_valuation_idx = ohlc_data.index.difference(
                    ohlc_data.dropna(
                        subset=[FLD.TURNOVER_RATE,],
                        axis=0,
                        how="all"
                    ).index
                )
            except TypeError:
                if ((len(ohlc_data)) < 1):
                    print(
                        f'\nCode:{code} ohlc_data.dropna() TypeError.\n',
                        ohlc_data.tail(10),
                    )
                    data_day_again, stock_name_faked = get_kline_price_v3(
                        code, start=start, end=end, verbose=True, realtime=False)
                    print(data_day_again.data.tail(10))
    else:
        if (ohlc_data is not None):
            missing_stock_valuation_idx = ohlc_data.index
        else:
            print(f'Code:{code} ohlc_data is None.')

    stock_ranking_pd = GQ_fetch_daily_metadata_reality(
        code,
        start='{}'.format(start)[:10],
        end='{}'.format(end)[:10],
        collections=DATABASE_GolemQ.stock_ranking,)
    if (stock_ranking_pd is not None):
        stock_ranking_pd = stock_ranking_pd.dropna(
            subset=[FLD.TURNOVER_RATE],
            axis=0,
            how="all")
        if (len(missing_stock_valuation_idx) > 1):
            if (verbose):
                print(u'需要追加换手率数据....{} to {} 从 stock_ranking'.format('{}'.format(start)[:10], 
                                                                            '{}'.format(end)[:10]))
            if (FLD.TURNOVER_RATE in ohlc_data.columns):
                missing_idx = ohlc_data.index.difference(
                    ohlc_data.dropna(
                        subset=[FLD.TURNOVER_RATE],
                        axis=0,
                        how="all").index)
            else:
                missing_idx = ohlc_data.index
            stock_ranking_idx = missing_idx.intersection(stock_ranking_pd.index)
            ohlc_data.loc[
                stock_ranking_idx,
                column_list]=stock_ranking_pd.loc[
                    stock_ranking_idx,
                    column_list]
            
            if (verbose):
                print(u'Fill stock_ranking:', 
                    ohlc_data[column_list].tail(3), 
                    u'\nmissing stock_valuation:\n', 
                    missing_stock_valuation_idx,
                u'\nwith stock_ranking:\n', 
                stock_ranking_pd.loc[stock_ranking_pd.index.intersection(missing_stock_valuation_idx),
                                                        column_list])

    missing_turnover_rate_idx = ohlc_data.index.difference(
            ohlc_data.dropna(
                subset=[FLD.TURNOVER_RATE,],
                axis=0,
                how="all"
            ).index
        )
    if (len(missing_turnover_rate_idx)):
        if (verbose):
            print(u'需要追加换手率数据....{} 从 akshare '.format(missing_turnover_rate_idx))
            print(ohlc_data.loc[missing_turnover_rate_idx, ])

        try:
            if (len(missing_turnover_rate_idx) > 1e-12):
                start_date = f"{missing_idx[0][0]}".replace('-', '')  # 开始日期
                end_date = f"{missing_idx[-1][0]}".replace('-', '')    # 结束日期

                res = ak.stock_zh_a_hist(symbol=symbol, period='daily', start_date=start_date, end_date=end_date)
                res = res[['日期', '股票代码', '换手率']].rename(
                    columns={
                        '日期': AKA.DATE,
                        '股票代码': AKA.CODE,
                        '换手率': FLD.TURNOVER_RATE,
                    }
                )
                recall = True
                stock_ranking_new = res.assign(date=pd.to_datetime(res.date)).drop_duplicates(([AKA.DATE,
                                    AKA.CODE])).set_index([AKA.DATE,
                                    AKA.CODE],
                                        drop=True)
                if (1.0 <= stock_ranking_new[FLD.TURNOVER_RATE].max() <= 100):
                    stock_ranking_new[FLD.TURNOVER_RATE] = stock_ranking_new[FLD.TURNOVER_RATE].astype(np.float32) / 100

                stock_ranking_to_save = stock_ranking_new.loc[missing_idx.intersection(missing_stock_valuation_idx), :]

            if (len(missing_idx) > 1e-12) and \
                (len(stock_ranking_to_save) > 1e-12):
                if (verbose):
                    print(code, stock_ranking_to_save)

                GQ_save_daily_metadata_reality(
                    stock_ranking_to_save,
                    collections=DATABASE_GolemQ.stock_ranking,
                )
        except Exception:
            pass

        try:
            if (verbose):
                print('GQ_featch_stock_valuation_from_baostock:', len(missing_stock_valuation_idx), missing_stock_valuation_idx)

            if (len(missing_stock_valuation_idx) > 1e-12):
                start_date = f"{missing_stock_valuation_idx[0][0]:%Y-%m-%d}"  # 开始日期
                end_date = f"{missing_stock_valuation_idx[-1][0]:%Y-%m-%d}"    # 结束日期

                baostock_st=True
                ret_valuation=GQ_featch_stock_valuation_from_baostock(
                    code, 
                    start='{}'.format(start_date)[:10], 
                    end='{}'.format(end_date)[:10],)
                baostock_st=False
                recall = True

                if (isinstance(ret_valuation, tuple)):
                    if ((GQ_util_get_last_day()-pd.to_datetime(start_date)) > timedelta(hours=24.1)):
                        print('query_history_k_data_plus respond error_code:', ret_valuation)
                        print('query_history_k_data_plus respond error_msg:'+ret_valuation[1], code)
                else:
                    if (verbose) and \
                        (pd.to_datetime(missing_stock_valuation_idx.get_level_values(level=0)[-1])-\
                        pd.to_datetime(missing_stock_valuation_idx.get_level_values(level=0)[0])>timedelta(days=8.5)):
                        print(ret_valuation.index.intersection(missing_stock_valuation_idx))
                        ret_valuation=ret_valuation.loc[ret_valuation.index.intersection(missing_stock_valuation_idx), :]
                        print(ret_valuation)
                    stock_valuation_to_save = ret_valuation.loc[ret_valuation.index.intersection(missing_stock_valuation_idx), :]
                    try:
                        GQ_save_daily_metadata_reality(
                            stock_valuation_to_save,
                            collections=DATABASE_GolemQ.stock_valuation,
                        )
                    except Exception:
                        traceback.print_exc()
                        print('Save error: {}'.format(ret_valuation))
                        print(code, start_date, end_date, ret_valuation.index.intersection(missing_stock_valuation_idx), missing_stock_valuation_idx)
                        pass
        except Exception:
            traceback.print_exc()
            pass

        # print('{}'.format(start)[:10], '{}'.format(end)[:10], ohlc_data[FLD.TURNOVER_RATE].tail(10))
        if (verbose):
            print(u'Fill stock_ranking_pd:', ohlc_data[column_list].tail(10))
                                                                                       
    return recall


def stock_min_aligned(
    verbose: bool = False
) -> dict:
    def get_missing_turnover(
        code, 
        data=None, 
        offset='', 
        verbose=False
    ):
        if (data is None) or (len(data) > 1920):
            if len(offset) == 0:
                start = '{}'.format(dt.now() - timedelta(days=2560))
            else:
                start = '{}'.format(pd.to_datetime(offset)) - timedelta(days=2560)
            data_day, stock_name_faked = get_kline_price_v3(
                code, start=start, verbose=verbose, realtime=True
            )
            ohlc_day = data_day.data     
        else:
            ohlc_day = data

        start = '{}'.format(
            ohlc_day.index.get_level_values(level=0)[0] - timedelta(hours=8.5)
        )
        end = '{}'.format(
            ohlc_day.index.get_level_values(level=0)[-1] + timedelta(hours=16)
        )

        stock_valuation_pd = GQ_fetch_daily_metadata_reality(
            code=code,
            start=start,
            end=end,
            collections=DATABASE_GolemQ.stock_valuation
        )

        column_list = [FLD.PE_RATION, FLD.TURNOVER_RATE]
        if (stock_valuation_pd is not None) and (len(stock_valuation_pd) > 0):
            stock_valuation_pd = stock_valuation_pd.rename(
                columns={
                    u'peTTM': FLD.PE_RATION,
                    AKA.TURNOVER: FLD.TURNOVER_RATE
                })
            stock_valuation_idx = ohlc_day.index.intersection(
                stock_valuation_pd.index
            )
            ohlc_day.loc[stock_valuation_idx, column_list] = stock_valuation_pd.loc[
                stock_valuation_idx, column_list
            ]
            missing_stock_valuation_idx = ohlc_day.index.difference(
                ohlc_day.dropna(
                    subset=[FLD.TURNOVER_RATE], axis=0, how="all"
                ).index
            )
            if (verbose) and (len(missing_stock_valuation_idx) > 0):
                print(
                    f'Phase 1: 缺 {len(missing_stock_valuation_idx)} 天的换手率', missing_stock_valuation_idx,
                    f"首:{ohlc_day.head(1).index.get_level_values(level=0)[0]} 尾:{ohlc_day.tail(1).index.get_level_values(level=0)[0]}")
        else:
            missing_stock_valuation_idx = ohlc_day.index

        if len(ohlc_day.index.difference(
            ohlc_day.dropna(
                subset=[FLD.TURNOVER_RATE], axis=0, how="all"
            ).index
        )) > 0:
            # 检查日期缺乏今天
            missing_date = ohlc_day.index.difference(
                ohlc_day.dropna(
                    subset=[FLD.TURNOVER_RATE], axis=0, how="all"
                ).index
            ).get_level_values(level=0)[0]
            if pd.to_datetime(missing_date).date() != pd.to_datetime(GQ_util_get_last_day()).date():
                tag1 = len(ohlc_day.index.difference(
                    ohlc_day.dropna(
                        subset=[FLD.TURNOVER_RATE], axis=0, how="all"
                    ).index))
                missing_stock_valuation_idx = ohlc_day.index.difference(
                    ohlc_day.dropna(
                        subset=[FLD.TURNOVER_RATE], axis=0, how="all"
                    ).index)
                if (verbose):
                    print(
                        f'缺 {tag1} 天的换手率', missing_stock_valuation_idx,
                        f"首:{ohlc_day.head(1).index.get_level_values(level=0)[0]} 尾:{ohlc_day.tail(1).index.get_level_values(level=0)[0]}")
                GQ_SU_crawl_stock_valuation(
                    code=code,
                    start=f'{missing_stock_valuation_idx.get_level_values(level=0)[0]:%Y-%m-%d}', 
                    end=f'{missing_stock_valuation_idx.get_level_values(level=0)[-1]:%Y-%m-%d}',
                )
    stock_cn_snapshot = GQ_stock_a_spot_em()

    # 创建心跳监控实例
    module = HeartbeatModule(
        module_name="stock_min_aligned",
        instance_id=f"stock_min_aligned_{dt.now().strftime('%Y%m%d_%H%M%S')}",
        timeout_seconds=300,  # 5分钟超时
    )

    if (module.mutex(verbose=True)):
        return False
    else:
        # 开始模块执行记录
        module.start(
            initial_message="个股数据对齐：换手率"
        )

    #  登陆系统
    print('\n stock_min_aligned ——> get_missing_turnover \n')
    lg = bs.login()

    # 显示登陆返回信息
    if verbose:
        print('login respond error_code:'+lg.error_code)
        print('login respond  error_msg:'+lg.error_msg)

    # 获取全A股代码
    if (stock_cn_snapshot is not None):
        codelist_candidate_all = stock_cn_snapshot.rename(
            columns={
                '代码': AKA.CODE,
                '名称': AKA.NAME,
                '成交量': AKA.VOLUME,
                '成交额': AKA.AMOUNT,
                '换手率': FLD.TURNOVER_RATE,
                '最新价': 'price',
                '涨跌幅': FLD.PCT_CHANGE,
            }
        )
    if (stock_cn_snapshot is not None) and \
        (AKA.CODE not in codelist_candidate_all.columns):
        codelist_candidate_all[AKA.CODE] = codelist_candidate_all.index.get_level_values(level=1)

    if (stock_cn_snapshot is not None) and \
        (FLD.TURNOVER_RATE in codelist_candidate_all.columns) and \
        (1.0 <= codelist_candidate_all[FLD.TURNOVER_RATE].max() <= 100):
        codelist_candidate_all[FLD.TURNOVER_RATE] = codelist_candidate_all[FLD.TURNOVER_RATE].astype(np.float32) / 100
        
    if (stock_cn_snapshot is None) or \
        (len(codelist_candidate_all) < 5000):
        # 原为 `QA.QA_fetch_stock_list()` —— 而本文件 `:60` 早就 import 了本地
        # 实现 `GQ_fetch_stock_list`，只是这处没换过来。两者读同一个集合。
        codelist_candidate_all = GQ_fetch_stock_list()[AKA.CODE].to_list()
    else:
        codelist_candidate_all = codelist_candidate_all[AKA.CODE].to_list()

    codelist_candidate_chronicles = {}
    endcall = False
    sync_count = 0

    # 指定 从 哪年开始抓取数据，baostock数据最早只到1999年。
    for year in range(dt.now().year, (dt.now().year-1) if (((pd.to_datetime(dt.now().date()) - GQ_util_get_last_day()) < timedelta(days=1.68))) else 1999, -1):
        start_date = '{}-01-01'.format(year)
        end_date = '{}-01-02'.format(year + 1)
        patched_code = []
        
        # 查询日线
        if (pd.to_datetime(GQ_util_get_last_day()) > pd.to_datetime(end_date)):
            code_candidate = GQ_fetch_stock_list_day(start='{}-12-20'.format(year), end=end_date)
            if code_candidate is None:
                print(f"Date:{'{}-12-20'.format(year)} code_candidate is None, cannot get stock list")
                curr_stocklists = []
            else:
                curr_stocklists = [code for code in code_candidate.index.get_level_values(level=1).unique()]
        else:
            code_candidate = GQ_fetch_stock_list()
            curr_stocklists = [code for code in code_candidate.index.unique()]
        curr_exclude_code_candidate = list(filter(None, [code if (code not in curr_stocklists) else None for code in codelist_candidate_all]))
        
        def checkpoint_stock_paused(
            start_date, end_date,
            curr_stocklists,
            curr_exclude_code_candidate,
        ):
            code_candidate_checkpoints = GQ_fetch_stock_list_day(
                code=curr_exclude_code_candidate,
                start=start_date,
                end=end_date)
            if (code_candidate_checkpoints is not None) and \
                (len(code_candidate_checkpoints) > 0):
                # 年底存在停牌的股票
                code_candidate_paused = code_candidate_checkpoints.index.get_level_values(level=1).unique()
                curr_stocklists = list(set([*curr_stocklists,
                                            *code_candidate_paused,]))
                curr_exclude_code_candidate = list(filter(None, [code if (code not in curr_stocklists) else None for code in codelist_candidate_all]))
            else:
                code_candidate_paused = []
                
            return curr_stocklists, curr_exclude_code_candidate
                
        curr_stocklists, curr_exclude_code_candidate = checkpoint_stock_paused(
            start_date='{}-06-20'.format(year),
            end_date='{}-12-31'.format(year),
            curr_stocklists=curr_stocklists,
            curr_exclude_code_candidate=curr_exclude_code_candidate,)
        
        curr_stocklists, curr_exclude_code_candidate = checkpoint_stock_paused(
            start_date='{}-01-01'.format(year),
            end_date='{}-06-01'.format(year),
            curr_stocklists=curr_stocklists,
            curr_exclude_code_candidate=curr_exclude_code_candidate,)
                        
        print(f'{year}年度，A股个股数量{len(curr_stocklists)}，被排除{len(curr_exclude_code_candidate)}')
        try:
            with tqdm(curr_stocklists, unit='stock') as overall_progress:
                # for asset in curr_stocklists:
                start_date = '{}-01-01'.format(year)
                end_date = '{}-12-31'.format(year)
                if (pd.to_datetime(GQ_util_get_last_day()) < pd.to_datetime(end_date)):
                    end_date = f'{GQ_util_get_last_day():%Y-%m-%d}'
                for symbol in overall_progress:
                    overall_progress.set_description(u"A股({})".format(symbol))
                    overall_progress.update(1)
                    # GQ_fetch_stock_empirical_data(symbol, verbose=True)
                    overall_progress.set_postfix_str(f'检查 {symbol} {start_date[:7]}~{end_date[:7]}年度 K线')
                    # overall_progress.write(f'\r检查 {symbol} {start_date}~{end_date} 年度 K线数据对其情况....')
                    try:
                        recall = ckpo_align_stock_turnover_rate(
                            symbol,
                            start=start_date,
                            end=end_date,
                            verbose=False,
                        )
                        if (recall):
                            patched_code.append(symbol)
                            time.sleep(2.5)
                        if (pd.to_datetime(GQ_util_get_last_day()) < pd.to_datetime('{}-12-31'.format(year))):
                            get_missing_turnover(
                                code=symbol, 
                                data=None, 
                                offset='', 
                                verbose=False
                            )
                            time.sleep(0.2)
                    except Exception as e:
                        stock_info = GQ_fetch_stock_info([symbol[:6]])
                        if (stock_info is not None):
                            # 检查是否存在 ipo_date 字段
                            if 'ipo_date' in stock_info.columns and not stock_info['ipo_date'].isna().all():
                                # 获取第一个有效的 ipo_date
                                ipo_date_val = stock_info['ipo_date'].iloc[0] if len(stock_info) > 0 else None
                                
                                if ipo_date_val is not None:
                                    try:
                                        # 处理不同类型的 ipo_date 值
                                        if pd.isna(ipo_date_val):
                                            print(f"{symbol} ipo_date 为空值，跳过")
                                            time.sleep(0.3)
                                            continue
                                        
                                        # 转换为字符串处理
                                        ipo_date_str = str(ipo_date_val).strip()
                                        
                                        # 检查是否为有效的日期字符串（8位数字）
                                        if ipo_date_str == '0' or ipo_date_str == '0.0':
                                            print(f"{symbol} ipo_date 为0，视为无效数据，跳过")
                                            time.sleep(0.3)
                                            continue
                                        
                                        # 检查是否为8位数字
                                        if len(ipo_date_str) == 8 and ipo_date_str.isdigit():
                                            # 将 ipo_date 转换为 datetime 格式
                                            ipo_date = pd.to_datetime(ipo_date_str, format='%Y%m%d')
                                            current_date = pd.to_datetime('today')
                                            
                                            # 计算上市天数
                                            days_since_ipo = (current_date - ipo_date).days
                                            
                                            # 判断是否在90天以内
                                            if days_since_ipo <= 90:
                                                print(f"{symbol} 上市时间较短 ({days_since_ipo}天)，跳过详细错误处理")
                                                time.sleep(0.3)
                                            else:
                                                # 上市超过90天，执行错误处理
                                                overall_progress.write(f'\r检查 {symbol} 换手率失败....{e}')
                                                traceback.print_exc()
                                                time.sleep(1)
                                        else:
                                            # 不是有效的8位数字格式
                                            print(f"{symbol} ipo_date 格式无效: {ipo_date_str}，跳过")
                                            time.sleep(0.3)
                                            
                                    except Exception as date_error:
                                        print(f"{symbol} 处理 ipo_date 时出错: {date_error}，跳过")
                                        time.sleep(0.3)
                                else:
                                    print(f"{symbol} ipo_date 为 None，跳过")
                                    time.sleep(0.3)
                            else:
                                print(f"{symbol} 没有 ipo_date 字段或全部为空，跳过")
                                time.sleep(0.3)
                            continue
                        continue

                    # 心跳签到
                    if (sync_count % 18 == 1):
                        module.checkin(
                            message=f"处理A股换手率数据，当前时间: {year}年。"
                        )
                    if (sync_count % 832 == 1):
                        lg = bs.login()
                    sync_count = sync_count + 1
        except KeyboardInterrupt:
            overall_progress.close()
            endcall = True
            # 用户终止运行
            module.complete(
                completion_message="用户按键Ctrl+C终止运行"
            )
        overall_progress.close()

        if (endcall):
            break
        
        # 逐年更新上一年的活动股票池(数量递减中)
        codelist_candidate_chronicles[year] = curr_stocklists
        codelist_candidate_all = curr_stocklists
        print(f"Year: {year}, 数量：{len(codelist_candidate_chronicles[year])} {codelist_candidate_chronicles[year][-10:]}")
        codelist_candidate_year_pd = pd.DataFrame({
            'year': year,
            'code': curr_stocklists,
            'kline': 0,
            'valuation': 0,
            'review': 0, })

        print(f"Year: {year}, patched 数量：{len(patched_code)} {patched_code[-60:]}")

    # 交易日结束，标记模块完成
    module.complete(
        completion_message=f"Year: {year}, patched 数量：{len(patched_code)} {patched_code[-60:]}"
    )

    # 登出系统
    bs.logout()
