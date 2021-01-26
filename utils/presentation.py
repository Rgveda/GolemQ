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
这里定义字符界面输出进度条等UI互动元素
"""
try:
    from joblib import Parallel, delayed
    import joblib
    import multiprocessing
except:
    print('Some command line args "-c" & "--pct_of_cores" need to run "pip install joblib" before use it.')
    print('Some command line args "-c" & "--pct_of_cores" need to run "pip install multiprocessing" before use it.')
    pass
from tqdm import tqdm
import pandas as pd
import numpy as np
import contextlib
from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST,
    FEATURES as FTR,
)

@contextlib.contextmanager
def tqdm_joblib(tqdm_object, label_template, step=1):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.set_description(label_template.format(tqdm_object.iterable[tqdm_object.n]))
            if ((tqdm_object.n + step) < len(tqdm_object.iterable)):
                tqdm_object.update(n=step)  # self.batch_size
            else:
                tqdm_object.update(n=(len(tqdm_object.iterable) - tqdm_object.n))  # self.batch_size
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def pandas_display_formatter():
    """
    将pandas.DataFrame的打印效果设置为中文unicode优化
    """
    pd.set_option('display.float_format',lambda x : '%.3f' % x)
    pd.set_option('display.max_columns', 22)
    pd.set_option("display.max_rows", 300)
    pd.set_option('display.width', 220)  # 设置打印宽度
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)


def featurs_printable_formatter(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    格式化，浓缩打印信息
    """
    ret_features = features

    ret_features[AKA.PRICE_PRNT] = ret_features.apply(lambda x:"{:.2f}/{:.2%}/{:.2%}".format(x.at[AKA.CLOSE],
                                                                                      x.at[FLD.ZEN_PEAK_RETURNS] + x.at[FLD.ZEN_PEAK_RETURNS_MAJOR],
                                                                                      x.at[FLD.ZEN_DASH_RETURNS],), 
                                                      axis=1)
    ret_features[FTR.TOP_RANK_PRNT] = ret_features.apply(lambda x:"{:4d}/{:4d}/{:2d}/{:2d}/{:2d}".format(int(x.at[FLD.FULLSTACK_TIMING_LAG]),
                                                                                                         int(x.at[FLD.HALFSTACK_TIMING_LAG]),
                                                                                                         int(x.at[FTR.TOP_RANK]),
                                                                                                         int(x.at[FLD.BOLL_JX_RANK]),
                                                                                                         int(x.at[FLD.BOOTSTRAP_GROUND_ZERO_RANK]),), 
                                                         axis=1)
    ret_features[FTR.RANK_PRNT] = ret_features.apply(lambda x:"{:2d}/{:2d}".format(int(x.at[FLD.BOLL_JX_RANK]),
                                                                                   int(x.at[FLD.BOOTSTRAP_GROUND_ZERO_RANK]),), 
                                                         axis=1)
    #ret_features[FTR.STACK_PRNT] = ret_features.apply(lambda
    #x:"{:2d}/{:2d}/{:2d}/{:2d}".format(int(x.at[FLD.HALFSTACK_TIMING_LAG]),
    #                                                                                         int(x.at[FTR.TOP_RANK]),
    #                                                                                         int(x.at[FLD.BOLL_JX_RANK]),
    #                                                                                         int(x.at[FLD.BOOTSTRAP_GROUND_ZERO_RANK]),),
    #                                                     axis=1)
    if (FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG in features.columns):
        ret_features[FLD.ZEN_DASH_PRNT] = ret_features.apply(lambda x:"{:3d}/{:3d}/{:3d}/{:3d}/{:3d}/{}".format(int(x.at[FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG]),
                                                                                                 int(x.at[FLD.ZEN_DASH_TIMING_LAG]),
                                                                                                 int(x.at[FLD.ZEN_PEAK_TIMING_LAG]),
                                                                                                 int(x.at[FLD.ZEN_DASH_TIMING_LAG_MAJOR]),
                                                                                                 int(x.at[FLD.ZEN_PEAK_TIMING_LAG_MAJOR]),
                                                                                                 x.at[ST.ZEN_BUY3]), 
                                                               axis=1)
    else:
        ret_features[FLD.ZEN_DASH_PRNT] = ret_features.apply(lambda x:"{:3d}/{:3d}/{:3d}/{:3d}/{}".format(int(x.at[FLD.ZEN_DASH_TIMING_LAG]),
                                                                                              int(x.at[FLD.ZEN_PEAK_TIMING_LAG]),
                                                                                              int(x.at[FLD.ZEN_DASH_TIMING_LAG_MAJOR]),
                                                                                              int(x.at[FLD.ZEN_PEAK_TIMING_LAG_MAJOR]),
                                                                                              x.at[ST.ZEN_BUY3]), 
                                                               axis=1)
    ret_features[FLD.ZEN_BOOST_PRNT] = ret_features.apply(lambda x:"{:3d}/{:3d}/{:3d}/{:3d}".format(int(x.at[FTR.BOOTSTRAP_ANCHOR_TIMING_LAG]),
                                                                                              int(x.at[FLD.ZEN_BOOST_TIMING_LAG]),
                                                                                              int(x.at[FLD.ZEN_WAVELET_TIMING_LAG]),
                                                                                              int(x.at[FLD.BOLL_RAISED_TIMING_LAG])), 
                                                               axis=1)
    if (FLD.POLYNOMIAL9_TIMING_LAG in features.columns):
        ret_features[FTR.POLYNOMIAL9_PRNT] = ret_features.apply(lambda x:"{:2d}/{:3d}/{:2d}/{:2d}".format(int(x.at[FLD.POLYNOMIAL9_TIMING_LAG]),
                                                                                                          int(x.at[FLD.DEA_ZERO_TIMING_LAG]),
                                                                                                          int(x.at[FLD.MACD_ZERO_TIMING_LAG]),
                                                                                                          int(x.at[FTR.POLYNOMIAL9_EMPIRICAL]),), 
                                                            axis=1)

    if (FLD.BOLL_JX_RSI in features.columns):
        ret_features[FLD.RSI_PRNT] = ret_features.apply(lambda x:"{:.2f}/{:.2f}".format(x.at[FLD.BOLL_JX_RSI],
                                                                                    x.at[FLD.BOLL_JX_MAXFACTOR],), 
                                                    axis=1)
    if (FLD.REGTREE_TIMING_LAG in features.columns):
        ret_features[FLD.REGTREE_PRNT] = ret_features.apply(lambda x:"{:.2f}/{:3d}/{:3d}".format(x.at[FLD.REGTREE_CORRCOEF],
                                                                                                       int(x.at[FLD.REGTREE_TIMING_LAG]),
                                                                                                       int(x.at[FLD.BOLL_LB_HMA5_TIMING_LAG]),), 
                                                        axis=1)
    if (FLD.DRAWDOWN_RATIO_MAJOR in features.columns):
        ret_features[FLD.DRAWDOWN_PRNT] = ret_features.apply(lambda x:"{:.2f}/{:.2f}/{:.2f}".format(x.at[FLD.DRAWDOWN_RATIO_MAJOR],
                                                                                                x.at[FLD.DRAWDOWN_RATIO],
                                                                                                x.at[FLD.DRAWDOWN_RATIO_MAJOR] + x.at[FLD.DRAWDOWN_RATIO] + x.at[FLD.MAPOWER30_MAJOR],), 
                                                                        axis=1)
    if (FLD.POLY9_REGTREE_DIVERGENCE in features.columns):
        ret_features[FLD.DIVERGENCE_PRNT] = ret_features.apply(lambda x:"{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(x.at[FLD.POLY9_REGTREE_DIVERGENCE],
                                                                                                         x.at[FLD.PEAK_OPEN],
                                                                                                         x.at[FLD.POLY9_MA30_DIVERGENCE],
                                                                                                         x.at[FLD.POLY9_MA90_DIVERGENCE],), 
                                                                        axis=1)

    if (FLD.PREDICT_PROB_LONG in features.columns):
        ret_features[FLD.PREDICT_PRNT] = ret_features.apply(lambda x:"{:.2f}/{}".format(x.at[FLD.PREDICT_PROB_LONG],
                                                                                        x.at[ST.PREDICT_LONG] if np.isnan(x.at[ST.PREDICT_LONG]) else int(x.at[ST.PREDICT_LONG])), 
                                                                        axis=1)
    return ret_features