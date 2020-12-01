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
import contextlib
            
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