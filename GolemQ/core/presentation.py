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
    import joblib
except Exception:
    print('joblib not installed.')
    pass
from tqdm import tqdm
import pandas as pd
import contextlib
import os


@contextlib.contextmanager
def tqdm_joblib(tqdm_object, label_template, step=1):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            try:
                tqdm_object.set_description(label_template.format(tqdm_object.iterable[tqdm_object.n]))
            except Exception:
                # print(len(tqdm_object.iterable), tqdm_object.n)
                pass
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


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def pandas_display_formatter():
    """
    将pandas.DataFrame的打印效果设置为中文unicode优化
    """
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    # pd.options.display.float_format = '{:.2%}'.format
    pd.set_option('display.max_columns', 22)
    pd.set_option("display.max_rows", 300)
    pd.set_option('display.width', 220)  # 设置打印宽度
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)


def stock_length_less_than_required(
        code, ohlc_data,
        log_msg='',
        verbose=False,
        ret_ohlc=False,
        ret_meta=False,):
    """
    停止运行，处理错误股票代码，提交给 RabbitMQ 或者记录到日志文件
    """
    if (isinstance(verbose, tqdm)):
        verbose.write(log_msg)
    elif (verbose):
        print(log_msg)
    if (ret_meta):
        return None, None, None
    elif (ret_ohlc):
        return None, None
    else:
        return None

