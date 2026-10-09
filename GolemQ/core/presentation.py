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
import datetime
import unicodedata
import io
import os
import sys


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


# ---------------------------------------------------------------------------
# 状态 banner —— 整条流水线的固定表头 + 表头之下的一块滚动状态窗
# ---------------------------------------------------------------------------
# ⚠️ 这是全树**唯一**一处 ANSI 转义定义。别处不要再起一套颜色常量。
# 背景：本次之前包内**没有任何** ANSI / 颜色 / TTY 检测（全树
# `colorama|termcolor|isatty` 零命中，唯一命中在 `.claude/helpers/*.sh` 那些
# shell 脚本里）。所以规则在这里一次定清：**只在真 TTY 上上色，非 TTY 一律不**。
_ANSI_GRAY = '\033[90m'
_ANSI_RED = '\033[31m'
_ANSI_GREEN = '\033[32m'
_ANSI_YELLOW = '\033[33m'
_ANSI_WHITE = '\033[97m'
_ANSI_RESET = '\033[0m'

#: 节点**三态**（用户 2026-10-09 定）—— 取数 banner（`--save`）用：
#:   * ``PENDING`` 队列中   —— 小点 ``·`` 灰
#:   * ``RUNNING`` 正在读取 —— 实点 ``●`` 绿
#:   * ``DONE``    读取完成（**含「未过期、本次不重取」**）—— 实点 ``●`` 白
#: **状态只由这个点表达**，节点名与阶段名都不上色、不挂文字。
PENDING = 'pending'
RUNNING = 'running'
DONE = 'done'

#: 自检 banner（`cli/bootstrap.py`）另加**三态**，与上面共用同一套渲染：
#:   * ``OK``   通过   —— 实点 ``●`` 绿
#:   * ``WARN`` 警告   —— 实点 ``●`` 黄（**不致命**：TTY 缺失、显式覆盖了线程上限…）
#:   * ``FAIL`` 失败   —— 实点 ``●`` 红
#:
#: ⚠️ 绿色的 ``RUNNING``（取数：正在读取）与 ``OK``（自检：通过）**同色不同义**。
#: 这不是笔误也不是冗余：两个 banner 从不同屏（自检的那块 `close` 完，取数的才开），
#: 而各自色系内部的语义是自洽的 —— 取数用「绿=在动」，自检用「绿=没问题」。
OK = 'ok'
WARN = 'warn'
FAIL = 'fail'

_MARKS = {
    PENDING: ('·', _ANSI_GRAY),
    RUNNING: ('●', _ANSI_GREEN),
    DONE:    ('●', _ANSI_WHITE),
    OK:      ('●', _ANSI_GREEN),
    WARN:    ('●', _ANSI_YELLOW),
    FAIL:    ('●', _ANSI_RED),
}

#: **无颜色时的符号**（非 TTY / `color=False`）。
#:
#: 取数三态（`·` / `●` / `●`）本来就不靠颜色也能读；**自检的四态不行** ——
#: 通过 / 警告 / 失败在高亮下是 绿/黄/红 三个同形的 `●`，一旦没有颜色就完全同形，
#: 而**日志正是出事后唯一能翻的东西**（用户 2026-10-09 的口径：「非TTY用符号标识」）。
#: 所以这里给后两态各换一个符号，与 :data:`_MARKS` 一一对应。
_PLAIN_MARKS = {
    PENDING: '·',
    RUNNING: '●',
    DONE:    '●',
    OK:      '●',
    WARN:    '!',
    FAIL:    '✗',
}

#: 表头静态行的时刻戳格式（`--save` 与自检共用）。
STAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

#: 表头之下那块滚动状态窗最多留几行。长跑时只看得见最近几条，
#: :meth:`Banner.close` 会把攒下的**全部**倒出来。
ECHO_WINDOW = 3

#: 同一个阶段行里，节点之间的分隔。
_NODE_SEP = '  '


def _vt_supported():
    """Windows 控制台是否**真的会解释** ANSI 转义（VT）；能打开就顺手打开。

    ⚠️ 这条是**必须的**，不是锦上添花：`isatty()` 为真**不代表**终端认 ANSI ——
    Windows 10 的 conhost 默认**没开** `ENABLE_VIRTUAL_TERMINAL_PROCESSING`，
    于是 `\\033[2A`（光标上移）不被执行、被当成普通字符或忽略，**每次重画都变成追加**
    —— 屏上就会出现两条 `source: pytdx`（2026-10-09 用户实报）。
    非 Windows 一律 True。
    """
    if sys.platform != 'win32':
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False                             # 不是控制台（已被重定向）
        enable_vt = 0x0004
        if mode.value & enable_vt:
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | enable_vt))
    except Exception:      # noqa: BLE001 开不了就当不支持，退回追加模式
        return False


def ansi_enabled(stream=None):
    """颜色/光标控制开关：**只对真 TTY 且真的支持 ANSI 时打开**。

    管道 / 重定向 / CI 日志里必须关掉 —— 否则转义码混进文本，`grep` 与日志
    切分都会看到 `^[[32m` 这种东西。Windows 上还要额外确认 VT（见 :func:`_vt_supported`）。

    >>> ansi_enabled(io.StringIO())          # 非 TTY
    False
    """
    stream = sys.stdout if stream is None else stream
    try:
        if not stream.isatty():
            return False
    except Exception:      # noqa: BLE001 有些包装过的 stream 没有 isatty
        return False
    return _vt_supported()


def display_width(text):
    """终端**显示宽度** —— 中日韩字符占两列。只为把阶段名 / 组名对齐。

    >>> display_width('K线')
    3
    >>> display_width('abcd')
    4
    """
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)


def _ljust(text, width):
    """按**显示宽度**左对齐补齐（`str.ljust` 只数字符个数，中文会错位）。"""
    return text + ' ' * max(0, width - display_width(text))


#: 阶段名那一列的**显示宽度**。跨 banner 共享 —— `环境自检`（自检那块）与
#: `参考数据`（取数那块）必须落在同一列上，否则屏上会参差。
#: 8 = `环境自检` 与 `参考数据` 的显示宽度，两者恰好相等不是巧合（都是四个汉字）。
PHASE_WIDTH = 8


def aligned_row(label, body, width=PHASE_WIDTH):
    """``阶段名`` + 值 的一行，与 :func:`render_pipeline_banner` 的阶段列**同宽**。

    用于那些「不是节点、没有状态点」的行 —— 目前只有 `数据源`：

    >>> aligned_row('数据源', 'pytdx')
    '数据源    pytdx'
    """
    return '{}  {}'.format(_ljust(label, width), body)


def stamp_text(when=None):
    """裸的时刻戳：``[2026-10-09 15:34:43]``。

    >>> import datetime as _dt
    >>> stamp_text(_dt.datetime(2026, 10, 9, 15, 12, 57))
    '[2026-10-09 15:12:57]'
    """
    when = datetime.datetime.now() if when is None else when
    return '[{}]'.format(when.strftime(STAMP_FORMAT))


def stamp(text, when=None):
    """给**静态头行**盖时刻戳：``[2026-10-09 15:34:43]: source: pytdx``。

    ⚠️ 只盖在**静态**的东西上（每个 banner 打一次的那行），**不盖**在会重画的
    状态行上 —— 那些行每次重画都会被重写，戳上去只会让「变化规则」失真
    （用户 2026-10-09 定：只管头行）。

    :param when: 可注入的时刻（测试用固定值；`None` = `datetime.now()`）。
        传 `datetime` 以外的对象只要它有 `strftime` 也行。

    >>> import datetime as _dt
    >>> stamp('bootstrap', _dt.datetime(2026, 10, 9, 15, 12, 57))
    '[2026-10-09 15:12:57]: bootstrap'
    """
    return '{}: {}'.format(stamp_text(when), text)


def dim(text, color=False):
    """**压暗**一行（`color=True` 时）。用 :data:`_ANSI_GRAY`，非 TTY 原样返回。

    ⚠️ **压暗只此一处** —— 身份行、阶段的起止行都走它。散着写 `'\\033[90m'`
    就会在"哪几行该压暗"上分叉（用户 2026-10-10 一次点了三处）。
    """
    if not color:
        return text
    return '{}{}{}'.format(_ANSI_GRAY, text, _ANSI_RESET)


def stamp_done(caption, when=None, color=False):
    """**收尾行**：``[2026-10-10 00:33:22]: bootstrap done.``（``color=True`` 时压暗）。

    与 :func:`stamp` 是**一对**：阶段开头打 ``[t]: bootstrap``，结束打这一行 ——
    于是日志里这个阶段**可检索**（起止各一行），而不是只有一堆状态行。

    ⚠️ **打它的那一刻 banner 必须已经 `close()`** —— 否则会被算进它的行数记账、
    下一次重画整块写花（`PITFALLS.md` P22）。

    >>> import datetime as _dt
    >>> stamp_done('bootstrap', _dt.datetime(2026, 10, 10, 0, 33, 22))
    '[2026-10-10 00:33:22]: bootstrap done.'
    """
    return dim(stamp('{} done.'.format(caption), when), color)


def identity(app='GolemQ', contact='', when=None, color=False):
    """**身份块** —— 每次运行开头那一小块，末尾带一个空行。

    ```
    GolemQ  [2026-10-09 17:29:39]
    Copyright (c) 2018-2026 azai/Rgveda/GolemQ(uant) | https://github.com/Rgveda | 知乎@阿财

    ```

    **为什么要这块**（2026-10-09 用户反馈"四行看着眼花"）：改版前是
    `版权行` + `[戳]: bootstrap` + 自检行 + `[戳]: source: pytdx` 四行，
    毛病有三：**两个戳紧挨着**（像同一件事说两遍）、**四种行同字重**（没有层次）、
    **数据源挤成第二行头**而不是一列。借的是 `terraform` / `gh` 的**身份块 + 空行分层**：
    名称与时刻一行、版权一行、**空行**、然后才是运行信息。

    ⚠️ 时刻戳从此**只出现一次**（在身份行上）—— 它标的是**这一次运行**，
    不是每个阶段；阶段行不再各挂一个。

    :param color: **把版权/联系行压暗**（只在真 TTY 上传 `True` —— 由调用方按
        :func:`ansi_enabled` 决定，本函数是纯的）。

        用 :data:`_ANSI_GRAY`（亮黑）而不是 SGR 的 ``\\033[2m``（faint）：两者观感同类，
        但 `90` 在 Windows conhost / 各种管道终端上渲染得**更可靠**，而 `2` 常被忽略成
        普通字重。也与本文件里 `·` 那个灰点**同一个灰** —— 颜色只该有一处定义。
    :param app: 产品名（``GolemQ``）
    :param contact: 版权/联系行（`cli/bootstrap.copyright_infos`）
    :param when: 可注入的时刻，`None` = 现在
    """
    lines = [dim('{}  {}'.format(app, stamp_text(when)), color)]
    if contact:
        lines.append(dim(contact, color))
    return '\n'.join(lines) + '\n\n'


def render_pipeline_banner(rows, states, color=True):
    """把「各阶段」渲染成多行表头（**纯函数**）。

    纯函数不看环境：上不上色由调用方（:class:`Banner`，按 TTY 定）传进来，
    这样才好测。

    ⚠️ **不含 `source:` 那一行**（2026-10-09）。那是**静态**的、永远不变，
    而本函数的输出会被**反复重画** —— 把它放进重画块里，任何一处重画没被终端
    正确执行（或输出被重定向/被日志捕获）都会让它**一行行堆出来**。
    它由 :meth:`Banner.render` **在画 banner 之前打一次**，之后再不碰。

    :param rows: 每行 ``(阶段名, 组名或 None, [节点键, ...])``。**连续同阶段名的
        行只在第一行打阶段名**，其余留白对齐（下例的 ``K线`` 两行）。组名不为
        ``None`` 时，节点显示成去掉组名前缀的短名（``stock_day`` → ``day``）。
    :param states: ``{节点键: 三态}``（:data:`PENDING` / :data:`RUNNING` / :data:`DONE`）；
        **没提到的节点算** :data:`PENDING`
    :param color: 是否给状态点着色。`False` 时按 :data:`_PLAIN_MARKS` 取符号 ——
        非 TTY 下**四态必须靠符号区分**（见那张表的说明）

    ⚠️ **状态只由那个点表达** —— 只给**点**上色，节点名与阶段名一律不上色、
    也不挂「已获取/未获取」文字。

    >>> rows = [('参考数据', None, ['stock_list', 'stock_info']),
    ...         ('K线', 'stock', ['stock_day', 'stock_1min']),
    ...         ('K线', 'index', ['index_day'])]
    >>> print(render_pipeline_banner(rows,
    ...       {'stock_list': DONE, 'stock_info': PENDING, 'stock_day': RUNNING},
    ...       color=False))
    参考数据  stock_list ●  stock_info ·
    K线       stock  day ●  1min ·
              index  day ·
    """
    phase_w = max([display_width(r[0]) for r in rows] + [0])
    group_w = max([display_width(r[1]) for r in rows if r[1]] + [0])

    lines = []
    prev_phase = None
    for phase, group, keys in rows:
        head = _ljust(phase, phase_w) if phase != prev_phase else ' ' * phase_w
        prev_phase = phase

        nodes = []
        for key in keys:
            label = key
            if group and key.startswith(group + '_'):
                label = key[len(group) + 1:]
            symbol, ansi = _MARKS.get(states.get(key), _MARKS[PENDING])
            if color:
                symbol = '{}{}{}'.format(ansi, symbol, _ANSI_RESET)
            else:
                symbol = _PLAIN_MARKS.get(states.get(key), _PLAIN_MARKS[PENDING])
            nodes.append('{} {}'.format(label, symbol))

        body = (_ljust(group, group_w) + '  ' if group else '') + _NODE_SEP.join(nodes)
        lines.append((head + '  ' + body).rstrip())

    return '\n'.join(lines)


class Banner:
    """整条流水线的**固定表头** + 表头之下的一块滚动状态窗，自己管光标行数原地重画。

    **为什么自己管光标**（而不是 `tqdm.write`）：终端没有「固定表头 + 下方滚动日志」
    这种东西 —— 表头一旦被下方的输出顶上去，按行数回退就会**插进别人的行里**。
    所以 banner 必须**独占自己那块区域**：它记得上一次画了几行（表头 + 状态窗），
    重画时精确回退那么多行再整体重写。

    ⚠️ **推论：banner 活跃期间，下方的一切输出都必须走 :meth:`echo`。**
    直接 `print` 会多出一行而 banner 不知道，下次重画的回退量就少 1 → 整块错位。
    生产侧（`refdata_save` / `kline_save`）因此都提供 `echo=` 形参，默认 `print`。

    TTY：表头 + 最近 ``echo_window`` 条状态原地重画；:meth:`close` 把攒下的
    **全部**状态行倒出来（否则三小时的长跑结束只剩最近三条）。
    非 TTY（管道/日志）：不上色、不重画，一切按原本顺序 `print` —— 日志里是全量、
    有序的，`close` 不补打。
    """

    def __init__(self, source, rows, stream=None, echo_window=ECHO_WINDOW,
                 caption=None, stamp_line=True, when=None, header=True):
        """:param caption: 静态头行的**全文**。给了就用它（自检 banner 传
            ``caption='bootstrap'``），不给则维持 ``source: {source}``（取数 banner
            的 ``source: pytdx`` 是记在文档与用户口径里的，不能改写）。
        :param stamp_line: 静态头行要不要盖时刻戳（见 :func:`stamp`）。
        :param when: 盖戳用的时刻，`None` = 现在。**只影响头行那一行**。
        :param header: ``False`` = **不打任何静态头行**，只有阶段行。

            CLI 用它的场景（2026-10-09 改版）：时刻戳与产品名由
            :func:`identity` 在**运行开头打一次**，阶段行（`环境自检` /
            `数据源` / `参考数据` / `K线` / `复权`）**跨两个 banner 对齐成一列**。
            若两个 banner 各打各的头行，屏上就是两条 `[戳]: …` 紧挨着 ——
            用户口径「像同一件事说了两遍」。

            默认 `True` 保持旧行为（单测与直接调用方不受影响）。
        """
        self._source = source
        self._caption = caption
        self._stamp_line = bool(stamp_line)
        self._when = when
        self._header = bool(header)
        self._rows = [(p, g, list(keys)) for p, g, keys in rows]
        self._stream = stream
        self._color = ansi_enabled(stream)
        self._echo_window = max(0, int(echo_window))
        self._states = {str(k): PENDING for _, _, keys in self._rows for k in keys}
        self._log = []           # 攒下的全部状态行
        self._drawn = 0          # 上一次画了几行（表头 + 状态窗）；0 = 还没画过

    @property
    def states(self):
        """当前状态快照（排障/测试用）。"""
        return dict(self._states)

    @property
    def log(self):
        """攒下的全部状态行（`close` 后仍可读；测试与排障用）。"""
        return list(self._log)

    def _write(self, text):
        out = sys.stdout if self._stream is None else self._stream
        out.write(text)
        flush = getattr(out, 'flush', None)
        if flush is not None:
            flush()

    def _header_lines(self):
        return render_pipeline_banner(self._rows, self._states,
                                      color=self._color).split('\n')

    def _paint(self, lines):
        for line in lines:
            self._write(line + '\033[K\n')      # 清掉本行残余 + 给 tqdm 留一行
        self._drawn = len(lines)

    def _draw(self):
        if not self._color:
            # 非 TTY：表头**只打一次**（= 这次要做哪些活儿），之后每次变化补一行
            # `  名字 符号`。**不再整块重打** —— 26 个节点 × 7 行表头 = 182 行噪声，
            # 而日志真正需要的只是「走到哪了」。
            if not self._drawn:
                for line in self._header_lines():
                    self._write(line + '\n')
                self._drawn = 1              # 只用来标记「表头已打过」
            return

        lines = self._header_lines()
        if self._echo_window:
            lines += self._log[-self._echo_window:]
        if self._drawn:                          # 回退到区块首行
            self._write('\033[{}A\r'.format(self._drawn))
        self._paint(lines)

    def render(self):
        """首次落盘：先打**一次**静态头行，再画表头（全部未获取）。

        ⚠️ 静态头行 **刻意不进重画块**（理由见 :func:`render_pipeline_banner`）。
        用户 2026-10-09 实报过连着 **9 行** `source: pytdx` —— 就是它跟着重画块
        反复输出的结果。它静态、永不变化，打一次就够，之后谁也别再碰它。

        头行 = `caption`（没给就是 `source: {source}`），再按 :func:`stamp` 盖时刻戳。
        ``header=False`` 时**一个头行都不打**（见构造器里那条说明）。
        """
        if self._header:
            head = ('source: {}'.format(self._source)
                    if self._caption is None else self._caption)
            if self._stamp_line:
                head = stamp(head, self._when)
            # 头行**压暗**（用户 2026-10-10：`[t]: bootstrap` 深灰）—— 只在真 TTY 上；
            # 非 TTY 那一行是**日志**，掺转义码是本项目颜色规则的第一条禁忌。
            self._write(dim(head, self._color) + '\n')
        self._draw()

    def mark(self, name, state=DONE):
        """把节点置为某个状态并重画。取值见 :data:`_MARKS` ——
        取数用 :data:`PENDING` / :data:`RUNNING` / :data:`DONE`，
        自检用 :data:`PENDING` / :data:`OK` / :data:`WARN` / :data:`FAIL`。
        传表里没有的状态**抛 `ValueError`**（别静默吞掉拼错的状态名）。

        不在表头里的名字直接忽略 —— 调用方可能传了被 `--save-collections` 之类
        缩掉、因而没进 `rows` 的集合。
        """
        if name not in self._states:
            return
        if state not in _MARKS:
            raise ValueError('未知状态 {!r}（应为 {}）'.format(state, sorted(_MARKS)))
        if self._states[name] == state:
            return
        self._states[name] = state
        if self._color:
            self._draw()
        else:
            # 非 TTY：一行一条，符号即状态（表头已在 `render()` 打过一次）。
            # ⚠️ 这里用 `_PLAIN_MARKS` 而不是 `_MARKS[..][0]` —— 四态后两态在高亮下
            # 是 黄/红 两个同形的 `●`，落到日志里必须换成 `!` / `✗` 才分得出来。
            self._write('  {} {}\n'.format(name, _PLAIN_MARKS[state]))

    def echo(self, text):
        """在状态窗里加一行 —— **banner 活跃时，下方的一切输出都走这里**。

        ⚠️ 直接 `print` 会让 :attr:`_drawn` 与实际屏上行数对不上，下一次重画整块错位。
        """
        text = str(text)
        self._log.append(text)
        if self._color:
            self._draw()
        else:
            self._write(text + '\n')             # 非 TTY：按原顺序全量打出

    def close(self):
        """收尾：TTY 下换成「表头 + 全部状态行」，把滚动窗里被挤掉的补回来。

        非 TTY 无事可做 —— 最后一次 `_draw` / `echo` 已经把终态打在日志里了。
        """
        if not self._color:
            return
        if self._drawn:
            self._write('\033[{}A\r'.format(self._drawn))
        self._paint(self._header_lines() + self._log)
        self._drawn = 0



