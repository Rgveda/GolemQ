# coding:utf-8
"""`EPOCH_DIVISOR` —— datetime → 秒的**唯一换算系数**（`DECISIONS.md` D52）。

pandas 3.0 把默认 `datetime64` 精度从 **ns** 改成 **µs**（本机 3.0.6 实测），
于是 `series.astype(np.int64) // 10 ** 9` **静默**少 1000 倍。

本用例钉三件事：

1. 那个系数确实**从 pandas 实际精度**推出来（不是写死、不是猜版本号字符串）；
2. `epoch_seconds` 的值对（两代 pandas 同结果）；
3. **全树不再出现写死的 `// 10 ** 9`** —— 这是「全系统检查」的常驻版：
   扫的是 **token 流**（跳过注释与字符串），所以讲这个坑的注释/docstring
   不会把它自己扫出来。
"""
from __future__ import annotations

import io
import re
import tokenize
import unittest
from pathlib import Path

from GolemQ.core.constants import EPOCH_DIVISOR, epoch_divisor


class TestTheConstantItself(unittest.TestCase):
    def test_divisor_matches_live_pandas_unit(self):
        """系数必须等于「当前 pandas 默认精度」的倒数 —— 不是某个写死的数。"""
        import numpy as np
        import pandas as pd
        unit = np.datetime_data(pd.to_datetime(['2000-01-01']).dtype)[0]
        expect = {'s': 1, 'ms': 10 ** 3, 'us': 10 ** 6, 'ns': 10 ** 9}[unit]
        self.assertEqual(EPOCH_DIVISOR, expect,
                         'pandas 默认 datetime64 是 {!r}，系数应为 {}'.format(unit, expect))

    def test_divisor_is_positive_int(self):
        self.assertIsInstance(EPOCH_DIVISOR, int)
        self.assertGreater(EPOCH_DIVISOR, 0)


class TestPerDtypeLookup(unittest.TestCase):
    """系数**按 dtype 查**，不是一个全局标量 —— 这条是本次的核心教训。"""

    def test_all_units(self):
        for dtype, want in (('datetime64[s]', 1),
                            ('datetime64[ms]', 10 ** 3),
                            ('datetime64[us]', 10 ** 6),
                            ('datetime64[ns]', 10 ** 9)):
            with self.subTest(dtype=dtype):
                self.assertEqual(epoch_divisor(dtype), want)

    def test_tz_aware_dtype(self):
        import pandas as pd
        dtype = pd.DatetimeIndex(['2026-01-05']).tz_localize('Asia/Shanghai').dtype
        self.assertEqual(epoch_divisor(dtype), 10 ** 6)

    def test_second_resolution_input_is_handled(self):
        """⚠️ 就是这条当初红的：`pd.to_datetime(date 对象)` 给 `datetime64[s]`。

        拿 "pandas 默认精度" 的 1e6 去除它 ⇒ `date_stamp` 变成 `1767`（实测）。
        """
        from datetime import date
        import pandas as pd
        from GolemQ.markets.StockCN.kline83 import epoch_seconds
        idx = pd.to_datetime(pd.Index([date(2026, 1, 5)]))
        self.assertEqual(str(idx.dtype), 'datetime64[s]')
        s = pd.Series(idx).dt.tz_localize('Asia/Shanghai')
        self.assertEqual(int(epoch_seconds(s).iloc[0]), 1767542400)


class TestEpochSecondsUsesIt(unittest.TestCase):
    def test_known_anchor(self):
        """北京 15:00 → 07:00Z = 1767596400（与 4.4 存量实测同值）。"""
        import pandas as pd
        from GolemQ.markets.StockCN.kline83 import epoch_seconds
        s = pd.Series(pd.to_datetime(['2026-01-05 15:00'])).dt.tz_localize('Asia/Shanghai')
        self.assertEqual(int(epoch_seconds(s).iloc[0]), 1767596400)

    def test_accepts_datetimeindex(self):
        import pandas as pd
        from GolemQ.markets.StockCN.kline83 import epoch_seconds
        idx = pd.DatetimeIndex(['2026-01-05 15:00']).tz_localize('Asia/Shanghai')
        self.assertEqual(int(epoch_seconds(idx).iloc[0]), 1767596400)

    def test_seconds_are_seconds_not_microseconds(self):
        """钉住**量级**：天数换算出来的秒必须落在 10 位数量级（不是 13 位）。"""
        import pandas as pd
        from GolemQ.markets.StockCN.kline83 import epoch_seconds
        s = pd.Series(pd.to_datetime(['2026-01-05'])).dt.tz_localize('Asia/Shanghai')
        value = int(epoch_seconds(s).iloc[0])
        self.assertEqual(len(str(abs(value))), 10, value)


def _code_tokens(path):
    """把源码压成 **token 串**，**跳过注释与字符串**。

    用 tokenize 而不是"按 `#` 切行"：本项目注释里大量出现被禁的写法
    （正是这个坑的说明），按行切会把它们自己扫出来；而字符串里也可能有。
    """
    text = path.read_text(encoding='utf-8')
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL,
                        tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            continue
        out.append(tok.string)
    return ' '.join(out)


#: 写死的「纳秒」除数。允许 `EPOCH_DIVISOR`（常量）与 `10 ** 3/6`（毫秒/微秒域）。
_HARDCODED_NS = re.compile(r'//\s*10\s*\*\*\s*9\b|//\s*1000000000\b|//\s*1e9\b')


class TestNoHardcodedNanosecondDivisor(unittest.TestCase):
    """全树守卫：`datetime → 秒` 只准走 `EPOCH_DIVISOR`。"""

    def test_tree_is_clean(self):
        root = Path(__file__).resolve().parents[1]      # GolemQ/ 包根
        offenders = []
        for path in sorted(root.rglob('*.py')):
            if 'test_cases' in path.parts:               # 本文件自己就带着这个正则
                continue
            code = _code_tokens(path)
            for match in _HARDCODED_NS.finditer(code):
                offenders.append('{}: …{}…'.format(
                    path.relative_to(root.parent),
                    code[max(0, match.start() - 40):match.end() + 20]))
        self.assertEqual(offenders, [],
                         '写死了纳秒除数（pandas 3 起会静默少 1000 倍）—— '
                         '改用 `constants.EPOCH_DIVISOR`：\n' + '\n'.join(offenders))


if __name__ == '__main__':
    unittest.main()
