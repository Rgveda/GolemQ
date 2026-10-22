# coding: utf-8
r"""ChartScan（YOLOv8 买卖点识别）**Streamlit 演示** —— 新树 / MongoDB 8.3 版

运行方式（⚠️ **必须在 GolemAI env** —— torch / ultralytics 只装在那儿）：

    conda activate GolemAI
    streamlit run examples/app_chartscan.py

参考 `Y:\Projects\ChartScanAI\app.py`（那份用 yfinance 取美股），本版把取数换成
**新树的 8.3 读口**，并把「生成 → 推理 → 画框」合成**一步**：点一次「识别」，
左栏是模型看到的原图，右栏是**框好的图**，下面是每个框的明细。

取数链路（**与线上扫描完全同一条**）：

    GolemQ.get_active_market().get_kline_price_v3 / get_kline_price_min   # 市场实例
        → markets/StockCN/kline83.py                                      # 8.3 时序集合，已前复权
    analysis/cv/render.GQ_generate_chart_stock                            # **与推理时同一个出图函数**
    analysis/cv/model.load_model().predict                                # YOLOv8
    analysis/cv/predict.calculate_i                                       # 像素 x → 第几根 bar

⚠️ **为什么坚持用 `GQ_generate_chart_stock` 出图**，而不是在 app 里另画一份：
YOLO 的像素 x 靠 `calculate_i` 映射回 bar（`(x-386)/6.5-1`），那两个常数只在
`figsize=(18, 6.5)` + `dpi=100` + `chunk_size=180` 下成立。**另画一份 = 坐标静默挪位**，
模型照样给出「可信」的框。

⚠️ **本 app 不落库**（只做可视化）。落库那条链是 `--server` / `--scan ChartScan`。
"""
import importlib.util
import os
import sys
from datetime import datetime, timedelta
from io import BytesIO

# --------------------------------------------------------------------------- #
# 让它能在「没装包、直接从源码跑」的情况下找到 GolemQ
# --------------------------------------------------------------------------- #
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import streamlit as st

from GolemQ import get_active_market
from GolemQ.analysis.cv.predict import CONFIDENCE, calculate_i
from GolemQ.analysis.cv.render import (
    CHUNK_SIZE, CLOSE_TIMES, GQ_generate_chart_stock,
)
from GolemQ.analysis.cv.model import weights_path
from GolemQ.markets.StockCN.reality_save import (
    UNIQUE_KEYS, fetch_reality_feat, reality_feat_name,
)
from GolemQ.markets.StockCN.symbol import is_stock_cn

#: 频率 → reality 族。三族**各有自己的列名**（搬自旧树，别再"统一"——
#: 8.3 里的历史数据就是按这套存的，见 `reality_save.CV_COLUMNS`）。
FAMILIES = ('day', '60min', '15min')
CV_FIELDS = {
    'day':   ('ClsMaj', 'ConfidenceMaj', 'CoordinatesMaj'),
    '60min': ('Class', 'Confidence', 'Coordinates'),
    '15min': ('ClsMin', 'ConfidenceMin', 'CoordinatesMin'),
}
#: 取多少自然日（与 `predict.GQ_predict_ChartScanAI` 的历史窗口同量级）。
LOOKBACK_DAYS = {'day': 512, '60min': 126, '15min': 63}

#: 推理能力只在 GolemAI env 有。**不用 `import ultralytics` 来判** —— 那会真的去载
#: torch（本 env 没有），直接崩在模块导入期。`find_spec` 只看"找不找得到"。
HAS_YOLO = importlib.util.find_spec('ultralytics') is not None


# --------------------------------------------------------------------------- #
# 取数（8.3）
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=600)
def load_kline(code: str, family: str):
    """8.3 的 K 线（已前复权）→ `(DataFrame|None, 展示名)`。"""
    market = get_active_market()
    start = (datetime.now() - timedelta(days=LOOKBACK_DAYS[family])).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    if family == 'day':
        res, name = market.get_kline_price_v3(code, start=start, end=end,
                                              realtime=True, if_fq='qfq', verbose=False)
    else:
        res, name = market.get_kline_price_min(code, start=start, end=end,
                                               frequency=family, realtime=True,
                                               if_fq='qfq', verbose=False)
    data = getattr(res, 'data', None)
    if data is None or len(data) == 0:
        return None, name
    return data, name


@st.cache_data(show_spinner=False, ttl=600)
def load_detections(code: str, family: str, days: int):
    """库里**已有**的 CV 检出（稀疏：只有识别到买卖点的那几根才有值）。"""
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    frame = fetch_reality_feat(code, start, end, family, is_stock_cn(code)[1])
    if frame is None:
        return pd.DataFrame()
    cols = [c for c in CV_FIELDS[family] if c in frame.columns]
    if not cols:
        return pd.DataFrame()
    return frame[cols].dropna(how='all')


@st.cache_resource(show_spinner='载入 YOLOv8 权重…')
def get_model():
    """进程内只载一次（52 MB）。`st.cache_resource` 跨 rerun 持有。"""
    from GolemQ.analysis.cv.model import load_model
    return load_model()


@st.cache_data(show_spinner='推理中…', ttl=600)
def detect(png: bytes, conf: float):
    """PNG 字节 → `(标注框后的 PNG 字节, 框列表)`。

    ⚠️ 框是 ultralytics 的 `res[0].plot()` 画的（**与线上同一套**），
    不是我们另画一遍 —— 另画只会和线上不一致。
    """
    from PIL import Image
    image = Image.open(BytesIO(png))
    res = get_model().predict(image, conf=conf)
    names, boxes = res[0].names, res[0].boxes
    rows = []
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        rows.append({'方向': names[int(b.cls[0])],
                     '置信度': round(float(b.conf[0]), 4),
                     'x1': round(x1, 1), 'y1': round(y1, 1),
                     'x2': round(x2, 1), 'y2': round(y2, 1)})
    buf = BytesIO()
    Image.fromarray(res[0].plot()[:, :, ::-1]).save(buf, format='PNG')
    return buf.getvalue(), rows


def _coords_to_time(values):
    """`Coordinates*` 是 **UTC 毫秒**；换成可读时间（非数字留 NaT）。"""
    return pd.to_datetime(pd.to_numeric(values, errors='coerce') / 1000,
                          unit='s', errors='coerce')


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #
st.set_page_config(page_title='ChartScanAI · 8.3', page_icon='📊', layout='wide',
                   initial_sidebar_state='expanded')

with st.sidebar:
    st.header('⚙️ 参数')
    code = st.text_input('股票代码（8.3）', value='600519',
                         help='6 位 A 股代码，如 600519 / 000001 / 510300')
    family = st.selectbox('K 线周期', FAMILIES, index=0,
                          help='day → ClsMaj 那套；60min → Class 那套；15min → ClsMin 那套')
    conf = float(st.slider('置信度阈值', 25, 100, int(CONFIDENCE * 100))) / 100
    run = st.button('🔍 识别', width='stretch', type='primary')
    st.caption(f'权重：`{weights_path()}`')
    if not HAS_YOLO:
        st.error('当前 env 没有 `ultralytics` —— 只能出图、不能画框。\n\n'
                 '```bash\nconda activate GolemAI\nstreamlit run examples/app_chartscan.py\n```')

st.title('📊 ChartScanAI · 8.3')
st.caption('YOLOv8 在 K 线图上找**买卖点形态**。取数与出图都走 GolemQ 新树 / MongoDB 8.3。')

# 一次「识别」= 取数 → 出图 → 推理 → 画框
if run:
    kline, name = load_kline(code, family)
    if kline is None:
        st.session_state.pop('result', None)
        st.error(f'`{code}` 在 `{family}` 上没有取到 K 线 —— 换只标的或换个频率试试。')
    else:
        label = code if name == code else f'{name}（{code}）'
        try:
            buf = GQ_generate_chart_stock(kline=kline, ticker=f'{name}({code}) ',
                                          chunk_size=CHUNK_SIZE, freq=family)
        except Exception as exc:                  # noqa: BLE001 演示里要给人看原因
            buf = None
            st.error(f'出图失败：{type(exc).__name__}: {exc}')
        if buf is not None:
            png = buf.getvalue()
            buf.close()
            result = {'code': code, 'family': family, 'label': label,
                      'png': png, 'kline': kline, 'conf': conf,
                      'detected': None, 'boxes': []}
            if HAS_YOLO:
                try:
                    result['detected'], result['boxes'] = detect(png, conf)
                except Exception as exc:          # noqa: BLE001
                    st.error(f'推理失败：{type(exc).__name__}: {exc}')
            st.session_state['result'] = result

res = st.session_state.get('result')

if res is None:
    st.info('👈 在侧边栏填代码 → 点 **识别**。左边是模型看到的图，右边是框好的图。')
    st.stop()

# --------------------------------------------------------------------------- #
# ① 原图 vs ② 识别框
# --------------------------------------------------------------------------- #
c1, c2 = st.columns(2)
with c1:
    st.subheader('① 模型看到的图')
    st.image(res['png'], width='stretch')
with c2:
    st.subheader('② 识别框')
    if res['detected']:
        st.image(res['detected'], width='stretch')
    elif HAS_YOLO:
        st.warning('这一次没有画出框（推理失败或没有检出）。')
    else:
        st.warning('当前 env 没有 `ultralytics`，无法画框 —— 见侧边栏的启动命令。')

st.caption(f"{res['label']} · {res['family']} · 最后 {CHUNK_SIZE} 根 · "
           f"`figsize=(18, 6.5)` / `dpi=100` —— YOLO 的像素 x 就按这套标定换算成第几根 bar")

# 下载（原图 + 框好的图都给）
d1, d2, _ = st.columns([1, 1, 3])
with d1:
    st.download_button('⬇️ 原图', data=res['png'],
                       file_name=f"{res['code']}_{res['family']}_raw.png",
                       mime='image/png', width='stretch')
with d2:
    if res['detected']:
        st.download_button('⬇️ 框好的图', data=res['detected'],
                           file_name=f"{res['code']}_{res['family']}_detected.png",
                           mime='image/png', width='stretch')

# --------------------------------------------------------------------------- #
# ③ 框明细
# --------------------------------------------------------------------------- #
st.subheader('③ 框明细')
if not res['boxes']:
    st.caption('没有检出（模型在这个阈值下没找到买卖点形态）。' if HAS_YOLO else '未运行推理。')
else:
    ts = res['kline'].tail(CHUNK_SIZE).index.get_level_values(0)
    rows = []
    for b in res['boxes']:
        i = calculate_i(b['x2'])
        rows.append({**b,
                     '窗口内第几根': i,
                     '对应时刻': ts[i] if -len(ts) <= i < len(ts) else None})
    table = pd.DataFrame(rows).sort_values('窗口内第几根')
    st.dataframe(table, width='stretch', hide_index=True)
    st.caption('⚠️「像素 x2 → 第几根」按 `calculate_i`（`(x-386)/6.5-1`）算，'
               '只在上面那套 `figsize/dpi/chunk_size` 标定下成立。')

# --------------------------------------------------------------------------- #
# ④ 库里已有检出（与本次推理对照）
# --------------------------------------------------------------------------- #
st.subheader('④ 库里已有的检出（8.3）')
st.caption(f'集合 `{reality_feat_name(res["family"], is_stock_cn(res["code"])[1])}` · '
           f'唯一键 `({", ".join(UNIQUE_KEYS[res["family"]])})` · '
           '⚠️ 检出**稀疏**，不是每根 bar 都有')

days = st.slider('回看天数', 30, 730, 180, step=30)
det = load_detections(res['code'], res['family'], days)
if det.empty:
    st.info(f'最近 {days} 天没有该标的的检出。')
else:
    cls_col, conf_col, coord_col = CV_FIELDS[res['family']]
    table = pd.DataFrame({
        '检出时刻': _coords_to_time(det[coord_col]) if coord_col in det else pd.NaT,
        '方向': det.get(cls_col),
        '置信度': pd.to_numeric(det.get(conf_col), errors='coerce'),
        '所在 bar': det.index.get_level_values(level=0),
    }).sort_values('所在 bar', ascending=False)
    st.dataframe(table, width='stretch', hide_index=True)
    st.caption('⚠️「检出时刻」(`Coordinates*`) 是 **UTC 毫秒**；「所在 bar」是文档的键'
               '（日族 `date_stamp` / 分钟族 `time_stamp`，**秒**）—— 单位不同，别直接比。')

if res['family'] != 'day':
    st.caption(f'60min 族只写收盘那几根（{", ".join(CLOSE_TIMES)}）；'
               '15min 的检出会**同时**往日族写一笔（`ClsMin`/`CoordinatesMin`）—— '
               '那是 `calc_ChartScaned_persistence` 判断「要不要重扫」的依据，不是重复写入。')

# --------------------------------------------------------------------------- #
# ⑤ 线上那条链（本 app 不落库）
# --------------------------------------------------------------------------- #
st.subheader('⑤ 线上扫描（落库）')
st.code(
    '# 服务端（多端口 = 多线程，每线程各载一份 52MB 权重）\n'
    'python -m GolemQ.cli --server ChartScan --port 5555\n\n'
    '# 客户端：按 scope 解析待扫任务，逐条 REQ\n'
    'python -m GolemQ.cli --scan ChartScan --eval fast --port 5555\n\n'
    '# scope: fast / block / etf / etc / etc_1 / etc_2\n'
    '# 一键起 1 服务端 + 4 客户端（Windows Terminal 标签页）：\n'
    'powershell -ExecutionPolicy Bypass -File chartscan.ps1',
    language='bash',
)
st.caption('⚠️ 两条 CLI 命令在 **GolemQ env** 里敲即可 —— 它们会用 GolemAI 的 python '
           '重起子进程（本 app 则**必须**直接跑在 GolemAI env）。见 `DECISIONS.md` D51。')
