# coding: utf-8
r"""ChartScan（YOLOv8 买卖点识别）**Streamlit 演示** —— 新树 / MongoDB 8.3 版

运行方式（⚠️ **必须在 GolemAI env** —— torch / ultralytics 只装在那儿）：

    conda activate GolemAI
    streamlit run examples/app_chartscan.py

参考 `Y:\Projects\ChartScanAI\app.py`（那份用 yfinance 取美股），本版把取数换成
**新树的 8.3 读口**，其余交互照旧：生成 K 线图 → 喂给模型 → 画出检出框。

取数链路（**与线上扫描完全同一条**）：

    GolemQ.get_active_market().get_kline_price_v3 / get_kline_price_min   # 市场实例
        → markets/StockCN/kline83.py                                      # 8.3 时序集合，已前复权
    chartscan/render.GQ_generate_chart_stock                              # **与推理时同一个出图函数**
    chartscan/model.load_model().predict                                  # YOLOv8
    chartscan/predict.calculate_i                                         # 像素 x → 第几根 bar

⚠️ **为什么坚持用 `GQ_generate_chart_stock` 出图**，而不是在 app 里另画一份：
YOLO 的像素 x 靠 `calculate_i` 映射回 bar（`(x-386)/6.5-1`），那两个常数只在
`figsize=(18, 6.5)` + `dpi=100` + `chunk_size=180` 下成立。**另画一份 = 坐标静默挪位**，
模型照样给出「可信」的框。上传**任意**图片时这条映射无意义，所以 app 只在
「自己生成的图」上显示 bar 映射。

⚠️ 本演示**不落库**（只做可视化）。要看落库的那条链，用
`python -m GolemQ.cli --server/--scan ChartScan`（见页面底部）。
"""
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
from GolemQ.markets.StockCN.chartscan.predict import CONFIDENCE, calculate_i
from GolemQ.markets.StockCN.chartscan.render import (
    CHUNK_SIZE, CLOSE_TIMES, GQ_generate_chart_stock,
)
from GolemQ.markets.StockCN.chartscan.model import load_model, weights_path
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
    return load_model()


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
    st.header('⚙️ 生成图表')
    code = st.text_input('股票代码（8.3）', value='600519',
                         help='6 位 A 股代码，如 600519 / 000001 / 510300')
    family = st.selectbox('K 线周期', FAMILIES, index=0,
                          help='day → ClsMaj 那套；60min → Class 那套；15min → ClsMin 那套')
    gen = st.button('生成图表', width='stretch')

    st.write('')
    st.subheader('🖼️ 上传图片检测')
    source_img = st.file_uploader('上传一张 K 线图…',
                                  type=('jpg', 'jpeg', 'png', 'bmp', 'webp'))

    confidence = float(st.slider('模型置信度阈值', 25, 100,
                                 int(CONFIDENCE * 100))) / 100
    detect = st.button('🎯 检测', width='stretch')
    st.caption(f'权重：`{weights_path()}`')

# 生成图：结果放进 session_state，rerun 后还在
if gen:
    if not code:
        st.sidebar.error('请输入代码。')
    else:
        kline, name = load_kline(code, family)
        if kline is None:
            st.session_state.pop('chart', None)
            st.error(f'`{code}` 在 `{family}` 上没有取到 K 线 —— 换只标的或换个频率试试。')
        else:
            label = code if name == code else f'{name}({code})'
            try:
                buf = GQ_generate_chart_stock(kline=kline, ticker=f'{label} ',
                                              chunk_size=CHUNK_SIZE, freq=family)
            except Exception as exc:              # noqa: BLE001 演示里要给人看原因
                buf = None
                st.error(f'出图失败：{type(exc).__name__}: {exc}')
            if buf is not None:
                st.session_state['chart'] = {
                    'png': buf.getvalue(), 'code': code, 'family': family,
                    'label': label, 'kline': kline,
                }
                buf.close()
                st.success(f'已生成 {label} · {family} · 最后 {CHUNK_SIZE} 根')

st.title('📊 ChartScanAI · 8.3')
st.caption('YOLOv8 在 K 线图上找**买卖点形态**。取数与出图都走 GolemQ 新树 / MongoDB 8.3；'
           '线上那条落库链路见页面底部。')

chart = st.session_state.get('chart')
col1, col2 = st.columns(2)

with col1:
    if source_img:
        st.image(source_img, caption='上传的图片', width='stretch')
    elif chart:
        st.image(chart['png'], caption=f"{chart['label']} · {chart['family']} · "
                                       f"最新 {CHUNK_SIZE} 根", width='stretch')
        st.download_button('下载这张图', data=chart['png'],
                           file_name=f"{chart['code']}_{chart['family']}_"
                                     f"{CHUNK_SIZE}.png", mime='image/png')

# 检测：优先用上传的图（与参考实现一致），否则用刚才生成的那张
target = None
if source_img:
    target = ('upload', source_img)
elif chart:
    target = ('chart', chart)

if detect:
    if target is None:
        st.error('先上传一张图，或在侧边栏点「生成图表」。')
    elif 'ultralytics' not in sys.modules:
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            st.error('检测需要 `ultralytics` + `torch` —— 它们只在 **GolemAI** env 里。\n\n'
                     '```bash\nconda activate GolemAI\nstreamlit run examples/app_chartscan.py\n```')
            st.stop()

    kind, payload = target
    try:
        from PIL import Image
        if kind == 'upload':
            payload.seek(0)
            image = Image.open(payload)
            mapping_ok = False
        else:
            image = Image.open(BytesIO(payload['png']))
            mapping_ok = True
        res = get_model().predict(image, conf=confidence)
    except Exception as exc:                      # noqa: BLE001
        st.error(f'推理失败：{type(exc).__name__}: {exc}')
        st.stop()

    with col2:
        st.image(res[0].plot()[:, :, ::-1], caption='检测结果', width='stretch')

    boxes = res[0].boxes
    with st.expander('检测明细', expanded=True):
        if len(boxes) == 0:
            st.info('没有检出（模型在这个阈值下没找到买卖点形态）。')
        else:
            rows = []
            for box in boxes:
                *_, x2, _y2 = [float(v) for v in box.xyxy[0]]
                row = {
                    '方向': res[0].names[int(box.cls[0])],
                    '置信度': round(float(box.conf[0]), 4),
                    '像素 x2': round(x2, 1),
                }
                if mapping_ok:
                    i = calculate_i(x2)
                    row['窗口内第几根'] = i
                    ts = payload['kline'].tail(CHUNK_SIZE).index.get_level_values(0)
                    row['对应时刻'] = ts[i] if -len(ts) <= i < len(ts) else None
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            if mapping_ok:
                st.caption('⚠️「像素 x2 → 第几根」是按 `figsize=(18, 6.5)` / `dpi=100` / '
                           '`chunk_size=180` 这套标定算的（`calculate_i`）。上传任意图片时'
                           '这条映射无意义，故不显示。')

# --------------------------------------------------------------------------- #
# 库内已有检出（与本次推理对照）
# --------------------------------------------------------------------------- #
if chart:
    st.subheader('📚 库里已有的检出（8.3）')
    st.caption(f'集合 `{reality_feat_name(chart["family"], is_stock_cn(chart["code"])[1])}` · '
               f'唯一键 `({", ".join(UNIQUE_KEYS[chart["family"]])})` · '
               f'⚠️ 检出**稀疏**，不是每根 bar 都有')

    days = st.slider('回看天数', 30, 730, 180, step=30)
    det = load_detections(chart['code'], chart['family'], days)
    if det.empty:
        st.info(f'最近 {days} 天没有该标的的检出。')
    else:
        cls_col, conf_col, coord_col = CV_FIELDS[chart['family']]
        table = pd.DataFrame({
            '检出时刻': (_coords_to_time(det[coord_col])
                       if coord_col in det else pd.NaT),
            '方向': det.get(cls_col),
            '置信度': pd.to_numeric(det.get(conf_col), errors='coerce'),
            '所在 bar': det.index.get_level_values(level=0),
        }).sort_values('所在 bar', ascending=False)
        st.dataframe(table, width='stretch', hide_index=True)
        st.caption('⚠️「检出时刻」(`Coordinates*`) 是 **UTC 毫秒**；「所在 bar」是文档的键'
                   '（日族 `date_stamp` / 分钟族 `time_stamp`，**秒**）—— 单位不同，别直接比。')

    if chart['family'] != 'day':
        st.caption(f'60min 族只写收盘那几根（{", ".join(CLOSE_TIMES)}）；'
                   '15min 的检出会**同时**往日族写一笔（`ClsMin`/`CoordinatesMin`）—— '
                   '那是 `calc_ChartScaned_persistence` 判断「要不要重扫」的依据，不是重复写入。')

# --------------------------------------------------------------------------- #
# 线上那条链（本 app 不落库）
# --------------------------------------------------------------------------- #
st.subheader('🚀 线上扫描（落库）')
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
