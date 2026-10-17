# coding:utf-8
"""ChartScan —— **CV 图形识别**（YOLOv8 扫 K 线图找形态），从 `Y:\\Projects\\GolemAI` 搬入。

| 模块 | 干什么 |
|:--|:--|
| `render.py` | K 线 → 蜡烛图 PNG（**只进内存**）；含 `calculate_i` 的像素↔bar 标定 |
| `model.py` | YOLO 权重路径 + 惰性加载 |
| `predict.py` | 取数（8.3）→ 出图 → 推理 → 写 `*_reality_feat_*` |
| `server.py` | ZeroMQ REP 服务端（多端口 = 每端口一个线程，各载一份模型） |
| `client.py` | ZeroMQ REQ 客户端（按 scope 解析出任务、逐条发） |

⚠️ **跑得起来的只有 `GolemAI` conda env** —— 那里才有 torch / ultralytics /
mplfinance（而 CV 要 torch，GolemQ env 是 NoMKL，两者 BLAS 冲突 ⇒ 只能分环境）。
故本包**模块级不 import 任何重依赖**（都在函数体里），这样 `import GolemQ...chartscan`
在 GolemQ env 下也成立 —— 只是调 `load_model()` / 出图会抛
`ModuleNotFoundError`，**那是预期的**（在错的 env 里本就不该能跑 CV）。

条目（**首选 CLI，它会自动切 env**，见 `DECISIONS.md` **D51** 五之二）：
    python -m GolemQ.cli --server ChartScan --port 6666
    python -m GolemQ.cli --scan   ChartScan --eval fast --port 6666

也可以直接跑模块（那时要**自己先** `conda activate GolemAI`）：
    python -m GolemQ.markets.StockCN.chartscan.server --port 6666
    python -m GolemQ.markets.StockCN.chartscan.client --eval fast --port 6666
"""
from .model import load_model, weights_path
from .predict import (GQ_markup_ChartScaned, GQ_markup_daily_ChartScaned,
                      GQ_predict_ChartScanAI, calc_ChartScanAI_reality,
                      calc_ChartScaned_persistence)
from .render import GQ_generate_chart_stock, calculate_i

__all__ = [
    'weights_path', 'load_model',
    'GQ_generate_chart_stock', 'calculate_i',
    'GQ_predict_ChartScanAI', 'calc_ChartScaned_persistence', 'calc_ChartScanAI_reality',
    'GQ_markup_ChartScaned', 'GQ_markup_daily_ChartScaned',
]
