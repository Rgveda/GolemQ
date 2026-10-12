# coding:utf-8
"""ChartScan 的 **YOLO 权重路径与惰性加载**。

为什么单独一个模块
================
权重路径的老实现用的是 `GolemAI` 那棵树的 `cache_path('models', date_subfix=None)`，
它内部 `basepath = os.getcwd()` —— 于是 `.ps1` **必须先 `CD` 到项目根**，否则找不到
权重。这里改成**按文件位置**推仓库根，与 cwd 无关。

⚠️ `ultralytics` 只在 **GolemAI** env 里（CV 要 torch，而 GolemQ env 是 NoMKL ⇒
两边的 BLAS 冲突，故 CV 只能在 GolemAI env 跑）。所以本模块**模块级不 import 它** ——
`load_model()` 里才 import，这样 `GolemQ/markets/StockCN/chartscan/` 在 GolemQ env 下
也 import 得动（只是调 `load_model` 会炸，那是预期的：在错的 env 里就是不该能跑 CV）。
"""
from __future__ import annotations

import os

__all__ = ['WEIGHTS_ENV_VAR', 'weights_path', 'load_model']

#: 覆盖权重路径的环境变量（不想把 52MB 放进仓库时用这个指到别处）。
WEIGHTS_ENV_VAR = 'GOLEMQ_CHARTSCAN_WEIGHTS'

#: 仓库内相对路径：`<仓库根>/datastore/models/weights/custom_yolov8.pt`。
#: ⚠️ `datastore/` 在 `.gitignore` 里 —— 权**不入库**。
_WEIGHTS_REL = os.path.join('datastore', 'models', 'weights', 'custom_yolov8.pt')


def _repo_root() -> str:
    """仓库根 = 本文件 `GolemQ/markets/StockCN/chartscan/model.py` 往上 **4** 层。

    **不用 `os.getcwd()`** —— 老实现用了，于是脚本必须先 `CD` 到项目根，
    换个工作目录就静默找不到权重。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))


def weights_path() -> str:
    """YOLO 权重路径。解析顺序：

    1. ``$GOLEMQ_CHARTSCAN_WEIGHTS``（显式覆盖，**不校验存在性** —— 给了就该用它）
    2. ``<仓库根>/datastore/models/weights/custom_yolov8.pt``

    >>> import os as _os
    >>> p = weights_path()
    >>> _os.path.basename(p)
    'custom_yolov8.pt'
    >>> _os.path.basename(_os.path.dirname(_os.path.dirname(p)))
    'models'
    """
    override = os.environ.get(WEIGHTS_ENV_VAR)
    if override:
        return override
    return os.path.join(_repo_root(), _WEIGHTS_REL)


def load_model(model=None):
    """取一个 YOLO 模型：给了 `model` 就原样返回，否则**惰性 import** ultralytics 载入。

    ⚠️ **本函数只在 GolemAI env 里能用**（`ultralytics` 不在 GolemQ env）。
    见模块 docstring。
    """
    if model is not None:
        return model
    from ultralytics import YOLO            # noqa: PLC0415  只在 GolemAI env 有
    return YOLO(weights_path())
