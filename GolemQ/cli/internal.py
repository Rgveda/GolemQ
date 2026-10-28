# coding:utf-8
"""**内部/调试入口** —— 用户入口不暴露的旗子只在这里。

    python -m GolemQ.cli.internal --review --eval fast --review-limit 5
    python -m GolemQ.cli.internal --migrate-metadata-merge --migrate-limit 1000

用户入口是 ``python -m GolemQ.cli``；它**不注册** `--review-limit` /
`--migrate-limit`（见 `commands/_internal.py`）。这样做是为了让 `--help`
只列用户真正会用的参数，避免「为了测试方便」的旗子被当成正常操作。
"""
from .__main__ import main

if __name__ == '__main__':
    main(internal=True)
