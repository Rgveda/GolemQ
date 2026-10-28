# coding:utf-8
"""**仅内部/调试用**的 CLI 参数 —— **不注册到用户入口**。

为什么要分开
============
用户 2026-10-23 指出：不少参数其实是「为了测试/试跑方便」加进来的，
它们不该出现在 `python -m GolemQ.cli --help` 里 —— 用户看到就会以为
那是正常操作的一部分，误用（如 `--review-limit 10`）会**只写一部分数据**。

规则
====
* **用户入口**：`python -m GolemQ.cli` —— 只注册 `commands/*.py` 的
  `add_arguments`，**不 include 本模块**；
* **内部入口**：`python -m GolemQ.cli.internal` —— 额外注册这里的旗子；
* 测试默认**直接调 `run_*(args)` / pipeline 函数**，不要为了小样本去加旗子；
  只有专门测 argparse 契约的用例才碰 parser。

目前收在这里的（用户 2026-10-23/24：**具体故障由 AI 智能体/测试排查，用户不再用**）：

* 所有 `--save*` 调参/试跑旗子：collections / start / turnover-days / margin-days /
  progress-every / jobs / no-adj / refresh / codes / targets / frequencies / dry-run；
* `--review-limit` / `--migrate-limit`；review 的 `--dry-run` / `--recheck-days`；
* `--moneyflow-dry-run` / `--migrate-dry-run`；
* `--fill-turnover-codes/start/end`。

**整条命令**（`Command.internal_only=True`）：`--save-coverage` / `--save-status` /
`--metadata-align` / `--review-runtime` / 全部 `--migrate-*`。

它们在用户 parser 里**不存在**，因此 `run_*` 一律用
``getattr(args, 'xxx', default)`` 读取。
"""
from __future__ import annotations

#: 只该出现在 internal parser 里的 dest。测试据此守「用户入口不暴露」。
INTERNAL_DESTS = ('review_limit', 'migrate_limit',
                  'save_collections', 'save_start', 'save_turnover_days',
                  'save_margin_days', 'save_progress_every', 'save_jobs',
                  'save_no_adj', 'save_refresh',
                  'save_codes', 'save_targets', 'save_frequencies',
                  'save_dry_run', 'moneyflow_dry_run', 'migrate_dry_run',
                  'fill_turnover_codes', 'fill_turnover_start',
                  'fill_turnover_end', 'dry_run', 'recheck_days')


def add_internal_arguments(parser) -> None:
    """把内部旗子注册到 parser —— **只由 `build_parser(internal=True)` 调用**。"""
    parser.add_argument('--review-limit', dest='review_limit', type=int, default=0,
                        help='[internal] 只跑前 N 只（调试/分批）。')
    parser.add_argument('--migrate-limit', dest='migrate_limit', type=int, default=None,
                        help='[internal] 只搬前 N 行（试跑用；不传 = 全量）。')
    parser.add_argument('--save-collections', dest='save_collections',
                        help='[internal] 限定要保存的参考集合，逗号分隔。')
    parser.add_argument('--save-start', dest='save_start', default='2015-01-01',
                        help='[internal] 分钟线首次灌库的起点。')
    parser.add_argument('--save-turnover-days', dest='save_turnover_days',
                        type=int, default=7,
                        help='[internal] 换手率增量刷新的回溯自然日数。')
    parser.add_argument('--save-margin-days', dest='save_margin_days',
                        type=int, default=0,
                        help='[internal] 增量窗口往前多算几个交易日。')
    parser.add_argument('--save-progress-every', dest='save_progress_every',
                        type=int, default=100,
                        help='[internal] 每处理 N 个代码打一行进度。')
    parser.add_argument('--save-jobs', dest='save_jobs', type=int, default=4,
                        help='[internal] 并发线程数。')
    parser.add_argument('--save-no-adj', dest='save_no_adj',
                        action='store_true', default=False,
                        help='[internal] 不重算 {stock,etf}_adj。')
    parser.add_argument('--save-refresh', dest='save_refresh',
                        action='store_true', default=False,
                        help='[internal] 忽略参考数据/复权的刷新闸，一律重取。')
    parser.add_argument('--save-codes', dest='save_codes',
                        help='[internal] 限定代码（试跑/排障用），逗号分隔。')
    parser.add_argument('--save-targets', dest='save_targets',
                        help='[internal] 限定标的族：stock,index,etf。')
    parser.add_argument('--save-frequencies', dest='save_frequencies',
                        help='[internal] 限定频率：day,1min,5min,15min,30min,60min。')
    parser.add_argument('--save-dry-run', dest='save_dry_run', action='store_true',
                        default=False,
                        help='[internal] 只取数不写库，与库内重叠窗口逐字段对拍。')
    parser.add_argument('--moneyflow-dry-run', dest='moneyflow_dry_run',
                        action='store_true', default=False,
                        help='[internal] 资金流/千股千评只取数统计，不写库。')
    parser.add_argument('--migrate-dry-run', dest='migrate_dry_run',
                        action='store_true', default=False,
                        help='[internal] 迁移只读 4.4 并统计，不写 8.3。')
    parser.add_argument('--fill-turnover-codes', dest='fill_turnover_codes',
                        default=None,
                        help='[internal] fill-turnover 的 6 位代码，逗号分隔。')
    parser.add_argument('--fill-turnover-start', dest='fill_turnover_start',
                        default=None, help='[internal] fill-turnover 左边界。')
    parser.add_argument('--fill-turnover-end', dest='fill_turnover_end',
                        default=None, help='[internal] fill-turnover 右边界。')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                        default=False,
                        help='[internal] review 只算不写（不写行、不写水位）。')
    parser.add_argument('--recheck-days', dest='recheck_days', type=int,
                        default=0,
                        help='[internal] review 水位前溯 N 天。')
