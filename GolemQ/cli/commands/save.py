# coding:utf-8
"""`--save <SOURCE>`：参考数据 + K 线 + xdxr/adj 的主流水线（`DECISIONS.md` D11/D14/D15/D16）。

本模块**同时拥有整个 `--save*` 参数家族**（12 个）—— 包括只读命令
（`--save-coverage` / `--save-status`）也会读的 `--save-targets` /
`--save-frequencies` / `--save-no-adj`。参数只在一处注册，别的命令读 `args` 即可。

这里只做**校验 + 委托**；取数逻辑全在 `markets/StockCN/{refdata_save,kline_save}.py`。
⚠️ 唯一「越界」的是 banner —— 但那是 `D14` 明确定的：「banner 由 CLI 构造后**注入**，
数据层不认识任何 UI」。
"""
from __future__ import annotations

import sys

from ._registry import Command, usage_error


def add_save_arguments(parser) -> None:
    parser.add_argument('--save-collections',
                        help="限定要保存的参考集合，逗号分隔；默认 = 该源能供的全部"
                             "（tdx/pytdx：stock_list,stock_info,stock_block,etf_list；"
                             "qmt：前三者）",
                        type=str,
                        default=None)

    # ⚠️ `choices=` 是**校验的唯一出处**（用户 2026-10-09）—— 别在 `run_save` 里
    # 再手写一遍 `if value not in (...)`：那样报错是「错误: ...」+ exit 1，
    # 而 argparse 给的是标准 usage + `invalid choice`（退出码 2），
    # 且可选值会自动进 `--help`。两处都写 = 两个真相源。
    # `type=str.lower` 让 `--save TDX` 照旧能用（老代码是 `.lower()`），
    # 它在 choices 校验**之前**生效。
    parser.add_argument('--save', nargs='?', const='tdx', default=None,
                        type=str.lower, choices=('tdx', 'pytdx', 'qmt'),
                        metavar='SOURCE',
                        help="保存 A 股行情到 MongoDB 8.3。tdx 与 pytdx **等价**："
                             "参考数据（stock_list/stock_info/stock_block/etf_list，"
                             "pytdx 侧、不碰 akshare）+ stock/index/etf 的 "
                             "day 与 1/5/15/30/60min + stock_xdxr（含 stock_adj 重算）；"
                             "qmt 只做参考数据（走 qmt 源），K 线只有 pytdx 供得了故跳过。"
                             "不带值时等同 tdx")

    parser.add_argument('--save-start', default='2015-01-01',
                        help="分钟线**首次**灌库的起点（库里没有该 code 时才用），默认 2015-01-01")

    parser.add_argument('--save-margin-days', type=int, default=0,
                        help="增量窗口往前多算几个交易日（默认 0 = 只覆盖水位当天那一根）。"
                             "调大只在「源端回补了更早历史」的修补场景有用，"
                             "代价是每只票每天多删多写一根")

    parser.add_argument('--save-progress-every', type=int, default=100,
                        help="每处理 N 个代码打一行进度（0 = 只打首尾），默认 100")

    parser.add_argument('--save-jobs', type=int, default=4,
                        help="并发线程数（= 同时打开的 pytdx 连接数），默认 4")

    parser.add_argument('--save-targets',
                        help="限定标的族，逗号分隔：stock,index,etf；默认全部")

    parser.add_argument('--save-frequencies',
                        help="限定频率，逗号分隔：day,1min,5min,15min,30min,60min；默认全部")

    parser.add_argument('--save-codes',
                        help="限定代码（试跑/排障用），逗号分隔；默认按标的名单全量")

    parser.add_argument('--save-dry-run', action='store_true', default=False,
                        help="**只取数不写库**，并与库内重叠窗口逐字段对拍（映射表的验收手段）")

    parser.add_argument('--save-no-adj', action='store_true', default=False,
                        help="不重算 {stock,etf}_adj（前复权因子）。默认对**事件变化**的 code 整条重算")

    parser.add_argument('--save-refresh', action='store_true', default=False,
                        help="忽略参考数据的**刷新闸**，本次一律重取。默认按上次**成功完成**"
                             "的时刻算：盘中 5 小时内 / 盘后 24 小时内不重取（压掉重复的 "
                             "HTTP 查询）。起算点是「上次成功完成」而非「上次启动」")


def run_save(args) -> None:
    # 取值合法性由 `add_save_arguments` 的 `choices=` 保证（argparse 会先报
    # `invalid choice` 并 exit 2），**这里不再重复校验**。
    value = args.save

    def _split(v):
        return [x.strip() for x in v.split(',') if x.strip()] if v else None

    targets = _split(args.save_targets)
    frequencies = _split(args.save_frequencies)
    codes = _split(args.save_codes)
    ref_collections = _split(args.save_collections)

    # 收盘行情下载是**长跑**：Ctrl-C 与数据形状异常（KeyError 等）都该给人话，
    # 不把堆栈砸到用户脸上。⚠️ `KeyboardInterrupt` **不是 `Exception` 子类**，
    # 必须单独捕获 —— 否则 Ctrl-C 照样打一堆 traceback。
    try:
        from GolemQ.core.presentation import (Banner, DONE, PENDING, RUNNING,
                                              aligned_row, ansi_enabled, dim, stamp,
                                              stamp_done)
        from GolemQ.markets.StockCN.kline_save import (
            FREQUENCIES as KLINE_FREQS,
            TARGETS as KLINE_TARGETS,
            allow_xdxr_shortcircuit,
            kline_sweep_age_hours,
            mark_kline_sweep,
            save_kline_tdx,
            save_adj,
            save_xdxr_tdx,
            target_collection_name,
        )
        from GolemQ.markets.StockCN.refdata_save import (
            ALL_REF_COLLECTIONS,
            REFDATA_BY_SOURCE,
            format_status,
            refdata_ttl_hours,
            save_refdata,
        )

        # 源名决定整条流程。**K 线只有 pytdx 供得了**（`kline_save` 全模块只走
        # tdx 适配器），故非 tdx 侧一律跳过 K 线/xdxr/adj，只做参考数据。
        tdx_like = value in ('tdx', 'pytdx')

        # 参考数据：只做该源**真正能供**的集合。表在 refdata_save.REFDATA_BY_SOURCE
        # —— 那是数据层的知识（它同时拥有 ALL_REF_COLLECTIONS / 优先级），
        # 放这里只留取用。逐条理由（含 etf_list 为何走 akshare）也写在那张表上。
        refdata_set = REFDATA_BY_SOURCE['pytdx' if tdx_like else 'qmt']

        # 名字压根不存在的（打错）要**硬报错**，不能混进「不在本命令范围」的软提示里
        # —— 否则 `--save-collections stock_lsit` 会被静默忽略。
        unknown = [c for c in (ref_collections or [])
                   if c not in ALL_REF_COLLECTIONS]
        if unknown:
            # **用法错**（集合名打错）→ `usage_error`，与 argparse 同形同码
            usage_error(
                'argument --save-collections: 未知集合 {}；可用: {}'.format(
                    unknown, list(ALL_REF_COLLECTIONS)),
                hint='提示: financial 不在本命令范围（4.4 一次性搬入，不随 --save 取数）。')

        ignored = [c for c in (ref_collections or []) if c not in refdata_set]
        if ignored:
            print('提示: --save {} 的参考数据只做 {}；{} 不在本命令范围'
                  '（financial 系 4.4 一次性搬入，--save 不提供）'.format(
                      value, '/'.join(refdata_set), '/'.join(ignored)))

        requested = [c for c in (ref_collections or refdata_set)
                     if c in refdata_set]
        do_refdata = bool(requested) and not args.save_dry_run

        # K 线的实际工作项：(族, 频率) → 集合名。默认取 kline_save 的两张表。
        k_targets = tuple(targets) if targets else KLINE_TARGETS
        k_freqs = tuple(frequencies) if frequencies else KLINE_FREQS
        do_xdxr = (tdx_like and not args.save_dry_run
                   and (targets is None or 'stock' in k_targets))

        # —— banner：把**整条流程**摆成表头，跑一步点一个 ——
        # 表在 CLI 拼，`refdata_save` / `kline_save` 都不认识 UI：它们只负责
        # `on_progress` 回调与 `echo` 输出汇。标题报**真实源名**而非 SOURCE 取值
        # （`--save tdx` 走的源就叫 pytdx，报 `source: pytdx` 才不误导）。
        rows = []
        if do_refdata:
            rows.append(('参考数据', None, requested))
        for tgt in (k_targets if tdx_like and k_targets else ()):
            rows.append(('K线', tgt,
                         [target_collection_name(tgt, f) for f in k_freqs]))
        if do_xdxr:
            # 股票与 ETF 各一对：xdxr（事件）→ adj（由其重算的前复权因子）。
            rows.append(('复权', None,
                         ['stock_xdxr', 'stock_adj', 'etf_xdxr', 'etf_adj']))

        # `数据源` 一行：与阶段名**对齐成一列**，取代旧 banner 自己那条头行
        # `source: pytdx`（2026-10-09 改版，见 `core.presentation.identity`）。
        # ⚠️ 必须打印在 `Banner` **建立之前** —— banner 一旦活着，`print` 就会把它的
        # 行数记账搞错（`PITFALLS.md` P22）。
        source_label = 'pytdx' if tdx_like else value
        # **阶段起止两行**（用户 2026-10-10，与 bootstrap 同构）：起 `[t]: …`、
        # 止 `[t]: … done.`（灰色，仅 TTY）。于是日志里 `--save` 这一段**可检索**。
        # ⚠️ 文案**按实际做的东西**取：`--save qmt` 只做参考数据、**不取 K 线**，
        # 给它打 "saving stock_cn klines" 就是假话（用户给的正是 tdx 路径那句）。
        stage = 'saving stock_cn klines' if tdx_like else 'saving stock_cn refdata'
        print(dim(stamp(stage + '...'), ansi_enabled()), flush=True)   # 起行带 `...`（进行中）
        print(aligned_row('数据源', source_label), flush=True)

        # `header=False`：时刻戳只在开头的**身份块**出现一次，这里只出阶段行。
        banner = Banner(source_label, rows, header=False) if rows else None
        if banner is not None:
            banner.render()
        # ⚠️ banner 活跃时，下方**一切**输出都得走它（见 Banner 的说明），
        # 否则它的行数记账会错位、整块写花。
        say = banner.echo if banner is not None else print

        # 参考数据的刷新闸：**每个集合各自的阈值**（`refdata_save.TTL_HOURS` ——
        # 名单类盘中 5h/盘后 24h，`stock_info` 恒定 24h），未过期的不重取。
        # 所以这里传的是**函数**而不是一个数：阈值要按集合现算。
        # 起算点是「上次**成功完成**」——见 refdata_save 的 `mark_refdata_success`。
        refdata_ttl = None if args.save_refresh else refdata_ttl_hours
        refdata = {}
        if do_refdata:
            # 刷新闸的**参数细节**只在 `-v` 下说 —— 不加 `--verbose` 别往外铺
            # 实现细节（用户 2026-10-09 明确）。默认下它的效果本来就体现在
            # 逐集合的状态里（命中的集合会显示为「读取完成」）。
            if refdata_ttl and args.verbose:
                say('[refdata] 刷新闸：**按集合**计时'
                    '（名单类 盘中5h/盘后24h，stock_info 恒定24h）；'
                    '要强制重取用 --save-refresh')
            # 只按「真取到行」点亮 —— failed/skipped 的节点留在未获取，
            # 免得 banner 替失败的集合谎报成功；逐集合的真实状态由 format_status 出。
            # ⚠️ 例外：被刷新闸拦下的（`cached`）**要点亮** —— 数据是好的、只是没重取，
            # 留灰会被读成「没取到」。
            refdata = save_refdata(
                collections=requested, codelist=codes,
                source=None if tdx_like else value,
                verbose=args.verbose, echo=say, ttl_hours=refdata_ttl,
                on_progress=lambda name, phase, entry: banner.mark(
                    name, RUNNING if phase == 'start' else (
                        DONE if (entry.get('rows') or entry.get('cached'))
                        else PENDING)))

        if not tdx_like:
            say('[save] 源 {}：K 线/xdxr/adj 只有 pytdx 供得了，本命令跳过'.format(value))
        else:
            report = save_kline_tdx(
                targets=targets, frequencies=frequencies, codes=codes,
                start_min=args.save_start, jobs=args.save_jobs,
                dry_run=args.save_dry_run,
                margin_days=args.save_margin_days,
                progress_every=args.save_progress_every,
                verbose=args.verbose, echo=say,
                # `--save-refresh` 一个旗子两处生效：参考数据的刷新闸（上面）、
                # K 线的短路（这里）。语义统一为「别信缓存，全查一遍」。
                force_refresh=args.save_refresh,
                on_progress=lambda nm, phase, st: (
                    banner.mark(nm, RUNNING if phase == 'start' else DONE)
                    if banner else None))
            if args.save_dry_run:
                for name, st in report['kline'].items():
                    say('[dry-run] {}: 可比 {} 行'.format(name, st.get('compared', 0)))
                    for field, entry in sorted(st.get('diffs', {}).items()):
                        say('    {} 差异行 {} 最大相对误差 {:.3g} 样例 {}'.format(
                            field, entry['n'], entry['max_rel'], entry['sample']))

            elif do_xdxr:
                # 股票与 ETF 走**同一条** xdxr → adj 编排（只有集合名与市场路由不同）。
                # ETF 那一支 2026-10-09 起由 pytdx 直供 —— 此前 `etf_xdxr`/`etf_adj`
                # 是 QMT 侧写的（`GQ_SU_save_etf_xdxr_qmt`），已实测 pytdx 供得了。
                for tgt, xdxr_node, adj_node in (
                        ('stock', 'stock_xdxr', 'stock_adj'),
                        ('etf', 'etf_xdxr', 'etf_adj')):
                    # **闸**（用户 2026-10-09 定）：`allow_xdxr_shortcircuit` = TTL **且**
                    # 「上次全查在**最近一次开盘之后**」—— 复权比 K 线**多这一条**，因为当天
                    # 除权信息**开盘前就该拿到**（`DECISIONS.md` D26）。
                    # ⚠️ 这推翻了 D25 原先的「复权只记账不开闸」——代价照实说：
                    # 新除权事件最多**滞后一个 TTL** 才被发现，那段窗口 `{tgt}_adj` 是旧基准
                    # （前复权因子一有新事件就要整条重标）。`--save-refresh` 可强制关掉闸。
                    #
                    # ⚠️ `adj` 节点**刻意不设闸**：它不是"定期全扫"，而是"**事件变了才重算**"
                    # （只在上面 `events_changed` 非空时才走到）—— 拿 TTL 拦它等于
                    # **该重标的时候不重标**，那正是它存在的理由。
                    if not args.save_refresh and allow_xdxr_shortcircuit(xdxr_node):
                        banner.mark(xdxr_node, DONE)   # 数据是好的、只是没重取（同参考数据的 `cached`）
                        # ⚠️ **`_adj` 也要点白**（用户 2026-10-10 实报「两个 _adj 运行过就该白」）：
                        # 闸跳过时这里 `continue`，**下面整个 `_adj` 段一次都不执行** ⇒ 它会留灰。
                        # 语义与"事件没变"同理：xdxr 集合未过期 ⇒ **事件与源端一致是上次已验的结论**
                        # ⇒ 由它算出的 `_adj` 也是最新的（同参考数据 `cached ⇒ DONE`）。
                        # 「最多滞后一个 TTL」是**闸本身**的代价，已体现在上面那个白点里（D25）。
                        banner.mark(adj_node, DONE)
                        # ⚠️ 刷新闸这句话**只在 `-v` 下打**（用户 2026-10-09 指出）——
                        # 与参考数据的刷新闸、K线的「整段跳过」**同一口径**：闸的细节是
                        # 排障信息，不是每次运行都要看的东西（age 也就在 -v 下才算）。
                        if args.verbose:
                            say('[xdxr] {} 刷新闸：上次全查 {:.1f}h 前，本次跳过'.format(
                                xdxr_node, kline_sweep_age_hours(xdxr_node) or 0.0))
                        continue
                    # ⚠️ **先点 RUNNING 再调**（2026-10-09 修）：这两个调用是**阻塞的**，
                    # 原先只在返回后 `mark(DONE)` —— 于是整段（几千只 × ~0.3s，分钟级）
                    # 屏上一直是个**灰点**，看不出在动。K线（`:211`）与参考数据（`:193`）
                    # 都靠 `on_progress` 的 `'start'` 点了绿点，唯独复权这条漏了。
                    # 不改成给 `save_xdxr_tdx` 加 `on_progress`：那函数内部的进度已经由
                    # tqdm 条承担，banner 这层只要粗粒度状态（绿=在跑 / 白=跑完）。
                    banner.mark(xdxr_node, RUNNING)
                    xdxr = save_xdxr_tdx(codes=codes, jobs=args.save_jobs,
                                         verbose=args.verbose, echo=say,
                                         target=tgt)
                    banner.mark(xdxr_node, DONE)
                    # 「上次**全查**于何时」逐节点记账。
                    # ⚠️ **只有全宇宙的跑法才记账**：复权的闸是**集合级**的（没有 K 线那样的
                    # 逐 code 水位兜底），一次 `--save-codes 600519` 的小范围跑若也记账，
                    # 就会把闸打开 → 之后**全量跑整段跳过 xdxr** → 新除权事件静默漏掉。
                    if codes is None:
                        mark_kline_sweep(xdxr_node)
                    if not xdxr['events_changed']:
                        # **事件没变 ⇒ `{target}_adj` 已经是最新的**（不用重算）。
                        # ⚠️ **点白，不是留灰**（用户 2026-10-10 指出）：灰点在这个 banner
                        # 里表示「队列中 / 未获取」，而这里的事实是「**检查过了、结论是不用做**」
                        # —— 留灰会被读成"这一步没做成"。与参考数据那条 `cached ⇒ DONE`
                        # **同一条口径**（`PENDING` 只留给"真的没轮上"）。
                        banner.mark(adj_node, DONE)
                    elif args.save_no_adj:
                        # ⚠️ 事件**变了**却按用户的 `--save-no-adj` 跳过重算 ⇒
                        # `{target}_adj` 此刻是**过期**的（旧因子）。**不许点白** ——
                        # 那才是真的谎报。留灰 + `-v` 说清。
                        if args.verbose:
                            say('[adj] {} 有事件变化，但 --save-no-adj ⇒ 因子停留在旧基准'
                                .format(adj_node))
                    else:
                        # 只对**事件变化**的 code 整条重算（见 save_adj 的护栏说明）
                        banner.mark(adj_node, RUNNING)      # 同 `xdxr_node`：阻塞调用前先点亮
                        save_adj(xdxr['events_changed'], verbose=args.verbose,
                                 echo=say, target=tgt)
                        banner.mark(adj_node, DONE)
                        if codes is None:
                            mark_kline_sweep(adj_node)

        if banner is not None:
            banner.close()
        # 阶段**收尾行**：与起行配对（用户 2026-10-10）。
        # ⚠️ **必须在 `close()` 之后** —— banner 活着时 `print` 会打乱它的行数记账（P22）；
        # 也**必须放在 `try` 里**：中途异常时这一行不该出现（那一段没 done，而兜底的
        # 「意外终止」已经报了）。
        print(stamp_done(stage, color=ansi_enabled()), flush=True)
        # 逐集合状态表：`-v` 下总是打；否则**只在有异常时**打 ——
        # 全是 ok / 命中刷新闸时，banner 上已经看得出来了，再铺一张表就是噪声。
        if refdata and (args.verbose or not all(
                e.get('status') == 'ok' or e.get('cached')
                for e in refdata.values())):
            print(format_status(refdata))

        # 「一个集合都没取到行」才算失败。部分失败不中断（K 线照跑），
        # 逐集合状态已在上面打出。
        # ⚠️ 判据**不能**写成 `status == 'failed'`：源不可用（`--save qmt` 的常态）
        # 走的是 `_pick_source` 抛异常那条路，`entry['status']` 停在初始的
        # **'skipped'** 而不是 'failed' —— 实测踩过：旧判据下 `--save qmt`
        # 三个集合全废却 `exit 0`。
        # `cached`（刷新闸拦下的）也算「有」—— 数据在那儿，只是没重取。
        if refdata and not any(e.get('rows') or e.get('cached')
                               for e in refdata.values()):
            print('失败: 所有参考集合都没取到行（见上面的逐集合说明）')
            sys.exit(1)

    except KeyboardInterrupt:
        # 用户按了 Ctrl-C ⇒ 首行**就该**说"被用户终止"
        print('收盘行情下载过程被用户终止')
        sys.exit(0)
    except Exception as exc:      # noqa: BLE001 常规兜底
        # ⚠️ **不能也说"被用户终止"**（用户 2026-10-09 指出）：程序错误被冠上
        # "用户终止"会把排查引到错的方向 —— 实测踩过，一个 `UnboundLocalError`
        # 就是这么被报成"用户按了 Ctrl-C"的。**两句分开写，各说各的真实情况。**
        print('收盘行情下载过程意外终止')
        print('  原因: {}: {}'.format(type(exc).__name__, exc))
        sys.exit(1)


# 命中判据用**默认的真值判据**即可：`choices=` 保证 `args.save` 只会是
# `None`（没给）或三个非空取值之一 —— 不存在 `--save ''` 这种边缘值了。
SAVE = Command('save', ('save',), add_save_arguments, run_save)
