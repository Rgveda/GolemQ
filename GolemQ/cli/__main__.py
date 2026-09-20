# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import sys
import time
from GolemQ.agents.messenger import setup_dingtalk_config_interactive, setup_serverchan_config_interactive, send_serverchan_message
from GolemQ.gateway.xtquant.config import setup_xtquant_config_interactive
from GolemQ.core.settings import setup_mongodb_config
from GolemQ.cli.tools import purge_mongodb_database
# 移除未使用的导入
from GolemQ.supervisor.scheduler import start_xtquant_sync_scheduler, stop_xtquant_sync_scheduler
from GolemQ import GQSUBSCRIBER
from GolemQ.cli.watchdog_manager import (
    parse_symbols,
    add_symbols_to_watchlist,
    remove_symbols_from_watchlist,
    list_watchlist_symbols
)
from GolemQ.supervisor.heartbeat import HeartbeatMonitor
from datetime import timedelta
from GolemQ.supervisor.function_checkin import checkin_function


def main() -> None:
    """主函数：处理命令行参数"""
    parser = argparse.ArgumentParser(
        description="GolemQ 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m GolemQ.cli --setup           # 初始化配置
  python -m GolemQ.cli --mongodb-init    # 初始化MongoDB配置
  python -m GolemQ.cli --dingtalk-init   # 初始化钉钉配置
  python -m GolemQ.cli --serverchan-init # 初始化Server酱配置
  python -m GolemQ.cli --xtquant-init    # 初始化XTQuant配置
  python -m GolemQ.cli --purge-l1           # 清理数据库
  python -m GolemQ.cli --purge-l1 --verbose # 详细模式清理数据库
  python -m GolemQ.cli --xtquant-sync    # 同步XTQuant持仓到MongoDB
  python -m GolemQ.cli --xtquant-sync-daemon  # 启动XTQuant定时同步守护进程
  python -m GolemQ.cli --eneloop-add --symbols "000001,000002" # 添加个股到关注列表
  python -m GolemQ.cli --eneloop-remove --symbols "000001,000002" # 删除个股并归档
  python -m GolemQ.cli --eneloop-list    # 查看当前关注列表
  python -m GolemQ.cli --sub l1_tencent  # 执行L1数据订阅功能
  python -m GolemQ.cli --stock-min-aligned  # 显示当前时间分钟对齐信息
  python -m GolemQ.cli --stock-min-aligned --symbol 000001.SZ  # 指定股票代码的对齐信息
  python -m GolemQ.cli --stock-min-aligned --interval 60min  # 使用60分钟间隔对齐
        """
    )
    
    # 添加命令行参数
    parser.add_argument('-s', '--setup',
                        help="初始化配置 (MongoDB和钉钉)",
                        action="store_true",
                        default=False)
    
    parser.add_argument('-i', '--init',
                        help="初始化配置 (MongoDB和钉钉) - 与 --setup 相同",
                        action="store_true",
                        default=False)
    
    parser.add_argument('-a', '--purge-l1',
                        help="清理 MongoDB 数据库",
                        action="store_true",
                        default=False)
    
    parser.add_argument("-v", "--verbose",
                        help="增加输出详细程度",
                        action="store_true",
                        default=False)
    
    parser.add_argument('--mongodb-init',
                        help="初始化MongoDB配置",
                        action="store_true",
                        default=False)
    
    parser.add_argument('--dingtalk-init',
                        help="初始化钉钉配置",
                        action="store_true",
                        default=False)

    parser.add_argument('--serverchan-init',
                        help="初始化Server酱配置",
                        action="store_true",
                        default=False)

    parser.add_argument('--xtquant-init',
                        help="初始化XTQuant配置",
                        action="store_true",
                        default=False)
    
    parser.add_argument('--xtquant-sync',
                        help="同步XTQuant持仓数据到MongoDB",
                        action="store_true",
                        default=False)
    
    parser.add_argument('--xtquant-sync-daemon',
                        help="启动XTQuant定时同步守护进程",
                        action="store_true",
                        default=False)
    
    parser.add_argument('--eneloop-add',
                        help="添加股票代码到关注列表",
                        action="store_true",
                        default=False)
    
    parser.add_argument('--eneloop-remove',
                        help="从关注列表删除股票代码并归档",
                        action="store_true",
                        default=False)
    
    parser.add_argument('--eneloop-list',
                        help="列出当前关注列表中的股票代码",
                        action="store_true",
                        default=False)
    
    parser.add_argument('--symbols',
                        help="股票代码列表，以逗号或换行分割",
                        type=str,
                        default="")
    
    parser.add_argument('--heartbeat-watchdog',
                        help="查看心跳监控状态",
                        action="store_true",
                        default=False)
    
    parser.add_argument('--sub',
                        help="执行指定的第三方行情订阅功能",
                        type=str,
                        metavar="SUBSCRIBER_KEY",
                        default=None)
    
    parser.add_argument('--stop-heartbeat-monitor',
                        help="停止所有心跳监控并清理资源",
                        action="store_true",
                        default=False)

    parser.add_argument('--stock-min-aligned',
                        help="股票分钟对齐功能",
                        action="store_true",
                        default=False)

    parser.add_argument('--symbol',
                        help="股票代码",
                        type=str,
                        default=None)

    parser.add_argument('--interval',
                        help="对齐间隔 (15min/60min)",
                        type=str,
                        default="15min")

    # ---- 参考数据（5 个集合）保存到 MongoDB 8.3 ----
    # 同时接受连字符与下划线两种写法：前者是 argparse 惯例，后者是习惯输入。
    parser.add_argument('--save-x', '--save_x',
                        help="保存 A 股参考数据到 MongoDB 8.3（按集合优先级自动选源）",
                        action="store_true",
                        default=False)

    parser.add_argument('--save-qmt', '--save_qmt',
                        help="保存 A 股参考数据到 MongoDB 8.3（强制走 MiniQMT，需客户端在线）",
                        action="store_true",
                        default=False)

    parser.add_argument('--save-collections',
                        help="限定要保存的集合，逗号分隔；默认全部 5 个 "
                             "(stock_list,stock_info,etf_list,stock_block,financial)",
                        type=str,
                        default=None)

    parser.add_argument('--save-status',
                        help="查看参考集合的库存量与各数据源可用性，不写库",
                        action="store_true",
                        default=False)

    # 解析参数
    args = parser.parse_args()
    
    # 处理参数
    if args.setup or args.init:
        if args.verbose:
            print("开始初始化配置...")
        setup_mongodb_config()
        setup_dingtalk_config_interactive()
        if args.verbose:
            print("配置初始化完成!")
    
    elif args.mongodb_init:
        setup_mongodb_config()
    
    elif args.dingtalk_init:
        setup_dingtalk_config_interactive()

    elif args.serverchan_init:
        setup_serverchan_config_interactive()

    elif args.xtquant_init:
        setup_xtquant_config_interactive()
    
    elif args.xtquant_sync:
        # 同步持仓数据到MongoDB（交易日循环执行，带心跳监控）
        from GolemQ.supervisor.messenger import send_alert
        from GolemQ.gateway.xtquant.xtquant_tools import xtquant_sync_during_trading_hours
        
        try:
            # 执行交易日循环同步
            success = xtquant_sync_during_trading_hours()
            
            if success:
                print("XTQuant持仓数据交易日同步成功!")
            else:
                # 发送失败通知
                send_alert(
                    title="XTQuant交易日同步失败",
                    message="XTQuant持仓数据交易日同步失败，请检查系统状态",
                    level="error"
                )
                print("XTQuant持仓数据交易日同步失败!")
                sys.exit(1)
                
        except Exception as e:
            # 发送异常通知
            send_alert(
                title="XTQuant交易日同步异常",
                message=f"XTQuant持仓数据交易日同步发生异常: {str(e)}",
                level="error"
            )
            print(f"XTQuant持仓数据交易日同步异常: {e}")
            sys.exit(1)
            
    elif args.xtquant_sync_daemon:
        # 启动定时同步守护进程
        print("启动XTQuant定时同步守护进程...")
        print("将在交易时间每1-15分钟自动同步持仓数据")
        print("按 Ctrl+C 停止守护进程")
        
        try:
            start_xtquant_sync_scheduler()
            
            # 保持主线程运行
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n停止XTQuant定时同步守护进程...")
            stop_xtquant_sync_scheduler()
            print("守护进程已停止")
            
        except Exception as e:
            print(f"守护进程运行异常: {e}")
            stop_xtquant_sync_scheduler()
            sys.exit(1)
    
    elif args.eneloop_add:
        # 添加股票到关注列表
        if not args.symbols:
            print("错误: 请使用 --symbols 参数指定要添加的股票代码")
            sys.exit(1)
        
        symbols = parse_symbols(args.symbols)
        if not symbols:
            print("错误: 没有有效的股票代码")
            sys.exit(1)
        
        print(f"准备添加 {len(symbols)} 个股票代码到关注列表:")
        for symbol in symbols:
            print(f"  {symbol}")
        
        if not args.verbose:
            confirm = input("确认添加? (y/N): ").strip().lower()
            if confirm != 'y':
                print("操作已取消")
                return
        
        success_count = add_symbols_to_watchlist(symbols, args.verbose)
        if success_count > 0:
            print("操作完成!")
        else:
            print("没有股票代码被添加")
    
    elif args.eneloop_remove:
        # 从关注列表删除股票并归档
        if not args.symbols:
            print("错误: 请使用 --symbols 参数指定要删除的股票代码")
            sys.exit(1)
        
        symbols = parse_symbols(args.symbols)
        if not symbols:
            print("错误: 没有有效的股票代码")
            sys.exit(1)
        
        print(f"准备从关注列表删除并归档 {len(symbols)} 个股票代码:")
        for symbol in symbols:
            print(f"  {symbol}")
        
        if not args.verbose:
            confirm = input("确认删除并归档? (y/N): ").strip().lower()
            if confirm != 'y':
                print("操作已取消")
                return
        
        success_count = remove_symbols_from_watchlist(symbols, args.verbose)
        if success_count > 0:
            print("操作完成!")
        else:
            print("没有股票代码被删除")
    
    elif args.eneloop_list:
        # 列出当前关注列表
        list_watchlist_symbols(args.verbose)
    
    elif args.purge_l1:
        # 确认操作
        if not args.verbose:
            confirm = input("警告: 这将删除所有 L1 数据! 确认操作? (y/N): ").strip().lower()
            if confirm != 'y':
                print("操作已取消")
                return
        
        purge_mongodb_database(args.verbose)
    
    elif args.sub:
        # 执行指定的第三方行情订阅器功能
        subscriber_key = args.sub
        if subscriber_key in GQSUBSCRIBER:
            try:
                print(f"执行第三方行情订阅器: {subscriber_key}")
                subscriber_func = GQSUBSCRIBER[subscriber_key]
                subscriber_func()
                print(f"第三方行情订阅器 {subscriber_key} 执行完成")
            except Exception as e:
                print(f"执行第三方行情订阅器 {subscriber_key} 时发生错误: {e}")
                sys.exit(1)
        else:
            print(f"错误: 第三方行情订阅器 '{subscriber_key}' 不存在")
            print("可用的第三方行情订阅器有:")
            for key in sorted(GQSUBSCRIBER.keys()):
                print(f"  {key}")
            sys.exit(1)
    
    elif args.heartbeat_watchdog:
        # 查看心跳监控状态
        monitor = HeartbeatMonitor()
        timeout_modules = []  # 存储超时模块信息
        
        print("=== 当前运行中的模块 ===")
        running_modules = monitor.get_running_modules()
        if running_modules:
            for module in running_modules:
                print(f"模块: {module['module_name']}")
                print(f"  实例ID: {module['instance_id']}")
                print(f"  状态: {module['status']}")
                print(f"  最后签到: {module['last_checkin_datetime']}")
                print(f"  签到次数: {module['checkin_count']}")
                print(f"  超时阈值: {module['timeout_seconds']}秒")
                print()
        else:
            print("  没有运行中的模块")
            print()
        
        print("=== 所有模块状态 ===")
        all_modules = monitor.get_all_modules()
        if all_modules:
            current_time = int(time.time())
            
            for module in all_modules:
                # 检查是否超时（状态为running但当前时间大于最后签到时间+超时时间）
                actual_status = module['status']
                if module['status'] == 'running':
                    last_checkin = module.get('last_checkin_timestamp', 0)
                    timeout_seconds = module.get('timeout_seconds', 300)  # 默认5分钟
                    if current_time > last_checkin + timeout_seconds:
                        actual_status = 'timeout (实际)'
                        status_icon = "🔴"
                    else:
                        status_icon = "🟢"
                else:
                    status_icon = "🟡" if module['status'] == 'completed' else \
                                 "🔴" if module['status'] == 'timeout' else "⚫"
                
                print(f"{status_icon} {module['module_name']} - {actual_status}")
                print(f"   最后签到: {module.get('last_checkin_datetime', 'N/A')}")
                if module['status'] != 'running':
                    print(f"   结束时间: {module.get('end_datetime', 'N/A')}")
                print(f"   超时阈值: {module.get('timeout_seconds', 300)}秒")
                
                # 显示超时信息（如果实际超时）
                if actual_status == 'timeout (实际)':
                    # 计算实际超时时间（当前时间 - 最后签到时间 - 超时阈值）
                    actual_timeout_seconds = current_time - last_checkin - timeout_seconds
                    timeout_minutes = actual_timeout_seconds // 60
                    print(f"   ⚠️ 已超时: {timeout_minutes}分钟 (超过阈值{timeout_seconds//60}分钟)")
                    
                    # 收集超时模块信息
                    timeout_modules.append({
                        'module_name': module['module_name'],
                        'instance_id': module['instance_id'],
                        'last_checkin': module.get('last_checkin_datetime', 'N/A'),
                        'timeout_threshold': f"{timeout_seconds//60}分钟",
                        'actual_timeout': f"{timeout_minutes}分钟"
                    })
                
                print()
        else:
            print("  没有模块记录")
            print()
        
        print("=== 最近模块执行历史 ===")
        recent_history = monitor.get_module_history(limit=10)
        if recent_history:
            for history in recent_history:
                status_icon = "✅" if history['status'] == 'completed' else \
                             "⏰" if history['status'] == 'timeout' else "❌"
                print(f"{status_icon} {history['module_name']} - {history['status']}")
                print(f"   开始: {history['start_datetime']}")
                print(f"   结束: {history['end_datetime']}")
                if 'completion_message' in history:
                    print(f"   消息: {history['completion_message']}")
                print()
        else:
            print("  没有历史记录")
        
        # 如果有超时模块，发送Server酱提醒
        if timeout_modules:
            _send_timeout_alert(timeout_modules)
    
    elif args.stop_heartbeat_monitor:
        # 停止所有心跳监控并清理资源
        monitor = HeartbeatMonitor()
        
        if not args.verbose:
            confirm = input("警告: 这将停止所有心跳监控并标记运行中的模块为停止状态! 确认操作? (y/N): ").strip().lower()
            if confirm != 'y':
                print("操作已取消")
                return
        
        print("正在停止心跳监控并清理资源...")
        monitor.stop_all_monitoring()
        print("心跳监控已停止，所有运行中的模块已标记为停止状态并转移到归档库")
    
    elif args.stock_min_aligned:
        # 股票分钟对齐功能
        try:
            from GolemQ.markets.StockCN.align import stock_min_aligned
            
            stock_min_aligned(
                verbose=args.verbose
            )
                    
        except Exception as e:
            print(f"股票分钟对齐功能执行失败: {e}")
            sys.exit(1)
            
    elif args.save_status:
        # 只读：报告库存量与各源可用性，不写库
        from GolemQ.pipeline.refdata import format_status
        print(format_status())

    elif args.save_x or args.save_qmt:
        # CLI 只做参数校验，业务逻辑在 pipeline/refdata.py
        from GolemQ.pipeline.refdata import format_status, save_refdata

        collections = None
        if args.save_collections:
            collections = [c.strip() for c in args.save_collections.split(',') if c.strip()]
            from GolemQ.pipeline.refdata import ALL_REF_COLLECTIONS
            unknown = [c for c in collections if c not in ALL_REF_COLLECTIONS]
            if unknown:
                print(f"未知集合: {unknown}；可用: {list(ALL_REF_COLLECTIONS)}")
                sys.exit(1)

        source = 'qmt' if args.save_qmt else None
        report = save_refdata(collections=collections, source=source,
                              verbose=args.verbose)
        print()
        print(format_status(report))

        # 有集合失败则以非零码退出，便于脚本/计划任务感知
        failed = [n for n, e in report.items() if e.get('status') == 'failed']
        if failed:
            print(f"\n失败的集合: {failed}")
            sys.exit(1)

    else:
        # 如果没有指定任何参数，显示帮助信息
        parser.print_help()


def _send_timeout_alert(timeout_modules: list) -> None:
    """发送超时模块提醒到Server酱
    
    Args:
        timeout_modules: 超时模块列表
    """
    if not timeout_modules:
        return
    
    # 导入频率控制模块
    try:
        # 检查调用频率（15分钟限制）
        checkin_result = checkin_function(
            function_name="_send_timeout_alert",
            expired_time=timedelta(minutes=15)
        )
        
        if not checkin_result["allowed"]:
            remaining_time = checkin_result['expired_timestamp'] - checkin_result['current_time']
            print(f"Server酱消息发送频率限制：距离下次可发送还有 {remaining_time} 秒")
            return
    except ImportError:
        # 如果导入失败，直接发送消息（向后兼容）
        print("警告: FunctionCheckinManager不可用，跳过频率限制")
    
    # 构建Markdown格式的消息
    title = f"⚠️ GolemQ 模块超时告警 ({len(timeout_modules)}个模块)"
    
    # 构建表格内容
    table_header = "| 模块名称 | 实例ID | 最后签到 | 超时阈值 | 实际超时 |\n"
    table_separator = "|----------|--------|----------|----------|----------|\n"
    table_rows = []
    
    for module in timeout_modules:
        # 截断过长的实例ID
        instance_id = module['instance_id']
        if len(instance_id) > 12:
            instance_id = instance_id[:12] + "..."
        
        table_rows.append(
            f"| {module['module_name']} | `{instance_id}` | {module['last_checkin']} | "
            f"{module['timeout_threshold']} | **{module['actual_timeout']}** |"
        )
    
    # 组合完整的Markdown内容
    markdown_content = (
        f"## 🚨 模块执行超时告警\n\n"
        f"检测到 **{len(timeout_modules)}** 个模块执行超时，请及时处理！\n\n"
        f"{table_header}{table_separator}{chr(10).join(table_rows)}\n\n"
        f"**处理建议:**\n"
        f"- 检查模块是否正常运行\n"
        f"- 确认网络连接是否正常\n"
        f"- 查看日志文件排查问题\n"
        f"- 必要时重启相关服务\n\n"
        f"⏰ 告警时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    try:
        # 发送Server酱消息
        result = send_serverchan_message(title, markdown_content)
        if 'error' not in result:
            print("✓ Server酱超时告警发送成功!")
        else:
            print(f"✗ Server酱消息发送失败: {result.get('error', '未知错误')}")
    except Exception as e:
        print(f"发送Server酱消息时出错: {e}")


if __name__ == "__main__":
    main()
