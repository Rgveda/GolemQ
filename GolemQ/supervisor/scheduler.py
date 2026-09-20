# coding:utf-8
#
# GolemQ 定时任务调度器
# 用于处理定时执行的任务，如 xtquant 同步
#

import time
import threading
import schedule
from datetime import datetime, time as dt_time
import pytz

from GolemQ.supervisor.heartbeat import HeartbeatMonitor
from GolemQ.supervisor.messenger import send_alert
from GolemQ.gateway.xtquant.xtquant_tools import export_xtquant_positions_to_mongodb


class TradingTimeChecker:
    """交易时间检查器"""
    
    def __init__(self, timezone='Asia/Shanghai'):
        self.timezone = pytz.timezone(timezone)
        
    def is_trading_time(self) -> bool:
        """
        检查当前是否为交易时间
        中国股市交易时间: 周一至周五 9:30-11:30, 13:00-15:00
        """
        now = datetime.now(self.timezone)
        
        # 检查是否为周末
        if now.weekday() >= 5:  # 5=周六, 6=周日
            return False
            
        # 检查是否为节假日 (这里需要扩展节假日判断逻辑)
        # 暂时只检查时间
        
        # 上午交易时间: 9:30-11:30
        morning_start = dt_time(9, 30)
        morning_end = dt_time(11, 30)
        
        # 下午交易时间: 13:00-15:00
        afternoon_start = dt_time(13, 0)
        afternoon_end = dt_time(15, 0)
        
        current_time = now.time()
        
        # 检查是否在交易时间内
        return ((morning_start <= current_time <= morning_end) or
                (afternoon_start <= current_time <= afternoon_end))


class XtquantSyncScheduler:
    """XTQuant 同步调度器"""
    
    def __init__(self):
        self.trading_checker = TradingTimeChecker()
        self.heartbeat_monitor = HeartbeatMonitor()
        self._running = False
        self._scheduler_thread = None
        self.min_interval_minutes = 15  # 最少15分钟运行一次
        self.max_interval_minutes = 1   # 最多1分钟运行一次
        
    def _sync_with_heartbeat(self) -> bool:
        """
        执行同步并记录心跳
        """
        instance_id = f"xtquant_sync_{int(time.time())}"
        
        # 开始模块执行记录
        self.heartbeat_monitor.start_module(
            module_name="xtquant_sync",
            instance_id=instance_id,
            timeout_seconds=300,  # 5分钟超时
            initial_message="开始XTQuant持仓同步"
        )
        
        try:
            # 执行同步
            success = export_xtquant_positions_to_mongodb()
            
            if success:
                # 同步成功，标记完成
                self.heartbeat_monitor.complete_module(
                    module_name="xtquant_sync",
                    instance_id=instance_id,
                    exit_code=0,
                    completion_message="XTQuant持仓同步成功完成"
                )
                return True
            else:
                # 同步失败，标记错误
                self.heartbeat_monitor.complete_module(
                    module_name="xtquant_sync", 
                    instance_id=instance_id,
                    exit_code=1,
                    completion_message="XTQuant持仓同步失败"
                )
                
                # 发送失败通知
                send_alert(
                    title="XTQuant同步失败",
                    message="XTQuant持仓数据同步失败，请检查系统状态",
                    level="error"
                )
                return False
                
        except Exception as e:
            # 发生异常，标记错误
            self.heartbeat_monitor.complete_module(
                module_name="xtquant_sync",
                instance_id=instance_id,
                exit_code=2,
                completion_message=f"XTQuant持仓同步异常: {str(e)}"
            )
            
            # 发送异常通知
            send_alert(
                title="XTQuant同步异常",
                message=f"XTQuant持仓数据同步发生异常: {str(e)}",
                level="error"
            )
            return False
            
    def _should_run_sync(self) -> bool:
        """
        判断是否应该运行同步
        """
        # 检查是否为交易时间
        if not self.trading_checker.is_trading_time():
            return False
            
        # 检查最近一次成功同步的时间
        # 这里可以添加更复杂的逻辑来确保每15分钟至少运行一次
        # 目前先简单返回True，在交易时间内都运行
        return True
        
    def _sync_job(self):
        """同步任务"""
        if self._should_run_sync():
            print(f"[{datetime.now()}] 执行XTQuant同步任务...")
            self._sync_with_heartbeat()
        else:
            print(f"[{datetime.now()}] 非交易时间，跳过XTQuant同步")
            
    def start_scheduling(self):
        """启动定时调度"""
        if self._running:
            return
            
        self._running = True
        
        # 配置调度器
        # 每1分钟检查一次，但只在交易时间且满足条件时执行
        schedule.every(self.max_interval_minutes).minutes.do(self._sync_job)
        
        # 启动监控
        self.heartbeat_monitor.start_monitoring()
        
        # 启动调度线程
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        print(f"XTQuant同步调度器已启动，将在交易时间每{self.max_interval_minutes}-{self.min_interval_minutes}分钟执行同步")
        
    def stop_scheduling(self):
        """停止定时调度"""
        self._running = False
        self.heartbeat_monitor.stop_monitoring()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
            
    def _scheduler_loop(self):
        """调度器循环"""
        while self._running:
            try:
                schedule.run_pending()
                time.sleep(1)  # 每秒检查一次
            except Exception as e:
                print(f"调度器循环异常: {e}")
                time.sleep(10)  # 异常后等待10秒再重试


# 全局调度器实例
global_scheduler = XtquantSyncScheduler()


def start_xtquant_sync_scheduler():
    """启动XTQuant同步调度器"""
    global_scheduler.start_scheduling()


def stop_xtquant_sync_scheduler():
    """停止XTQuant同步调度器"""
    global_scheduler.stop_scheduling()