# coding:utf-8
#
# GolemQ 函数调用频率控制管理器
#

import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from collections import defaultdict
from datetime import timedelta
import socket
import requests
from functools import lru_cache
import traceback
import subprocess
import platform


@lru_cache(maxsize=1)
def get_gateway():
    """获取默认网关地址"""
    try:
        # 根据不同操作系统使用不同方法
        system = platform.system()
        
        if system == "Windows":
            # Windows系统
            # ⚠️ `errors='replace'` 是**必须的**：本项目要求 PYTHONUTF8=1，
            # 而 `text=True` 会按 UTF-8 解码 —— zh-CN 的 `ipconfig` 吐的是 GBK，
            # 解码失败会**打死 subprocess 的读线程**，于是 `result.stdout` 变 None，
            # 这里 `.split` 报 `NoneType`、并且每次调用都在屏上留一段线程 traceback。
            result = subprocess.run(['ipconfig'], capture_output=True, text=True,
                                    errors='replace')
            lines = result.stdout.split('\n')
            for line in lines:
                if '默认网关' in line or 'Default Gateway' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        return parts[1].strip()
        
        elif system == "Linux" or system == "Darwin":  # Darwin是macOS
            # Linux/macOS系统
            result = subprocess.run(['ip', 'route'], capture_output=True, text=True,
                                    errors='replace')
            lines = result.stdout.split('\n')
            for line in lines:
                if 'default via' in line:
                    parts = line.split()
                    return parts[2]  # default via 192.168.1.1 dev eth0
            
        # 备用方法：通过连接外部地址获取本地网关信息
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            
        # 假设网关是本地IP的最后一个字节改为1
        ip_parts = local_ip.split('.')
        ip_parts[3] = '1'
        return '.'.join(ip_parts)
        
    except Exception as e:
        print(f"获取网关失败: {e}")
        return "192.168.1.1"  # 默认返回常见网关
    

@lru_cache(maxsize=1)
def get_optimized_caller_ip():
    """
    优化的IP获取方案，优先零延迟方法，备用快速国内服务
    """
    # 第一阶段：零延迟本地方法
    try:
        # 尝试获取主机名IP
        host_ip = socket.gethostbyname(socket.gethostname())
        if host_ip != '127.0.0.1':
            return host_ip
    except Exception:
        pass
    
    # 第二阶段：快速国内DNS连接（100ms超时）
    domestic_servers = [
        ("114.114.114.114", 53),
        ("223.5.5.5", 53),
        ("119.29.29.29", 53)
    ]
    
    for server, port in domestic_servers:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)  # 100ms超时
            s.connect((server, port))
            ip = s.getsockname()[0]
            s.close()
            if ip and ip != '127.0.0.1':
                return ip
        except Exception:
            continue
    
    # 第三阶段：回退到127.0.0.1
    return socket.gethostbyname(socket.gethostname())


def resolve_caller_ip() -> str:
    """本机在签到表里的 `caller_ip` 标识。

    从 :meth:`FunctionCheckinManager.checkin_function` 里**提出来**的 ——
    查询侧（:func:`last_checkin`）也要用同一个值，否则查的是另一台机器的记录。
    """
    try:
        # 首先尝试获取本地IP（无需网络请求）
        caller_ip = socket.gethostbyname(socket.gethostname())

        if caller_ip.startswith('127.') or caller_ip.startswith('192.') or caller_ip == '0.0.0.0':
            # 使用优化版本
            caller_ip = get_optimized_caller_ip()

        if caller_ip.startswith('192.168'):
            # 如果192.168开头
            caller_ip = get_gateway()

        return caller_ip
    except (socket.gaierror, requests.RequestException):
        # 如果所有方法都失败，使用默认回环地址
        fallback = socket.gethostbyname(socket.gethostname())
        traceback.print_exc()
        return fallback


class FunctionCheckinManager:
    """函数调用频率控制管理器"""
    
    def __init__(self):
        """
        初始化函数签到管理器
        """
        # 函数级导入（非必需，但保持与其它消费者一致）：`GOLEMQ` 是 8.3 的运维库。
        from GolemQ.core.settings import GOLEMQ
        self.db = GOLEMQ
        self.function_collection = self.db.function_checkins
        self.archive_collection = self.db.function_checkins_archive
        
        # 内存中的并发计数器（用于快速检查）
        self._concurrent_counters = defaultdict(int)
        self._last_cleanup_time = time.time()
        
        # 创建索引
        self._create_indexes()
    
    def _create_indexes(self):
        """创建必要的索引"""
        # 函数签到表索引
        self.function_collection.create_index([
            ("function_name", 1),
            ("caller_ip", 1)
        ], unique=True)
        
        # 为function_name和caller_ip单独创建索引，优化查询性能
        self.function_collection.create_index([("function_name", 1)])
        self.function_collection.create_index([("caller_ip", 1)])
        
        self.function_collection.create_index([
            ("last_call_timestamp", -1)
        ])
        
        # 归档表索引
        self.archive_collection.create_index([
            ("function_name", 1),
            ("caller_ip", 1)
        ])
        
        self.archive_collection.create_index([("function_name", 1)])
        self.archive_collection.create_index([("caller_ip", 1)])
        
        self.archive_collection.create_index([
            ("last_call_timestamp", -1)
        ])
    
    def expired_time(self, expired_time: Optional[Union[int, timedelta]] = None) -> int:
        """
        计算过期时间戳
        
        Args:
            expired_time: 过期时间，可以是秒数(int)或timedelta对象，None表示默认15分钟
            
        Returns:
            int: 过期时间戳（秒）
        """
        if expired_time is None:
            # 默认15分钟
            return int(time.time()) + (15 * 60)
        elif isinstance(expired_time, timedelta):
            return int(time.time()) + int(expired_time.total_seconds())
        else:
            # 假设是秒数
            return int(time.time()) + expired_time

    def checkin_function(
        self,
        function_name: str,
        expired_time: Optional[Union[int, timedelta]] = None,
        caller_ip: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        函数调用签到

        Args:
            function_name: 函数名称
            caller_ip: 调用者IP地址

        Returns:
            Dict: 包含检查结果的信息
        """
        current_time = int(time.time())
        current_datetime = f"{datetime.now():%Y-%m-%d %H:%M:%S}"

        # 调用方可以**钉住**这个键（默认 None = 沿用既有解析）。
        # 为什么要能钉：`resolve_caller_ip` 在「本机是 192.168.*」时返回的是
        # **网关**而不是本机，且 `get_gateway()` 失败会退到硬编码的 `192.168.1.1`
        # —— 同一会话可以先后算出两个不同的值（实测 2026-10-09：`192.168.1.1`
        # 与 `192.168.50.1` 各写了一条）。做**计时/限流**的调用方必须钉一个稳定的值，
        # 否则记录查不到、闸门静默失效。
        caller_ip = caller_ip or resolve_caller_ip()

        # 检查现有记录
        existing_record = self.function_collection.find_one({
            "function_name": function_name,
            "caller_ip": caller_ip
        })
        
        if existing_record:
            # 检查是否在过期时间内
            if current_time < existing_record.get("expired_timestamp", 0):
                return {
                    "function_name": function_name,
                    "caller_ip": caller_ip,
                    "allowed": False,
                    "reason": "expired_time_not_reached",
                    "current_time": current_time,
                    "expired_timestamp": existing_record["expired_timestamp"]
                }
        
        # 计算新的过期时间
        expired_timestamp = self.expired_time(expired_time)
        
        # 更新或创建记录
        update_data = {
            "$set": {
                "last_call_timestamp": current_time,
                "last_call_datetime": current_datetime,
                "expired_timestamp": expired_timestamp,
                "updated_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}"
            },
            "$inc": {"call_count": 1},
            "$setOnInsert": {
                "created_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}"
            }
        }
        
        self.function_collection.update_one(
            {
                "function_name": function_name,
                "caller_ip": caller_ip
            },
            update_data,
            upsert=True
        )
        
        # 清理过期的并发计数（每5分钟清理一次）
        if current_time - self._last_cleanup_time > 300:
            self._cleanup_concurrent_counters()
            self._last_cleanup_time = current_time
        
        return {
            "function_name": function_name,
            "caller_ip": caller_ip,
            "allowed": True,
            "expired_timestamp": expired_timestamp,
            "current_time": current_time
        }
    
    def complete_function_call(self, function_name: str, caller_ip: str):
        """
        标记函数调用完成（减少并发计数）
        
        Args:
            function_name: 函数名称
            caller_ip: 调用者IP地址
        """
        concurrent_key = f"{function_name}_{caller_ip}"
        if self._concurrent_counters[concurrent_key] > 0:
            self._concurrent_counters[concurrent_key] -= 1
    
    def _cleanup_concurrent_counters(self):
        """清理过期的并发计数器"""
        # 移除计数为0的项
        keys_to_remove = [k for k, v in self._concurrent_counters.items() if v == 0]
        for key in keys_to_remove:
            del self._concurrent_counters[key]
    
    def get_function_stats(self, function_name: str,
                           caller_ip: Optional[str] = None) -> Optional[Dict[str, Any]]:
        caller_ip = caller_ip or resolve_caller_ip()
        """获取函数的调用统计信息"""
        return self.function_collection.find_one({
            "function_name": function_name,
            "caller_ip": caller_ip
        })
    
    def get_all_function_stats(self) -> List[Dict[str, Any]]:
        """获取所有函数的调用统计信息"""
        return list(self.function_collection.find())
    
    def reset_function_stats(self, function_name: str, caller_ip: str) -> bool:
        """重置函数的调用统计"""
        result = self.function_collection.delete_one({
            "function_name": function_name,
            "caller_ip": caller_ip
        })
        return result.deleted_count > 0
    
    def archive_old_records(self, days: int = 7):
        """归档旧的调用记录"""
        cutoff_time = int(time.time()) - (days * 24 * 3600)
        
        # 查找需要归档的记录
        old_records = self.function_collection.find({
            "last_call_timestamp": {"$lt": cutoff_time}
        })
        
        archived_count = 0
        for record in old_records:
            # 插入到归档表
            self.archive_collection.insert_one(record)
            # 从活动表删除
            self.function_collection.delete_one({"_id": record["_id"]})
            archived_count += 1
        
        return archived_count
    
    def cleanup_archive(self, days: int = 30):
        """清理旧的归档记录"""
        cutoff_time = int(time.time()) - (days * 24 * 3600)
        
        # 清理归档表
        result = self.archive_collection.delete_many({
            "last_call_timestamp": {"$lt": cutoff_time}
        })
        
        return result.deleted_count
    
    def get_concurrent_count(self, function_name: str, caller_ip: str) -> int:
        """获取当前的并发调用数"""
        concurrent_key = f"{function_name}_{caller_ip}"
        return self._concurrent_counters[concurrent_key]
    
    def get_call_frequency(self, function_name: str, caller_ip: str) -> float:
        """计算当前的调用频率（次/分钟）"""
        record = self.get_function_stats(function_name, caller_ip)
        if not record or record["call_count"] < 2:
            return 0.0
        
        current_time = int(time.time())
        time_elapsed = current_time - record["last_call_timestamp"]
        
        if time_elapsed <= 0:
            return float('inf')
        
        # 基于最后两次调用的时间间隔计算频率
        return 60.0 / time_elapsed
    
    def is_function_expired(self, function_name: str, caller_ip: str) -> bool:
        """
        检查函数调用是否已过期
        
        Args:
            function_name: 函数名称
            caller_ip: 调用者IP地址
            
        Returns:
            bool: True表示已过期，False表示未过期
        """
        record = self.get_function_stats(function_name, caller_ip)
        if not record:
            return True  # 没有记录视为已过期
        
        current_time = int(time.time())
        expired_timestamp = record.get("expired_timestamp", 0)
        
        return current_time >= expired_timestamp


# 全局函数签到管理器实例
#: 全局单例**惰性构造**：`FunctionCheckinManager.__init__` 会连库（4.4 句柄），
#: 而模块级 `= FunctionCheckinManager()` 会让**每个** import 本模块的路径都去构造
#: QUANTAXIS（实测多加载 200+ 个模块，见 `PITFALLS.md` P18）。
_global_manager = None


def _manager():
    global _global_manager
    if _global_manager is None:
        _global_manager = FunctionCheckinManager()
    return _global_manager


def __getattr__(name):
    """兼容旧写法 `global_function_manager`（PEP 562 惰性属性）。"""
    if name == 'global_function_manager':
        return _manager()
    raise AttributeError('module {!r} has no attribute {!r}'.format(__name__, name))


def stable_caller_key() -> str:
    """做**计时 / 限流**的调用方该用的 `caller_ip` —— **钉死成主机名**，不走 `resolve_caller_ip()`。

    ⚠️ 那个函数在「本机是 192.168.*」时返回的是**网关**而不是本机，且 `get_gateway()`
    失败会退到硬编码的 `192.168.1.1` —— 同一会话能先后算出两个值（实测 2026-10-09：
    `192.168.1.1` 与 `192.168.50.1` 各写了一条 `refdata:*`）。**计时键抖动 = 记录查不到
    = 刷新闸静默失效**，且顺带每次调用都要起一个 `ipconfig` 子进程。
    主机名稳定、可读，且天然区分机器。

    **为什么放在这一层**（2026-10-09 从 `refdata_save._caller_key` 上移）：
    上面对 `resolve_caller_ip` 的警告是**承重的**，而 kline 的刷新闸也要用同一个键 ——
    两处各写一份，就是「平行实现不会报错，只会分叉」的又一个实例。
    """
    import platform
    return 'host:' + (platform.node() or 'unknown')


def checkin_function(function_name: str, expired_time: Optional[Union[int, timedelta]] = None,
                     caller_ip: Optional[str] = None) -> Dict[str, Any]:
    """使用全局函数管理器进行调用签到。

    `caller_ip` 可显式**钉住**（见 :meth:`FunctionCheckinManager.checkin_function`）——
    做计时/限流的调用方**应该**传 :func:`stable_caller_key`，默认的
    `resolve_caller_ip()` 不稳定。
    """
    return _manager().checkin_function(function_name, expired_time, caller_ip)


def checkin_age_hours(function_name: str, caller_ip: Optional[str] = None,
                      now=None) -> Optional[float]:
    """**只读**：`function_name` 上次签到距今多少小时；**从没签过 → ``None``**（= 该做）。

    「先判后写」里的那个**读口** —— 与 :func:`checkin_function` 的分工见 :func:`last_checkin`。
    `now` 只为测试注入（默认取当前时间）。
    """
    rec = last_checkin(function_name, caller_ip=caller_ip)
    if not rec or not rec.get('last_call_timestamp'):
        return None
    now_ts = (now or datetime.now()).timestamp()
    return (now_ts - float(rec['last_call_timestamp'])) / 3600.0


def mark_checkin(function_name: str, hours: float,
                 caller_ip: Optional[str] = None) -> bool:
    """记账：把「**刚刚成功完成**」写进签到表。返回是否记上。

    ⚠️ **失败绝不抛**：签到表在**运维库**（`GOLEMQ`）里，连不上 / 索引建不出来都有可能。
    这是记账，不是数据 —— 为它把一次**成功的取数**变成失败是本末倒置。
    没记上的代价只是「下次照常重取」，可忽略。
    """
    try:
        checkin_function(function_name, timedelta(hours=hours), caller_ip=caller_ip)
        return True
    except Exception:      # noqa: BLE001 记账失败不许影响数据通路
        return False


def complete_function_call(function_name: str, caller_ip: str):
    """使用全局函数管理器标记调用完成"""
    _manager().complete_function_call(function_name, caller_ip)


def last_checkin(function_name: str,
                 caller_ip: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """**只读**：本机最近一次签到记录（不调用、不写库）；从没签过 → ``None``。

    与 :func:`checkin_function` 的分工（`--save` 的参考数据刷新闸就靠这一对）：

    * 想**先判后写**（例如「只有真成功才计时」）就用本函数读，
      再在成功之后调 :func:`checkin_function` 记账；
    * :func:`checkin_function` 是**调用即记账**的（它的 `allowed` 语义是
      「抢到这次名额」），所以**不能**拿它当「上次成功于何时」的读口。
    """
    return _manager().get_function_stats(function_name, caller_ip)