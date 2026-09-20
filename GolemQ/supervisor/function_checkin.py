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
from GolemQ.core.settings import (
    DATABASE as DATABASE_GolemQ,
)
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
            result = subprocess.run(['ipconfig'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for line in lines:
                if '默认网关' in line or 'Default Gateway' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        return parts[1].strip()
        
        elif system == "Linux" or system == "Darwin":  # Darwin是macOS
            # Linux/macOS系统
            result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
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


class FunctionCheckinManager:
    """函数调用频率控制管理器"""
    
    def __init__(self):
        """
        初始化函数签到管理器
        """
        self.db = DATABASE_GolemQ
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
        expired_time: Optional[Union[int, timedelta]] = None
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
        
        # 获取调用者IP地址，优先获取本地IP
        try:
            # 首先尝试获取本地IP（无需网络请求）
            caller_ip = socket.gethostbyname(socket.gethostname())
            
            if caller_ip.startswith('127.') or caller_ip.startswith('192.') or caller_ip == '0.0.0.0':
                # 使用优化版本
                caller_ip = get_optimized_caller_ip()

            if caller_ip.startswith('192.168'):
                # 如果192.168开头
                caller_ip = get_gateway()
                
        except (socket.gaierror, requests.RequestException):
            # 如果所有方法都失败，使用默认回环地址
            caller_ip = socket.gethostbyname(socket.gethostname())
            traceback.print_exc()

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
    
    def get_function_stats(self, function_name: str, caller_ip: str) -> Optional[Dict[str, Any]]:
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
global_function_manager = FunctionCheckinManager()


def checkin_function(function_name: str, expired_time: Optional[Union[int, timedelta]] = None) -> Dict[str, Any]:
    """使用全局函数管理器进行调用签到"""
    return global_function_manager.checkin_function(function_name, expired_time)


def complete_function_call(function_name: str, caller_ip: str):
    """使用全局函数管理器标记调用完成"""
    global_function_manager.complete_function_call(function_name, caller_ip)