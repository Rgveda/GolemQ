# coding:utf-8
#
# GolemQ 功能模块心跳监控
#

import time
import threading
import socket
import os
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List

from GolemQ.core.settings import GOLEMQ
from .messenger import Messenger


class HeartbeatMonitor:
    """功能模块心跳监控器"""
    
    def __init__(self, check_interval: int = 60, timeout_threshold: int = 300):
        """
        初始化心跳监控器
        
        Args:
            check_interval: 检查间隔（秒）
            timeout_threshold: 默认超时阈值（秒）
        """
        self.db = GOLEMQ
        self.module_collection = self.db.module_heartbeats
        self.archive_collection = self.db.module_heartbeats_archive
        self.messenger = Messenger()
        self.check_interval = check_interval
        self.default_timeout = timeout_threshold
        self._running = False
        self._check_thread = None
        self.inserted_id = None
        
        # 创建索引
        self._create_indexes()
    
    def _create_indexes(self):
        """创建必要的索引"""
        # 模块心跳表索引
        self.module_collection.create_index([
            ("module_name", 1), 
            ("instance_id", 1)
        ], unique=True)
        
        self.module_collection.create_index([
            ("status", 1), 
            ("last_checkin_timestamp", 1)
        ])
        
        self.module_collection.create_index([
            ("start_timestamp", -1)
        ])
        
        # 归档表索引
        self.archive_collection.create_index([
            ("module_name", 1), 
            ("instance_id", 1)
        ])
        
        self.archive_collection.create_index([
            ("start_timestamp", -1)
        ])
        
    def start_module(
        self, 
        module_name: str, 
        instance_id: str, 
        timeout_seconds: Optional[int] = None,
        initial_message: str = "Module started"
    ) -> str:
        """
        开始一个模块的执行记录
        
        Args:
            module_name: 模块名称
            instance_id: 实例唯一标识符
            timeout_seconds: 超时时间（秒），None使用默认值
            initial_message: 初始消息
            
        Returns:
            str: 记录ID
        """
        current_time = int(time.time())
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hashed_instance_id = instance_id
        
        record = {
            "module_name": module_name,
            "instance_id": hashed_instance_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            
            # 时间信息（双记录格式）
            "start_timestamp": current_time,
            "start_datetime": current_datetime,
            "last_checkin_timestamp": current_time,
            "last_checkin_datetime": current_datetime,
            
            "timeout_seconds": timeout_seconds or self.default_timeout,
            "status": "running",
            
            # 心跳信息
            "checkin_count": 1,
            "last_checkin_message": initial_message,
            
            "created_at": current_time,
            "updated_at": current_time
        }

        result = self.module_collection.insert_one(record)
        self.inserted_id = result.inserted_id
        return str(hashed_instance_id)
    
    def checkin(
        self, 
        module_name: str, 
        instance_id: str, 
        message: str = "Heartbeat checkin"
    ) -> bool:
        """
        模块签到（心跳）
        
        Args:
            module_name: 模块名称
            instance_id: 实例唯一标识符
            message: 签到消息
            
        Returns:
            bool: 是否签到成功
        """
        current_time = int(time.time())
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hashed_instance_id = instance_id
        
        result = self.module_collection.update_one(
            {
                "module_name": module_name,
                "instance_id": hashed_instance_id,
                "status": "running"
            },
            {
                "$set": {
                    "last_checkin_timestamp": current_time,
                    "last_checkin_datetime": current_datetime,
                    "last_checkin_message": message,
                    "updated_at": current_time
                },
                "$inc": {"checkin_count": 1}
            }
        )
        
        return result.modified_count > 0
    
    def complete_module(
        self,
        module_name: str,
        instance_id: str,
        exit_code: int = 0,
        completion_message: str = "Module completed successfully"
    ) -> bool:
        """
        标记模块执行完成
        
        Args:
            module_name: 模块名称
            instance_id: 实例唯一标识符
            exit_code: 退出代码
            completion_message: 完成消息
            
        Returns:
            bool: 是否成功标记完成
        """
        current_time = int(time.time())
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hashed_instance_id = instance_id
        
        # 获取当前记录
        query_id = {
            "module_name": module_name,
            "instance_id": hashed_instance_id,
            "status": "running"
        }
        record = self.module_collection.find_one(query_id)
        
        if not record:
            print(f'HeartbeatMonitor.complete_module 没有找到模块心跳记录：{query_id}')
            return False
        
        # 更新记录为完成状态
        result = self.module_collection.update_one(
            query_id,
            {
                "$set": {
                    "status": "completed",
                    "end_timestamp": current_time,
                    "end_datetime": current_datetime,
                    "exit_code": exit_code,
                    "completion_message": completion_message,
                    "updated_at": current_time
                }
            }
        )
        
        if result.modified_count > 0:
            # 归档已完成记录
            self._archive_completed_record(module_name, instance_id)
            return True
        
        return False
    
    def _archive_completed_record(self, module_name: str, instance_id: str):
        """归档已完成的记录"""
        hashed_instance_id = instance_id
        record = self.module_collection.find_one({
            "module_name": module_name,
            "instance_id": hashed_instance_id
        })
        
        if record:
            # 插入到归档表
            self.archive_collection.insert_one(record)
            # 从活动表删除
            self.module_collection.delete_one({"_id": record["_id"]})
    
    def _check_timeouts(self):
        """检查超时的模块"""
        current_time = int(time.time())
        
        # 查找所有运行中且超时的记录
        timeout_records = self.module_collection.find({
            "status": "running",
            "last_checkin_timestamp": {
                "$lt": current_time - self.default_timeout
            }
        })
        
        for record in timeout_records:
            # 标记为超时
            self.module_collection.update_one(
                {"_id": record["_id"]},
                {
                    "$set": {
                        "status": "timeout",
                        "end_timestamp": current_time,
                        "end_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "completion_message": "Module timeout",
                        "updated_at": current_time
                    }
                }
            )
            
            # 发送超时告警
            message = f"模块超时告警: {record['module_name']} ({record['instance_id']})\n" \
                      f"最后签到时间: {record['last_checkin_datetime']}\n" \
                      f"超时阈值: {record.get('timeout_seconds', self.default_timeout)}秒"
            
            self.messenger.send_alert(
                title="模块执行超时",
                message=message,
                level="error"
            )
            
            # 归档超时记录
            self._archive_completed_record(record['module_name'], record['instance_id'])
    
    def start_monitoring(self):
        """启动监控线程"""
        if self._running:
            return
        
        self._running = True
        self._check_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._check_thread.start()
    
    def stop_monitoring(self):
        """停止监控线程"""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                self._check_timeouts()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"监控循环异常: {e}")
                time.sleep(60)  # 异常后等待1分钟再重试
    
    def get_running_modules(self) -> List[Dict[str, Any]]:
        """获取所有运行中的模块"""
        return list(self.module_collection.find({"status": "running"}))
    
    def get_all_modules(self) -> List[Dict[str, Any]]:
        """获取所有模块状态（包括运行中、已完成、超时、错误的模块）"""
        return list(self.module_collection.find({}))
    
    def get_module_history(self, module_name: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取模块执行历史
        
        Args:
            module_name: 模块名称，None表示所有模块
            limit: 返回记录数量限制
            
        Returns:
            模块历史记录列表
        """
        query = {}
        if module_name:
            query["module_name"] = module_name
            
        return list(self.archive_collection.find(query)
                    .sort("end_timestamp", -1)
                    .limit(limit))
    
    def get_module_status(self, module_name: str, instance_id: str) -> Optional[Dict[str, Any]]:
        """获取特定模块的状态"""
        hashed_instance_id = instance_id
        return self.module_collection.find_one({
            "module_name": module_name,
            "instance_id": hashed_instance_id
        })
    
    def cleanup_old_records(self, days: int = 30):
        """清理旧的归档记录"""
        cutoff_time = int(time.time()) - (days * 24 * 3600)
        
        # 清理归档表
        self.archive_collection.delete_many({
            "end_timestamp": {"$lt": cutoff_time}
        })
    
    def update_module_status(
        self,
        module_name: str,
        instance_id: str,
        status: str,
        message: str = None
    ) -> bool:
        """
        更新模块状态
        
        Args:
            module_name: 模块名称
            instance_id: 实例ID
            status: 新状态 (running, completed, timeout, error, paused)
            message: 状态消息
            
        Returns:
            bool: 是否更新成功
        """
        current_time = int(time.time())
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hashed_instance_id = instance_id
        
        update_data = {
            "status": status,
            "updated_at": current_time
        }
        
        if message:
            update_data["last_checkin_message"] = message
        
        if status in ["completed", "timeout", "error"]:
            update_data["end_timestamp"] = current_time
            update_data["end_datetime"] = current_datetime
            if message:
                update_data["completion_message"] = message
        
        result = self.module_collection.update_one(
            {
                "module_name": module_name,
                "instance_id": hashed_instance_id
            },
            {"$set": update_data}
        )
        
        return result.modified_count > 0
    
    def stop_all_monitoring(self):
        """
        停止所有监控并清理资源，将所有需要归档的模块（包括运行中和已停止的）转移到归档库
        """
        self.stop_monitoring()
        current_time = int(time.time())
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 获取所有需要归档的模块（运行中和已停止的）
        modules_to_archive = list(self.module_collection.find({
            "status": {"$in": ["running", "stopped"]}
        }))
        
        # 将需要归档的模块转移到归档库
        for module in modules_to_archive:
            # 如果模块还在运行中，先更新状态为停止
            if module["status"] == "running":
                self.module_collection.update_one(
                    {"_id": module["_id"]},
                    {
                        "$set": {
                            "status": "stopped",
                            "end_timestamp": current_time,
                            "end_datetime": current_datetime,
                            "completion_message": "监控服务停止",
                            "updated_at": current_time
                        }
                    }
                )
                # 获取更新后的记录
                module = self.module_collection.find_one({"_id": module["_id"]})
            
            # 插入到归档表
            if module:
                self.archive_collection.insert_one(module)
                # 从活动表删除
                self.module_collection.delete_one({"_id": module["_id"]})


monitor = HeartbeatMonitor()


class HeartbeatModule:
    """功能模块心跳监控器"""
    def __init__(self, module_name: str, instance_id: str, timeout_seconds: int = 10):
        """
        初始化心跳监控器
        
        Args:
            module_name: 模块名称
            instance_id: 实例ID
            timeout_threshold: 默认超时阈值（秒）
        """
        self.monitor = monitor
        self.module_name = module_name
        self.instance_id = self._hash_instance_id(instance_id)
        self.default_timeout = timeout_seconds
        self._running = False
        self._check_thread = None

    def _hash_instance_id(self, instance_id: str) -> str:
        """
        对实例ID进行哈希处理
        
        Args:
            instance_id: 原始实例ID
            
        Returns:
            str: 哈希后的实例ID
        """
        return hashlib.sha256(instance_id.encode()).hexdigest()
    
    def mutex(self, verbose: bool = False):
        # 检查是否有旧的运行实例
        running_modules = self.monitor.get_running_modules()
        old_instances = [m for m in running_modules if m['module_name'] == self.module_name]
        
        if old_instances:
            current_time = int(time.time())
            for old_instance in old_instances:
                last_checkin = old_instance.get('last_checkin_timestamp', 0)
                timeout_seconds = old_instance.get('timeout_seconds', self.default_timeout)
                
                # 检查是否超时
                if current_time > last_checkin + timeout_seconds:
                    # 超时实例，清理并继续
                    if verbose:
                        print(f"发现超时的旧实例 {old_instance['instance_id']}，最后签到时间: {old_instance.get('last_checkin_datetime', 'N/A')}")
                        print("清理超时实例并继续执行...")
                    self.monitor.update_module_status(
                        module_name=self.module_name,
                        instance_id=old_instance['instance_id'],
                        status="timeout",
                        message="检测到新实例启动，清理超时旧实例"
                    )
                    self.monitor._archive_completed_record(self.module_name, old_instance['instance_id'])
                else:
                    # 未超时实例，提示并退出
                    self.remaining_time = (last_checkin + timeout_seconds) - current_time
                    if verbose:
                        print(f"model:{self.module_name} 发现未超时的运行实例 {old_instance['instance_id']}")
                        print(f"最后签到时间: {old_instance.get('last_checkin_datetime', 'N/A')}")
                        print(f"超时阈值: {timeout_seconds}秒，剩余时间: {self.remaining_time}秒")
                        print("错误: 只能运行一个实例，请等待当前实例完成或超时后再启动")
                    return True
                
        return False

    def start(self, initial_message: str):
        """
        开始一个模块的执行记录
        
        Args:
            initial_message: 初始消息
            
        Returns:
            str: 记录ID
        """
        self.instance_id = self.monitor.start_module(
            module_name=self.module_name,
            instance_id=self.instance_id,
            initial_message=initial_message,
            timeout_seconds=self.default_timeout,
        )

    def complete(self, completion_message: str, exit_code: int = 0):
        """
        标记模块执行完成
        
        Args:
            exit_code: 退出代码
            completion_message: 完成消息
            
        Returns:
            bool: 是否成功标记完成
        """
        result = self.monitor.complete_module(
            module_name=self.module_name,
            instance_id=self.instance_id,
            exit_code=exit_code,
            completion_message=completion_message
        )
        return result

    def checkin(self, message: str):
        """
        模块签到（心跳）
        
        Args:
            message: 签到消息
            
        Returns:
            bool: 是否签到成功
        """
        result = self.monitor.checkin(
            module_name=self.module_name,
            instance_id=self.instance_id,
            message=message
        )
        return result
