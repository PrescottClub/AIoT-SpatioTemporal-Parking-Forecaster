"""
日志系统模块

提供统一的日志配置和管理功能。

Author: AI Assistant
Date: 2025-07-29
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化为JSON格式"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class LoggerManager:
    """日志管理器"""
    
    _loggers: Dict[str, logging.Logger] = {}
    _initialized: bool = False
    
    @classmethod
    def setup_logging(
        cls,
        log_level: str = "INFO",
        log_dir: str = "logs",
        log_format: Optional[str] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = False,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> None:
        """设置全局日志配置
        
        Args:
            log_level: 日志级别
            log_dir: 日志目录
            log_format: 日志格式
            enable_console: 是否启用控制台输出
            enable_file: 是否启用文件输出
            enable_json: 是否启用JSON格式
            max_bytes: 日志文件最大大小
            backup_count: 备份文件数量
        """
        if cls._initialized:
            return
        
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 设置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 默认格式
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # 控制台处理器
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            
            if sys.stdout.isatty():  # 终端支持颜色
                console_formatter = ColoredFormatter(log_format)
            else:
                console_formatter = logging.Formatter(log_format)
            
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # 文件处理器
        if enable_file:
            from logging.handlers import RotatingFileHandler
            
            # 普通日志文件
            file_handler = RotatingFileHandler(
                log_path / "app.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
            # 错误日志文件
            error_handler = RotatingFileHandler(
                log_path / "error.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            root_logger.addHandler(error_handler)
            
            # JSON格式日志文件
            if enable_json:
                json_handler = RotatingFileHandler(
                    log_path / "app.json",
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                json_handler.setLevel(getattr(logging, log_level.upper()))
                json_formatter = JSONFormatter()
                json_handler.setFormatter(json_formatter)
                root_logger.addHandler(json_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """获取日志器
        
        Args:
            name: 日志器名称
            
        Returns:
            日志器实例
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def set_level(cls, level: str) -> None:
        """设置全局日志级别
        
        Args:
            level: 日志级别
        """
        logging.getLogger().setLevel(getattr(logging, level.upper()))
        for logger in cls._loggers.values():
            logger.setLevel(getattr(logging, level.upper()))


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        """初始化实验日志器
        
        Args:
            experiment_name: 实验名称
            log_dir: 日志目录
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / "experiments"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建实验特定的日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.json"
        
        self.metrics = {}
        self.config = {}
        self.artifacts = []
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """记录配置信息
        
        Args:
            config: 配置字典
        """
        self.config = config
        self._save_log()
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """记录指标
        
        Args:
            name: 指标名称
            value: 指标值
            step: 步骤数
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        if step is not None:
            metric_entry['step'] = step
        
        self.metrics[name].append(metric_entry)
        self._save_log()
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "file") -> None:
        """记录产物
        
        Args:
            artifact_path: 产物路径
            artifact_type: 产物类型
        """
        artifact_entry = {
            'path': artifact_path,
            'type': artifact_type,
            'timestamp': datetime.now().isoformat()
        }
        
        self.artifacts.append(artifact_entry)
        self._save_log()
    
    def _save_log(self) -> None:
        """保存日志到文件"""
        log_data = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'metrics': self.metrics,
            'artifacts': self.artifacts,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)


# 便捷函数
def setup_logger(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True
) -> None:
    """设置日志系统
    
    Args:
        log_level: 日志级别
        log_dir: 日志目录
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
    """
    LoggerManager.setup_logging(
        log_level=log_level,
        log_dir=log_dir,
        enable_console=enable_console,
        enable_file=enable_file
    )


def get_logger(name: str) -> logging.Logger:
    """获取日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器实例
    """
    return LoggerManager.get_logger(name)


# 创建默认日志器
logger = get_logger(__name__)
