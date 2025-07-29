"""
评估指标模块

提供各种评估指标的计算功能。

Author: AI Assistant
Date: 2025-07-29
"""

import numpy as np
import torch
from typing import Dict, List, Union, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings


def MAE(y_true: Union[np.ndarray, torch.Tensor], 
        y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """计算平均绝对误差 (Mean Absolute Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        MAE值
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return float(mean_absolute_error(y_true, y_pred))


def RMSE(y_true: Union[np.ndarray, torch.Tensor], 
         y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """计算均方根误差 (Root Mean Square Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        RMSE值
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def MAPE(y_true: Union[np.ndarray, torch.Tensor], 
         y_pred: Union[np.ndarray, torch.Tensor],
         epsilon: float = 1e-8) -> float:
    """计算平均绝对百分比误差 (Mean Absolute Percentage Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        epsilon: 避免除零的小值
        
    Returns:
        MAPE值 (百分比)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 避免除零
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100)


def R2(y_true: Union[np.ndarray, torch.Tensor], 
       y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """计算决定系数 (R-squared)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        R²值
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return float(r2_score(y_true, y_pred))


def SMAPE(y_true: Union[np.ndarray, torch.Tensor], 
          y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """计算对称平均绝对百分比误差 (Symmetric Mean Absolute Percentage Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        SMAPE值 (百分比)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # 避免除零
    denominator = np.where(denominator == 0, 1e-8, denominator)
    
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def calculate_metrics(
    y_true: Union[np.ndarray, torch.Tensor], 
    y_pred: Union[np.ndarray, torch.Tensor],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """计算多个评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        metrics: 要计算的指标列表，默认为所有指标
        
    Returns:
        包含各指标值的字典
    """
    if metrics is None:
        metrics = ['mae', 'rmse', 'mape', 'r2', 'smape']
    
    results = {}
    
    # 确保输入格式正确
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 展平数组
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 计算各指标
    metric_functions = {
        'mae': MAE,
        'rmse': RMSE,
        'mape': MAPE,
        'r2': R2,
        'smape': SMAPE
    }
    
    for metric in metrics:
        if metric.lower() in metric_functions:
            try:
                results[metric.upper()] = metric_functions[metric.lower()](y_true, y_pred)
            except Exception as e:
                warnings.warn(f"计算{metric}时出错: {e}")
                results[metric.upper()] = float('nan')
        else:
            warnings.warn(f"未知指标: {metric}")
    
    return results


class MetricsTracker:
    """指标跟踪器
    
    用于在训练过程中跟踪和记录各种指标。
    """
    
    def __init__(self):
        """初始化指标跟踪器"""
        self.metrics_history: Dict[str, List[float]] = {}
        self.current_metrics: Dict[str, float] = {}
    
    def update(self, metrics: Dict[str, float]) -> None:
        """更新指标
        
        Args:
            metrics: 指标字典
        """
        self.current_metrics.update(metrics)
        
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(value)
    
    def get_current(self) -> Dict[str, float]:
        """获取当前指标
        
        Returns:
            当前指标字典
        """
        return self.current_metrics.copy()
    
    def get_history(self, metric_name: Optional[str] = None) -> Union[Dict[str, List[float]], List[float]]:
        """获取指标历史
        
        Args:
            metric_name: 指标名称，如果为None则返回所有指标历史
            
        Returns:
            指标历史
        """
        if metric_name is None:
            return self.metrics_history.copy()
        else:
            return self.metrics_history.get(metric_name, [])
    
    def get_best(self, metric_name: str, mode: str = 'min') -> Tuple[float, int]:
        """获取最佳指标值
        
        Args:
            metric_name: 指标名称
            mode: 'min' 或 'max'
            
        Returns:
            (最佳值, 最佳值的索引)
        """
        if metric_name not in self.metrics_history:
            raise ValueError(f"指标 {metric_name} 不存在")
        
        history = self.metrics_history[metric_name]
        if not history:
            raise ValueError(f"指标 {metric_name} 历史为空")
        
        if mode == 'min':
            best_value = min(history)
            best_index = history.index(best_value)
        elif mode == 'max':
            best_value = max(history)
            best_index = history.index(best_value)
        else:
            raise ValueError("mode 必须是 'min' 或 'max'")
        
        return best_value, best_index
    
    def get_average(self, metric_name: str, last_n: Optional[int] = None) -> float:
        """获取指标平均值
        
        Args:
            metric_name: 指标名称
            last_n: 最近n个值的平均，如果为None则计算所有值的平均
            
        Returns:
            平均值
        """
        if metric_name not in self.metrics_history:
            raise ValueError(f"指标 {metric_name} 不存在")
        
        history = self.metrics_history[metric_name]
        if not history:
            raise ValueError(f"指标 {metric_name} 历史为空")
        
        if last_n is None:
            return float(np.mean(history))
        else:
            return float(np.mean(history[-last_n:]))
    
    def reset(self) -> None:
        """重置所有指标"""
        self.metrics_history.clear()
        self.current_metrics.clear()
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """获取指标摘要
        
        Returns:
            包含各指标统计信息的字典
        """
        summary = {}
        
        for metric_name, history in self.metrics_history.items():
            if history:
                summary[metric_name] = {
                    'current': history[-1],
                    'best': min(history),
                    'worst': max(history),
                    'mean': float(np.mean(history)),
                    'std': float(np.std(history)),
                    'count': len(history)
                }
        
        return summary
