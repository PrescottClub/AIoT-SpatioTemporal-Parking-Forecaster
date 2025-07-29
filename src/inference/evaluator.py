"""
模型评估器模块

实现模型性能评估和分析功能。

Author: AI Assistant
Date: 2025-07-29
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .predictor import InferencePredictor
from ..utils.logger import get_logger
from ..utils.metrics import calculate_metrics, MetricsTracker

logger = get_logger(__name__)


class ModelEvaluator:
    """模型评估器
    
    提供全面的模型性能评估功能。
    
    Args:
        predictor: 推理预测器
        metrics: 要计算的指标列表
        
    Example:
        >>> evaluator = ModelEvaluator(predictor)
        >>> results = evaluator.evaluate(test_data, targets)
    """
    
    def __init__(
        self,
        predictor: InferencePredictor,
        metrics: List[str] = None
    ):
        self.predictor = predictor
        self.metrics = metrics or ['mae', 'rmse', 'mape', 'r2']
        
        logger.info(f"模型评估器初始化，评估指标: {self.metrics}")
    
    def evaluate(
        self,
        test_data: List[Dict[str, torch.Tensor]],
        targets: List[torch.Tensor],
        detailed: bool = True
    ) -> Dict[str, Any]:
        """评估模型性能
        
        Args:
            test_data: 测试数据
            targets: 真实目标值
            detailed: 是否返回详细结果
            
        Returns:
            评估结果字典
        """
        logger.info(f"开始评估模型，测试样本数: {len(test_data)}")
        
        # 预测
        start_time = time.time()
        predictions = self.predictor.predict_batch(test_data)
        inference_time = time.time() - start_time
        
        # 计算指标
        predictions_tensor = torch.stack(predictions)
        targets_tensor = torch.stack(targets)

        # 确保形状匹配
        if predictions_tensor.shape != targets_tensor.shape:
            logger.warning(f"预测和目标形状不匹配: {predictions_tensor.shape} vs {targets_tensor.shape}")
            # 调整形状以匹配
            min_shape = tuple(min(p, t) for p, t in zip(predictions_tensor.shape, targets_tensor.shape))
            predictions_tensor = predictions_tensor[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
            targets_tensor = targets_tensor[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]

        all_predictions = predictions_tensor.numpy().flatten()
        all_targets = targets_tensor.numpy().flatten()
        
        metrics_result = calculate_metrics(
            all_targets, all_predictions, self.metrics
        )
        
        results = {
            'metrics': metrics_result,
            'inference_time': inference_time,
            'samples_per_second': len(test_data) / inference_time,
            'num_samples': len(test_data)
        }
        
        if detailed:
            results.update(self._detailed_analysis(predictions, targets))
        
        logger.info(f"评估完成，主要指标: {metrics_result}")
        return results
    
    def _detailed_analysis(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """详细分析
        
        Args:
            predictions: 预测结果
            targets: 真实目标
            
        Returns:
            详细分析结果
        """
        pred_array = torch.stack(predictions).numpy()
        target_array = torch.stack(targets).numpy()
        
        # 按时间步分析
        timestep_metrics = {}
        for t in range(pred_array.shape[1]):
            timestep_pred = pred_array[:, t, :, :].flatten()
            timestep_target = target_array[:, t, :, :].flatten()
            
            timestep_metrics[f'timestep_{t}'] = calculate_metrics(
                timestep_target, timestep_pred, self.metrics
            )
        
        # 按节点分析
        node_metrics = {}
        for n in range(pred_array.shape[2]):
            node_pred = pred_array[:, :, n, :].flatten()
            node_target = target_array[:, :, n, :].flatten()
            
            node_metrics[f'node_{n}'] = calculate_metrics(
                node_target, node_pred, self.metrics
            )
        
        # 误差分布
        errors = pred_array - target_array
        error_stats = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'percentiles': {
                '25': float(np.percentile(errors, 25)),
                '50': float(np.percentile(errors, 50)),
                '75': float(np.percentile(errors, 75)),
                '95': float(np.percentile(errors, 95))
            }
        }
        
        return {
            'timestep_metrics': timestep_metrics,
            'node_metrics': node_metrics,
            'error_statistics': error_stats,
            'prediction_shape': pred_array.shape,
            'target_shape': target_array.shape
        }
    
    def evaluate_by_conditions(
        self,
        test_data: List[Dict[str, torch.Tensor]],
        targets: List[torch.Tensor],
        conditions: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """按条件评估
        
        Args:
            test_data: 测试数据
            targets: 真实目标
            conditions: 评估条件
            
        Returns:
            按条件分组的评估结果
        """
        # 这里可以根据不同条件（如时间段、节点类型等）分组评估
        # 占位实现
        return {'overall': self.evaluate(test_data, targets)}
    
    def compare_models(
        self,
        predictors: Dict[str, InferencePredictor],
        test_data: List[Dict[str, torch.Tensor]],
        targets: List[torch.Tensor]
    ) -> Dict[str, Dict[str, Any]]:
        """比较多个模型
        
        Args:
            predictors: 预测器字典
            test_data: 测试数据
            targets: 真实目标
            
        Returns:
            模型比较结果
        """
        results = {}
        
        for name, predictor in predictors.items():
            logger.info(f"评估模型: {name}")
            
            # 临时替换预测器
            original_predictor = self.predictor
            self.predictor = predictor
            
            try:
                results[name] = self.evaluate(test_data, targets, detailed=False)
            finally:
                self.predictor = original_predictor
        
        return results


class PerformanceAnalyzer:
    """性能分析器
    
    分析模型的推理性能和资源使用情况。
    
    Args:
        predictor: 推理预测器
    """
    
    def __init__(self, predictor: InferencePredictor):
        self.predictor = predictor
        
        logger.info("性能分析器初始化完成")
    
    def benchmark_inference(
        self,
        test_data: List[Dict[str, torch.Tensor]],
        batch_sizes: List[int] = None,
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """推理性能基准测试
        
        Args:
            test_data: 测试数据
            batch_sizes: 要测试的批次大小列表
            num_runs: 运行次数
            
        Returns:
            性能基准结果
        """
        batch_sizes = batch_sizes or [1, 4, 8, 16, 32]
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"测试批次大小: {batch_size}")
            
            times = []
            for run in range(num_runs):
                # 准备批次数据
                batch_data = test_data[:batch_size]
                
                # 计时
                start_time = time.time()
                _ = self.predictor.predict_batch(batch_data, batch_size)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / avg_time
            
            results[f'batch_size_{batch_size}'] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'throughput': throughput,
                'times': times
            }
        
        return results
    
    def analyze_memory_usage(
        self,
        test_data: List[Dict[str, torch.Tensor]],
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """分析内存使用情况
        
        Args:
            test_data: 测试数据
            batch_size: 批次大小
            
        Returns:
            内存使用分析结果
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，无法分析GPU内存使用")
            return {'gpu_available': False}
        
        # 清空缓存
        torch.cuda.empty_cache()
        
        # 记录初始内存
        initial_memory = torch.cuda.memory_allocated()
        
        # 执行推理
        batch_data = test_data[:batch_size]
        _ = self.predictor.predict_batch(batch_data, batch_size)
        
        # 记录峰值内存
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        
        return {
            'gpu_available': True,
            'initial_memory_mb': initial_memory / 1024**2,
            'peak_memory_mb': peak_memory / 1024**2,
            'current_memory_mb': current_memory / 1024**2,
            'memory_increase_mb': (current_memory - initial_memory) / 1024**2
        }
    
    def profile_model_components(
        self,
        test_data: Dict[str, torch.Tensor],
        num_runs: int = 10
    ) -> Dict[str, float]:
        """分析模型各组件的耗时
        
        Args:
            test_data: 单个测试样本
            num_runs: 运行次数
            
        Returns:
            各组件耗时分析
        """
        # 这里可以使用torch.profiler进行详细的性能分析
        # 占位实现
        total_times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.predictor.predict(
                test_data['node_features'],
                test_data['edge_index'],
                test_data['time_series'].unsqueeze(0)
            )
            total_times.append(time.time() - start_time)
        
        return {
            'total_time_avg': np.mean(total_times),
            'total_time_std': np.std(total_times),
            'total_time_min': np.min(total_times),
            'total_time_max': np.max(total_times)
        }
