"""
预测结果后处理器模块

实现预测结果的后处理、格式转换和可视化功能。

Author: AI Assistant
Date: 2025-07-29
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
from pathlib import Path

from ..data.preprocessor import DataPreprocessor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PredictionPostProcessor:
    """预测结果后处理器
    
    处理模型预测结果，包括反标准化、格式转换、置信区间计算等。
    
    Args:
        preprocessor: 数据预处理器（用于反标准化）
        output_format: 输出格式 ('tensor', 'numpy', 'dataframe', 'json')
        
    Example:
        >>> processor = PredictionPostProcessor(preprocessor)
        >>> results = processor.process(predictions, metadata)
    """
    
    def __init__(
        self,
        preprocessor: Optional[DataPreprocessor] = None,
        output_format: str = 'dataframe'
    ):
        self.preprocessor = preprocessor
        self.output_format = output_format
        
        logger.info(f"预测结果后处理器初始化，输出格式: {output_format}")
    
    def process(
        self,
        predictions: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, np.ndarray, pd.DataFrame, Dict[str, Any]]:
        """处理预测结果
        
        Args:
            predictions: 预测结果 [batch_size, pred_len, num_nodes, output_dim]
            metadata: 元数据信息
            confidence: 置信度 (可选)
            
        Returns:
            处理后的预测结果
        """
        # 反标准化
        if self.preprocessor is not None:
            predictions = self._denormalize(predictions)
            if confidence is not None:
                confidence = self._denormalize_confidence(confidence)
        
        # 格式转换
        if self.output_format == 'tensor':
            return predictions
        elif self.output_format == 'numpy':
            return predictions.numpy()
        elif self.output_format == 'dataframe':
            return self._to_dataframe(predictions, metadata, confidence)
        elif self.output_format == 'json':
            return self._to_json(predictions, metadata, confidence)
        else:
            raise ValueError(f"不支持的输出格式: {self.output_format}")
    
    def _denormalize(self, predictions: torch.Tensor) -> torch.Tensor:
        """反标准化预测结果
        
        Args:
            predictions: 标准化的预测结果
            
        Returns:
            反标准化的预测结果
        """
        if not hasattr(self.preprocessor, 'target_scaler'):
            logger.warning("预处理器没有目标缩放器，跳过反标准化")
            return predictions
        
        # 获取原始形状
        original_shape = predictions.shape
        
        # 重塑为2D进行反标准化
        predictions_2d = predictions.view(-1, original_shape[-1])
        
        # 反标准化
        predictions_denorm = torch.from_numpy(
            self.preprocessor.target_scaler.inverse_transform(
                predictions_2d.numpy()
            )
        ).float()
        
        # 恢复原始形状
        return predictions_denorm.view(original_shape)
    
    def _denormalize_confidence(self, confidence: torch.Tensor) -> torch.Tensor:
        """反标准化置信度
        
        Args:
            confidence: 标准化的置信度
            
        Returns:
            反标准化的置信度
        """
        # 置信度通常不需要反标准化，但可以根据需要调整
        return confidence
    
    def _to_dataframe(
        self,
        predictions: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: Optional[torch.Tensor] = None
    ) -> pd.DataFrame:
        """转换为DataFrame格式
        
        Args:
            predictions: 预测结果
            metadata: 元数据
            confidence: 置信度
            
        Returns:
            DataFrame格式的预测结果
        """
        batch_size, pred_len, num_nodes, output_dim = predictions.shape
        
        results = []
        
        # 获取时间信息
        start_time = metadata.get('start_time', datetime.now()) if metadata else datetime.now()
        time_interval = metadata.get('time_interval', timedelta(minutes=10)) if metadata else timedelta(minutes=10)
        
        # 获取节点信息
        node_ids = metadata.get('node_ids', list(range(num_nodes))) if metadata else list(range(num_nodes))
        
        for batch_idx in range(batch_size):
            for time_idx in range(pred_len):
                prediction_time = start_time + time_interval * time_idx
                
                for node_idx in range(num_nodes):
                    node_id = node_ids[node_idx]
                    
                    for dim_idx in range(output_dim):
                        row = {
                            'batch_id': batch_idx,
                            'timestamp': prediction_time,
                            'time_step': time_idx,
                            'node_id': node_id,
                            'dimension': dim_idx,
                            'prediction': predictions[batch_idx, time_idx, node_idx, dim_idx].item()
                        }
                        
                        if confidence is not None:
                            row['confidence'] = confidence[batch_idx, time_idx, node_idx, dim_idx].item()
                        
                        results.append(row)
        
        return pd.DataFrame(results)
    
    def _to_json(
        self,
        predictions: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """转换为JSON格式
        
        Args:
            predictions: 预测结果
            metadata: 元数据
            confidence: 置信度
            
        Returns:
            JSON格式的预测结果
        """
        batch_size, pred_len, num_nodes, output_dim = predictions.shape
        
        result = {
            'metadata': {
                'batch_size': batch_size,
                'prediction_length': pred_len,
                'num_nodes': num_nodes,
                'output_dimension': output_dim,
                'timestamp': datetime.now().isoformat(),
                'has_confidence': confidence is not None
            },
            'predictions': []
        }
        
        # 添加额外的元数据
        if metadata:
            result['metadata'].update(metadata)
        
        # 转换预测结果
        for batch_idx in range(batch_size):
            batch_predictions = {
                'batch_id': batch_idx,
                'time_series': []
            }
            
            for time_idx in range(pred_len):
                time_step = {
                    'time_step': time_idx,
                    'nodes': []
                }
                
                for node_idx in range(num_nodes):
                    node_data = {
                        'node_id': node_idx,
                        'values': predictions[batch_idx, time_idx, node_idx, :].tolist()
                    }
                    
                    if confidence is not None:
                        node_data['confidence'] = confidence[batch_idx, time_idx, node_idx, :].tolist()
                    
                    time_step['nodes'].append(node_data)
                
                batch_predictions['time_series'].append(time_step)
            
            result['predictions'].append(batch_predictions)
        
        return result
    
    def calculate_confidence_intervals(
        self,
        predictions: torch.Tensor,
        confidence: torch.Tensor,
        confidence_level: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算置信区间
        
        Args:
            predictions: 预测结果
            confidence: 置信度
            confidence_level: 置信水平
            
        Returns:
            下界和上界
        """
        # 简单的置信区间计算（假设正态分布）
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        
        margin = z_score * confidence
        lower_bound = predictions - margin
        upper_bound = predictions + margin
        
        return lower_bound, upper_bound
    
    def aggregate_predictions(
        self,
        predictions: torch.Tensor,
        aggregation_method: str = 'mean',
        axis: Optional[int] = None
    ) -> torch.Tensor:
        """聚合预测结果
        
        Args:
            predictions: 预测结果
            aggregation_method: 聚合方法 ('mean', 'sum', 'max', 'min')
            axis: 聚合轴
            
        Returns:
            聚合后的预测结果
        """
        if aggregation_method == 'mean':
            return torch.mean(predictions, dim=axis, keepdim=True)
        elif aggregation_method == 'sum':
            return torch.sum(predictions, dim=axis, keepdim=True)
        elif aggregation_method == 'max':
            return torch.max(predictions, dim=axis, keepdim=True)[0]
        elif aggregation_method == 'min':
            return torch.min(predictions, dim=axis, keepdim=True)[0]
        else:
            raise ValueError(f"不支持的聚合方法: {aggregation_method}")
    
    def save_predictions(
        self,
        predictions: Union[pd.DataFrame, Dict[str, Any]],
        filepath: str,
        format: str = 'auto'
    ) -> None:
        """保存预测结果
        
        Args:
            predictions: 预测结果
            filepath: 保存路径
            format: 保存格式 ('auto', 'csv', 'json', 'parquet')
        """
        filepath = Path(filepath)
        
        # 自动检测格式
        if format == 'auto':
            format = filepath.suffix.lower().lstrip('.')
        
        # 确保目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(predictions, pd.DataFrame):
            if format == 'csv':
                predictions.to_csv(filepath, index=False)
            elif format == 'parquet':
                predictions.to_parquet(filepath, index=False)
            elif format == 'json':
                predictions.to_json(filepath, orient='records', date_format='iso')
            else:
                raise ValueError(f"DataFrame不支持格式: {format}")
        
        elif isinstance(predictions, dict):
            if format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, indent=2, ensure_ascii=False, default=str)
            else:
                raise ValueError(f"字典不支持格式: {format}")
        
        else:
            raise ValueError(f"不支持的预测结果类型: {type(predictions)}")
        
        logger.info(f"预测结果已保存到: {filepath}")
    
    def create_summary_report(
        self,
        predictions: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建预测摘要报告
        
        Args:
            predictions: 预测结果
            metadata: 元数据
            
        Returns:
            摘要报告
        """
        pred_array = predictions.numpy()
        
        summary = {
            'shape': pred_array.shape,
            'statistics': {
                'mean': float(np.mean(pred_array)),
                'std': float(np.std(pred_array)),
                'min': float(np.min(pred_array)),
                'max': float(np.max(pred_array)),
                'median': float(np.median(pred_array))
            },
            'percentiles': {
                '25': float(np.percentile(pred_array, 25)),
                '75': float(np.percentile(pred_array, 75)),
                '95': float(np.percentile(pred_array, 95)),
                '99': float(np.percentile(pred_array, 99))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            summary['metadata'] = metadata
        
        return summary
