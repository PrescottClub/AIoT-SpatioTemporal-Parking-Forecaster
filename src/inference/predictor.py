"""
推理预测器模块

实现模型推理和预测功能，包括单样本推理、批量推理和实时预测。

Author: AI Assistant
Date: 2025-07-29
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import time
from datetime import datetime, timedelta

from ..models.spatiotemporal_model import SpatioTemporalPredictor as TrainingPredictor
from ..data.preprocessor import DataPreprocessor
from ..utils.logger import get_logger
from ..utils.metrics import calculate_metrics
from ..config import Config

logger = get_logger(__name__)


class InferencePredictor:
    """推理预测器

    提供模型推理和预测功能的高级接口。

    Args:
        model_path: 训练好的模型路径
        config: 模型配置
        device: 推理设备

    Example:
        >>> predictor = InferencePredictor('models/best_model.pth')
        >>> predictions = predictor.predict(node_features, edge_index, time_series)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[TrainingPredictor] = None,
        config: Optional[Config] = None,
        device: Optional[str] = None
    ):
        self.config = config or Config()
        self.device = torch.device(device or self.config.system.device)

        # 加载模型
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("必须提供model或model_path")

        self.model.to(self.device)
        self.model.eval()

        # 预处理器
        self.preprocessor = None

        logger.info(f"推理预测器初始化完成，设备: {self.device}")

    def _load_model(self, model_path: str) -> TrainingPredictor:
        """加载训练好的模型

        Args:
            model_path: 模型文件路径

        Returns:
            加载的模型
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        # 从检查点恢复配置
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            # 更新当前配置
            for key, value in saved_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # 创建模型
        model = TrainingPredictor(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"模型已从 {model_path} 加载")
        return model

    def set_preprocessor(self, preprocessor: DataPreprocessor) -> None:
        """设置数据预处理器

        Args:
            preprocessor: 已拟合的数据预处理器
        """
        self.preprocessor = preprocessor
        logger.info("数据预处理器已设置")

    def predict(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        time_series: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_confidence: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """单次预测

        Args:
            node_features: 节点特征 [num_nodes, node_features]
            edge_index: 边索引 [2, num_edges]
            time_series: 时间序列 [batch_size, seq_len, features]
            edge_attr: 边特征 [num_edges, edge_features] (可选)
            return_confidence: 是否返回置信度

        Returns:
            预测结果 [batch_size, pred_len, num_nodes, output_dim]
            如果return_confidence=True，还返回置信度
        """
        # 移动数据到设备
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        time_series = time_series.to(self.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)

        with torch.no_grad():
            predictions = self.model.model(
                node_features, edge_index, time_series, edge_attr
            )

        if return_confidence:
            # 简单的置信度估计（基于预测方差）
            confidence = torch.ones_like(predictions) * 0.8  # 占位实现
            return predictions, confidence

        return predictions

    def predict_batch(
        self,
        batch_data: List[Dict[str, torch.Tensor]],
        batch_size: int = 32
    ) -> List[torch.Tensor]:
        """批量预测

        Args:
            batch_data: 批量数据列表
            batch_size: 批次大小

        Returns:
            预测结果列表
        """
        predictions = []

        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]

            # 合并批次数据
            node_features = batch[0]['node_features']  # 所有样本共享
            edge_index = batch[0]['edge_index']  # 所有样本共享
            time_series = torch.stack([item['time_series'] for item in batch])

            edge_attr = None
            if 'edge_attr' in batch[0]:
                edge_attr = batch[0]['edge_attr']

            # 预测
            batch_predictions = self.predict(
                node_features, edge_index, time_series, edge_attr
            )

            # 分解批次结果
            for j in range(len(batch)):
                predictions.append(batch_predictions[j])

        return predictions

    def predict_realtime(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        time_series: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """实时预测

        Args:
            node_features: 节点特征
            edge_index: 边索引
            time_series: 时间序列
            edge_attr: 边特征 (可选)

        Returns:
            包含预测结果和元信息的字典
        """
        start_time = time.time()

        predictions = self.predict(
            node_features, edge_index, time_series, edge_attr
        )

        inference_time = time.time() - start_time

        return {
            'predictions': predictions,
            'inference_time': inference_time,
            'timestamp': datetime.now(),
            'input_shape': time_series.shape,
            'output_shape': predictions.shape
        }

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        seq_len: int,
        pred_len: int,
        edge_attr: Optional[torch.Tensor] = None
    ) -> pd.DataFrame:
        """从DataFrame预测

        Args:
            df: 输入数据DataFrame
            node_features: 节点特征
            edge_index: 边索引
            seq_len: 序列长度
            pred_len: 预测长度
            edge_attr: 边特征 (可选)

        Returns:
            包含预测结果的DataFrame
        """
        if self.preprocessor is None:
            raise ValueError("需要设置预处理器才能处理DataFrame")

        # 预处理数据
        df_processed = self.preprocessor.preprocess(df, fit=False)

        # 创建时间序列
        time_series_data = self.preprocessor.create_time_series(
            df_processed, seq_len=seq_len, pred_len=pred_len
        )

        # 预测
        predictions = self.predict(
            node_features, edge_index, time_series_data.sequences
        )

        # 转换为DataFrame
        results = []
        for i, pred in enumerate(predictions):
            for t in range(pred_len):
                for node in range(pred.shape[1]):
                    results.append({
                        'sample_id': i,
                        'time_step': t,
                        'node_id': node,
                        'prediction': pred[t, node, 0].item()
                    })

        return pd.DataFrame(results)


class BatchPredictor:
    """批量预测器

    专门用于大规模批量推理的高效预测器。

    Args:
        predictor: 基础预测器
        batch_size: 批次大小
        num_workers: 并行工作进程数
    """

    def __init__(
        self,
        predictor: InferencePredictor,
        batch_size: int = 64,
        num_workers: int = 1
    ):
        self.predictor = predictor
        self.batch_size = batch_size
        self.num_workers = num_workers

        logger.info(f"批量预测器初始化，批次大小: {batch_size}")

    def predict_dataset(
        self,
        dataset: List[Dict[str, torch.Tensor]],
        show_progress: bool = True
    ) -> List[torch.Tensor]:
        """预测整个数据集

        Args:
            dataset: 数据集
            show_progress: 是否显示进度

        Returns:
            预测结果列表
        """
        predictions = []
        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            batch_predictions = self.predictor.predict_batch(batch, len(batch))
            predictions.extend(batch_predictions)

            if show_progress:
                batch_num = i // self.batch_size + 1
                logger.info(f"批次 {batch_num}/{total_batches} 完成")

        return predictions

    def predict_streaming(
        self,
        data_stream,
        buffer_size: int = 100
    ):
        """流式预测

        Args:
            data_stream: 数据流生成器
            buffer_size: 缓冲区大小

        Yields:
            预测结果
        """
        buffer = []

        for data in data_stream:
            buffer.append(data)

            if len(buffer) >= buffer_size:
                predictions = self.predictor.predict_batch(buffer, self.batch_size)
                for pred in predictions:
                    yield pred
                buffer = []

        # 处理剩余数据
        if buffer:
            predictions = self.predictor.predict_batch(buffer, self.batch_size)
            for pred in predictions:
                yield pred


class SpatioTemporalPredictor:
    """时空预测器（兼容性别名）

    为了保持向后兼容性而提供的别名。
    """

    def __init__(self, *args, **kwargs):
        self._predictor = InferencePredictor(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._predictor, name)
