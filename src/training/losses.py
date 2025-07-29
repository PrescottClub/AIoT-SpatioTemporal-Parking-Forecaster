"""
损失函数模块

实现各种适用于时空预测的损失函数。

Author: AI Assistant
Date: 2025-07-29
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MSELoss(nn.Module):
    """均方误差损失"""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.mse(predictions, targets)


class MAELoss(nn.Module):
    """平均绝对误差损失"""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.mae = nn.L1Loss(reduction=reduction)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.mae(predictions, targets)


class HuberLoss(nn.Module):
    """Huber损失（对异常值更鲁棒）"""

    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.huber = nn.SmoothL1Loss(reduction=reduction, beta=delta)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.huber(predictions, targets)


class MAPELoss(nn.Module):
    """平均绝对百分比误差损失"""

    def __init__(self, epsilon: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 避免除零
        targets_safe = torch.where(torch.abs(targets) < self.epsilon,
                                 torch.sign(targets) * self.epsilon, targets)

        mape = torch.abs((targets - predictions) / targets_safe)

        if self.reduction == 'mean':
            return torch.mean(mape)
        elif self.reduction == 'sum':
            return torch.sum(mape)
        else:
            return mape


class QuantileLoss(nn.Module):
    """分位数损失（用于不确定性估计）"""

    def __init__(self, quantile: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.quantile = quantile
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        errors = targets - predictions
        loss = torch.where(errors >= 0,
                          self.quantile * errors,
                          (self.quantile - 1) * errors)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class WeightedMSELoss(nn.Module):
    """加权均方误差损失

    可以根据时间步或节点重要性进行加权。
    """

    def __init__(self,
                 temporal_weights: Optional[torch.Tensor] = None,
                 spatial_weights: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.temporal_weights = temporal_weights
        self.spatial_weights = spatial_weights
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions/targets shape: [batch_size, pred_len, num_nodes, output_dim]
        mse = (predictions - targets) ** 2

        # 应用时间权重
        if self.temporal_weights is not None:
            temporal_weights = self.temporal_weights.to(mse.device)
            if temporal_weights.dim() == 1:
                temporal_weights = temporal_weights.view(1, -1, 1, 1)
            mse = mse * temporal_weights

        # 应用空间权重
        if self.spatial_weights is not None:
            spatial_weights = self.spatial_weights.to(mse.device)
            if spatial_weights.dim() == 1:
                spatial_weights = spatial_weights.view(1, 1, -1, 1)
            mse = mse * spatial_weights

        if self.reduction == 'mean':
            return torch.mean(mse)
        elif self.reduction == 'sum':
            return torch.sum(mse)
        else:
            return mse


class FocalLoss(nn.Module):
    """Focal损失（关注难样本）"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 计算基础损失（MSE）
        mse = (predictions - targets) ** 2

        # 计算调制因子
        pt = torch.exp(-mse)  # 预测置信度
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        focal_loss = focal_weight * mse

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class SpatioTemporalLoss(nn.Module):
    """时空损失函数

    结合多种损失函数，专门为时空预测任务设计。

    Args:
        base_loss: 基础损失函数类型
        temporal_weight: 时间维度权重
        spatial_weight: 空间维度权重
        consistency_weight: 一致性损失权重
        smoothness_weight: 平滑性损失权重
    """

    def __init__(self,
                 base_loss: str = 'mse',
                 temporal_weight: float = 1.0,
                 spatial_weight: float = 1.0,
                 consistency_weight: float = 0.1,
                 smoothness_weight: float = 0.1,
                 **loss_kwargs):
        super().__init__()

        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight

        # 创建基础损失函数
        self.base_loss = self._create_base_loss(base_loss, **loss_kwargs)

        logger.info(f"时空损失函数初始化: {base_loss}, "
                   f"权重 - 时间: {temporal_weight}, 空间: {spatial_weight}, "
                   f"一致性: {consistency_weight}, 平滑性: {smoothness_weight}")

    def _create_base_loss(self, loss_type: str, **kwargs) -> nn.Module:
        """创建基础损失函数"""
        if loss_type == 'mse':
            return MSELoss(**kwargs)
        elif loss_type == 'mae':
            return MAELoss(**kwargs)
        elif loss_type == 'huber':
            return HuberLoss(**kwargs)
        elif loss_type == 'mape':
            return MAPELoss(**kwargs)
        elif loss_type == 'quantile':
            return QuantileLoss(**kwargs)
        elif loss_type == 'focal':
            return FocalLoss(**kwargs)
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                node_features: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            predictions: 预测值 [batch_size, pred_len, num_nodes, output_dim]
            targets: 目标值 [batch_size, pred_len, num_nodes, output_dim]
            node_features: 节点特征 (可选)
            edge_index: 边索引 (可选)

        Returns:
            损失字典
        """
        losses = {}

        # 1. 基础预测损失
        base_loss = self.base_loss(predictions, targets)
        losses['base_loss'] = base_loss

        # 2. 时间一致性损失
        if self.consistency_weight > 0:
            temporal_consistency = self._temporal_consistency_loss(predictions)
            losses['temporal_consistency'] = temporal_consistency

        # 3. 空间平滑性损失
        if self.smoothness_weight > 0 and edge_index is not None:
            spatial_smoothness = self._spatial_smoothness_loss(predictions, edge_index)
            losses['spatial_smoothness'] = spatial_smoothness

        # 4. 总损失
        total_loss = (self.temporal_weight * base_loss +
                     self.consistency_weight * losses.get('temporal_consistency', 0) +
                     self.smoothness_weight * losses.get('spatial_smoothness', 0))

        losses['total_loss'] = total_loss

        return losses

    def _temporal_consistency_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """时间一致性损失

        鼓励相邻时间步的预测保持平滑。
        """
        if predictions.size(1) < 2:
            return torch.tensor(0.0, device=predictions.device)

        # 计算相邻时间步的差异
        temporal_diff = predictions[:, 1:] - predictions[:, :-1]

        # L2正则化
        consistency_loss = torch.mean(temporal_diff ** 2)

        return consistency_loss

    def _spatial_smoothness_loss(self,
                                predictions: torch.Tensor,
                                edge_index: torch.Tensor) -> torch.Tensor:
        """空间平滑性损失

        鼓励相邻节点的预测保持相似。
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=predictions.device)

        # 获取边的源节点和目标节点
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        # 计算相邻节点预测的差异
        source_pred = predictions[:, :, source_nodes, :]  # [batch, time, num_edges, dim]
        target_pred = predictions[:, :, target_nodes, :]  # [batch, time, num_edges, dim]

        spatial_diff = source_pred - target_pred
        smoothness_loss = torch.mean(spatial_diff ** 2)

        return smoothness_loss


class MultiTaskLoss(nn.Module):
    """多任务损失函数

    用于同时预测多个目标（如占用率、流量等）。
    """

    def __init__(self, task_weights: Dict[str, float], loss_functions: Dict[str, nn.Module]):
        super().__init__()
        self.task_weights = task_weights
        self.loss_functions = nn.ModuleDict(loss_functions)

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            predictions: 预测字典
            targets: 目标字典

        Returns:
            损失字典
        """
        losses = {}
        total_loss = 0.0

        for task_name in self.task_weights:
            if task_name in predictions and task_name in targets:
                task_loss = self.loss_functions[task_name](
                    predictions[task_name], targets[task_name]
                )
                losses[f'{task_name}_loss'] = task_loss
                total_loss += self.task_weights[task_name] * task_loss

        losses['total_loss'] = total_loss
        return losses


def create_loss_function(loss_config: Dict[str, Any]) -> nn.Module:
    """创建损失函数的工厂函数

    Args:
        loss_config: 损失函数配置

    Returns:
        损失函数实例
    """
    loss_type = loss_config.get('type', 'mse')

    if loss_type == 'spatiotemporal':
        return SpatioTemporalLoss(**loss_config.get('params', {}))
    elif loss_type == 'multitask':
        return MultiTaskLoss(**loss_config.get('params', {}))
    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(**loss_config.get('params', {}))
    else:
        # 简单损失函数
        loss_classes = {
            'mse': MSELoss,
            'mae': MAELoss,
            'huber': HuberLoss,
            'mape': MAPELoss,
            'quantile': QuantileLoss,
            'focal': FocalLoss
        }

        if loss_type in loss_classes:
            return loss_classes[loss_type](**loss_config.get('params', {}))
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
