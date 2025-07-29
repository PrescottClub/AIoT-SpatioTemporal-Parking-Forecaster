"""
训练器模块

实现模型训练的完整流程，包括训练循环、验证、早停等功能。

Author: AI Assistant
Date: 2025-07-29
"""

import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import numpy as np
from datetime import datetime
import json

from ..models.spatiotemporal_model import SpatioTemporalPredictor
from ..utils.logger import get_logger
from ..utils.metrics import MetricsTracker, calculate_metrics
from ..config import Config

logger = get_logger(__name__)


def spatiotemporal_collate_fn(batch):
    """自定义的批次整理函数

    确保node_features和edge_index不被批次化，因为它们对所有样本都是相同的。
    """
    # 获取第一个样本作为模板
    first_sample = batch[0]

    # node_features和edge_index对所有样本都相同，不需要批次化
    collated = {
        'node_features': first_sample['node_features'],
        'edge_index': first_sample['edge_index'],
    }

    # 对时间序列和目标进行批次化
    collated['time_series'] = torch.stack([sample['time_series'] for sample in batch])
    collated['targets'] = torch.stack([sample['targets'] for sample in batch])

    # 处理可选的边特征
    if 'edge_attr' in first_sample:
        collated['edge_attr'] = first_sample['edge_attr']

    return collated


class SpatioTemporalDataset(Dataset):
    """时空数据集类

    用于PyTorch DataLoader的数据集包装器。

    Args:
        node_features: 节点特征 [num_nodes, node_features]
        edge_index: 边索引 [2, num_edges]
        time_series: 时间序列数据 [num_samples, seq_len, features]
        targets: 目标值 [num_samples, pred_len, num_nodes, output_dim]
        edge_attr: 边特征 [num_edges, edge_features] (可选)
    """

    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        time_series: torch.Tensor,
        targets: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ):
        self.node_features = node_features
        self.edge_index = edge_index
        self.time_series = time_series
        self.targets = targets
        self.edge_attr = edge_attr

        # 验证数据形状
        assert len(time_series) == len(targets), "时间序列和目标数量不匹配"

    def __len__(self) -> int:
        return len(self.time_series)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = {
            'node_features': self.node_features,
            'edge_index': self.edge_index,
            'time_series': self.time_series[idx],
            'targets': self.targets[idx]
        }

        if self.edge_attr is not None:
            batch['edge_attr'] = self.edge_attr

        return batch


class EarlyStopping:
    """早停机制

    监控验证指标，在指标不再改善时停止训练。

    Args:
        patience: 容忍的epoch数
        min_delta: 最小改善幅度
        mode: 'min' 或 'max'
        restore_best_weights: 是否恢复最佳权重
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

        self.compare = self._get_compare_fn()

    def _get_compare_fn(self) -> Callable:
        """获取比较函数"""
        if self.mode == 'min':
            return lambda current, best: current < (best - self.min_delta)
        else:
            return lambda current, best: current > (best + self.min_delta)

    def __call__(self, score: float, model: nn.Module) -> bool:
        """检查是否应该早停

        Args:
            score: 当前验证分数
            model: 模型实例

        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("恢复最佳模型权重")

        return self.early_stop


class LearningRateScheduler:
    """学习率调度器包装器"""

    def __init__(self, optimizer: optim.Optimizer, scheduler_type: str, **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        if scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
        elif scheduler_type == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")

    def step(self, metric: Optional[float] = None):
        """更新学习率"""
        if self.scheduler_type == 'plateau':
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_last_lr(self) -> List[float]:
        """获取当前学习率"""
        return self.scheduler.get_last_lr()


class Trainer:
    """时空模型训练器

    提供完整的模型训练功能，包括训练循环、验证、早停、模型保存等。

    Args:
        model: 要训练的模型
        config: 训练配置
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器 (可选)

    Example:
        >>> trainer = Trainer(model, config, train_loader, val_loader)
        >>> trainer.train()
    """

    def __init__(
        self,
        model: SpatioTemporalPredictor,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 设备配置
        self.device = torch.device(config.system.device)
        self.model.to(self.device)

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 早停
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            mode=config.training.mode,
            restore_best_weights=True
        )

        # 指标跟踪
        self.metrics_tracker = MetricsTracker()

        # 训练状态
        self.current_epoch = 0
        self.best_score = float('inf') if config.training.mode == 'min' else float('-inf')
        self.training_history = []

        # 日志和保存路径
        self.checkpoint_dir = Path(config.system.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"训练器初始化完成，设备: {self.device}")
        logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_type = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay

        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_type}")

    def _create_scheduler(self) -> Optional[LearningRateScheduler]:
        """创建学习率调度器"""
        if not hasattr(self.config.training, 'scheduler'):
            return None

        scheduler_type = self.config.training.scheduler

        if scheduler_type == 'cosine':
            return LearningRateScheduler(
                self.optimizer, 'cosine',
                T_max=self.config.training.epochs
            )
        elif scheduler_type == 'step':
            return LearningRateScheduler(
                self.optimizer, 'step',
                step_size=self.config.training.epochs // 3,
                gamma=0.1
            )
        elif scheduler_type == 'plateau':
            return LearningRateScheduler(
                self.optimizer, 'plateau',
                mode=self.config.training.mode,
                patience=self.config.training.patience // 2,
                factor=0.5
            )
        else:
            return None

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch

        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # 移动数据到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # 前向传播
            self.optimizer.zero_grad()
            result = self.model(batch)
            loss = result['loss']

            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"检测到无效损失值: {loss.item()}")
                continue

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if hasattr(self.config.training, 'gradient_clip_val'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_val
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 记录训练进度
            if batch_idx % self.config.training.log_every_n_steps == 0:
                logger.debug(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        return {'train_loss': avg_loss}

    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch

        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # 前向传播
                result = self.model(batch)
                loss = result['loss']
                predictions = result['predictions']

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1

                    # 收集预测和目标用于指标计算
                    all_predictions.append(predictions.cpu())
                    all_targets.append(batch['targets'].cpu())

        # 计算平均损失
        avg_loss = total_loss / max(num_batches, 1)

        # 计算评估指标
        metrics = {'val_loss': avg_loss}

        if all_predictions and all_targets:
            predictions_tensor = torch.cat(all_predictions, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)

            # 计算各种指标
            eval_metrics = calculate_metrics(
                targets_tensor.numpy().flatten(),
                predictions_tensor.numpy().flatten(),
                metrics=['mae', 'rmse', 'mape', 'r2']
            )

            # 添加前缀以区分验证指标
            for key, value in eval_metrics.items():
                metrics[f'val_{key.lower()}'] = value

        return metrics

    def train(self) -> Dict[str, Any]:
        """执行完整的训练过程

        Returns:
            训练历史和最终结果
        """
        logger.info(f"开始训练，共 {self.config.training.epochs} 个epoch")
        start_time = time.time()

        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # 训练阶段
            train_metrics = self.train_epoch()

            # 验证阶段
            val_metrics = self.validate_epoch()

            # 合并指标
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch
            epoch_metrics['lr'] = self.optimizer.param_groups[0]['lr']

            # 更新指标跟踪器
            self.metrics_tracker.update(epoch_metrics)

            # 学习率调度
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'scheduler_type') and self.scheduler.scheduler_type == 'plateau':
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            # 早停检查
            monitor_metric = val_metrics.get(self.config.training.monitor_metric, val_metrics['val_loss'])
            if self.early_stopping(monitor_metric, self.model):
                logger.info(f"早停触发，在第 {epoch + 1} 个epoch停止训练")
                break

            # 保存检查点
            if self._should_save_checkpoint(monitor_metric):
                self.save_checkpoint(epoch, monitor_metric)

            # 记录进度
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_progress(epoch, epoch_metrics, epoch_time)

            # 保存历史
            self.training_history.append(epoch_metrics)

        total_time = time.time() - start_time
        logger.info(f"训练完成，总耗时: {total_time:.2f}秒")

        # 最终评估
        final_results = self._final_evaluation()

        return {
            'training_history': self.training_history,
            'final_results': final_results,
            'total_time': total_time,
            'best_score': self.best_score
        }

    def _should_save_checkpoint(self, current_score: float) -> bool:
        """判断是否应该保存检查点"""
        if self.config.training.mode == 'min':
            return current_score < self.best_score
        else:
            return current_score > self.best_score

    def save_checkpoint(self, epoch: int, score: float, filename: Optional[str] = None) -> str:
        """保存模型检查点

        Args:
            epoch: 当前epoch
            score: 当前分数
            filename: 文件名 (可选)

        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch_{epoch:03d}_{timestamp}.pth"

        filepath = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': score,
            'config': self.config.to_dict(),
            'training_history': self.training_history,
            'metrics_tracker': self.metrics_tracker.get_history()
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.scheduler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"检查点已保存: {filepath}")

        # 更新最佳分数
        self.best_score = score

        return str(filepath)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """加载模型检查点

        Args:
            filepath: 检查点文件路径

        Returns:
            检查点信息
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_score = checkpoint['score']
        self.training_history = checkpoint.get('training_history', [])

        logger.info(f"检查点已加载: {filepath}")
        return checkpoint

    def _log_epoch_progress(self, epoch: int, metrics: Dict[str, float], epoch_time: float):
        """记录epoch进度"""
        log_msg = f"Epoch {epoch + 1}/{self.config.training.epochs} "
        log_msg += f"({epoch_time:.2f}s) - "

        for key, value in metrics.items():
            if key != 'epoch':
                log_msg += f"{key}: {value:.4f} "

        logger.info(log_msg)

    def _final_evaluation(self) -> Dict[str, Any]:
        """最终评估"""
        results = {}

        # 在测试集上评估（如果有）
        if self.test_loader is not None:
            logger.info("在测试集上进行最终评估...")
            test_metrics = self.evaluate(self.test_loader, prefix='test')
            results['test_metrics'] = test_metrics

        # 获取最佳验证指标
        best_metrics = self.metrics_tracker.summary()
        results['best_metrics'] = best_metrics

        return results

    def evaluate(self, data_loader: DataLoader, prefix: str = 'eval') -> Dict[str, float]:
        """在给定数据集上评估模型

        Args:
            data_loader: 数据加载器
            prefix: 指标前缀

        Returns:
            评估指标
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                result = self.model(batch)
                loss = result['loss']
                predictions = result['predictions']

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1

                    all_predictions.append(predictions.cpu())
                    all_targets.append(batch['targets'].cpu())

        # 计算指标
        avg_loss = total_loss / max(num_batches, 1)
        metrics = {f'{prefix}_loss': avg_loss}

        if all_predictions and all_targets:
            predictions_tensor = torch.cat(all_predictions, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)

            eval_metrics = calculate_metrics(
                targets_tensor.numpy().flatten(),
                predictions_tensor.numpy().flatten(),
                metrics=['mae', 'rmse', 'mape', 'r2']
            )

            for key, value in eval_metrics.items():
                metrics[f'{prefix}_{key.lower()}'] = value

        return metrics
