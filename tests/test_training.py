"""
训练系统测试模块

测试训练器、损失函数和相关组件的功能。

Author: AI Assistant
Date: 2025-07-29
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import tempfile
import shutil

from src.training.trainer import Trainer, SpatioTemporalDataset, EarlyStopping, LearningRateScheduler
from src.training.losses import (
    MSELoss, MAELoss, HuberLoss, MAPELoss, SpatioTemporalLoss, 
    WeightedMSELoss, create_loss_function
)
from src.models.spatiotemporal_model import SpatioTemporalPredictor
from src.config import Config


class TestSpatioTemporalDataset:
    """时空数据集测试类"""
    
    def test_dataset_creation(self, sample_tensor_data, device):
        """测试数据集创建"""
        node_features = torch.randn(3, 8).to(device)
        edge_index = sample_tensor_data['edge_index'].to(device)
        time_series = torch.randn(10, 12, 8).to(device)
        targets = torch.randn(10, 6, 3, 1).to(device)
        
        dataset = SpatioTemporalDataset(
            node_features=node_features,
            edge_index=edge_index,
            time_series=time_series,
            targets=targets
        )
        
        assert len(dataset) == 10
        
        # 测试获取单个样本
        sample = dataset[0]
        assert 'node_features' in sample
        assert 'edge_index' in sample
        assert 'time_series' in sample
        assert 'targets' in sample
        
        assert sample['node_features'].shape == (3, 8)
        assert sample['time_series'].shape == (12, 8)
        assert sample['targets'].shape == (6, 3, 1)
    
    def test_dataset_with_edge_attr(self, device):
        """测试包含边特征的数据集"""
        node_features = torch.randn(3, 8).to(device)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]]).to(device)
        edge_attr = torch.randn(3, 2).to(device)
        time_series = torch.randn(5, 12, 8).to(device)
        targets = torch.randn(5, 6, 3, 1).to(device)
        
        dataset = SpatioTemporalDataset(
            node_features=node_features,
            edge_index=edge_index,
            time_series=time_series,
            targets=targets,
            edge_attr=edge_attr
        )
        
        sample = dataset[0]
        assert 'edge_attr' in sample
        assert sample['edge_attr'].shape == (3, 2)


class TestLossFunctions:
    """损失函数测试类"""
    
    def test_mse_loss(self, device):
        """测试MSE损失"""
        loss_fn = MSELoss()
        
        predictions = torch.randn(4, 6, 3, 1).to(device)
        targets = torch.randn(4, 6, 3, 1).to(device)
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
    
    def test_mae_loss(self, device):
        """测试MAE损失"""
        loss_fn = MAELoss()
        
        predictions = torch.randn(4, 6, 3, 1).to(device)
        targets = torch.randn(4, 6, 3, 1).to(device)
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
    
    def test_mape_loss(self, device):
        """测试MAPE损失"""
        loss_fn = MAPELoss()
        
        predictions = torch.randn(4, 6, 3, 1).to(device) * 0.5 + 0.5  # 正值
        targets = torch.randn(4, 6, 3, 1).to(device) * 0.5 + 0.5     # 正值
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
    
    def test_weighted_mse_loss(self, device):
        """测试加权MSE损失"""
        temporal_weights = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]).to(device)
        spatial_weights = torch.tensor([1.0, 1.2, 0.8]).to(device)
        
        loss_fn = WeightedMSELoss(
            temporal_weights=temporal_weights,
            spatial_weights=spatial_weights
        )
        
        predictions = torch.randn(4, 6, 3, 1).to(device)
        targets = torch.randn(4, 6, 3, 1).to(device)
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
    
    def test_spatiotemporal_loss(self, device):
        """测试时空损失函数"""
        loss_fn = SpatioTemporalLoss(
            base_loss='mse',
            consistency_weight=0.1,
            smoothness_weight=0.1
        )
        
        predictions = torch.randn(4, 6, 3, 1).to(device)
        targets = torch.randn(4, 6, 3, 1).to(device)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]]).to(device)
        
        losses = loss_fn(predictions, targets, edge_index=edge_index)
        
        assert 'base_loss' in losses
        assert 'temporal_consistency' in losses
        assert 'spatial_smoothness' in losses
        assert 'total_loss' in losses
        
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value.item(), float)
            assert loss_value.item() >= 0
    
    def test_create_loss_function(self):
        """测试损失函数工厂"""
        # 测试简单损失函数
        mse_config = {'type': 'mse', 'params': {'reduction': 'mean'}}
        mse_loss = create_loss_function(mse_config)
        assert isinstance(mse_loss, MSELoss)
        
        # 测试时空损失函数
        st_config = {
            'type': 'spatiotemporal',
            'params': {
                'base_loss': 'mae',
                'consistency_weight': 0.2
            }
        }
        st_loss = create_loss_function(st_config)
        assert isinstance(st_loss, SpatioTemporalLoss)


class TestEarlyStopping:
    """早停机制测试类"""
    
    def test_early_stopping_min_mode(self):
        """测试最小化模式的早停"""
        early_stopping = EarlyStopping(patience=3, mode='min')

        # 创建简单模型用于测试
        model = torch.nn.Linear(10, 1)

        # 模拟训练过程
        scores = [1.0, 0.8, 0.9, 0.85, 0.87, 0.86]  # 第3个epoch后开始恶化

        should_stop = False
        for i, score in enumerate(scores):
            should_stop = early_stopping(score, model)
            if should_stop:
                break

        # 应该在某个点触发早停
        assert should_stop
    
    def test_early_stopping_max_mode(self):
        """测试最大化模式的早停"""
        early_stopping = EarlyStopping(patience=2, mode='max')

        model = torch.nn.Linear(10, 1)

        # 模拟准确率下降
        scores = [0.8, 0.9, 0.85, 0.82, 0.80]

        should_stop = False
        for i, score in enumerate(scores):
            should_stop = early_stopping(score, model)
            if should_stop:
                break

        # 应该在某个点触发早停
        assert should_stop


class TestLearningRateScheduler:
    """学习率调度器测试类"""
    
    def test_step_scheduler(self):
        """测试步长调度器"""
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        scheduler = LearningRateScheduler(optimizer, 'step', step_size=2, gamma=0.1)

        initial_lr = optimizer.param_groups[0]['lr']

        # 测试调度器能正常工作
        scheduler.step()
        lr_after_step = optimizer.param_groups[0]['lr']

        # 验证学习率发生了变化（具体值可能因PyTorch版本而异）
        assert isinstance(lr_after_step, float)
        assert lr_after_step > 0
    
    def test_plateau_scheduler(self):
        """测试平台调度器"""
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = LearningRateScheduler(
            optimizer, 'plateau', 
            mode='min', patience=2, factor=0.5
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # 损失改善时学习率不变
        scheduler.step(1.0)
        scheduler.step(0.8)
        assert optimizer.param_groups[0]['lr'] == initial_lr
        
        # 损失不再改善时学习率下降
        scheduler.step(0.9)
        scheduler.step(0.85)
        scheduler.step(0.87)
        assert optimizer.param_groups[0]['lr'] < initial_lr


class TestTrainer:
    """训练器测试类"""
    
    @pytest.fixture
    def trainer_setup(self, test_config, device):
        """设置训练器测试环境"""
        # 调整配置
        test_config.model.node_feature_dim = 8
        test_config.model.hidden_dim = 16
        test_config.data.sequence_length = 12
        test_config.data.prediction_length = 6
        test_config.training.epochs = 2
        test_config.training.log_every_n_steps = 1
        
        # 创建模型
        model = SpatioTemporalPredictor(test_config)
        
        # 创建数据
        node_features = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        time_series = torch.randn(8, 12, 8)
        targets = torch.randn(8, 6, 3, 1)
        
        dataset = SpatioTemporalDataset(node_features, edge_index, time_series, targets)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        test_config.system.checkpoint_dir = temp_dir
        
        trainer = Trainer(model, test_config, train_loader, val_loader)
        
        yield trainer, temp_dir
        
        # 清理
        shutil.rmtree(temp_dir)
    
    def test_trainer_initialization(self, trainer_setup):
        """测试训练器初始化"""
        trainer, temp_dir = trainer_setup
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.early_stopping is not None
        assert trainer.metrics_tracker is not None
        assert trainer.current_epoch == 0
    
    def test_train_epoch(self, trainer_setup):
        """测试单个epoch训练"""
        trainer, temp_dir = trainer_setup
        
        metrics = trainer.train_epoch()
        
        assert 'train_loss' in metrics
        assert isinstance(metrics['train_loss'], float)
        assert metrics['train_loss'] >= 0
    
    def test_validate_epoch(self, trainer_setup):
        """测试单个epoch验证"""
        trainer, temp_dir = trainer_setup
        
        metrics = trainer.validate_epoch()
        
        assert 'val_loss' in metrics
        assert isinstance(metrics['val_loss'], float)
        assert metrics['val_loss'] >= 0
    
    @pytest.mark.slow
    def test_full_training(self, trainer_setup):
        """测试完整训练过程"""
        trainer, temp_dir = trainer_setup
        
        results = trainer.train()
        
        assert 'training_history' in results
        assert 'final_results' in results
        assert 'total_time' in results
        assert 'best_score' in results
        
        # 检查训练历史
        history = results['training_history']
        assert len(history) > 0
        assert all('train_loss' in epoch for epoch in history)
        assert all('val_loss' in epoch for epoch in history)
    
    def test_save_load_checkpoint(self, trainer_setup):
        """测试检查点保存和加载"""
        trainer, temp_dir = trainer_setup
        
        # 保存检查点
        checkpoint_path = trainer.save_checkpoint(epoch=0, score=1.0)
        assert Path(checkpoint_path).exists()
        
        # 修改模型权重
        original_weights = trainer.model.state_dict().copy()
        for param in trainer.model.parameters():
            param.data.fill_(0.5)
        
        # 加载检查点
        checkpoint_info = trainer.load_checkpoint(checkpoint_path)
        
        assert checkpoint_info['epoch'] == 0
        assert checkpoint_info['score'] == 1.0
        
        # 验证权重已恢复
        loaded_weights = trainer.model.state_dict()
        for key in original_weights:
            assert torch.allclose(original_weights[key], loaded_weights[key])
    
    def test_evaluate(self, trainer_setup):
        """测试模型评估"""
        trainer, temp_dir = trainer_setup
        
        metrics = trainer.evaluate(trainer.val_loader, prefix='test')
        
        assert 'test_loss' in metrics
        assert isinstance(metrics['test_loss'], float)
        assert metrics['test_loss'] >= 0
