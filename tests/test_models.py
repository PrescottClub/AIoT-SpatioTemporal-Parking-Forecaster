"""
模型测试模块

测试GAT层、Transformer层和时空融合模型的功能。

Author: AI Assistant
Date: 2025-07-29
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from src.models.gat_layer import GATLayer, MultiHeadGATLayer
from src.models.transformer_layer import (
    PositionalEncoding, MultiHeadAttention, TransformerEncoder, TemporalTransformer
)
from src.models.spatiotemporal_model import (
    SpatialEncoder, TemporalEncoder, SpatioTemporalFusion, 
    SpatioTemporalModel, SpatioTemporalPredictor
)
from src.config import Config


class TestGATLayer:
    """GAT层测试类"""
    
    def test_gat_layer_init(self):
        """测试GAT层初始化"""
        gat = GATLayer(
            in_features=64,
            out_features=32,
            num_heads=8,
            dropout=0.1,
            alpha=0.2
        )
        
        assert gat.in_features == 64
        assert gat.out_features == 32
        assert gat.num_heads == 8
        assert gat.head_dim == 4  # 32 // 8
        assert gat.dropout == 0.1
        assert gat.alpha == 0.2
    
    def test_gat_layer_forward(self, device):
        """测试GAT层前向传播"""
        gat = GATLayer(
            in_features=16,
            out_features=32,
            num_heads=4,
            dropout=0.0  # 测试时关闭dropout
        ).to(device)
        
        # 创建测试数据
        num_nodes = 5
        num_edges = 8
        
        node_features = torch.randn(num_nodes, 16).to(device)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 1, 2, 3],
            [1, 2, 3, 4, 0, 0, 1, 2]
        ]).to(device)
        
        # 前向传播
        output = gat(node_features, edge_index)
        
        assert output.shape == (num_nodes, 32)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gat_layer_attention_weights(self, device):
        """测试GAT层注意力权重返回"""
        gat = GATLayer(
            in_features=8,
            out_features=16,
            num_heads=2
        ).to(device)
        
        num_nodes = 3
        node_features = torch.randn(num_nodes, 8).to(device)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]]).to(device)
        
        output, attention_weights = gat(
            node_features, edge_index, return_attention_weights=True
        )
        
        assert output.shape == (num_nodes, 16)
        assert attention_weights.shape == (3, 2)  # num_edges, num_heads
        
        # 检查注意力权重的基本属性
        assert torch.all(attention_weights >= 0), "注意力权重应该非负"
        assert not torch.isnan(attention_weights).any(), "注意力权重不应包含NaN"
        assert not torch.isinf(attention_weights).any(), "注意力权重不应包含Inf"


class TestTransformerLayer:
    """Transformer层测试类"""
    
    def test_positional_encoding(self, device):
        """测试位置编码"""
        d_model = 64
        max_len = 100
        pe = PositionalEncoding(d_model, max_len, dropout=0.0).to(device)
        
        seq_len = 50
        batch_size = 8
        x = torch.randn(seq_len, batch_size, d_model).to(device)
        
        output = pe(x)
        
        assert output.shape == (seq_len, batch_size, d_model)
        assert not torch.isnan(output).any()
    
    def test_multi_head_attention(self, device):
        """测试多头注意力"""
        d_model = 64
        num_heads = 8
        attention = MultiHeadAttention(d_model, num_heads, dropout=0.0).to(device)
        
        seq_len = 20
        batch_size = 4
        x = torch.randn(seq_len, batch_size, d_model).to(device)
        
        output, attention_weights = attention(x, x, x)
        
        assert output.shape == (seq_len, batch_size, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        assert not torch.isnan(output).any()
    
    def test_transformer_encoder(self, device):
        """测试Transformer编码器"""
        d_model = 128
        num_heads = 8
        num_layers = 4
        
        encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=0.0
        ).to(device)
        
        seq_len = 30
        batch_size = 6
        x = torch.randn(seq_len, batch_size, d_model).to(device)
        
        output = encoder(x)
        
        assert output.shape == (seq_len, batch_size, d_model)
        assert not torch.isnan(output).any()
    
    def test_temporal_transformer(self, device):
        """测试时间Transformer"""
        input_dim = 32
        d_model = 64
        output_dim = 16
        
        temporal_transformer = TemporalTransformer(
            input_dim=input_dim,
            d_model=d_model,
            output_dim=output_dim,
            num_heads=4,
            num_layers=2,
            dropout=0.0
        ).to(device)
        
        batch_size = 8
        seq_len = 24
        x = torch.randn(batch_size, seq_len, input_dim).to(device)
        
        output = temporal_transformer(x)
        
        assert output.shape == (batch_size, seq_len, output_dim)
        assert not torch.isnan(output).any()


class TestSpatioTemporalModel:
    """时空模型测试类"""
    
    def test_spatial_encoder(self, device):
        """测试空间编码器"""
        node_features = 28
        hidden_dim = 64
        num_gat_layers = 2
        
        spatial_encoder = SpatialEncoder(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_gat_layers=num_gat_layers,
            dropout=0.0
        ).to(device)
        
        num_nodes = 5
        node_feat = torch.randn(num_nodes, node_features).to(device)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).to(device)
        
        output = spatial_encoder(node_feat, edge_index)
        
        assert output.shape == (num_nodes, hidden_dim)
        assert not torch.isnan(output).any()
    
    def test_temporal_encoder(self, device):
        """测试时间编码器"""
        input_dim = 28
        hidden_dim = 64
        
        temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_transformer_layers=2,
            dropout=0.0
        ).to(device)
        
        batch_size = 4
        seq_len = 24
        x = torch.randn(batch_size, seq_len, input_dim).to(device)
        
        output = temporal_encoder(x)
        
        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(output).any()
    
    def test_spatiotemporal_fusion(self, device):
        """测试时空融合模块"""
        spatial_dim = 64
        temporal_dim = 64
        hidden_dim = 128
        output_dim = 1
        
        fusion = SpatioTemporalFusion(
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            fusion_method='concat',
            dropout=0.0
        ).to(device)
        
        batch_size = 4
        num_nodes = 5
        seq_len = 24
        
        spatial_features = torch.randn(batch_size, num_nodes, spatial_dim).to(device)
        temporal_features = torch.randn(batch_size, seq_len, temporal_dim).to(device)
        
        output = fusion(spatial_features, temporal_features)
        
        assert output.shape == (batch_size, seq_len, num_nodes, output_dim)
        assert not torch.isnan(output).any()
    
    def test_spatiotemporal_model_forward(self, device):
        """测试完整时空模型前向传播"""
        node_features = 28
        hidden_dim = 64
        seq_len = 24
        pred_len = 6
        
        model = SpatioTemporalModel(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_gat_layers=1,
            num_transformer_layers=2,
            seq_len=seq_len,
            pred_len=pred_len,
            dropout=0.0
        ).to(device)
        
        # 创建测试数据
        batch_size = 4
        num_nodes = 5
        
        node_feat = torch.randn(num_nodes, node_features).to(device)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).to(device)
        time_series = torch.randn(batch_size, seq_len, node_features).to(device)
        
        # 前向传播
        output = model(node_feat, edge_index, time_series)
        
        assert output.shape == (batch_size, pred_len, num_nodes, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_spatiotemporal_model_predict(self, device):
        """测试模型预测接口"""
        model = SpatioTemporalModel(
            node_features=16,
            hidden_dim=32,
            seq_len=12,
            pred_len=3,
            dropout=0.0
        ).to(device)
        
        num_nodes = 3
        node_feat = torch.randn(num_nodes, 16).to(device)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]]).to(device)
        time_series = torch.randn(2, 12, 16).to(device)
        
        # 预测模式
        predictions = model.predict(node_feat, edge_index, time_series)
        
        assert predictions.shape == (2, 3, num_nodes, 1)
        assert not torch.isnan(predictions).any()
    
    def test_spatiotemporal_predictor(self, test_config, device):
        """测试时空预测器"""
        # 调整配置以适应测试
        test_config.model.node_feature_dim = 16
        test_config.model.hidden_dim = 32
        test_config.data.sequence_length = 12
        test_config.data.prediction_length = 3
        
        predictor = SpatioTemporalPredictor(test_config).to(device)
        
        # 创建测试批次
        batch = {
            'node_features': torch.randn(3, 16).to(device),
            'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]]).to(device),
            'time_series': torch.randn(2, 12, 16).to(device),
            'targets': torch.randn(2, 3, 3, 1).to(device)
        }
        
        # 前向传播
        result = predictor(batch)
        
        assert 'predictions' in result
        assert 'loss' in result
        assert result['predictions'].shape == (2, 3, 3, 1)
        assert isinstance(result['loss'].item(), float)
        
        # 预测接口
        predictions = predictor.predict(batch)
        assert predictions.shape == (2, 3, 3, 1)


class TestModelIntegration:
    """模型集成测试类"""
    
    def test_model_with_real_data_shapes(self, sample_tensor_data, device):
        """测试模型与真实数据形状的兼容性"""
        # 使用样本数据的形状创建模型
        batch_size, seq_len, num_nodes, num_features = sample_tensor_data['node_features'].shape
        
        model = SpatioTemporalModel(
            node_features=num_features,
            hidden_dim=64,
            seq_len=seq_len,
            pred_len=6,
            dropout=0.0
        ).to(device)
        
        # 准备输入数据
        node_features = sample_tensor_data['node_features'][0, 0, :, :].to(device)  # [num_nodes, features]
        edge_index = sample_tensor_data['edge_index'].to(device)
        time_series = sample_tensor_data['node_features'][:, :, 0, :].to(device)  # [batch, seq, features]
        
        # 前向传播
        output = model(node_features, edge_index, time_series)
        
        assert output.shape == (batch_size, 6, num_nodes, 1)
        assert not torch.isnan(output).any()
    
    @pytest.mark.slow
    def test_model_training_step(self, test_config, device):
        """测试模型训练步骤"""
        # 创建小型模型进行快速测试
        test_config.model.node_feature_dim = 8
        test_config.model.hidden_dim = 16
        test_config.data.sequence_length = 6
        test_config.data.prediction_length = 2
        
        model = SpatioTemporalPredictor(test_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 创建训练数据
        batch = {
            'node_features': torch.randn(2, 8).to(device),
            'edge_index': torch.tensor([[0, 1], [1, 0]]).to(device),
            'time_series': torch.randn(4, 6, 8).to(device),
            'targets': torch.randn(4, 2, 2, 1).to(device)
        }
        
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        
        result = model(batch)
        loss = result['loss']
        
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 检查梯度
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "模型参数没有梯度"
