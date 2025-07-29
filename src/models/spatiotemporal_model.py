"""
时空融合模型模块

实现融合GAT和Transformer的时空预测模型，用于停车场占用率预测。

Author: AI Assistant
Date: 2025-07-29
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from .gat_layer import GATLayer, MultiHeadGATLayer
from .transformer_layer import TransformerEncoder, TemporalTransformer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SpatialEncoder(nn.Module):
    """空间编码器

    使用GAT层学习停车场之间的空间关系。

    Args:
        node_features: 节点特征维度
        hidden_dim: 隐藏层维度
        num_gat_layers: GAT层数
        num_heads: 注意力头数
        dropout: Dropout率
        use_residual: 是否使用残差连接
    """

    def __init__(
        self,
        node_features: int,
        hidden_dim: int,
        num_gat_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super(SpatialEncoder, self).__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.use_residual = use_residual

        # 输入投影层
        self.input_projection = nn.Linear(node_features, hidden_dim)

        # GAT层
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            if i == 0:
                # 第一层：从输入维度到隐藏维度
                gat_layer = GATLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            else:
                # 后续层：隐藏维度到隐藏维度
                gat_layer = GATLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            self.gat_layers.append(gat_layer)

        # 层归一化
        if use_residual:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)
            ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播

        Args:
            node_features: 节点特征 [num_nodes, node_features]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_features] (可选)

        Returns:
            空间编码后的特征 [num_nodes, hidden_dim]
        """
        # 输入投影
        x = self.input_projection(node_features)
        x = self.dropout(x)

        # 通过GAT层
        for i, gat_layer in enumerate(self.gat_layers):
            residual = x if self.use_residual else None

            # GAT前向传播
            x_new = gat_layer(x, edge_index, edge_attr)

            # 残差连接和层归一化
            if self.use_residual and residual is not None:
                x = self.layer_norms[i](x_new + residual)
            else:
                x = x_new

            x = self.dropout(x)

        return x


class TemporalEncoder(nn.Module):
    """时间编码器

    使用Transformer学习时间序列中的时间依赖关系。

    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        num_transformer_layers: Transformer层数
        num_heads: 注意力头数
        d_ff: 前馈网络维度
        max_seq_len: 最大序列长度
        dropout: Dropout率
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 512,
        max_seq_len: int = 1000,
        dropout: float = 0.1
    ):
        super(TemporalEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 时间Transformer
        self.temporal_transformer = TemporalTransformer(
            input_dim=input_dim,
            d_model=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            mask: 注意力掩码

        Returns:
            时间编码后的特征 [batch_size, seq_len, hidden_dim]
        """
        return self.temporal_transformer(x, mask)


class SpatioTemporalFusion(nn.Module):
    """时空融合模块

    融合空间和时间特征，生成最终的预测结果。

    Args:
        spatial_dim: 空间特征维度
        temporal_dim: 时间特征维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        fusion_method: 融合方法 ('concat', 'add', 'attention')
        dropout: Dropout率
    """

    def __init__(
        self,
        spatial_dim: int,
        temporal_dim: int,
        hidden_dim: int,
        output_dim: int,
        fusion_method: str = 'attention',
        dropout: float = 0.1
    ):
        super(SpatioTemporalFusion, self).__init__()

        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method

        if fusion_method == 'concat':
            # 连接融合
            self.fusion_layer = nn.Sequential(
                nn.Linear(spatial_dim + temporal_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif fusion_method == 'add':
            # 加法融合（需要维度匹配）
            assert spatial_dim == temporal_dim, "Spatial and temporal dimensions must match for add fusion"
            self.fusion_layer = nn.Sequential(
                nn.Linear(spatial_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif fusion_method == 'attention':
            # 注意力融合
            self.spatial_proj = nn.Linear(spatial_dim, hidden_dim)
            self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(
        self,
        spatial_features: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> torch.Tensor:
        """前向传播

        Args:
            spatial_features: 空间特征 [batch_size, num_nodes, spatial_dim]
            temporal_features: 时间特征 [batch_size, seq_len, temporal_dim]

        Returns:
            融合后的特征 [batch_size, seq_len, num_nodes, output_dim]
        """
        batch_size = spatial_features.size(0)
        num_nodes = spatial_features.size(1)
        seq_len = temporal_features.size(1)

        if self.fusion_method == 'concat':
            # 扩展维度以匹配
            spatial_expanded = spatial_features.unsqueeze(1).expand(-1, seq_len, -1, -1)
            temporal_expanded = temporal_features.unsqueeze(2).expand(-1, -1, num_nodes, -1)

            # 连接特征
            fused = torch.cat([spatial_expanded, temporal_expanded], dim=-1)
            fused = self.fusion_layer(fused)

        elif self.fusion_method == 'add':
            # 扩展维度并相加
            spatial_expanded = spatial_features.unsqueeze(1).expand(-1, seq_len, -1, -1)
            temporal_expanded = temporal_features.unsqueeze(2).expand(-1, -1, num_nodes, -1)

            fused = spatial_expanded + temporal_expanded
            fused = self.fusion_layer(fused)

        elif self.fusion_method == 'attention':
            # 投影到相同维度
            spatial_proj = self.spatial_proj(spatial_features)  # [batch_size, num_nodes, hidden_dim]
            temporal_proj = self.temporal_proj(temporal_features)  # [batch_size, seq_len, hidden_dim]

            # 重塑为注意力输入格式
            spatial_flat = spatial_proj.view(batch_size, num_nodes, self.hidden_dim)
            temporal_flat = temporal_proj.view(batch_size, seq_len, self.hidden_dim)

            # 交叉注意力（简化版本）
            # 这里使用空间特征作为query，时间特征作为key和value
            spatial_attended = []
            for i in range(num_nodes):
                query = spatial_flat[:, i:i+1, :].transpose(0, 1)  # [1, batch_size, hidden_dim]
                key_value = temporal_flat.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]

                attended, _ = self.attention(query, key_value, key_value)
                spatial_attended.append(attended.transpose(0, 1))  # [batch_size, 1, hidden_dim]

            # 组合结果
            spatial_attended = torch.cat(spatial_attended, dim=1)  # [batch_size, num_nodes, hidden_dim]

            # 扩展到时空维度
            fused = spatial_attended.unsqueeze(1).expand(-1, seq_len, -1, -1)
            fused = self.fusion_layer(fused)

        # 输出层
        output = self.output_layer(fused)

        return output


class SpatioTemporalModel(nn.Module):
    """时空预测模型

    融合GAT和Transformer的完整时空预测模型，用于停车场占用率预测。

    Args:
        node_features: 节点特征维度
        hidden_dim: 隐藏层维度
        num_gat_layers: GAT层数
        num_transformer_layers: Transformer层数
        seq_len: 输入序列长度
        pred_len: 预测序列长度
        num_heads: 注意力头数
        dropout: Dropout率
        fusion_method: 时空融合方法

    Example:
        >>> model = SpatioTemporalModel(
        ...     node_features=28, hidden_dim=128, seq_len=24, pred_len=6
        ... )
        >>> # 输入数据
        >>> node_features = torch.randn(5, 28)  # 5个节点，28维特征
        >>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 边索引
        >>> time_series = torch.randn(32, 24, 28)  # 批次大小32，序列长度24
        >>> # 前向传播
        >>> output = model(node_features, edge_index, time_series)
    """

    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 128,
        num_gat_layers: int = 2,
        num_transformer_layers: int = 4,
        seq_len: int = 168,
        pred_len: int = 24,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_method: str = 'attention',
        output_dim: int = 1
    ):
        super(SpatioTemporalModel, self).__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_dim = output_dim

        # 空间编码器（GAT）
        self.spatial_encoder = SpatialEncoder(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_gat_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_residual=True
        )

        # 时间编码器（Transformer）
        self.temporal_encoder = TemporalEncoder(
            input_dim=node_features,
            hidden_dim=hidden_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            d_ff=hidden_dim * 4,
            max_seq_len=seq_len * 2,
            dropout=dropout
        )

        # 时空融合模块
        self.fusion_module = SpatioTemporalFusion(
            spatial_dim=hidden_dim,
            temporal_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,  # 使用正确的output_dim
            fusion_method=fusion_method,
            dropout=dropout
        )

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(output_dim, max(output_dim * 2, 8)),  # 使用output_dim作为输入
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(output_dim * 2, 8), pred_len * output_dim)
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用更保守的初始化
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        time_series: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播

        Args:
            node_features: 节点特征 [num_nodes, node_features]
            edge_index: 边索引 [2, num_edges]
            time_series: 时间序列 [batch_size, seq_len, node_features]
            edge_attr: 边特征 [num_edges, edge_features] (可选)
            mask: 注意力掩码 (可选)

        Returns:
            预测结果 [batch_size, pred_len, num_nodes, output_dim]
        """
        batch_size = time_series.size(0)
        num_nodes = node_features.size(0)

        # 1. 空间编码
        spatial_features = self.spatial_encoder(
            node_features, edge_index, edge_attr
        )  # [num_nodes, hidden_dim]

        # 扩展到批次维度
        # 检查spatial_features的维度
        if spatial_features.dim() == 2:
            # [num_nodes, hidden_dim] -> [batch_size, num_nodes, hidden_dim]
            spatial_features = spatial_features.unsqueeze(0).expand(batch_size, -1, -1)
        elif spatial_features.dim() == 3:
            # [1, num_nodes, hidden_dim] -> [batch_size, num_nodes, hidden_dim]
            spatial_features = spatial_features.expand(batch_size, -1, -1)
        else:
            # 处理其他维度情况
            spatial_features = spatial_features.view(batch_size, num_nodes, -1)

        # 2. 时间编码
        temporal_features = self.temporal_encoder(
            time_series, mask
        )  # [batch_size, seq_len, hidden_dim]

        # 3. 时空融合
        fused_features = self.fusion_module(
            spatial_features, temporal_features
        )  # [batch_size, seq_len, num_nodes, output_dim]

        # 4. 预测
        # 使用最后一个时间步的特征进行预测
        last_features = fused_features[:, -1, :, :]  # [batch_size, num_nodes, output_dim]

        # 调试信息
        actual_batch_size, actual_num_nodes, actual_feature_dim = last_features.shape
        expected_size = batch_size * num_nodes
        actual_size = actual_batch_size * actual_num_nodes

        # 使用实际的维度进行reshape
        predictions = self.prediction_head(
            last_features.reshape(actual_size, -1)
        )  # [actual_batch_size * actual_num_nodes, pred_len * output_dim]

        # 重塑输出
        predictions = predictions.view(
            actual_batch_size, actual_num_nodes, self.pred_len, self.output_dim
        ).transpose(1, 2)  # [batch_size, pred_len, num_nodes, output_dim]

        return predictions

    def predict(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        time_series: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """预测接口（推理模式）

        Args:
            node_features: 节点特征
            edge_index: 边索引
            time_series: 时间序列
            edge_attr: 边特征 (可选)

        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            return self.forward(node_features, edge_index, time_series, edge_attr)

    def get_attention_weights(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        time_series: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """获取注意力权重（用于可视化）

        Args:
            node_features: 节点特征
            edge_index: 边索引
            time_series: 时间序列
            edge_attr: 边特征 (可选)

        Returns:
            包含各种注意力权重的字典
        """
        # 这里可以扩展以返回GAT和Transformer的注意力权重
        # 目前返回空字典作为占位
        return {}

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'node_features={self.node_features}, '
                f'hidden_dim={self.hidden_dim}, '
                f'seq_len={self.seq_len}, '
                f'pred_len={self.pred_len}, '
                f'output_dim={self.output_dim})')


class SpatioTemporalPredictor(nn.Module):
    """时空预测器

    包装SpatioTemporalModel的高级接口，提供更便捷的使用方式。

    Args:
        config: 模型配置
    """

    def __init__(self, config):
        super(SpatioTemporalPredictor, self).__init__()

        self.config = config

        # 创建主模型
        self.model = SpatioTemporalModel(
            node_features=config.model.node_feature_dim,
            hidden_dim=config.model.hidden_dim,
            num_gat_layers=config.model.num_gat_layers,
            num_transformer_layers=config.model.num_transformer_layers,
            seq_len=config.data.sequence_length,
            pred_len=config.data.prediction_length,
            num_heads=config.model.gat_heads,
            dropout=config.model.gat_dropout,
            output_dim=config.model.output_dim
        )

        # 损失函数
        self.criterion = self._create_criterion()

    def _create_criterion(self):
        """创建损失函数"""
        if hasattr(self.config.training, 'loss_function'):
            loss_type = self.config.training.loss_function
        else:
            loss_type = 'mse'

        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'mae':
            return nn.L1Loss()
        elif loss_type == 'huber':
            return nn.SmoothL1Loss()
        else:
            return nn.MSELoss()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            batch: 包含输入数据的批次字典

        Returns:
            包含预测结果和损失的字典
        """
        # 提取输入数据
        node_features = batch['node_features']
        edge_index = batch['edge_index']
        time_series = batch['time_series']
        targets = batch.get('targets', None)
        edge_attr = batch.get('edge_attr', None)

        # 模型预测
        predictions = self.model(node_features, edge_index, time_series, edge_attr)

        result = {'predictions': predictions}

        # 计算损失（如果有目标值）
        if targets is not None:
            loss = self.criterion(predictions, targets)
            result['loss'] = loss

        return result

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测接口

        Args:
            batch: 输入批次

        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(batch)
            return result['predictions']
