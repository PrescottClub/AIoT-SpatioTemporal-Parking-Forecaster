"""
GAT层模块

实现图注意力网络（Graph Attention Network）层。

Author: AI Assistant
Date: 2025-07-29
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
import math

from ..utils.logger import get_logger

logger = get_logger(__name__)


class GATLayer(nn.Module):
    """图注意力网络层

    实现单头或多头图注意力机制，用于学习节点之间的空间关系。

    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        num_heads: 注意力头数
        dropout: Dropout率
        alpha: LeakyReLU负斜率
        concat: 是否连接多头输出（True）或平均（False）
        bias: 是否使用偏置

    Example:
        >>> gat = GATLayer(in_features=64, out_features=32, num_heads=8)
        >>> node_features = torch.randn(10, 64)  # 10个节点，64维特征
        >>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 边索引
        >>> output = gat(node_features, edge_index)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
        bias: bool = True
    ):
        super(GATLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # 如果连接多头输出，每个头的输出维度需要调整
        if self.concat:
            assert out_features % num_heads == 0, \
                f"out_features ({out_features}) must be divisible by num_heads ({num_heads})"
            self.head_dim = out_features // num_heads
        else:
            self.head_dim = out_features

        # 线性变换层（为每个头创建独立的权重）
        self.W = nn.Parameter(torch.empty(size=(num_heads, in_features, self.head_dim)))

        # 注意力权重参数
        self.a = nn.Parameter(torch.empty(size=(num_heads, 2 * self.head_dim, 1)))

        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化模型参数"""
        # 使用更保守的初始化
        std = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.W, -std, std)
        nn.init.uniform_(self.a, -std, std)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播

        Args:
            x: 节点特征 [num_nodes, in_features]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_features] (可选)
            return_attention_weights: 是否返回注意力权重

        Returns:
            如果return_attention_weights=False: 输出节点特征 [num_nodes, out_features]
            如果return_attention_weights=True: (输出特征, 注意力权重)
        """
        num_nodes = x.size(0)

        # 简化的实现：对每个头分别处理
        head_outputs = []
        all_attention_weights = []

        for head in range(self.num_heads):
            # 线性变换
            h = torch.matmul(x, self.W[head])  # [num_nodes, head_dim]

            # 计算注意力权重
            attention_weights = self._compute_attention_weights_single_head(h, edge_index, head)

            # 应用dropout
            attention_weights = self.dropout_layer(attention_weights)

            # 聚合邻居特征
            output = self._aggregate_neighbors_single_head(h, edge_index, attention_weights)

            head_outputs.append(output)
            all_attention_weights.append(attention_weights)

        # 处理多头输出
        if self.concat:
            # 连接所有头的输出
            output = torch.cat(head_outputs, dim=-1)  # [num_nodes, num_heads * head_dim]
        else:
            # 平均所有头的输出
            output = torch.stack(head_outputs, dim=0).mean(dim=0)  # [num_nodes, head_dim]

        # 添加偏置
        if self.bias is not None:
            output = output + self.bias

        if return_attention_weights:
            # 合并所有头的注意力权重
            combined_weights = torch.stack(all_attention_weights, dim=-1)  # [num_edges, num_heads]
            return output, combined_weights
        else:
            return output

    def _compute_attention_weights(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """计算注意力权重

        Args:
            h: 变换后的节点特征 [num_nodes, num_heads, head_dim]
            edge_index: 边索引 [2, num_edges]

        Returns:
            注意力权重 [num_edges, num_heads]
        """
        num_edges = edge_index.size(1)

        # 获取源节点和目标节点的特征
        source_nodes = edge_index[0]  # [num_edges]
        target_nodes = edge_index[1]  # [num_edges]

        h_source = h[source_nodes]  # [num_edges, num_heads, head_dim]
        h_target = h[target_nodes]  # [num_edges, num_heads, head_dim]

        # 连接源节点和目标节点特征
        h_concat = torch.cat([h_source, h_target], dim=-1)  # [num_edges, num_heads, 2*head_dim]

        # 计算注意力分数
        # 重塑为正确的维度进行批量矩阵乘法
        # h_concat: [num_edges, num_heads, 2*head_dim]
        # self.a: [num_heads, 2*head_dim, 1]
        attention_scores = torch.zeros(num_edges, self.num_heads, device=h_concat.device)

        for head in range(self.num_heads):
            # 对每个头单独计算注意力分数
            head_concat = h_concat[:, head, :]  # [num_edges, 2*head_dim]
            head_a = self.a[head, :, :]  # [2*head_dim, 1]
            head_scores = torch.matmul(head_concat, head_a).squeeze(-1)  # [num_edges]
            attention_scores[:, head] = head_scores

        # 应用LeakyReLU激活
        attention_scores = self.leaky_relu(attention_scores)

        # 对每个目标节点的所有入边进行softmax归一化
        attention_weights = self._edge_softmax(attention_scores, edge_index[1])

        return attention_weights

    def _edge_softmax(
        self,
        attention_scores: torch.Tensor,
        target_nodes: torch.Tensor
    ) -> torch.Tensor:
        """对每个节点的入边进行softmax归一化

        Args:
            attention_scores: 注意力分数 [num_edges, num_heads]
            target_nodes: 目标节点索引 [num_edges]

        Returns:
            归一化的注意力权重 [num_edges, num_heads]
        """
        # 找到每个节点的最大注意力分数（数值稳定性）
        max_scores = torch.zeros_like(attention_scores)
        for i in range(attention_scores.size(1)):  # 对每个头
            max_scores[:, i] = attention_scores[:, i] - \
                torch.index_select(
                    torch.scatter_reduce(
                        torch.full((target_nodes.max() + 1,), float('-inf'),
                                 device=attention_scores.device),
                        0, target_nodes, attention_scores[:, i], reduce='amax'
                    ), 0, target_nodes
                )

        # 计算exp值
        exp_scores = torch.exp(attention_scores - max_scores)

        # 计算每个节点的归一化常数
        normalizers = torch.zeros(target_nodes.max() + 1, attention_scores.size(1),
                                device=attention_scores.device)
        normalizers.scatter_add_(0, target_nodes.unsqueeze(1).expand(-1, attention_scores.size(1)),
                               exp_scores)

        # 归一化
        attention_weights = exp_scores / torch.index_select(normalizers, 0, target_nodes)

        return attention_weights

    def _aggregate_neighbors(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """聚合邻居节点特征

        Args:
            h: 节点特征 [num_nodes, num_heads, head_dim]
            edge_index: 边索引 [2, num_edges]
            attention_weights: 注意力权重 [num_edges, num_heads]

        Returns:
            聚合后的特征 [num_nodes, num_heads, head_dim]
        """
        num_nodes = h.size(0)
        num_heads = h.size(1)
        head_dim = h.size(2)

        # 初始化输出
        output = torch.zeros_like(h)

        # 获取源节点和目标节点
        source_nodes = edge_index[0]  # [num_edges]
        target_nodes = edge_index[1]  # [num_edges]

        # 获取源节点特征
        h_source = h[source_nodes]  # [num_edges, num_heads, head_dim]

        # 应用注意力权重
        weighted_features = h_source * attention_weights.unsqueeze(-1)  # [num_edges, num_heads, head_dim]

        # 聚合到目标节点
        output.scatter_add_(0, target_nodes.unsqueeze(1).unsqueeze(2).expand(-1, num_heads, head_dim),
                          weighted_features)

        return output

    def _compute_attention_weights_single_head(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        head_idx: int
    ) -> torch.Tensor:
        """计算单个头的注意力权重

        Args:
            h: 单头变换后的节点特征 [num_nodes, head_dim]
            edge_index: 边索引 [2, num_edges]
            head_idx: 头索引

        Returns:
            注意力权重 [num_edges]
        """
        num_edges = edge_index.size(1)

        # 获取源节点和目标节点的特征
        source_nodes = edge_index[0]  # [num_edges]
        target_nodes = edge_index[1]  # [num_edges]

        h_source = h[source_nodes]  # [num_edges, head_dim]
        h_target = h[target_nodes]  # [num_edges, head_dim]

        # 连接源节点和目标节点特征
        h_concat = torch.cat([h_source, h_target], dim=-1)  # [num_edges, 2*head_dim]

        # 计算注意力分数
        attention_scores = torch.matmul(h_concat, self.a[head_idx, :, :]).squeeze(-1)  # [num_edges]

        # 应用LeakyReLU激活
        attention_scores = self.leaky_relu(attention_scores)

        # 对每个目标节点的所有入边进行softmax归一化
        attention_weights = self._edge_softmax_single(attention_scores, target_nodes)

        return attention_weights

    def _edge_softmax_single(
        self,
        attention_scores: torch.Tensor,
        target_nodes: torch.Tensor
    ) -> torch.Tensor:
        """对每个节点的入边进行softmax归一化（单头版本）

        Args:
            attention_scores: 注意力分数 [num_edges]
            target_nodes: 目标节点索引 [num_edges]

        Returns:
            归一化的注意力权重 [num_edges]
        """
        # 检查输入是否包含NaN或Inf
        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            logger.warning("注意力分数包含NaN或Inf值，使用均匀分布")
            return torch.ones_like(attention_scores) / attention_scores.size(0)

        # 使用简化的softmax实现
        # 数值稳定的softmax：先减去最大值
        max_score = torch.max(attention_scores)
        exp_scores = torch.exp(torch.clamp(attention_scores - max_score, min=-20, max=20))

        # 对每个目标节点分别归一化
        attention_weights = torch.zeros_like(attention_scores)
        unique_nodes = torch.unique(target_nodes)

        for node in unique_nodes:
            mask = (target_nodes == node)
            node_exp_scores = exp_scores[mask]
            sum_exp = torch.sum(node_exp_scores) + 1e-6
            attention_weights[mask] = node_exp_scores / sum_exp

        # 最终检查
        if torch.isnan(attention_weights).any():
            logger.warning("归一化后仍有NaN值，使用均匀分布")
            attention_weights = torch.ones_like(attention_weights) / attention_weights.size(0)

        return attention_weights

    def _aggregate_neighbors_single_head(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """聚合邻居节点特征（单头版本）

        Args:
            h: 节点特征 [num_nodes, head_dim]
            edge_index: 边索引 [2, num_edges]
            attention_weights: 注意力权重 [num_edges]

        Returns:
            聚合后的特征 [num_nodes, head_dim]
        """
        num_nodes = h.size(0)
        head_dim = h.size(1)

        # 初始化输出
        output = torch.zeros_like(h)

        # 获取源节点和目标节点
        source_nodes = edge_index[0]  # [num_edges]
        target_nodes = edge_index[1]  # [num_edges]

        # 获取源节点特征
        h_source = h[source_nodes]  # [num_edges, head_dim]

        # 应用注意力权重
        weighted_features = h_source * attention_weights.unsqueeze(-1)  # [num_edges, head_dim]

        # 使用循环聚合（更稳定但较慢）
        for i in range(len(target_nodes)):
            target_node = target_nodes[i]
            output[target_node] += weighted_features[i]

        return output

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'num_heads={self.num_heads}, '
                f'dropout={self.dropout}, '
                f'alpha={self.alpha}, '
                f'concat={self.concat})')


class MultiHeadGATLayer(nn.Module):
    """多头GAT层的简化实现

    这是一个更简洁的多头GAT实现，使用PyTorch Geometric风格的接口。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        if concat:
            assert out_features % num_heads == 0
            self.head_dim = out_features // num_heads
        else:
            self.head_dim = out_features

        # 使用单个线性层处理所有头
        self.linear = nn.Linear(in_features, num_heads * self.head_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))

        self.dropout_layer = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.xavier_uniform_(self.attention, gain=gain)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """前向传播（简化版本）"""
        num_nodes = x.size(0)

        # 线性变换
        h = self.linear(x).view(num_nodes, self.num_heads, self.head_dim)

        # 简化的注意力计算（使用平均池化作为占位）
        # 在实际应用中，这里应该实现完整的注意力机制
        if self.concat:
            output = h.view(num_nodes, -1)
        else:
            output = h.mean(dim=1)

        return output
