"""
Transformer层模块

实现用于时间序列建模的Transformer编码器。

Author: AI Assistant
Date: 2025-07-29
"""

import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """位置编码模块

    为时间序列数据添加位置信息，使Transformer能够理解序列中的时间顺序。

    Args:
        d_model: 模型维度
        max_len: 最大序列长度
        dropout: Dropout率
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        # 应用sin和cos函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]

        # 注册为buffer，不参与梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 [seq_len, batch_size, d_model]

        Returns:
            添加位置编码后的张量 [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制

    实现标准的多头自注意力机制，用于捕获时间序列中的长期依赖关系。

    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        dropout: Dropout率
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            query: 查询张量 [seq_len, batch_size, d_model]
            key: 键张量 [seq_len, batch_size, d_model]
            value: 值张量 [seq_len, batch_size, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]

        Returns:
            (输出张量, 注意力权重)
        """
        seq_len, batch_size, d_model = query.size()

        # 线性变换
        Q = self.w_q(query)  # [seq_len, batch_size, d_model]
        K = self.w_k(key)    # [seq_len, batch_size, d_model]
        V = self.w_v(value)  # [seq_len, batch_size, d_model]

        # 重塑为多头格式
        Q = Q.view(seq_len, batch_size, self.num_heads, self.d_k).transpose(0, 1).transpose(1, 2)
        K = K.view(seq_len, batch_size, self.num_heads, self.d_k).transpose(0, 1).transpose(1, 2)
        V = V.view(seq_len, batch_size, self.num_heads, self.d_k).transpose(0, 1).transpose(1, 2)
        # 现在形状为 [batch_size, num_heads, seq_len, d_k]

        # 计算注意力
        attention_output, attention_weights = self._scaled_dot_product_attention(
            Q, K, V, mask
        )

        # 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        ).transpose(0, 1)  # 转换回 [seq_len, batch_size, d_model]

        # 最终线性变换
        output = self.w_o(attention_output)

        return output, attention_weights

    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """缩放点积注意力

        Args:
            Q: 查询 [batch_size, num_heads, seq_len, d_k]
            K: 键 [batch_size, num_heads, seq_len, d_k]
            V: 值 [batch_size, num_heads, seq_len, d_k]
            mask: 掩码 [batch_size, seq_len, seq_len]

        Returns:
            (注意力输出, 注意力权重)
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [batch_size, num_heads, seq_len, seq_len]

        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        attention_output = torch.matmul(attention_weights, V)

        return attention_output, attention_weights


class FeedForward(nn.Module):
    """前馈网络

    Transformer中的位置前馈网络，包含两个线性变换和一个激活函数。

    Args:
        d_model: 模型维度
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout率
        activation: 激活函数类型
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # 选择激活函数
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 [seq_len, batch_size, d_model]

        Returns:
            输出张量 [seq_len, batch_size, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层

    单个Transformer编码器层，包含多头自注意力和前馈网络。

    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout率
        activation: 激活函数类型
        layer_norm_eps: LayerNorm的epsilon值
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播

        Args:
            src: 输入张量 [seq_len, batch_size, d_model]
            src_mask: 源序列掩码 [batch_size, seq_len, seq_len]

        Returns:
            输出张量 [seq_len, batch_size, d_model]
        """
        # 多头自注意力 + 残差连接 + LayerNorm
        attn_output, _ = self.self_attention(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_output))

        # 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_output))

        return src


class TransformerEncoder(nn.Module):
    """Transformer编码器

    完整的Transformer编码器，包含多个编码器层和位置编码。
    用于处理时间序列数据，捕获长期时间依赖关系。

    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        num_layers: 编码器层数
        d_ff: 前馈网络隐藏层维度
        max_seq_len: 最大序列长度
        dropout: Dropout率
        activation: 激活函数类型
        layer_norm_eps: LayerNorm的epsilon值

    Example:
        >>> encoder = TransformerEncoder(d_model=128, num_heads=8, num_layers=4)
        >>> x = torch.randn(50, 32, 128)  # [seq_len, batch_size, d_model]
        >>> output = encoder(x)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5
    ):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # 编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(num_layers)
        ])

        # 最终层归一化
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # 参数初始化
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播

        Args:
            src: 输入序列 [seq_len, batch_size, d_model]
            mask: 注意力掩码 [seq_len, seq_len]
            src_key_padding_mask: 键填充掩码 [batch_size, seq_len]

        Returns:
            编码后的序列 [seq_len, batch_size, d_model]
        """
        # 检查输入维度
        if src.size(-1) != self.d_model:
            raise ValueError(f"Expected input feature dimension {self.d_model}, "
                           f"but got {src.size(-1)}")

        # 添加位置编码
        src = self.pos_encoding(src * math.sqrt(self.d_model))

        # 处理填充掩码
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(1)
            if mask is None:
                mask = src_key_padding_mask
            else:
                mask = mask.unsqueeze(0).unsqueeze(0) | src_key_padding_mask

        # 通过编码器层
        output = src
        for layer in self.layers:
            output = layer(output, mask)

        # 最终层归一化
        output = self.norm(output)

        return output

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成因果掩码（下三角掩码）

        Args:
            sz: 序列长度

        Returns:
            掩码张量 [sz, sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class TemporalTransformer(nn.Module):
    """时间Transformer

    专门用于时间序列预测的Transformer模型，包含输入投影和输出投影。

    Args:
        input_dim: 输入特征维度
        d_model: 模型维度
        output_dim: 输出特征维度
        num_heads: 注意力头数
        num_layers: 编码器层数
        d_ff: 前馈网络隐藏层维度
        max_seq_len: 最大序列长度
        dropout: Dropout率
        activation: 激活函数类型
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super(TemporalTransformer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim

        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)

        # Transformer编码器
        self.transformer = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            activation=activation
        )

        # 输出投影
        self.output_projection = nn.Linear(d_model, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入序列 [batch_size, seq_len, input_dim] 或 [seq_len, batch_size, input_dim]
            mask: 注意力掩码

        Returns:
            输出序列 [batch_size, seq_len, output_dim] 或 [seq_len, batch_size, output_dim]
        """
        # 检查输入格式并转换为 [seq_len, batch_size, input_dim]
        if x.dim() == 3 and x.size(0) != x.size(1):
            # 假设输入是 [batch_size, seq_len, input_dim]
            x = x.transpose(0, 1)  # -> [seq_len, batch_size, input_dim]
            transpose_output = True
        else:
            transpose_output = False

        # 输入投影
        x = self.input_projection(x)  # [seq_len, batch_size, d_model]
        x = self.dropout(x)

        # Transformer编码
        x = self.transformer(x, mask)  # [seq_len, batch_size, d_model]

        # 输出投影
        x = self.output_projection(x)  # [seq_len, batch_size, output_dim]

        # 如果需要，转换回原始格式
        if transpose_output:
            x = x.transpose(0, 1)  # -> [batch_size, seq_len, output_dim]

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'input_dim={self.input_dim}, '
                f'd_model={self.d_model}, '
                f'output_dim={self.output_dim}, '
                f'num_layers={self.transformer.num_layers})')
