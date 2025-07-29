"""
数据集调试脚本

检查数据集中的实际数据形状和内容。

Author: AI Assistant
Date: 2025-07-29
"""

import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import SpatioTemporalDataset, spatiotemporal_collate_fn


def debug_dataset():
    """调试数据集"""
    print("开始数据集调试...")
    
    # 创建测试数据
    num_samples = 10
    num_nodes = 3
    seq_len = 12
    pred_len = 6
    node_features = 8
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    node_feat = torch.randn(num_nodes, node_features) * 0.1
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    time_series = torch.randn(num_samples, seq_len, node_features) * 0.1
    targets = torch.randn(num_samples, pred_len, num_nodes, 1) * 0.1
    
    print(f"原始数据形状:")
    print(f"  node_feat: {node_feat.shape}")
    print(f"  edge_index: {edge_index.shape}")
    print(f"  time_series: {time_series.shape}")
    print(f"  targets: {targets.shape}")
    
    print(f"\n边索引内容:")
    print(f"  edge_index: {edge_index}")
    print(f"  最大节点索引: {edge_index.max().item()}")
    print(f"  最小节点索引: {edge_index.min().item()}")
    
    # 创建数据集
    dataset = SpatioTemporalDataset(node_feat, edge_index, time_series, targets)
    
    print(f"\n数据集长度: {len(dataset)}")
    
    # 检查第一个样本
    sample = dataset[0]
    print(f"\n第一个样本:")
    for key, value in sample.items():
        print(f"  {key}: {value.shape}")
        if key == 'edge_index':
            print(f"    内容: {value}")
            print(f"    最大索引: {value.max().item()}")
    
    # 创建数据加载器
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=spatiotemporal_collate_fn)
    
    # 检查第一个批次
    batch = next(iter(data_loader))
    print(f"\n第一个批次:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
        if key == 'edge_index':
            print(f"    内容: {value}")
            print(f"    最大索引: {value.max().item()}")
        elif key == 'node_features':
            print(f"    节点数量: {value.shape[0]}")


if __name__ == '__main__':
    debug_dataset()
