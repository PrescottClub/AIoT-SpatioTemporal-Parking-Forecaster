"""
模型数值稳定性测试脚本

测试模型是否存在NaN问题。

Author: AI Assistant
Date: 2025-07-29
"""

import sys
from pathlib import Path
import torch
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.spatiotemporal_model import SpatioTemporalModel


def test_model_stability():
    """测试模型数值稳定性"""
    print("测试模型数值稳定性...")
    
    # 创建简单的测试数据
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 模型参数
    node_features = 8
    hidden_dim = 32
    seq_len = 12
    pred_len = 3
    
    # 创建模型
    model = SpatioTemporalModel(
        node_features=node_features,
        hidden_dim=hidden_dim,
        num_gat_layers=1,
        num_transformer_layers=1,
        seq_len=seq_len,
        pred_len=pred_len,
        num_heads=2,
        dropout=0.0  # 关闭dropout进行测试
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建测试输入
    batch_size = 2
    num_nodes = 3
    
    node_feat = torch.randn(num_nodes, node_features) * 0.1  # 小的初始值
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    time_series = torch.randn(batch_size, seq_len, node_features) * 0.1
    
    print(f"输入形状: node_features={node_feat.shape}, edge_index={edge_index.shape}, time_series={time_series.shape}")
    
    # 检查输入是否包含NaN
    print(f"输入包含NaN: {torch.isnan(node_feat).any() or torch.isnan(time_series).any()}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        try:
            output = model(node_feat, edge_index, time_series)
            print(f"输出形状: {output.shape}")
            print(f"输出包含NaN: {torch.isnan(output).any()}")
            print(f"输出包含Inf: {torch.isinf(output).any()}")
            print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            if torch.isnan(output).any():
                print("❌ 模型输出包含NaN值")
                return False
            else:
                print("✅ 模型输出正常")
                return True
                
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return False


def test_individual_components():
    """测试各个组件"""
    print("\n测试各个组件...")
    
    from src.models.gat_layer import GATLayer
    from src.models.transformer_layer import TemporalTransformer
    
    # 测试GAT层
    print("测试GAT层...")
    gat = GATLayer(in_features=8, out_features=16, num_heads=2, dropout=0.0)
    
    node_feat = torch.randn(3, 8) * 0.1
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    
    gat_output = gat(node_feat, edge_index)
    print(f"GAT输出包含NaN: {torch.isnan(gat_output).any()}")
    
    # 测试Transformer
    print("测试Transformer...")
    transformer = TemporalTransformer(
        input_dim=8, d_model=16, output_dim=8, 
        num_heads=2, num_layers=1, dropout=0.0
    )
    
    time_input = torch.randn(2, 12, 8) * 0.1
    transformer_output = transformer(time_input)
    print(f"Transformer输出包含NaN: {torch.isnan(transformer_output).any()}")


if __name__ == '__main__':
    success = test_model_stability()
    test_individual_components()
    
    if success:
        print("\n✅ 模型数值稳定性测试通过")
    else:
        print("\n❌ 模型存在数值稳定性问题")
