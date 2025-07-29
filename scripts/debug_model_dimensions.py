"""
模型维度调试脚本

详细分析模型各层的维度传递，定位具体的维度不匹配问题。

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

from src.models.spatiotemporal_model import SpatioTemporalModel, SpatioTemporalPredictor
from src.config import Config
from src.utils.logger import setup_logger, get_logger


def debug_model_dimensions():
    """调试模型维度传递"""
    setup_logger(log_level="DEBUG")
    logger = get_logger(__name__)
    
    logger.info("开始模型维度调试...")
    
    # 配置
    config = Config()
    config.model.node_feature_dim = 8
    config.model.hidden_dim = 16
    config.data.sequence_length = 12
    config.data.prediction_length = 6
    
    logger.info(f"配置信息:")
    logger.info(f"  node_feature_dim: {config.model.node_feature_dim}")
    logger.info(f"  hidden_dim: {config.model.hidden_dim}")
    logger.info(f"  output_dim: {config.model.output_dim}")
    logger.info(f"  sequence_length: {config.data.sequence_length}")
    logger.info(f"  prediction_length: {config.data.prediction_length}")
    
    # 创建测试数据
    batch_size = 4
    num_nodes = 3
    seq_len = config.data.sequence_length
    node_features = config.model.node_feature_dim
    
    node_feat = torch.randn(num_nodes, node_features) * 0.1
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    time_series = torch.randn(batch_size, seq_len, node_features) * 0.1
    
    logger.info(f"输入数据形状:")
    logger.info(f"  node_features: {node_feat.shape}")
    logger.info(f"  edge_index: {edge_index.shape}")
    logger.info(f"  time_series: {time_series.shape}")
    
    # 创建模型
    try:
        model = SpatioTemporalModel(
            node_features=node_features,
            hidden_dim=config.model.hidden_dim,
            num_gat_layers=1,
            num_transformer_layers=1,
            seq_len=seq_len,
            pred_len=config.data.prediction_length,
            num_heads=2,
            dropout=0.0,
            output_dim=config.model.output_dim
        )
        logger.info("模型创建成功")
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        return
    
    # 逐步调试各个组件
    model.eval()
    
    try:
        with torch.no_grad():
            logger.info("\n=== 调试空间编码器 ===")
            spatial_features = model.spatial_encoder(node_feat, edge_index)
            logger.info(f"空间编码器输出: {spatial_features.shape}")
            
            logger.info("\n=== 调试时间编码器 ===")
            temporal_features = model.temporal_encoder(time_series)
            logger.info(f"时间编码器输出: {temporal_features.shape}")
            
            logger.info("\n=== 调试时空融合 ===")
            # 扩展空间特征到批次维度
            if spatial_features.dim() == 2:
                spatial_expanded = spatial_features.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                spatial_expanded = spatial_features.expand(batch_size, -1, -1)
            
            logger.info(f"扩展后的空间特征: {spatial_expanded.shape}")
            
            fused_features = model.fusion_module(spatial_expanded, temporal_features)
            logger.info(f"融合模块输出: {fused_features.shape}")
            
            logger.info("\n=== 调试预测头 ===")
            last_features = fused_features[:, -1, :, :]
            logger.info(f"最后时间步特征: {last_features.shape}")
            
            reshaped_features = last_features.reshape(batch_size * num_nodes, -1)
            logger.info(f"重塑后特征: {reshaped_features.shape}")
            
            logger.info(f"预测头期望输入维度: {model.prediction_head[0].in_features}")
            logger.info(f"实际输入维度: {reshaped_features.shape[-1]}")
            
            if reshaped_features.shape[-1] != model.prediction_head[0].in_features:
                logger.error("❌ 维度不匹配！")
                logger.error(f"期望: {model.prediction_head[0].in_features}, 实际: {reshaped_features.shape[-1]}")
            else:
                logger.info("✅ 维度匹配")
                
                predictions = model.prediction_head(reshaped_features)
                logger.info(f"预测头输出: {predictions.shape}")
                
                final_predictions = predictions.view(batch_size, config.data.prediction_length, num_nodes, config.model.output_dim)
                logger.info(f"最终预测形状: {final_predictions.shape}")
            
    except Exception as e:
        logger.error(f"前向传播调试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 调试融合模块的详细配置
    logger.info("\n=== 融合模块配置 ===")
    fusion = model.fusion_module
    logger.info(f"spatial_dim: {fusion.spatial_dim}")
    logger.info(f"temporal_dim: {fusion.temporal_dim}")
    logger.info(f"hidden_dim: {fusion.hidden_dim}")
    logger.info(f"output_dim: {fusion.output_dim}")
    logger.info(f"fusion_method: {fusion.fusion_method}")
    
    # 检查输出层配置
    if hasattr(fusion, 'output_layer'):
        output_layer = fusion.output_layer
        logger.info(f"输出层结构:")
        for i, layer in enumerate(output_layer):
            logger.info(f"  {i}: {layer}")
            if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                logger.info(f"    输入维度: {layer.in_features}, 输出维度: {layer.out_features}")


if __name__ == '__main__':
    debug_model_dimensions()
