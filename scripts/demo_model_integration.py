"""
模型集成演示脚本

演示GAT、Transformer和时空融合模型的完整集成。

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

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.graph_builder import GraphBuilder
from src.models.spatiotemporal_model import SpatioTemporalModel, SpatioTemporalPredictor
from src.config import Config
from src.utils.logger import setup_logger, get_logger
from src.utils.metrics import calculate_metrics


def main():
    """主函数"""
    # 设置日志
    setup_logger(log_level="INFO")
    logger = get_logger(__name__)
    
    logger.info("开始模型集成演示...")
    
    try:
        # 1. 初始化配置
        config = Config()
        logger.info("配置初始化完成")
        
        # 2. 加载和预处理数据
        logger.info("加载数据...")
        data_loader = DataLoader(config)
        df = data_loader.load_parking_data("sample_data/parking_data.csv")
        topology = data_loader.load_graph_topology("sample_data/graph.json")
        
        preprocessor = DataPreprocessor(config)
        df_processed = preprocessor.preprocess(df, fit=True)
        
        graph_builder = GraphBuilder(config)
        graph_data = graph_builder.build_graph(df_processed, topology, method='hybrid')
        
        logger.info(f"数据加载完成: {len(df)} 条记录, {graph_data.num_nodes} 个节点")
        
        # 3. 创建时间序列数据
        logger.info("创建时间序列数据...")
        time_series_data = preprocessor.create_time_series(
            df_processed,
            seq_len=24,  # 4小时历史数据
            pred_len=6,  # 预测1小时
            stride=6     # 每小时创建一个样本
        )
        
        logger.info(f"时间序列数据: {time_series_data.sequences.shape}")
        
        # 4. 创建模型
        logger.info("创建时空预测模型...")
        
        # 获取特征维度
        num_features = time_series_data.sequences.shape[-1]
        
        model = SpatioTemporalModel(
            node_features=num_features,
            hidden_dim=64,
            num_gat_layers=2,
            num_transformer_layers=2,
            seq_len=24,
            pred_len=6,
            num_heads=4,
            dropout=0.1,
            fusion_method='attention'
        )
        
        logger.info(f"模型创建完成: {sum(p.numel() for p in model.parameters())} 个参数")
        
        # 5. 准备输入数据
        logger.info("准备模型输入...")
        
        # 节点特征（使用所有数值特征的平均值）
        # 获取所有数值特征列
        numeric_cols = [col for col in df_processed.columns
                       if df_processed[col].dtype in ['float32', 'float64', 'int32', 'int64', 'bool']
                       and col not in ['timestamp', 'parking_id']]

        logger.info(f"使用的特征列: {len(numeric_cols)} 个")

        node_features = df_processed.groupby('parking_id')[numeric_cols].mean()
        node_features = torch.FloatTensor(node_features.values)

        logger.info(f"节点特征形状: {node_features.shape}")
        
        # 边索引
        edge_index = graph_data.edge_index
        
        # 时间序列（取前8个样本作为批次）
        batch_size = min(8, time_series_data.sequences.shape[0])

        # 检查实际的张量形状
        logger.info(f"原始序列形状: {time_series_data.sequences.shape}")
        logger.info(f"原始目标形状: {time_series_data.targets.shape}")

        # 根据实际形状调整索引
        if time_series_data.sequences.dim() == 4:
            time_series = time_series_data.sequences[:batch_size, :, 0, :]  # [batch, seq, features]
        else:
            time_series = time_series_data.sequences[:batch_size, :, :]  # [batch, seq, features]

        if time_series_data.targets.dim() == 3:
            targets = time_series_data.targets[:batch_size, :, :]  # [batch, pred, 1]
        else:
            targets = time_series_data.targets[:batch_size, :]  # [batch, pred]
        
        logger.info(f"输入形状: node_features={node_features.shape}, "
                   f"edge_index={edge_index.shape}, time_series={time_series.shape}")
        
        # 6. 模型前向传播
        logger.info("执行模型前向传播...")
        
        model.eval()
        with torch.no_grad():
            predictions = model(node_features, edge_index, time_series)
        
        logger.info(f"预测输出形状: {predictions.shape}")
        
        # 7. 计算评估指标
        logger.info("计算评估指标...")
        
        # 重塑目标值以匹配预测形状
        targets_reshaped = targets.unsqueeze(-1).expand(-1, -1, graph_data.num_nodes, -1)
        
        metrics = calculate_metrics(
            targets_reshaped.numpy().flatten(),
            predictions.numpy().flatten(),
            metrics=['mae', 'rmse', 'mape', 'r2']
        )
        
        logger.info("评估指标:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # 8. 测试训练模式
        logger.info("测试训练模式...")

        # 更新配置以匹配实际特征维度
        config.model.node_feature_dim = num_features
        config.model.hidden_dim = 64
        config.data.sequence_length = 24
        config.data.prediction_length = 6

        # 创建预测器
        predictor = SpatioTemporalPredictor(config)
        
        # 准备训练批次
        batch = {
            'node_features': node_features,
            'edge_index': edge_index,
            'time_series': time_series,
            'targets': targets_reshaped
        }
        
        # 前向传播（包含损失计算）
        predictor.train()
        result = predictor(batch)
        
        logger.info(f"训练模式 - 损失: {result['loss'].item():.4f}")
        
        # 9. 测试梯度计算
        logger.info("测试梯度计算...")
        
        optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
        
        optimizer.zero_grad()
        loss = result['loss']
        loss.backward()
        
        # 检查梯度
        total_grad_norm = 0
        param_count = 0
        for param in predictor.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
                param_count += 1
        
        total_grad_norm = total_grad_norm ** 0.5
        logger.info(f"梯度范数: {total_grad_norm:.4f}, 参数数量: {param_count}")
        
        optimizer.step()
        
        # 10. 性能分析
        logger.info("性能分析...")
        
        import time
        
        # 推理性能
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                _ = model(node_features, edge_index, time_series)
        
        inference_time = (time.time() - start_time) / 100
        logger.info(f"平均推理时间: {inference_time*1000:.2f} ms")
        
        # 内存使用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            logger.info(f"GPU内存使用: {memory_allocated:.2f} MB")
        
        logger.info("模型集成演示完成！")
        
        return {
            'model': model,
            'predictor': predictor,
            'predictions': predictions,
            'metrics': metrics,
            'inference_time': inference_time
        }
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    results = main()
    
    print("\n" + "="*50)
    print("模型集成演示总结")
    print("="*50)
    print(f"模型参数数量: {sum(p.numel() for p in results['model'].parameters()):,}")
    print(f"预测输出形状: {results['predictions'].shape}")
    print(f"平均推理时间: {results['inference_time']*1000:.2f} ms")
    print("\n评估指标:")
    for metric, value in results['metrics'].items():
        print(f"  {metric.upper()}: {value:.4f}")
    print("\n✅ 所有模型组件集成成功！")
