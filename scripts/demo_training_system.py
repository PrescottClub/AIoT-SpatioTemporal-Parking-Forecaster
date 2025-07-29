"""
训练系统演示脚本

演示完整的训练流程，包括数据准备、模型训练、验证和保存。

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

from src.data.data_loader import DataLoader as ParkingDataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.graph_builder import GraphBuilder
from src.models.spatiotemporal_model import SpatioTemporalPredictor
from src.training.trainer import Trainer, SpatioTemporalDataset, spatiotemporal_collate_fn
from src.training.losses import create_loss_function
from src.config import Config
from src.utils.logger import setup_logger, get_logger


def create_simple_dataset(config, device):
    """创建简化的训练数据集"""
    # 简化的数据参数
    num_samples = 50
    num_nodes = 3
    seq_len = 12
    pred_len = 6
    node_features = 8
    
    # 生成模拟数据
    torch.manual_seed(42)
    np.random.seed(42)
    
    node_feat = torch.randn(num_nodes, node_features).to(device) * 0.1
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]]).to(device)
    time_series = torch.randn(num_samples, seq_len, node_features).to(device) * 0.1
    targets = torch.randn(num_samples, pred_len, num_nodes, 1).to(device) * 0.1
    
    # 创建数据集
    dataset = SpatioTemporalDataset(node_feat, edge_index, time_series, targets)
    
    # 分割数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=spatiotemporal_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=spatiotemporal_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=spatiotemporal_collate_fn)
    
    return train_loader, val_loader, test_loader


def main():
    """主函数"""
    # 设置日志
    setup_logger(log_level="INFO")
    logger = get_logger(__name__)
    
    logger.info("开始训练系统演示...")
    
    try:
        # 1. 初始化配置
        config = Config()
        
        # 调整配置以适应演示
        config.model.node_feature_dim = 8
        config.model.hidden_dim = 16
        config.data.sequence_length = 12
        config.data.prediction_length = 6
        config.training.epochs = 5
        config.training.learning_rate = 0.001
        config.training.patience = 3
        config.training.log_every_n_steps = 1
        config.training.monitor_metric = 'val_loss'
        config.training.mode = 'min'
        
        logger.info("配置初始化完成")
        
        # 2. 设备配置
        device = torch.device('cpu')  # 强制使用CPU进行调试
        config.system.device = 'cpu'  # 确保配置也使用CPU
        logger.info(f"使用设备: {device}")
        
        # 3. 创建数据集
        logger.info("创建训练数据集...")
        train_loader, val_loader, test_loader = create_simple_dataset(config, device)
        
        logger.info(f"训练集: {len(train_loader.dataset)} 样本")
        logger.info(f"验证集: {len(val_loader.dataset)} 样本")
        logger.info(f"测试集: {len(test_loader.dataset)} 样本")
        
        # 4. 创建模型
        logger.info("创建时空预测模型...")
        model = SpatioTemporalPredictor(config)
        
        # 配置损失函数
        model.loss_fn = create_loss_function({
            'type': 'spatiotemporal',
            'params': {
                'base_loss': 'mse',
                'consistency_weight': 0.1,
                'smoothness_weight': 0.1
            }
        })
        
        logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 5. 创建训练器
        logger.info("初始化训练器...")
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        # 6. 开始训练
        logger.info("开始模型训练...")
        training_results = trainer.train()
        
        # 7. 分析训练结果
        logger.info("分析训练结果...")
        
        history = training_results['training_history']
        final_results = training_results['final_results']
        total_time = training_results['total_time']
        best_score = training_results['best_score']
        
        logger.info(f"训练完成，总耗时: {total_time:.2f}秒")
        logger.info(f"最佳验证分数: {best_score:.4f}")
        
        # 打印训练历史
        logger.info("训练历史:")
        for epoch_data in history:
            epoch = epoch_data['epoch']
            train_loss = epoch_data['train_loss']
            val_loss = epoch_data['val_loss']
            lr = epoch_data['lr']
            logger.info(f"  Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                       f"val_loss={val_loss:.4f}, lr={lr:.6f}")
        
        # 最终评估
        if 'test_metrics' in final_results:
            test_metrics = final_results['test_metrics']
            logger.info("测试集评估结果:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # 8. 保存最终模型
        logger.info("保存最终模型...")
        final_checkpoint = trainer.save_checkpoint(
            epoch=len(history) - 1,
            score=best_score,
            filename="final_model.pth"
        )
        logger.info(f"最终模型已保存: {final_checkpoint}")
        
        # 9. 演示模型推理
        logger.info("演示模型推理...")
        model.eval()
        
        # 获取一个测试批次
        test_batch = next(iter(test_loader))
        test_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in test_batch.items()}
        
        with torch.no_grad():
            result = model(test_batch)
            predictions = result['predictions']
            targets = test_batch['targets']
            
            logger.info(f"推理输入形状: {test_batch['time_series'].shape}")
            logger.info(f"推理输出形状: {predictions.shape}")
            logger.info(f"目标形状: {targets.shape}")
            
            # 计算简单指标
            mae = torch.mean(torch.abs(predictions - targets)).item()
            mse = torch.mean((predictions - targets) ** 2).item()
            
            logger.info(f"推理MAE: {mae:.4f}")
            logger.info(f"推理MSE: {mse:.4f}")
        
        logger.info("训练系统演示完成！")
        
        return {
            'training_results': training_results,
            'model': model,
            'trainer': trainer,
            'final_checkpoint': final_checkpoint
        }
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    results = main()
    
    print("\n" + "="*60)
    print("训练系统演示总结")
    print("="*60)
    
    training_results = results['training_results']
    history = training_results['training_history']
    
    print(f"训练轮数: {len(history)}")
    print(f"总训练时间: {training_results['total_time']:.2f}秒")
    print(f"最佳验证分数: {training_results['best_score']:.4f}")
    
    if history:
        final_epoch = history[-1]
        print(f"最终训练损失: {final_epoch['train_loss']:.4f}")
        print(f"最终验证损失: {final_epoch['val_loss']:.4f}")
    
    print(f"模型检查点: {results['final_checkpoint']}")
    print("\n✅ 训练系统功能验证成功！")
