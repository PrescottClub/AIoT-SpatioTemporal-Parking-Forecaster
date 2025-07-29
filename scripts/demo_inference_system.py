"""
推理系统演示脚本

演示完整的推理流程，包括模型加载、预测、评估和结果处理。

Author: AI Assistant
Date: 2025-07-29
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import InferencePredictor, BatchPredictor
from src.inference.evaluator import ModelEvaluator, PerformanceAnalyzer
from src.inference.postprocessor import PredictionPostProcessor
from src.models.spatiotemporal_model import SpatioTemporalPredictor as TrainingPredictor
from src.data.preprocessor import DataPreprocessor
from src.config import Config
from src.utils.logger import setup_logger, get_logger


def create_demo_data(config, device):
    """创建演示数据"""
    # 模拟数据参数
    num_samples = 20
    num_nodes = 5
    seq_len = config.data.sequence_length
    pred_len = config.data.prediction_length
    node_features = config.model.node_feature_dim
    
    # 生成模拟数据
    torch.manual_seed(42)
    np.random.seed(42)
    
    node_feat = torch.randn(num_nodes, node_features).to(device) * 0.1
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).to(device)
    
    # 创建测试数据集
    test_data = []
    targets = []
    
    for i in range(num_samples):
        time_series = torch.randn(seq_len, node_features).to(device) * 0.1
        target = torch.randn(pred_len, num_nodes, 1).to(device) * 0.1
        
        test_data.append({
            'node_features': node_feat,
            'edge_index': edge_index,
            'time_series': time_series
        })
        targets.append(target)
    
    # 创建元数据
    metadata = {
        'node_ids': [f'parking_lot_{i}' for i in range(num_nodes)],
        'start_time': datetime(2025, 1, 1, 0, 0, 0),
        'time_interval': timedelta(minutes=10),
        'feature_names': [f'feature_{i}' for i in range(node_features)]
    }
    
    return test_data, targets, metadata


def main():
    """主函数"""
    # 设置日志
    setup_logger(log_level="INFO")
    logger = get_logger(__name__)
    
    logger.info("开始推理系统演示...")
    
    try:
        # 1. 初始化配置
        config = Config()
        config.model.node_feature_dim = 8
        config.model.hidden_dim = 32
        config.data.sequence_length = 12
        config.data.prediction_length = 6
        config.system.device = 'cpu'  # 使用CPU进行演示
        
        device = torch.device(config.system.device)
        logger.info(f"使用设备: {device}")
        
        # 2. 创建训练模型（模拟已训练的模型）
        logger.info("创建模拟训练模型...")
        training_model = TrainingPredictor(config)
        
        # 3. 创建推理预测器
        logger.info("初始化推理预测器...")
        predictor = InferencePredictor(model=training_model, config=config)
        
        # 4. 创建演示数据
        logger.info("创建演示数据...")
        test_data, targets, metadata = create_demo_data(config, device)
        
        logger.info(f"测试数据: {len(test_data)} 个样本")
        logger.info(f"节点数量: {len(metadata['node_ids'])}")
        
        # 5. 单次预测演示
        logger.info("\n=== 单次预测演示 ===")
        
        single_data = test_data[0]
        predictions = predictor.predict(
            single_data['node_features'],
            single_data['edge_index'],
            single_data['time_series'].unsqueeze(0)
        )
        
        logger.info(f"单次预测输入形状: {single_data['time_series'].shape}")
        logger.info(f"单次预测输出形状: {predictions.shape}")
        
        # 6. 实时预测演示
        logger.info("\n=== 实时预测演示 ===")
        
        realtime_result = predictor.predict_realtime(
            single_data['node_features'],
            single_data['edge_index'],
            single_data['time_series'].unsqueeze(0)
        )
        
        logger.info(f"实时预测耗时: {realtime_result['inference_time']*1000:.2f} ms")
        logger.info(f"预测时间戳: {realtime_result['timestamp']}")
        
        # 7. 批量预测演示
        logger.info("\n=== 批量预测演示 ===")
        
        batch_predictor = BatchPredictor(predictor, batch_size=8)
        batch_predictions = batch_predictor.predict_dataset(test_data, show_progress=False)
        
        logger.info(f"批量预测结果: {len(batch_predictions)} 个预测")
        logger.info(f"单个预测形状: {batch_predictions[0].shape}")
        
        # 8. 模型评估演示
        logger.info("\n=== 模型评估演示 ===")
        
        evaluator = ModelEvaluator(predictor)
        eval_results = evaluator.evaluate(test_data[:10], targets[:10], detailed=True)
        
        logger.info("评估指标:")
        for metric, value in eval_results['metrics'].items():
            if not np.isnan(value):
                logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"推理性能: {eval_results['samples_per_second']:.2f} 样本/秒")
        
        # 9. 性能分析演示
        logger.info("\n=== 性能分析演示 ===")
        
        analyzer = PerformanceAnalyzer(predictor)
        benchmark_results = analyzer.benchmark_inference(
            test_data[:8], batch_sizes=[1, 2, 4], num_runs=3
        )
        
        logger.info("性能基准测试:")
        for batch_size, result in benchmark_results.items():
            logger.info(f"  {batch_size}: {result['throughput']:.2f} 样本/秒")
        
        # 10. 结果后处理演示
        logger.info("\n=== 结果后处理演示 ===")
        
        # DataFrame格式
        postprocessor = PredictionPostProcessor(output_format='dataframe')
        df_result = postprocessor.process(
            predictions, metadata
        )
        
        logger.info(f"DataFrame结果形状: {df_result.shape}")
        logger.info("DataFrame前5行:")
        logger.info(f"\n{df_result.head()}")
        
        # JSON格式
        postprocessor.output_format = 'json'
        json_result = postprocessor.process(predictions, metadata)
        
        logger.info(f"JSON结果键: {list(json_result.keys())}")
        logger.info(f"元数据: {json_result['metadata']}")
        
        # 11. 置信区间计算
        logger.info("\n=== 置信区间计算演示 ===")
        
        # 模拟置信度
        confidence = torch.randn_like(predictions) * 0.1 + 0.8
        lower, upper = postprocessor.calculate_confidence_intervals(
            predictions, confidence, confidence_level=0.95
        )
        
        logger.info(f"置信区间形状: {lower.shape}")
        logger.info(f"平均置信区间宽度: {torch.mean(upper - lower).item():.4f}")
        
        # 12. 摘要报告
        logger.info("\n=== 摘要报告演示 ===")
        
        summary = postprocessor.create_summary_report(predictions, metadata)
        
        logger.info("预测摘要:")
        logger.info(f"  形状: {summary['shape']}")
        logger.info(f"  均值: {summary['statistics']['mean']:.4f}")
        logger.info(f"  标准差: {summary['statistics']['std']:.4f}")
        logger.info(f"  范围: [{summary['statistics']['min']:.4f}, {summary['statistics']['max']:.4f}]")
        
        # 13. 保存结果演示
        logger.info("\n=== 保存结果演示 ===")
        
        # 创建输出目录
        output_dir = Path("outputs/inference_demo")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存DataFrame
        postprocessor.save_predictions(
            df_result, 
            output_dir / "predictions.csv",
            format='csv'
        )
        
        # 保存JSON
        postprocessor.save_predictions(
            json_result,
            output_dir / "predictions.json",
            format='json'
        )
        
        logger.info(f"结果已保存到: {output_dir}")
        
        logger.info("推理系统演示完成！")
        
        return {
            'predictor': predictor,
            'batch_predictor': batch_predictor,
            'evaluator': evaluator,
            'analyzer': analyzer,
            'postprocessor': postprocessor,
            'predictions': predictions,
            'eval_results': eval_results,
            'benchmark_results': benchmark_results,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    results = main()
    
    print("\n" + "="*60)
    print("推理系统演示总结")
    print("="*60)
    
    eval_results = results['eval_results']
    benchmark_results = results['benchmark_results']
    
    print(f"评估样本数: {eval_results['num_samples']}")
    print(f"推理性能: {eval_results['samples_per_second']:.2f} 样本/秒")
    
    print("\n性能基准:")
    for batch_size, result in benchmark_results.items():
        print(f"  {batch_size}: {result['throughput']:.2f} 样本/秒")
    
    print(f"\n预测形状: {results['predictions'].shape}")
    print(f"摘要统计: 均值={results['summary']['statistics']['mean']:.4f}, "
          f"标准差={results['summary']['statistics']['std']:.4f}")
    
    print("\n✅ 推理系统功能验证成功！")
