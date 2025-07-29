"""
数据处理演示脚本

演示数据加载、预处理和图构建功能。

Author: AI Assistant
Date: 2025-07-29
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.graph_builder import GraphBuilder
from src.config import Config
from src.utils.logger import setup_logger, get_logger


def main():
    """主函数"""
    # 设置日志
    setup_logger(log_level="INFO")
    logger = get_logger(__name__)
    
    logger.info("开始数据处理演示...")
    
    try:
        # 1. 初始化配置
        config = Config()
        logger.info("配置初始化完成")
        
        # 2. 初始化数据加载器
        data_loader = DataLoader(config)
        logger.info("数据加载器初始化完成")
        
        # 3. 加载停车场数据
        parking_data_file = "sample_data/parking_data.csv"
        df = data_loader.load_parking_data(parking_data_file)
        logger.info(f"成功加载停车场数据: {len(df)} 条记录")
        
        # 4. 加载图拓扑数据
        graph_topology_file = "sample_data/graph.json"
        topology = data_loader.load_graph_topology(graph_topology_file)
        logger.info("成功加载图拓扑数据")
        
        # 5. 显示数据统计
        stats = data_loader.get_data_statistics()
        logger.info("数据统计信息:")
        logger.info(f"  - 总记录数: {stats['total_records']}")
        logger.info(f"  - 停车场数量: {stats['num_parking_lots']}")
        logger.info(f"  - 时间范围: {stats['time_range']['start']} 到 {stats['time_range']['end']}")
        logger.info(f"  - 平均占用率: {stats['occupancy_stats']['mean']:.3f}")
        logger.info(f"  - 周末数据比例: {stats['weekend_ratio']:.3f}")
        
        # 6. 数据预处理
        preprocessor = DataPreprocessor(config)
        logger.info("开始数据预处理...")
        
        df_processed = preprocessor.preprocess(df, fit=True)
        logger.info(f"数据预处理完成: {len(df_processed)} 条记录")
        
        # 7. 构建图结构
        graph_builder = GraphBuilder(config)
        logger.info("开始构建图结构...")
        
        graph_data = graph_builder.build_graph(df_processed, topology, method='hybrid')
        logger.info("图结构构建完成")
        
        # 8. 显示图统计
        graph_stats = graph_builder.get_graph_statistics(graph_data)
        logger.info("图结构统计:")
        logger.info(f"  - 节点数: {graph_stats['num_nodes']}")
        logger.info(f"  - 边数: {graph_stats['num_edges']}")
        logger.info(f"  - 平均度: {graph_stats['avg_degree']:.2f}")
        logger.info(f"  - 图密度: {graph_stats['density']:.3f}")
        
        # 9. 创建时间序列数据
        logger.info("创建时间序列数据...")
        time_series_data = preprocessor.create_time_series(
            df_processed,
            seq_len=24,  # 24个时间步（4小时）
            pred_len=6,  # 预测6个时间步（1小时）
            stride=6     # 每小时创建一个样本
        )
        logger.info(f"时间序列数据创建完成: {time_series_data.sequences.shape}")
        
        # 10. 显示张量信息
        logger.info("张量信息:")
        logger.info(f"  - 输入序列形状: {time_series_data.sequences.shape}")
        logger.info(f"  - 目标序列形状: {time_series_data.targets.shape}")
        logger.info(f"  - 节点映射: {time_series_data.node_mapping}")
        
        logger.info("数据处理演示完成！")
        
        return {
            'parking_data': df_processed,
            'graph_data': graph_data,
            'time_series_data': time_series_data,
            'stats': stats,
            'graph_stats': graph_stats
        }
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    results = main()
    print("\n演示成功完成！")
    print(f"处理了 {results['stats']['total_records']} 条停车场数据")
    print(f"构建了包含 {results['graph_stats']['num_nodes']} 个节点的图结构")
    print(f"生成了 {results['time_series_data'].sequences.shape[0]} 个时间序列样本")
