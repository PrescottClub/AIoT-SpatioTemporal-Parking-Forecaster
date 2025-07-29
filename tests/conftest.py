"""
pytest配置文件

提供测试的全局配置和fixture。

Author: AI Assistant
Date: 2025-07-29
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any, Tuple
import json

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


@pytest.fixture(scope="session")
def test_config():
    """测试配置fixture"""
    from src.config import Config
    
    config = Config()
    # 使用较小的参数进行快速测试
    config.data.batch_size = 4
    config.data.sequence_length = 24
    config.data.prediction_length = 6
    config.model.hidden_dim = 32
    config.model.num_gat_layers = 1
    config.model.num_transformer_layers = 1
    config.training.epochs = 2
    
    return config


@pytest.fixture(scope="session")
def sample_parking_data():
    """生成样本停车场数据"""
    np.random.seed(42)
    
    # 生成48小时的数据，每10分钟一个数据点
    timestamps = pd.date_range(
        start="2025-07-28 00:00:00",
        periods=288,  # 48小时 * 6个数据点/小时
        freq="10T"
    )
    
    parking_ids = ["P001", "P002", "P003"]
    data = []
    
    for timestamp in timestamps:
        for parking_id in parking_ids:
            # 模拟占用率的日周期性
            hour = timestamp.hour
            base_occupancy = 0.3 + 0.4 * np.sin(2 * np.pi * hour / 24)
            noise = np.random.normal(0, 0.1)
            occupancy = np.clip(base_occupancy + noise, 0, 1)
            
            data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "parking_id": parking_id,
                "occupancy": round(occupancy, 3),
                "avg_confidence": round(np.random.uniform(0.9, 1.0), 3),
                "static_capacity": {"P001": 500, "P002": 300, "P003": 800}[parking_id],
                "static_price_level": {"P001": 3, "P002": 2, "P003": 4}[parking_id],
                "is_weekend": 1 if timestamp.weekday() >= 5 else 0,
                "weather_condition": np.random.choice(["Clear", "Cloudy", "Rainy"]),
                "poi_hotness": round(np.random.uniform(0.1, 0.9), 1)
            })
    
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_graph_data():
    """生成样本图数据"""
    return {
        "nodes": {
            "P001": {
                "lat": 39.9042,
                "lon": 116.4074,
                "district": "central",
                "type": "street"
            },
            "P002": {
                "lat": 39.9142,
                "lon": 116.4174,
                "district": "central",
                "type": "mall"
            },
            "P003": {
                "lat": 39.8942,
                "lon": 116.3974,
                "district": "west",
                "type": "office"
            }
        },
        "edges": {
            "P001": ["P002", "P003"],
            "P002": ["P001"],
            "P003": ["P001"]
        },
        "edge_weights": {
            "P001-P002": 0.8,
            "P001-P003": 0.6,
            "P002-P001": 0.8,
            "P003-P001": 0.6
        }
    }


@pytest.fixture(scope="function")
def temp_data_dir(sample_parking_data, sample_graph_data):
    """创建临时数据目录"""
    temp_dir = tempfile.mkdtemp()
    
    # 创建数据文件
    data_file = Path(temp_dir) / "parking_data.csv"
    sample_parking_data.to_csv(data_file, index=False)
    
    graph_file = Path(temp_dir) / "graph.json"
    with open(graph_file, 'w', encoding='utf-8') as f:
        json.dump(sample_graph_data, f, indent=2)
    
    yield temp_dir
    
    # 清理
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def sample_tensor_data():
    """生成样本张量数据"""
    batch_size = 4
    seq_len = 24
    num_nodes = 3
    node_features = 8
    
    # 节点特征
    node_features_tensor = torch.randn(batch_size, seq_len, num_nodes, node_features)
    
    # 边索引 (COO格式)
    edge_index = torch.tensor([
        [0, 0, 1, 2],  # 源节点
        [1, 2, 0, 0]   # 目标节点
    ], dtype=torch.long)
    
    # 边特征
    edge_attr = torch.randn(4, 2)  # 4条边，每条边2个特征
    
    # 目标值
    targets = torch.randn(batch_size, 6, num_nodes, 1)  # 预测6个时间步
    
    return {
        "node_features": node_features_tensor,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "targets": targets
    }


@pytest.fixture(scope="function")
def mock_model():
    """创建模拟模型"""
    from src.models.spatiotemporal_model import SpatioTemporalModel
    
    model = SpatioTemporalModel(
        node_features=8,
        hidden_dim=32,
        num_gat_layers=1,
        num_transformer_layers=1,
        seq_len=24,
        pred_len=6,
        num_heads=4
    )
    
    return model


@pytest.fixture(scope="function")
def device():
    """获取测试设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """每个测试前重置随机种子"""
    torch.manual_seed(42)
    np.random.seed(42)


# 测试标记
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    # 为GPU测试添加跳过条件
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


# 性能测试辅助函数
@pytest.fixture
def benchmark_timer():
    """性能测试计时器"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.end_time - self.start_time
        
        @property
        def elapsed(self):
            if self.start_time is None:
                return 0
            if self.end_time is None:
                return time.time() - self.start_time
            return self.end_time - self.start_time
    
    return Timer()


# 内存使用监控
@pytest.fixture
def memory_monitor():
    """内存使用监控"""
    import psutil
    import os
    
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.initial_memory = self.process.memory_info().rss
        
        def get_current_usage(self):
            return self.process.memory_info().rss
        
        def get_memory_increase(self):
            return self.get_current_usage() - self.initial_memory
        
        def get_memory_usage_mb(self):
            return self.get_current_usage() / 1024 / 1024
    
    return MemoryMonitor()
