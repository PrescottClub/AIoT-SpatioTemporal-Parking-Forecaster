"""
数据加载器测试模块

测试数据加载和处理功能。

Author: AI Assistant
Date: 2025-07-29
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import torch
from pathlib import Path
from datetime import datetime

from src.data.data_loader import DataLoader, ParkingRecord, GraphData, TimeSeriesData
from src.config import Config


class TestDataLoader:
    """数据加载器测试类"""
    
    def test_init(self):
        """测试初始化"""
        loader = DataLoader()
        assert loader.config is None
        assert loader._parking_data_cache is None
        assert loader._graph_topology_cache is None
        
        config = Config()
        loader_with_config = DataLoader(config)
        assert loader_with_config.config is config
    
    def test_load_parking_data_success(self, temp_data_dir):
        """测试成功加载停车场数据"""
        loader = DataLoader()
        data_file = Path(temp_data_dir) / "parking_data.csv"
        
        df = loader.load_parking_data(data_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'parking_id' in df.columns
        assert 'occupancy' in df.columns
        assert 'timestamp' in df.columns
        
        # 检查数据类型
        assert df['occupancy'].dtype == np.float32
        assert df['timestamp'].dtype.name.startswith('datetime')
        assert df['parking_id'].dtype.name == 'category'
    
    def test_load_parking_data_file_not_found(self):
        """测试文件不存在的情况"""
        loader = DataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_parking_data("nonexistent_file.csv")
    
    def test_load_parking_data_cache(self, temp_data_dir):
        """测试数据缓存功能"""
        loader = DataLoader()
        data_file = Path(temp_data_dir) / "parking_data.csv"
        
        # 第一次加载
        df1 = loader.load_parking_data(data_file, use_cache=True)
        assert loader._parking_data_cache is not None
        
        # 第二次加载（使用缓存）
        df2 = loader.load_parking_data(data_file, use_cache=True)
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_load_graph_topology_success(self, temp_data_dir):
        """测试成功加载图拓扑数据"""
        loader = DataLoader()
        graph_file = Path(temp_data_dir) / "graph.json"
        
        topology = loader.load_graph_topology(graph_file)
        
        assert isinstance(topology, dict)
        assert 'edges' in topology
        assert isinstance(topology['edges'], dict)
    
    def test_load_graph_topology_file_not_found(self):
        """测试图拓扑文件不存在的情况"""
        loader = DataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_graph_topology("nonexistent_graph.json")
    
    def test_validate_parking_data_missing_columns(self):
        """测试缺少必需列的情况"""
        loader = DataLoader()
        
        # 创建缺少列的DataFrame
        df = pd.DataFrame({
            'timestamp': ['2025-07-28 10:00:00'],
            'parking_id': ['P001']
            # 缺少其他必需列
        })
        
        with pytest.raises(ValueError, match="缺少必需的列"):
            loader._validate_parking_data(df)
    
    def test_validate_parking_data_empty(self):
        """测试空数据的情况"""
        loader = DataLoader()
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="数据文件为空"):
            loader._validate_parking_data(df)
    
    def test_validate_parking_data_invalid_capacity(self):
        """测试无效容量的情况"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'timestamp': ['2025-07-28 10:00:00'],
            'parking_id': ['P001'],
            'occupancy': [0.5],
            'avg_confidence': [0.9],
            'static_capacity': [0],  # 无效容量
            'static_price_level': [3],
            'is_weekend': [False],
            'weather_condition': ['Clear'],
            'poi_hotness': [0.5]
        })
        
        with pytest.raises(ValueError, match="停车场容量必须大于0"):
            loader._validate_parking_data(df)
    
    def test_validate_graph_topology_missing_edges(self):
        """测试缺少边信息的情况"""
        loader = DataLoader()
        
        topology = {'nodes': {}}  # 缺少edges字段
        
        with pytest.raises(ValueError, match="图拓扑数据缺少'edges'字段"):
            loader._validate_graph_topology(topology)
    
    def test_validate_graph_topology_invalid_edges(self):
        """测试无效边格式的情况"""
        loader = DataLoader()
        
        topology = {'edges': 'invalid'}  # edges不是字典
        
        with pytest.raises(ValueError, match="'edges'字段必须是字典类型"):
            loader._validate_graph_topology(topology)
    
    def test_get_parking_ids(self, sample_parking_data):
        """测试获取停车场ID列表"""
        loader = DataLoader()
        loader._parking_data_cache = sample_parking_data
        
        parking_ids = loader.get_parking_ids()
        
        assert isinstance(parking_ids, list)
        assert len(parking_ids) > 0
        assert all(isinstance(pid, str) for pid in parking_ids)
        assert parking_ids == sorted(parking_ids)  # 应该是排序的
    
    def test_get_parking_ids_no_cache(self):
        """测试没有缓存数据时获取停车场ID"""
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="没有可用的停车场数据"):
            loader.get_parking_ids()
    
    def test_get_time_range(self, sample_parking_data):
        """测试获取时间范围"""
        loader = DataLoader()
        loader._parking_data_cache = sample_parking_data
        
        start_time, end_time = loader.get_time_range()
        
        assert isinstance(start_time, datetime)
        assert isinstance(end_time, datetime)
        assert start_time <= end_time
    
    def test_get_data_statistics(self, sample_parking_data):
        """测试获取数据统计信息"""
        loader = DataLoader()
        loader._parking_data_cache = sample_parking_data
        
        stats = loader.get_data_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_records' in stats
        assert 'num_parking_lots' in stats
        assert 'parking_ids' in stats
        assert 'time_range' in stats
        assert 'occupancy_stats' in stats
        assert 'confidence_stats' in stats
        assert 'weather_conditions' in stats
        assert 'weekend_ratio' in stats
        
        # 检查统计值的合理性
        assert stats['total_records'] > 0
        assert stats['num_parking_lots'] > 0
        assert 0 <= stats['weekend_ratio'] <= 1
        assert 0 <= stats['occupancy_stats']['mean'] <= 1
    
    def test_clear_cache(self, temp_data_dir):
        """测试清除缓存"""
        loader = DataLoader()
        data_file = Path(temp_data_dir) / "parking_data.csv"
        graph_file = Path(temp_data_dir) / "graph.json"
        
        # 加载数据到缓存
        loader.load_parking_data(data_file)
        loader.load_graph_topology(graph_file)
        
        assert loader._parking_data_cache is not None
        assert loader._graph_topology_cache is not None
        
        # 清除缓存
        loader.clear_cache()
        
        assert loader._parking_data_cache is None
        assert loader._graph_topology_cache is None


class TestParkingRecord:
    """停车场记录测试类"""
    
    def test_parking_record_creation(self):
        """测试停车场记录创建"""
        timestamp = datetime(2025, 7, 28, 10, 0, 0)
        
        record = ParkingRecord(
            timestamp=timestamp,
            parking_id="P001",
            occupancy=0.75,
            avg_confidence=0.98,
            static_capacity=500,
            static_price_level=3,
            is_weekend=False,
            weather_condition="Clear",
            poi_hotness=0.8
        )
        
        assert record.timestamp == timestamp
        assert record.parking_id == "P001"
        assert record.occupancy == 0.75
        assert record.avg_confidence == 0.98
        assert record.static_capacity == 500
        assert record.static_price_level == 3
        assert record.is_weekend is False
        assert record.weather_condition == "Clear"
        assert record.poi_hotness == 0.8
    
    def test_parking_record_to_dict(self):
        """测试停车场记录转换为字典"""
        timestamp = datetime(2025, 7, 28, 10, 0, 0)
        
        record = ParkingRecord(
            timestamp=timestamp,
            parking_id="P001",
            occupancy=0.75,
            avg_confidence=0.98,
            static_capacity=500,
            static_price_level=3,
            is_weekend=False,
            weather_condition="Clear",
            poi_hotness=0.8
        )
        
        record_dict = record.to_dict()
        
        assert isinstance(record_dict, dict)
        assert record_dict['timestamp'] == timestamp
        assert record_dict['parking_id'] == "P001"
        assert record_dict['occupancy'] == 0.75


class TestGraphData:
    """图数据测试类"""
    
    def test_graph_data_creation(self, sample_tensor_data):
        """测试图数据创建"""
        edge_index = sample_tensor_data['edge_index']
        edge_attr = sample_tensor_data['edge_attr']
        node_features = torch.randn(3, 8)
        node_ids = ['P001', 'P002', 'P003']
        
        graph_data = GraphData(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_features=node_features,
            node_ids=node_ids,
            num_nodes=3,
            num_edges=4
        )
        
        assert torch.equal(graph_data.edge_index, edge_index)
        assert torch.equal(graph_data.edge_attr, edge_attr)
        assert torch.equal(graph_data.node_features, node_features)
        assert graph_data.node_ids == node_ids
        assert graph_data.num_nodes == 3
        assert graph_data.num_edges == 4
    
    def test_graph_data_to_device(self, sample_tensor_data, device):
        """测试图数据设备转换"""
        edge_index = sample_tensor_data['edge_index']
        edge_attr = sample_tensor_data['edge_attr']
        node_features = torch.randn(3, 8)
        node_ids = ['P001', 'P002', 'P003']
        
        graph_data = GraphData(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_features=node_features,
            node_ids=node_ids,
            num_nodes=3,
            num_edges=4
        )
        
        graph_data_device = graph_data.to(device)
        
        assert graph_data_device.edge_index.device.type == device.type
        assert graph_data_device.edge_attr.device.type == device.type
        assert graph_data_device.node_features.device.type == device.type
        assert graph_data_device.node_ids == node_ids  # 列表不变


class TestTimeSeriesData:
    """时间序列数据测试类"""
    
    def test_time_series_data_creation(self, sample_tensor_data):
        """测试时间序列数据创建"""
        sequences = sample_tensor_data['node_features']
        targets = sample_tensor_data['targets']
        timestamps = [datetime.now()]
        node_mapping = {'P001': 0, 'P002': 1, 'P003': 2}
        
        ts_data = TimeSeriesData(
            sequences=sequences,
            targets=targets,
            timestamps=timestamps,
            node_mapping=node_mapping
        )
        
        assert torch.equal(ts_data.sequences, sequences)
        assert torch.equal(ts_data.targets, targets)
        assert ts_data.timestamps == timestamps
        assert ts_data.node_mapping == node_mapping
    
    def test_time_series_data_to_device(self, sample_tensor_data, device):
        """测试时间序列数据设备转换"""
        sequences = sample_tensor_data['node_features']
        targets = sample_tensor_data['targets']
        timestamps = [datetime.now()]
        node_mapping = {'P001': 0, 'P002': 1, 'P003': 2}
        
        ts_data = TimeSeriesData(
            sequences=sequences,
            targets=targets,
            timestamps=timestamps,
            node_mapping=node_mapping
        )
        
        ts_data_device = ts_data.to(device)
        
        assert ts_data_device.sequences.device.type == device.type
        assert ts_data_device.targets.device.type == device.type
        assert ts_data_device.timestamps == timestamps  # 列表不变
        assert ts_data_device.node_mapping == node_mapping  # 字典不变
