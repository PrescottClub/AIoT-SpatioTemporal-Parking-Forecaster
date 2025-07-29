"""
数据加载器模块

负责加载和处理停车场数据和图拓扑数据。

Author: AI Assistant
Date: 2025-07-29
"""

import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

from ..utils.logger import get_logger
from ..config import Config

logger = get_logger(__name__)


@dataclass
class ParkingRecord:
    """停车场记录数据类"""
    timestamp: datetime
    parking_id: str
    occupancy: float
    avg_confidence: float
    static_capacity: int
    static_price_level: int
    is_weekend: bool
    weather_condition: str
    poi_hotness: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp,
            'parking_id': self.parking_id,
            'occupancy': self.occupancy,
            'avg_confidence': self.avg_confidence,
            'static_capacity': self.static_capacity,
            'static_price_level': self.static_price_level,
            'is_weekend': self.is_weekend,
            'weather_condition': self.weather_condition,
            'poi_hotness': self.poi_hotness
        }


@dataclass
class GraphData:
    """图数据结构"""
    edge_index: torch.Tensor  # [2, num_edges]
    edge_attr: torch.Tensor   # [num_edges, edge_features]
    node_features: torch.Tensor  # [num_nodes, node_features]
    node_ids: List[str]
    num_nodes: int
    num_edges: int
    
    def to(self, device: torch.device) -> 'GraphData':
        """移动到指定设备"""
        return GraphData(
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            node_features=self.node_features.to(device),
            node_ids=self.node_ids,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges
        )


@dataclass
class TimeSeriesData:
    """时间序列数据结构"""
    sequences: torch.Tensor  # [batch_size, seq_len, num_nodes, num_features]
    targets: torch.Tensor    # [batch_size, pred_len, num_nodes, num_targets]
    timestamps: List[datetime]
    node_mapping: Dict[str, int]
    
    def to(self, device: torch.device) -> 'TimeSeriesData':
        """移动到指定设备"""
        return TimeSeriesData(
            sequences=self.sequences.to(device),
            targets=self.targets.to(device),
            timestamps=self.timestamps,
            node_mapping=self.node_mapping
        )


class DataLoader:
    """数据加载器
    
    负责加载停车场数据和图拓扑数据，并转换为模型所需的格式。
    
    Example:
        >>> loader = DataLoader(config)
        >>> df = loader.load_parking_data("sample_data/parking_data.csv")
        >>> graph_data = loader.load_graph_topology("sample_data/graph.json")
    """
    
    def __init__(self, config: Optional[Config] = None):
        """初始化数据加载器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
        # 缓存
        self._parking_data_cache: Optional[pd.DataFrame] = None
        self._graph_topology_cache: Optional[Dict[str, Any]] = None
    
    def load_parking_data(
        self, 
        file_path: Union[str, Path],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """加载停车场数据
        
        Args:
            file_path: CSV文件路径
            use_cache: 是否使用缓存
            
        Returns:
            停车场数据DataFrame
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误
        """
        file_path = Path(file_path)
        
        # 检查缓存
        if use_cache and self._parking_data_cache is not None:
            self.logger.info("使用缓存的停车场数据")
            return self._parking_data_cache.copy()
        
        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        self.logger.info(f"加载停车场数据: {file_path}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 验证数据格式
            self._validate_parking_data(df)
            
            # 数据类型转换
            df = self._convert_parking_data_types(df)
            
            # 缓存数据
            if use_cache:
                self._parking_data_cache = df.copy()
            
            self.logger.info(f"成功加载 {len(df)} 条停车场数据记录")
            return df
            
        except Exception as e:
            self.logger.error(f"加载停车场数据失败: {e}")
            raise
    
    def load_graph_topology(
        self, 
        file_path: Union[str, Path],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """加载图拓扑结构
        
        Args:
            file_path: JSON文件路径
            use_cache: 是否使用缓存
            
        Returns:
            图拓扑数据字典
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误
        """
        file_path = Path(file_path)
        
        # 检查缓存
        if use_cache and self._graph_topology_cache is not None:
            self.logger.info("使用缓存的图拓扑数据")
            return self._graph_topology_cache.copy()
        
        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"图拓扑文件不存在: {file_path}")
        
        self.logger.info(f"加载图拓扑数据: {file_path}")
        
        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # 验证数据格式
            self._validate_graph_topology(graph_data)
            
            # 缓存数据
            if use_cache:
                self._graph_topology_cache = graph_data.copy()
            
            self.logger.info(f"成功加载图拓扑数据，包含 {len(graph_data.get('nodes', {}))} 个节点")
            return graph_data
            
        except Exception as e:
            self.logger.error(f"加载图拓扑数据失败: {e}")
            raise
    
    def _validate_parking_data(self, df: pd.DataFrame) -> None:
        """验证停车场数据格式
        
        Args:
            df: 停车场数据DataFrame
            
        Raises:
            ValueError: 数据格式错误
        """
        required_columns = [
            'timestamp', 'parking_id', 'occupancy', 'avg_confidence',
            'static_capacity', 'static_price_level', 'is_weekend',
            'weather_condition', 'poi_hotness'
        ]
        
        # 检查数据类型和范围
        if df.empty:
            raise ValueError("数据文件为空")

        # 检查必需列
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"缺少必需的列: {missing_columns}")
        
        # 检查占用率范围
        if not df['occupancy'].between(0, 1).all():
            warnings.warn("发现占用率超出[0,1]范围的数据")
        
        # 检查置信度范围
        if not df['avg_confidence'].between(0, 1).all():
            warnings.warn("发现置信度超出[0,1]范围的数据")
        
        # 检查停车场容量
        if (df['static_capacity'] <= 0).any():
            raise ValueError("停车场容量必须大于0")
        
        self.logger.debug("停车场数据格式验证通过")
    
    def _validate_graph_topology(self, graph_data: Dict[str, Any]) -> None:
        """验证图拓扑数据格式
        
        Args:
            graph_data: 图拓扑数据字典
            
        Raises:
            ValueError: 数据格式错误
        """
        # 检查必需字段
        if 'edges' not in graph_data:
            raise ValueError("图拓扑数据缺少'edges'字段")
        
        edges = graph_data['edges']
        if not isinstance(edges, dict):
            raise ValueError("'edges'字段必须是字典类型")
        
        # 检查节点一致性
        if 'nodes' in graph_data:
            nodes = set(graph_data['nodes'].keys())
            edge_nodes = set(edges.keys())
            
            # 检查边中的节点是否都在节点列表中
            for node, neighbors in edges.items():
                if node not in nodes:
                    warnings.warn(f"边中的节点 {node} 不在节点列表中")
                
                for neighbor in neighbors:
                    if neighbor not in nodes:
                        warnings.warn(f"邻居节点 {neighbor} 不在节点列表中")
        
        self.logger.debug("图拓扑数据格式验证通过")
    
    def _convert_parking_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换停车场数据类型
        
        Args:
            df: 原始DataFrame
            
        Returns:
            类型转换后的DataFrame
        """
        df = df.copy()
        
        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 转换数值类型
        df['occupancy'] = df['occupancy'].astype(np.float32)
        df['avg_confidence'] = df['avg_confidence'].astype(np.float32)
        df['static_capacity'] = df['static_capacity'].astype(np.int32)
        df['static_price_level'] = df['static_price_level'].astype(np.int32)
        df['is_weekend'] = df['is_weekend'].astype(bool)
        df['poi_hotness'] = df['poi_hotness'].astype(np.float32)
        
        # 转换分类变量
        df['parking_id'] = df['parking_id'].astype('category')
        df['weather_condition'] = df['weather_condition'].astype('category')
        
        return df
    
    def get_parking_ids(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """获取停车场ID列表
        
        Args:
            df: 停车场数据DataFrame，如果为None则使用缓存数据
            
        Returns:
            停车场ID列表
        """
        if df is None:
            if self._parking_data_cache is None:
                raise ValueError("没有可用的停车场数据")
            df = self._parking_data_cache
        
        return sorted(df['parking_id'].unique().tolist())
    
    def get_time_range(self, df: Optional[pd.DataFrame] = None) -> Tuple[datetime, datetime]:
        """获取数据时间范围
        
        Args:
            df: 停车场数据DataFrame，如果为None则使用缓存数据
            
        Returns:
            (开始时间, 结束时间)
        """
        if df is None:
            if self._parking_data_cache is None:
                raise ValueError("没有可用的停车场数据")
            df = self._parking_data_cache
        
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()

        # 确保转换为datetime对象
        if isinstance(min_time, str):
            min_time = pd.to_datetime(min_time).to_pydatetime()
        elif hasattr(min_time, 'to_pydatetime'):
            min_time = min_time.to_pydatetime()

        if isinstance(max_time, str):
            max_time = pd.to_datetime(max_time).to_pydatetime()
        elif hasattr(max_time, 'to_pydatetime'):
            max_time = max_time.to_pydatetime()

        return min_time, max_time
    
    def get_data_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """获取数据统计信息
        
        Args:
            df: 停车场数据DataFrame，如果为None则使用缓存数据
            
        Returns:
            数据统计信息字典
        """
        if df is None:
            if self._parking_data_cache is None:
                raise ValueError("没有可用的停车场数据")
            df = self._parking_data_cache
        
        start_time, end_time = self.get_time_range(df)
        parking_ids = self.get_parking_ids(df)
        
        stats = {
            'total_records': len(df),
            'num_parking_lots': len(parking_ids),
            'parking_ids': parking_ids,
            'time_range': {
                'start': start_time,
                'end': end_time,
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            },
            'occupancy_stats': {
                'mean': float(df['occupancy'].mean()),
                'std': float(df['occupancy'].std()),
                'min': float(df['occupancy'].min()),
                'max': float(df['occupancy'].max())
            },
            'confidence_stats': {
                'mean': float(df['avg_confidence'].mean()),
                'std': float(df['avg_confidence'].std()),
                'min': float(df['avg_confidence'].min()),
                'max': float(df['avg_confidence'].max())
            },
            'weather_conditions': df['weather_condition'].value_counts().to_dict(),
            'weekend_ratio': float(df['is_weekend'].mean())
        }
        
        return stats
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self._parking_data_cache = None
        self._graph_topology_cache = None
        self.logger.info("数据缓存已清除")
