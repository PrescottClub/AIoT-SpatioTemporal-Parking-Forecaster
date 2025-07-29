"""
图构建器模块

负责构建停车场之间的图结构，包括邻接矩阵、边特征等。

Author: AI Assistant
Date: 2025-07-29
"""

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from .data_loader import GraphData
from ..utils.logger import get_logger
from ..config import Config

logger = get_logger(__name__)


class GraphBuilder:
    """图构建器
    
    负责根据停车场数据和拓扑信息构建图结构。
    
    Example:
        >>> builder = GraphBuilder(config)
        >>> graph_data = builder.build_graph(df, topology)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """初始化图构建器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
        # 图构建参数
        self.distance_threshold = 5.0  # 公里
        self.similarity_threshold = 0.5
        
    def build_graph(
        self,
        df: pd.DataFrame,
        topology: Dict[str, Any],
        method: str = 'hybrid'
    ) -> GraphData:
        """构建图数据结构
        
        Args:
            df: 停车场数据DataFrame
            topology: 图拓扑信息
            method: 图构建方法 ('topology', 'distance', 'similarity', 'hybrid')
            
        Returns:
            图数据对象
        """
        self.logger.info(f"使用 {method} 方法构建图结构")
        
        # 获取节点列表
        node_ids = sorted(df['parking_id'].unique())
        num_nodes = len(node_ids)
        
        if num_nodes == 0:
            raise ValueError("没有找到停车场节点")
        
        # 构建节点特征
        node_features = self._build_node_features(df, node_ids)
        
        # 构建边
        if method == 'topology':
            edge_index, edge_attr = self._build_edges_from_topology(topology, node_ids)
        elif method == 'distance':
            edge_index, edge_attr = self._build_edges_from_distance(topology, node_ids)
        elif method == 'similarity':
            edge_index, edge_attr = self._build_edges_from_similarity(df, node_ids)
        elif method == 'hybrid':
            edge_index, edge_attr = self._build_edges_hybrid(df, topology, node_ids)
        else:
            raise ValueError(f"未知的图构建方法: {method}")
        
        num_edges = edge_index.shape[1]
        
        self.logger.info(f"构建完成: {num_nodes} 个节点, {num_edges} 条边")
        
        return GraphData(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_features=node_features,
            node_ids=node_ids,
            num_nodes=num_nodes,
            num_edges=num_edges
        )
    
    def _build_node_features(
        self, 
        df: pd.DataFrame, 
        node_ids: List[str]
    ) -> torch.Tensor:
        """构建节点特征
        
        Args:
            df: 停车场数据DataFrame
            node_ids: 节点ID列表
            
        Returns:
            节点特征张量 [num_nodes, num_features]
        """
        node_features = []
        
        for node_id in node_ids:
            node_data = df[df['parking_id'] == node_id]
            
            if len(node_data) == 0:
                # 如果没有数据，使用默认特征
                features = [0.5, 0.9, 500, 3, 0.5, 0.5, 0.0, 0.5]
            else:
                # 计算统计特征
                features = [
                    float(node_data['occupancy'].mean()),           # 平均占用率
                    float(node_data['avg_confidence'].mean()),      # 平均置信度
                    float(node_data['static_capacity'].iloc[0]),    # 静态容量
                    float(node_data['static_price_level'].iloc[0]), # 价格等级
                    float(node_data['poi_hotness'].mean()),         # POI热度
                    float(node_data['is_weekend'].mean()),          # 周末比例
                    float(node_data['occupancy'].std()),            # 占用率标准差
                    float(node_data['occupancy'].max())             # 最大占用率
                ]
            
            node_features.append(features)
        
        return torch.FloatTensor(node_features)
    
    def _build_edges_from_topology(
        self, 
        topology: Dict[str, Any], 
        node_ids: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据拓扑信息构建边
        
        Args:
            topology: 拓扑信息
            node_ids: 节点ID列表
            
        Returns:
            (边索引, 边特征)
        """
        if 'edges' not in topology:
            raise ValueError("拓扑信息中缺少边信息")
        
        edges = topology['edges']
        edge_weights = topology.get('edge_weights', {})
        
        # 创建节点ID到索引的映射
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        
        edge_list = []
        edge_features = []
        
        for source, targets in edges.items():
            if source not in node_to_idx:
                continue
                
            source_idx = node_to_idx[source]
            
            for target in targets:
                if target not in node_to_idx:
                    continue
                    
                target_idx = node_to_idx[target]
                
                # 添加边
                edge_list.append([source_idx, target_idx])
                
                # 边特征：权重和类型
                weight_key = f"{source}-{target}"
                weight = edge_weights.get(weight_key, 1.0)
                edge_features.append([weight, 1.0])  # [权重, 拓扑边标识]
        
        if not edge_list:
            # 如果没有边，创建自环
            edge_list = [[i, i] for i in range(len(node_ids))]
            edge_features = [[1.0, 0.0] for _ in range(len(node_ids))]
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_attr = torch.FloatTensor(edge_features)
        
        return edge_index, edge_attr
    
    def _build_edges_from_distance(
        self, 
        topology: Dict[str, Any], 
        node_ids: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据地理距离构建边
        
        Args:
            topology: 拓扑信息（包含坐标）
            node_ids: 节点ID列表
            
        Returns:
            (边索引, 边特征)
        """
        if 'nodes' not in topology:
            raise ValueError("拓扑信息中缺少节点坐标信息")
        
        nodes_info = topology['nodes']
        
        # 提取坐标
        coordinates = []
        valid_nodes = []
        
        for node_id in node_ids:
            if node_id in nodes_info and 'lat' in nodes_info[node_id] and 'lon' in nodes_info[node_id]:
                lat = nodes_info[node_id]['lat']
                lon = nodes_info[node_id]['lon']
                coordinates.append([lat, lon])
                valid_nodes.append(node_id)
        
        if len(coordinates) < 2:
            self.logger.warning("坐标信息不足，使用默认连接")
            return self._build_default_edges(node_ids)
        
        # 计算距离矩阵（使用哈弗辛公式）
        distances = self._calculate_haversine_distances(np.array(coordinates))
        
        # 构建边
        edge_list = []
        edge_features = []
        
        for i in range(len(valid_nodes)):
            for j in range(len(valid_nodes)):
                if i != j and distances[i, j] <= self.distance_threshold:
                    edge_list.append([i, j])
                    # 边特征：距离和距离边标识
                    edge_features.append([distances[i, j], 0.0])
        
        # 添加自环
        for i in range(len(valid_nodes)):
            edge_list.append([i, i])
            edge_features.append([0.0, 0.0])
        
        if not edge_list:
            return self._build_default_edges(node_ids)
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_attr = torch.FloatTensor(edge_features)
        
        return edge_index, edge_attr
    
    def _build_edges_from_similarity(
        self, 
        df: pd.DataFrame, 
        node_ids: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据特征相似性构建边
        
        Args:
            df: 停车场数据DataFrame
            node_ids: 节点ID列表
            
        Returns:
            (边索引, 边特征)
        """
        # 计算每个停车场的特征向量
        feature_vectors = []
        
        for node_id in node_ids:
            node_data = df[df['parking_id'] == node_id]
            
            if len(node_data) == 0:
                # 默认特征向量
                features = [0.5, 0.9, 500, 3, 0.5]
            else:
                features = [
                    float(node_data['occupancy'].mean()),
                    float(node_data['avg_confidence'].mean()),
                    float(node_data['static_capacity'].iloc[0]) / 1000,  # 归一化
                    float(node_data['static_price_level'].iloc[0]) / 5,   # 归一化
                    float(node_data['poi_hotness'].mean())
                ]
            
            feature_vectors.append(features)
        
        # 计算余弦相似性
        similarity_matrix = cosine_similarity(feature_vectors)
        
        # 构建边
        edge_list = []
        edge_features = []
        
        for i in range(len(node_ids)):
            for j in range(len(node_ids)):
                if i != j and similarity_matrix[i, j] >= self.similarity_threshold:
                    edge_list.append([i, j])
                    # 边特征：相似性和相似性边标识
                    edge_features.append([similarity_matrix[i, j], 0.0])
        
        # 添加自环
        for i in range(len(node_ids)):
            edge_list.append([i, i])
            edge_features.append([1.0, 0.0])
        
        if not edge_list:
            return self._build_default_edges(node_ids)
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_attr = torch.FloatTensor(edge_features)
        
        return edge_index, edge_attr
    
    def _build_edges_hybrid(
        self, 
        df: pd.DataFrame, 
        topology: Dict[str, Any], 
        node_ids: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """混合方法构建边
        
        Args:
            df: 停车场数据DataFrame
            topology: 拓扑信息
            node_ids: 节点ID列表
            
        Returns:
            (边索引, 边特征)
        """
        all_edges = set()
        all_edge_features = {}
        
        # 1. 拓扑边
        try:
            topo_edge_index, topo_edge_attr = self._build_edges_from_topology(topology, node_ids)
            for i in range(topo_edge_index.shape[1]):
                src, dst = topo_edge_index[0, i].item(), topo_edge_index[1, i].item()
                all_edges.add((src, dst))
                all_edge_features[(src, dst)] = [topo_edge_attr[i, 0].item(), 1.0, 0.0]
        except Exception as e:
            self.logger.warning(f"构建拓扑边失败: {e}")
        
        # 2. 距离边
        try:
            dist_edge_index, dist_edge_attr = self._build_edges_from_distance(topology, node_ids)
            for i in range(dist_edge_index.shape[1]):
                src, dst = dist_edge_index[0, i].item(), dist_edge_index[1, i].item()
                if (src, dst) not in all_edges:
                    all_edges.add((src, dst))
                    all_edge_features[(src, dst)] = [dist_edge_attr[i, 0].item(), 0.0, 1.0]
        except Exception as e:
            self.logger.warning(f"构建距离边失败: {e}")
        
        # 3. 相似性边
        try:
            sim_edge_index, sim_edge_attr = self._build_edges_from_similarity(df, node_ids)
            for i in range(sim_edge_index.shape[1]):
                src, dst = sim_edge_index[0, i].item(), sim_edge_index[1, i].item()
                if (src, dst) not in all_edges:
                    all_edges.add((src, dst))
                    all_edge_features[(src, dst)] = [sim_edge_attr[i, 0].item(), 0.0, 0.0]
        except Exception as e:
            self.logger.warning(f"构建相似性边失败: {e}")
        
        # 确保有自环
        for i in range(len(node_ids)):
            if (i, i) not in all_edges:
                all_edges.add((i, i))
                all_edge_features[(i, i)] = [1.0, 0.0, 0.0]
        
        if not all_edges:
            return self._build_default_edges(node_ids)
        
        # 转换为张量
        edge_list = list(all_edges)
        edge_features = [all_edge_features[edge] for edge in edge_list]
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_attr = torch.FloatTensor(edge_features)
        
        return edge_index, edge_attr
    
    def _build_default_edges(self, node_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建默认边（全连接图）
        
        Args:
            node_ids: 节点ID列表
            
        Returns:
            (边索引, 边特征)
        """
        num_nodes = len(node_ids)
        edge_list = []
        edge_features = []
        
        # 全连接图
        for i in range(num_nodes):
            for j in range(num_nodes):
                edge_list.append([i, j])
                weight = 1.0 if i == j else 0.5  # 自环权重为1，其他为0.5
                edge_features.append([weight, 0.0])
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_attr = torch.FloatTensor(edge_features)
        
        return edge_index, edge_attr
    
    def _calculate_haversine_distances(self, coordinates: np.ndarray) -> np.ndarray:
        """计算哈弗辛距离
        
        Args:
            coordinates: 坐标数组 [num_points, 2] (lat, lon)
            
        Returns:
            距离矩阵 [num_points, num_points]
        """
        def haversine(lat1, lon1, lat2, lon2):
            """计算两点间的哈弗辛距离（公里）"""
            R = 6371  # 地球半径（公里）
            
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        
        num_points = len(coordinates)
        distances = np.zeros((num_points, num_points))
        
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    distances[i, j] = haversine(
                        coordinates[i, 0], coordinates[i, 1],
                        coordinates[j, 0], coordinates[j, 1]
                    )
        
        return distances
    
    def visualize_graph(
        self, 
        graph_data: GraphData, 
        save_path: Optional[str] = None
    ) -> None:
        """可视化图结构
        
        Args:
            graph_data: 图数据对象
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            self.logger.warning("缺少可视化依赖，跳过图可视化")
            return
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加节点
        for i, node_id in enumerate(graph_data.node_ids):
            G.add_node(i, label=node_id)
        
        # 添加边
        edge_index = graph_data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src != dst:  # 跳过自环
                weight = graph_data.edge_attr[i, 0].item()
                G.add_edge(src, dst, weight=weight)
        
        # 绘制图
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 绘制节点
        nx.draw_nodes(G, pos, node_color='lightblue', 
                     node_size=1000, alpha=0.8)
        
        # 绘制边
        nx.draw_edges(G, pos, alpha=0.5, width=1)
        
        # 添加标签
        labels = {i: node_id for i, node_id in enumerate(graph_data.node_ids)}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title(f"停车场图结构 ({graph_data.num_nodes} 节点, {graph_data.num_edges} 边)")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"图可视化已保存到: {save_path}")
        
        plt.show()
    
    def get_graph_statistics(self, graph_data: GraphData) -> Dict[str, Any]:
        """获取图统计信息
        
        Args:
            graph_data: 图数据对象
            
        Returns:
            图统计信息字典
        """
        edge_index = graph_data.edge_index.numpy()
        
        # 计算度
        degrees = np.bincount(edge_index[0], minlength=graph_data.num_nodes)
        
        # 计算连通性
        adj_matrix = np.zeros((graph_data.num_nodes, graph_data.num_nodes))
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            adj_matrix[src, dst] = 1
        
        stats = {
            'num_nodes': graph_data.num_nodes,
            'num_edges': graph_data.num_edges,
            'avg_degree': float(np.mean(degrees)),
            'max_degree': int(np.max(degrees)),
            'min_degree': int(np.min(degrees)),
            'density': graph_data.num_edges / (graph_data.num_nodes * (graph_data.num_nodes - 1)),
            'self_loops': int(np.sum(edge_index[0] == edge_index[1]))
        }
        
        return stats
