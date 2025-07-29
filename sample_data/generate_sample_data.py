"""
样本数据生成脚本

生成用于演示的停车场数据和图拓扑数据。

Author: AI Assistant
Date: 2025-07-29
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# 设置随机种子
np.random.seed(42)


def generate_parking_data(
    num_parking_lots: int = 5,
    num_days: int = 3,
    interval_minutes: int = 10
) -> pd.DataFrame:
    """生成停车场数据
    
    Args:
        num_parking_lots: 停车场数量
        num_days: 数据天数
        interval_minutes: 数据间隔（分钟）
        
    Returns:
        停车场数据DataFrame
    """
    print(f"生成 {num_parking_lots} 个停车场 {num_days} 天的数据...")
    
    # 停车场基本信息
    parking_lots = []
    for i in range(num_parking_lots):
        parking_lots.append({
            'id': f'P{i+1:03d}',
            'capacity': np.random.choice([200, 300, 500, 800, 1000]),
            'price_level': np.random.choice([1, 2, 3, 4, 5]),
            'type': np.random.choice(['street', 'mall', 'office', 'residential']),
            'base_occupancy': np.random.uniform(0.2, 0.8)
        })
    
    # 生成时间序列
    start_time = datetime(2025, 7, 26, 0, 0, 0)  # 周六开始
    time_points = []
    current_time = start_time
    
    while current_time < start_time + timedelta(days=num_days):
        time_points.append(current_time)
        current_time += timedelta(minutes=interval_minutes)
    
    # 生成数据
    data = []
    
    for time_point in time_points:
        hour = time_point.hour
        day_of_week = time_point.weekday()
        is_weekend = day_of_week >= 5
        
        # 天气条件（简单模拟）
        weather_conditions = ['Clear', 'Cloudy', 'Rainy']
        weather_weights = [0.6, 0.3, 0.1]
        weather = np.random.choice(weather_conditions, p=weather_weights)
        
        for parking_lot in parking_lots:
            # 基础占用率模式
            base_occupancy = parking_lot['base_occupancy']
            
            # 时间模式（日周期）
            time_factor = 0.3 * np.sin(2 * np.pi * hour / 24) + 0.2 * np.sin(4 * np.pi * hour / 24)
            
            # 工作日/周末模式
            if is_weekend:
                if parking_lot['type'] == 'office':
                    weekend_factor = -0.4  # 办公区周末占用率低
                elif parking_lot['type'] == 'mall':
                    weekend_factor = 0.2   # 商场周末占用率高
                else:
                    weekend_factor = 0.0
            else:
                if parking_lot['type'] == 'office':
                    # 办公区工作日高峰
                    if 8 <= hour <= 10 or 17 <= hour <= 19:
                        weekend_factor = 0.3
                    else:
                        weekend_factor = 0.0
                else:
                    weekend_factor = 0.0
            
            # 天气影响
            if weather == 'Rainy':
                weather_factor = 0.15  # 雨天占用率稍高
            elif weather == 'Cloudy':
                weather_factor = 0.05
            else:
                weather_factor = 0.0
            
            # 随机噪声
            noise = np.random.normal(0, 0.1)
            
            # 计算最终占用率
            occupancy = base_occupancy + time_factor + weekend_factor + weather_factor + noise
            occupancy = np.clip(occupancy, 0.0, 1.0)
            
            # 置信度（模拟检测置信度）
            confidence = np.random.uniform(0.85, 0.99)
            
            # POI热度（根据停车场类型和时间）
            if parking_lot['type'] == 'mall':
                poi_base = 0.7
            elif parking_lot['type'] == 'office':
                poi_base = 0.6
            elif parking_lot['type'] == 'street':
                poi_base = 0.5
            else:
                poi_base = 0.4
            
            poi_hotness = poi_base + 0.2 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 0.1)
            poi_hotness = np.clip(poi_hotness, 0.0, 1.0)
            
            # 添加记录
            data.append({
                'timestamp': time_point.strftime('%Y-%m-%d %H:%M:%S'),
                'parking_id': parking_lot['id'],
                'occupancy': round(occupancy, 3),
                'avg_confidence': round(confidence, 3),
                'static_capacity': parking_lot['capacity'],
                'static_price_level': parking_lot['price_level'],
                'is_weekend': 1 if is_weekend else 0,
                'weather_condition': weather,
                'poi_hotness': round(poi_hotness, 3)
            })
    
    df = pd.DataFrame(data)
    print(f"生成了 {len(df)} 条数据记录")
    return df


def generate_graph_topology(parking_ids: List[str]) -> Dict[str, Any]:
    """生成图拓扑数据
    
    Args:
        parking_ids: 停车场ID列表
        
    Returns:
        图拓扑数据字典
    """
    print(f"生成 {len(parking_ids)} 个停车场的图拓扑...")
    
    # 生成节点信息（模拟北京市区坐标）
    base_lat, base_lon = 39.9042, 116.4074  # 天安门坐标
    
    nodes = {}
    for i, parking_id in enumerate(parking_ids):
        # 在基准点周围随机分布
        lat_offset = np.random.uniform(-0.05, 0.05)  # 约5公里范围
        lon_offset = np.random.uniform(-0.05, 0.05)
        
        nodes[parking_id] = {
            'lat': round(base_lat + lat_offset, 6),
            'lon': round(base_lon + lon_offset, 6),
            'district': np.random.choice(['central', 'north', 'south', 'east', 'west']),
            'type': np.random.choice(['street', 'mall', 'office', 'residential'])
        }
    
    # 生成边（基于距离和类型相似性）
    edges = {}
    edge_weights = {}
    
    for i, source in enumerate(parking_ids):
        neighbors = []
        
        for j, target in enumerate(parking_ids):
            if i != j:
                # 计算距离
                lat1, lon1 = nodes[source]['lat'], nodes[source]['lon']
                lat2, lon2 = nodes[target]['lat'], nodes[target]['lon']
                
                # 简化距离计算
                distance = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
                
                # 类型相似性
                type_similarity = 1.0 if nodes[source]['type'] == nodes[target]['type'] else 0.5
                
                # 决定是否连接（距离近或类型相似）
                if distance < 0.03 or (type_similarity > 0.5 and np.random.random() < 0.3):
                    neighbors.append(target)
                    
                    # 计算边权重
                    weight = type_similarity * (1.0 - min(distance / 0.05, 1.0))
                    edge_weights[f"{source}-{target}"] = round(weight, 3)
        
        # 确保每个节点至少有一个邻居
        if not neighbors and len(parking_ids) > 1:
            # 连接到最近的节点
            distances = []
            for target in parking_ids:
                if target != source:
                    lat1, lon1 = nodes[source]['lat'], nodes[source]['lon']
                    lat2, lon2 = nodes[target]['lat'], nodes[target]['lon']
                    distance = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
                    distances.append((distance, target))
            
            distances.sort()
            nearest = distances[0][1]
            neighbors.append(nearest)
            edge_weights[f"{source}-{nearest}"] = 0.5
        
        edges[source] = neighbors
    
    topology = {
        'nodes': nodes,
        'edges': edges,
        'edge_weights': edge_weights,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'num_nodes': len(parking_ids),
            'num_edges': sum(len(neighbors) for neighbors in edges.values()),
            'description': 'Generated sample parking lot topology for demonstration'
        }
    }
    
    print(f"生成了 {len(edges)} 个节点和 {len(edge_weights)} 条边")
    return topology


def main():
    """主函数"""
    print("开始生成样本数据...")
    
    # 创建输出目录
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    # 生成停车场数据
    df = generate_parking_data(
        num_parking_lots=5,
        num_days=3,
        interval_minutes=10
    )
    
    # 保存停车场数据
    parking_data_file = output_dir / 'parking_data.csv'
    df.to_csv(parking_data_file, index=False)
    print(f"停车场数据已保存到: {parking_data_file}")
    
    # 生成图拓扑数据
    parking_ids = sorted(df['parking_id'].unique())
    topology = generate_graph_topology(parking_ids)
    
    # 保存图拓扑数据
    graph_file = output_dir / 'graph.json'
    with open(graph_file, 'w', encoding='utf-8') as f:
        json.dump(topology, f, indent=2, ensure_ascii=False)
    print(f"图拓扑数据已保存到: {graph_file}")
    
    # 打印数据统计
    print("\n数据统计:")
    print(f"- 停车场数量: {len(parking_ids)}")
    print(f"- 时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"- 总记录数: {len(df)}")
    print(f"- 平均占用率: {df['occupancy'].mean():.3f}")
    print(f"- 占用率范围: {df['occupancy'].min():.3f} - {df['occupancy'].max():.3f}")
    
    print("\n停车场信息:")
    for parking_id in parking_ids:
        parking_data = df[df['parking_id'] == parking_id]
        capacity = parking_data['static_capacity'].iloc[0]
        price_level = parking_data['static_price_level'].iloc[0]
        avg_occupancy = parking_data['occupancy'].mean()
        print(f"- {parking_id}: 容量={capacity}, 价格等级={price_level}, 平均占用率={avg_occupancy:.3f}")
    
    print("\n样本数据生成完成！")


if __name__ == '__main__':
    main()
