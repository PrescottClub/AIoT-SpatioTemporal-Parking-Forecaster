"""
数据预处理器模块

负责数据的预处理、特征工程和时间序列构建。

Author: AI Assistant
Date: 2025-07-29
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from datetime import datetime, timedelta
import warnings

from .data_loader import TimeSeriesData
from ..utils.logger import get_logger
from ..config import Config

logger = get_logger(__name__)


class DataPreprocessor:
    """数据预处理器
    
    负责数据清洗、特征工程、标准化和时间序列构建。
    
    Example:
        >>> preprocessor = DataPreprocessor(config)
        >>> processed_data = preprocessor.preprocess(df)
        >>> time_series = preprocessor.create_time_series(processed_data)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """初始化数据预处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
        # 缩放器
        self.scalers: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        # 特征配置
        self.numerical_features = [
            'occupancy', 'avg_confidence', 'static_capacity',
            'static_price_level', 'poi_hotness'
        ]
        self.categorical_features = ['weather_condition']
        self.boolean_features = ['is_weekend']
        
        # 是否已拟合
        self.is_fitted = False
    
    def preprocess(
        self, 
        df: pd.DataFrame,
        fit: bool = True,
        handle_missing: str = 'interpolate'
    ) -> pd.DataFrame:
        """预处理数据
        
        Args:
            df: 原始数据DataFrame
            fit: 是否拟合预处理器
            handle_missing: 缺失值处理方式 ('drop', 'interpolate', 'fill')
            
        Returns:
            预处理后的DataFrame
        """
        self.logger.info("开始数据预处理")
        
        df_processed = df.copy()
        
        # 1. 处理缺失值
        df_processed = self._handle_missing_values(df_processed, method=handle_missing)
        
        # 2. 特征工程
        df_processed = self._feature_engineering(df_processed)
        
        # 3. 异常值处理
        df_processed = self._handle_outliers(df_processed)
        
        # 4. 数据标准化
        if fit:
            df_processed = self._fit_transform_features(df_processed)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("预处理器尚未拟合，请先调用fit=True")
            df_processed = self._transform_features(df_processed)
        
        self.logger.info(f"数据预处理完成，处理了 {len(df_processed)} 条记录")
        return df_processed
    
    def _handle_missing_values(
        self, 
        df: pd.DataFrame, 
        method: str = 'interpolate'
    ) -> pd.DataFrame:
        """处理缺失值
        
        Args:
            df: 数据DataFrame
            method: 处理方法
            
        Returns:
            处理后的DataFrame
        """
        missing_counts = df.isnull().sum()
        if missing_counts.sum() == 0:
            return df
        
        self.logger.info(f"发现缺失值: {missing_counts[missing_counts > 0].to_dict()}")
        
        df_clean = df.copy()
        
        if method == 'drop':
            df_clean = df_clean.dropna()
        elif method == 'interpolate':
            # 对数值列进行插值
            for col in self.numerical_features:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    df_clean[col] = df_clean.groupby('parking_id')[col].transform(
                        lambda x: x.interpolate(method='linear')
                    )
            
            # 对分类列进行前向填充
            for col in self.categorical_features:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    df_clean[col] = df_clean.groupby('parking_id')[col].transform(
                        lambda x: x.fillna(method='ffill').fillna(method='bfill')
                    )
        elif method == 'fill':
            # 使用默认值填充
            fill_values = {
                'occupancy': 0.5,
                'avg_confidence': 0.9,
                'weather_condition': 'Clear',
                'poi_hotness': 0.5
            }
            df_clean = df_clean.fillna(fill_values)
        
        return df_clean
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程
        
        Args:
            df: 数据DataFrame
            
        Returns:
            特征工程后的DataFrame
        """
        df_features = df.copy()
        
        # 时间特征
        df_features['hour'] = df_features['timestamp'].dt.hour
        df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
        df_features['month'] = df_features['timestamp'].dt.month
        df_features['is_rush_hour'] = df_features['hour'].isin([7, 8, 9, 17, 18, 19])
        
        # 周期性特征（正弦余弦编码）
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        # 容量利用率
        df_features['capacity_utilization'] = df_features['occupancy']
        
        # 价格等级归一化
        df_features['price_level_norm'] = df_features['static_price_level'] / 5.0
        
        # 滞后特征（如果数据足够）
        df_features = self._create_lag_features(df_features)
        
        # 滑动窗口统计特征
        df_features = self._create_rolling_features(df_features)
        
        return df_features
    
    def _create_lag_features(
        self, 
        df: pd.DataFrame, 
        lag_periods: List[int] = [1, 2, 6, 12, 24]
    ) -> pd.DataFrame:
        """创建滞后特征
        
        Args:
            df: 数据DataFrame
            lag_periods: 滞后周期列表
            
        Returns:
            包含滞后特征的DataFrame
        """
        df_lag = df.copy()
        
        for parking_id in df['parking_id'].unique():
            mask = df_lag['parking_id'] == parking_id
            parking_data = df_lag[mask].sort_values('timestamp')
            
            for lag in lag_periods:
                col_name = f'occupancy_lag_{lag}'
                df_lag.loc[mask, col_name] = parking_data['occupancy'].shift(lag)
        
        return df_lag
    
    def _create_rolling_features(
        self, 
        df: pd.DataFrame,
        windows: List[int] = [6, 12, 24]
    ) -> pd.DataFrame:
        """创建滑动窗口统计特征
        
        Args:
            df: 数据DataFrame
            windows: 窗口大小列表
            
        Returns:
            包含滑动窗口特征的DataFrame
        """
        df_rolling = df.copy()
        
        for parking_id in df['parking_id'].unique():
            mask = df_rolling['parking_id'] == parking_id
            parking_data = df_rolling[mask].sort_values('timestamp')
            
            for window in windows:
                # 滑动平均
                col_mean = f'occupancy_rolling_mean_{window}'
                df_rolling.loc[mask, col_mean] = parking_data['occupancy'].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # 滑动标准差
                col_std = f'occupancy_rolling_std_{window}'
                df_rolling.loc[mask, col_std] = parking_data['occupancy'].rolling(
                    window=window, min_periods=1
                ).std().fillna(0)
        
        return df_rolling
    
    def _handle_outliers(
        self, 
        df: pd.DataFrame, 
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """处理异常值
        
        Args:
            df: 数据DataFrame
            method: 异常值检测方法 ('iqr', 'zscore')
            threshold: 阈值
            
        Returns:
            处理后的DataFrame
        """
        df_clean = df.copy()
        
        for col in ['occupancy', 'avg_confidence']:
            if col not in df_clean.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outliers = z_scores > threshold
            
            if outliers.sum() > 0:
                self.logger.warning(f"发现 {outliers.sum()} 个 {col} 异常值")
                # 用中位数替换异常值
                df_clean.loc[outliers, col] = df_clean[col].median()
        
        return df_clean
    
    def _fit_transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换特征
        
        Args:
            df: 数据DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        df_scaled = df.copy()
        
        # 数值特征标准化
        numerical_cols = [col for col in df_scaled.columns 
                         if df_scaled[col].dtype in ['float32', 'float64', 'int32', 'int64']
                         and col not in ['timestamp', 'parking_id']]
        
        if numerical_cols:
            self.scalers['numerical'] = StandardScaler()
            df_scaled[numerical_cols] = self.scalers['numerical'].fit_transform(
                df_scaled[numerical_cols]
            )
        
        # 分类特征编码
        for col in self.categorical_features:
            if col in df_scaled.columns:
                self.label_encoders[col] = LabelEncoder()
                df_scaled[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                    df_scaled[col].astype(str)
                )
        
        return df_scaled
    
    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换特征（使用已拟合的转换器）
        
        Args:
            df: 数据DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        df_scaled = df.copy()
        
        # 数值特征标准化
        numerical_cols = [col for col in df_scaled.columns 
                         if df_scaled[col].dtype in ['float32', 'float64', 'int32', 'int64']
                         and col not in ['timestamp', 'parking_id']]
        
        if numerical_cols and 'numerical' in self.scalers:
            df_scaled[numerical_cols] = self.scalers['numerical'].transform(
                df_scaled[numerical_cols]
            )
        
        # 分类特征编码
        for col in self.categorical_features:
            if col in df_scaled.columns and col in self.label_encoders:
                # 处理未见过的类别
                unique_values = set(df_scaled[col].astype(str))
                known_values = set(self.label_encoders[col].classes_)
                unknown_values = unique_values - known_values
                
                if unknown_values:
                    self.logger.warning(f"发现未知类别 {unknown_values}，将使用默认值")
                    # 用最常见的类别替换未知值
                    most_common = self.label_encoders[col].classes_[0]
                    df_scaled[col] = df_scaled[col].astype(str).replace(
                        list(unknown_values), most_common
                    )
                
                df_scaled[f'{col}_encoded'] = self.label_encoders[col].transform(
                    df_scaled[col].astype(str)
                )
        
        return df_scaled
    
    def create_time_series(
        self,
        df: pd.DataFrame,
        seq_len: Optional[int] = None,
        pred_len: Optional[int] = None,
        stride: int = 1,
        target_column: str = 'occupancy'
    ) -> TimeSeriesData:
        """创建时间序列数据
        
        Args:
            df: 预处理后的数据DataFrame
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            stride: 滑动步长
            target_column: 目标列名
            
        Returns:
            时间序列数据对象
        """
        if seq_len is None:
            seq_len = self.config.data.sequence_length if self.config else 168
        if pred_len is None:
            pred_len = self.config.data.prediction_length if self.config else 24
        
        self.logger.info(f"创建时间序列数据: seq_len={seq_len}, pred_len={pred_len}")
        
        # 获取停车场列表和数值特征列
        parking_ids = sorted(df['parking_id'].unique())

        # 只选择数值特征
        feature_columns = []
        for col in df.columns:
            if col not in ['timestamp', 'parking_id', 'weather_condition']:
                if df[col].dtype in ['float32', 'float64', 'int32', 'int64', 'bool']:
                    feature_columns.append(col)
                elif col.endswith('_encoded'):  # 编码后的分类特征
                    feature_columns.append(col)
        
        sequences = []
        targets = []
        timestamps = []
        
        for parking_id in parking_ids:
            parking_data = df[df['parking_id'] == parking_id].sort_values('timestamp')
            
            if len(parking_data) < seq_len + pred_len:
                self.logger.warning(f"停车场 {parking_id} 数据不足，跳过")
                continue
            
            # 创建滑动窗口
            for i in range(0, len(parking_data) - seq_len - pred_len + 1, stride):
                # 输入序列
                seq_data = parking_data.iloc[i:i+seq_len][feature_columns].values
                sequences.append(seq_data)
                
                # 目标序列
                target_data = parking_data.iloc[i+seq_len:i+seq_len+pred_len][target_column].values
                targets.append(target_data)
                
                # 时间戳
                timestamps.append(parking_data.iloc[i+seq_len]['timestamp'])
        
        if not sequences:
            raise ValueError("无法创建时间序列数据，请检查数据长度和参数")
        
        # 确保数据类型正确
        sequences_array = np.array(sequences, dtype=np.float32)
        targets_array = np.array(targets, dtype=np.float32)

        # 转换为张量
        sequences_tensor = torch.FloatTensor(sequences_array)
        targets_tensor = torch.FloatTensor(targets_array)

        # 重新整形为 [batch_size, seq_len, num_nodes, num_features]
        batch_size = len(sequences)
        num_nodes = len(parking_ids)
        num_features = len(feature_columns)

        # 这里需要重新组织数据结构以支持多节点
        # 暂时使用简化版本
        sequences_tensor = sequences_tensor.unsqueeze(2)  # 添加节点维度
        targets_tensor = targets_tensor.unsqueeze(-1)     # 添加特征维度
        
        # 创建节点映射
        node_mapping = {parking_id: i for i, parking_id in enumerate(parking_ids)}
        
        self.logger.info(f"成功创建 {batch_size} 个时间序列样本")
        
        return TimeSeriesData(
            sequences=sequences_tensor,
            targets=targets_tensor,
            timestamps=timestamps,
            node_mapping=node_mapping
        )
    
    def inverse_transform(
        self, 
        data: torch.Tensor, 
        feature_name: str = 'occupancy'
    ) -> torch.Tensor:
        """反向转换数据
        
        Args:
            data: 标准化后的数据
            feature_name: 特征名称
            
        Returns:
            原始尺度的数据
        """
        if not self.is_fitted:
            raise ValueError("预处理器尚未拟合")
        
        if 'numerical' not in self.scalers:
            return data
        
        # 这里需要根据具体的标准化方式进行反向转换
        # 简化实现
        return data
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表
        
        Returns:
            特征名称列表
        """
        features = []
        features.extend(self.numerical_features)
        features.extend([f'{col}_encoded' for col in self.categorical_features])
        features.extend(self.boolean_features)
        
        # 添加工程特征
        features.extend(['hour', 'day_of_week', 'month', 'is_rush_hour'])
        features.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos'])
        features.extend(['capacity_utilization', 'price_level_norm'])
        
        return features
