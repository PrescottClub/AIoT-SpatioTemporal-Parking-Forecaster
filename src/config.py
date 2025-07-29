"""
配置管理模块

提供项目的所有配置参数管理，包括数据、模型、训练和系统配置。

Author: AI Assistant
Date: 2025-07-29
"""

import os
import torch
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class DataConfig:
    """数据配置类"""
    
    # 数据路径
    data_path: str = "sample_data/parking_data.csv"
    graph_path: str = "sample_data/graph.json"
    
    # 时间序列参数
    sequence_length: int = 168  # 7天 * 24小时
    prediction_length: int = 24  # 预测未来24小时
    
    # 数据分割比例
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 数据加载参数
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # 特征配置
    node_features: List[str] = None
    edge_features: List[str] = None
    target_feature: str = "occupancy"
    
    def __post_init__(self):
        """初始化后处理"""
        if self.node_features is None:
            self.node_features = [
                "occupancy", "avg_confidence", "static_capacity",
                "static_price_level", "is_weekend", "poi_hotness"
            ]
        if self.edge_features is None:
            self.edge_features = ["distance", "similarity"]


@dataclass
class ModelConfig:
    """模型配置类"""
    
    # 基础参数
    node_feature_dim: int = 8  # 节点特征维度
    edge_feature_dim: int = 2  # 边特征维度
    hidden_dim: int = 128
    output_dim: int = 1  # 预测目标维度
    
    # GAT参数
    num_gat_layers: int = 2
    gat_heads: int = 8
    gat_dropout: float = 0.1
    gat_alpha: float = 0.2  # LeakyReLU负斜率
    
    # Transformer参数
    num_transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_dim_feedforward: int = 512
    transformer_dropout: float = 0.1
    
    # 其他参数
    activation: str = "relu"
    use_residual: bool = True
    use_layer_norm: bool = True


@dataclass
class TrainingConfig:
    """训练配置类"""
    
    # 基础训练参数
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # 优化器配置
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, step, plateau
    
    # 早停参数
    patience: int = 10
    min_delta: float = 1e-4
    
    # 梯度相关
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    
    # 模型保存
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    mode: str = "min"  # min or max
    
    # 验证频率
    val_check_interval: float = 1.0  # 每个epoch验证一次
    log_every_n_steps: int = 50
    
    # 损失函数
    loss_function: str = "mse"  # mse, mae, huber
    loss_weights: Optional[Dict[str, float]] = None


@dataclass
class SystemConfig:
    """系统配置类"""
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_ids: List[int] = None
    
    # 随机种子
    seed: int = 42
    deterministic: bool = True
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 路径配置
    project_root: str = "."
    checkpoint_dir: str = "models/checkpoints"
    log_dir: str = "logs"
    output_dir: str = "outputs"
    
    # 性能配置
    num_threads: int = 4
    benchmark: bool = True  # 启用cudnn benchmark
    
    def __post_init__(self):
        """初始化后处理"""
        if self.gpu_ids is None:
            self.gpu_ids = [0] if torch.cuda.is_available() else []
        
        # 确保目录存在
        for dir_path in [self.checkpoint_dir, self.log_dir, self.output_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


class Config:
    """主配置类
    
    统一管理所有配置参数，提供配置的加载、保存和验证功能。
    
    Example:
        >>> config = Config()
        >>> config.data.batch_size = 64
        >>> config.model.hidden_dim = 256
        >>> config.save_to_file("config.yaml")
    """
    
    def __init__(self):
        """初始化配置"""
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.system = SystemConfig()
        
        # 设置环境变量
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """设置环境变量"""
        # 设置随机种子
        torch.manual_seed(self.system.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.system.seed)
            torch.cuda.manual_seed_all(self.system.seed)
        
        # 设置线程数
        torch.set_num_threads(self.system.num_threads)
        
        # 设置cudnn
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = self.system.deterministic
            torch.backends.cudnn.benchmark = self.system.benchmark
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            包含所有配置的字典
        """
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "system": asdict(self.system)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            Config实例
        """
        config = cls()
        
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])
        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        if "system" in config_dict:
            config.system = SystemConfig(**config_dict["system"])
        
        # 重新设置环境
        config._setup_environment()
        return config
    
    def save_to_file(self, filepath: str) -> None:
        """保存配置到文件
        
        Args:
            filepath: 文件路径，支持.yaml和.json格式
        """
        import yaml
        import json
        
        config_dict = self.to_dict()
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Config':
        """从文件加载配置
        
        Args:
            filepath: 文件路径
            
        Returns:
            Config实例
        """
        import yaml
        import json
        
        filepath = Path(filepath)
        
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls.from_dict(config_dict)
    
    def validate(self) -> bool:
        """验证配置的有效性
        
        Returns:
            配置是否有效
            
        Raises:
            ValueError: 配置无效时抛出异常
        """
        # 验证数据分割比例
        total_ratio = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"数据分割比例之和必须为1.0，当前为{total_ratio}")
        
        # 验证序列长度
        if self.data.sequence_length <= 0:
            raise ValueError("序列长度必须大于0")
        
        if self.data.prediction_length <= 0:
            raise ValueError("预测长度必须大于0")
        
        # 验证模型参数
        if self.model.hidden_dim <= 0:
            raise ValueError("隐藏层维度必须大于0")
        
        if self.model.num_gat_layers <= 0:
            raise ValueError("GAT层数必须大于0")
        
        if self.model.num_transformer_layers <= 0:
            raise ValueError("Transformer层数必须大于0")
        
        # 验证训练参数
        if self.training.epochs <= 0:
            raise ValueError("训练轮数必须大于0")
        
        if self.training.learning_rate <= 0:
            raise ValueError("学习率必须大于0")
        
        return True
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"Config(data={self.data}, model={self.model}, training={self.training}, system={self.system})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


# 创建默认配置实例
default_config = Config()


def get_config() -> Config:
    """获取默认配置实例
    
    Returns:
        默认配置实例
    """
    return default_config


def create_config_from_args(args: Any) -> Config:
    """从命令行参数创建配置
    
    Args:
        args: 命令行参数对象
        
    Returns:
        Config实例
    """
    config = Config()
    
    # 从args更新配置
    if hasattr(args, 'config_file') and args.config_file:
        config = Config.load_from_file(args.config_file)
    
    # 覆盖特定参数
    if hasattr(args, 'batch_size'):
        config.data.batch_size = args.batch_size
    if hasattr(args, 'learning_rate'):
        config.training.learning_rate = args.learning_rate
    if hasattr(args, 'epochs'):
        config.training.epochs = args.epochs
    
    return config
