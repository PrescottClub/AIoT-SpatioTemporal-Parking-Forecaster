"""
训练模块

包含模型训练器、损失函数等训练相关功能。
"""

from .trainer import Trainer
from .losses import SpatioTemporalLoss

__all__ = [
    "Trainer",
    "SpatioTemporalLoss"
]
