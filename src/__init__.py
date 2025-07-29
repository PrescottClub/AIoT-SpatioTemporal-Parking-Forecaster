"""
AIoT 时空预测模型本地复现项目

这是一个用于演示GAT+Transformer时空预测模型的项目，
专门用于停车场占用率预测的本地实现。

Author: AI Assistant
Version: 1.0.0
Date: 2025-07-29
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"
__description__ = "AIoT Spatio-Temporal Parking Forecaster"

# 导入主要模块
from . import data
from . import models
from . import training
from . import inference
from . import utils

__all__ = [
    "data",
    "models", 
    "training",
    "inference",
    "utils"
]
