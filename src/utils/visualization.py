"""
可视化工具模块

提供训练过程和预测结果的可视化功能。

Author: AI Assistant
Date: 2025-07-29
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import warnings

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    title: str = "训练过程指标变化"
) -> None:
    """绘制训练过程中的指标变化曲线
    
    Args:
        metrics_history: 指标历史字典
        save_path: 保存路径
        figsize: 图像大小
        title: 图像标题
    """
    if not metrics_history:
        warnings.warn("指标历史为空，无法绘制")
        return
    
    # 计算子图数量
    n_metrics = len(metrics_history)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_metrics > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # 绘制每个指标
    for i, (metric_name, values) in enumerate(metrics_history.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        epochs = range(1, len(values) + 1)
        
        ax.plot(epochs, values, 'b-', linewidth=2, label=metric_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} 变化曲线')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 标注最佳值
        if 'loss' in metric_name.lower() or 'error' in metric_name.lower():
            best_idx = np.argmin(values)
            best_value = values[best_idx]
        else:
            best_idx = np.argmax(values)
            best_value = values[best_idx]
        
        ax.annotate(
            f'Best: {best_value:.4f}',
            xy=(best_idx + 1, best_value),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
    
    # 隐藏多余的子图
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    
    plt.show()


def plot_predictions(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    timestamps: Optional[List[str]] = None,
    parking_ids: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 8),
    title: str = "预测结果对比"
) -> None:
    """绘制预测结果对比图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        timestamps: 时间戳列表
        parking_ids: 停车场ID列表
        save_path: 保存路径
        figsize: 图像大小
        title: 图像标题
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("真实值和预测值长度不匹配")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. 时间序列对比图
    ax1 = axes[0, 0]
    x = range(len(y_true))
    if timestamps:
        x = pd.to_datetime(timestamps)
    
    ax1.plot(x, y_true, 'b-', label='真实值', linewidth=2, alpha=0.8)
    ax1.plot(x, y_pred, 'r--', label='预测值', linewidth=2, alpha=0.8)
    ax1.set_xlabel('时间')
    ax1.set_ylabel('占用率')
    ax1.set_title('时间序列对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 散点图
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # 添加对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('真实值')
    ax2.set_ylabel('预测值')
    ax2.set_title('预测值 vs 真实值')
    ax2.grid(True, alpha=0.3)
    
    # 添加R²值
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. 误差分布直方图
    ax3 = axes[1, 0]
    errors = y_pred - y_true
    ax3.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('预测误差')
    ax3.set_ylabel('频次')
    ax3.set_title('误差分布')
    ax3.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    ax3.text(0.05, 0.95, f'均值: {mean_error:.4f}\n标准差: {std_error:.4f}',
             transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. 误差随时间变化
    ax4 = axes[1, 1]
    ax4.plot(x, errors, 'g-', linewidth=1, alpha=0.7)
    ax4.axhline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('时间')
    ax4.set_ylabel('预测误差')
    ax4.set_title('误差随时间变化')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图已保存到: {save_path}")
    
    plt.show()


def plot_attention_weights(
    attention_weights: np.ndarray,
    node_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "注意力权重热力图"
) -> None:
    """绘制注意力权重热力图
    
    Args:
        attention_weights: 注意力权重矩阵 [num_nodes, num_nodes]
        node_names: 节点名称列表
        save_path: 保存路径
        figsize: 图像大小
        title: 图像标题
    """
    if attention_weights.ndim != 2:
        raise ValueError("注意力权重必须是2D矩阵")
    
    plt.figure(figsize=figsize)
    
    # 创建热力图
    sns.heatmap(
        attention_weights,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        square=True,
        linewidths=0.5,
        cbar_kws={'label': '注意力权重'},
        xticklabels=node_names if node_names else False,
        yticklabels=node_names if node_names else False
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('目标节点')
    plt.ylabel('源节点')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力权重图已保存到: {save_path}")
    
    plt.show()


def plot_loss_landscape(
    loss_values: np.ndarray,
    param1_range: np.ndarray,
    param2_range: np.ndarray,
    param1_name: str = "参数1",
    param2_name: str = "参数2",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """绘制损失函数地形图
    
    Args:
        loss_values: 损失值矩阵
        param1_range: 参数1的取值范围
        param2_range: 参数2的取值范围
        param1_name: 参数1名称
        param2_name: 参数2名称
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 2D热力图
    im1 = ax1.contourf(param1_range, param2_range, loss_values, levels=20, cmap='viridis')
    ax1.set_xlabel(param1_name)
    ax1.set_ylabel(param2_name)
    ax1.set_title('损失函数等高线图')
    plt.colorbar(im1, ax=ax1, label='损失值')
    
    # 3D表面图
    from mpl_toolkits.mplot3d import Axes3D
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(param1_range, param2_range)
    surf = ax2.plot_surface(X, Y, loss_values, cmap='viridis', alpha=0.8)
    ax2.set_xlabel(param1_name)
    ax2.set_ylabel(param2_name)
    ax2.set_zlabel('损失值')
    ax2.set_title('损失函数3D图')
    plt.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"损失地形图已保存到: {save_path}")
    
    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "特征重要性"
) -> None:
    """绘制特征重要性图
    
    Args:
        feature_names: 特征名称列表
        importance_scores: 重要性分数列表
        save_path: 保存路径
        figsize: 图像大小
        title: 图像标题
    """
    if len(feature_names) != len(importance_scores):
        raise ValueError("特征名称和重要性分数长度不匹配")
    
    # 按重要性排序
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_scores = [importance_scores[i] for i in sorted_indices]
    
    plt.figure(figsize=figsize)
    
    # 创建条形图
    bars = plt.bar(range(len(sorted_names)), sorted_scores, 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.xlabel('特征')
    plt.ylabel('重要性分数')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {save_path}")
    
    plt.show()


def create_dashboard(
    metrics_history: Dict[str, List[float]],
    predictions_data: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 15)
) -> None:
    """创建综合仪表板
    
    Args:
        metrics_history: 指标历史
        predictions_data: 预测数据字典
        save_path: 保存路径
        figsize: 图像大小
    """
    fig = plt.figure(figsize=figsize)
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. 训练损失曲线
    ax1 = fig.add_subplot(gs[0, :2])
    if 'train_loss' in metrics_history:
        ax1.plot(metrics_history['train_loss'], label='训练损失', linewidth=2)
    if 'val_loss' in metrics_history:
        ax1.plot(metrics_history['val_loss'], label='验证损失', linewidth=2)
    ax1.set_title('损失函数变化')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 评估指标
    ax2 = fig.add_subplot(gs[0, 2:])
    metric_names = [k for k in metrics_history.keys() if 'loss' not in k.lower()]
    if metric_names:
        for metric in metric_names[:3]:  # 最多显示3个指标
            ax2.plot(metrics_history[metric], label=metric, linewidth=2)
        ax2.set_title('评估指标变化')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 预测对比
    if 'y_true' in predictions_data and 'y_pred' in predictions_data:
        ax3 = fig.add_subplot(gs[1, :2])
        y_true = predictions_data['y_true']
        y_pred = predictions_data['y_pred']
        
        x = range(len(y_true))
        ax3.plot(x, y_true, 'b-', label='真实值', linewidth=2, alpha=0.8)
        ax3.plot(x, y_pred, 'r--', label='预测值', linewidth=2, alpha=0.8)
        ax3.set_title('预测结果对比')
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('占用率')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 散点图
        ax4 = fig.add_subplot(gs[1, 2:])
        ax4.scatter(y_true, y_pred, alpha=0.6, s=20)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax4.set_xlabel('真实值')
        ax4.set_ylabel('预测值')
        ax4.set_title('预测值 vs 真实值')
        ax4.grid(True, alpha=0.3)
    
    # 5. 误差分析
    if 'y_true' in predictions_data and 'y_pred' in predictions_data:
        ax5 = fig.add_subplot(gs[2, :2])
        errors = np.array(predictions_data['y_pred']) - np.array(predictions_data['y_true'])
        ax5.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.axvline(0, color='red', linestyle='--', linewidth=2)
        ax5.set_title('误差分布')
        ax5.set_xlabel('预测误差')
        ax5.set_ylabel('频次')
        ax5.grid(True, alpha=0.3)
        
        # 6. 性能指标摘要
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')
        
        # 计算指标
        from .metrics import calculate_metrics
        metrics = calculate_metrics(predictions_data['y_true'], predictions_data['y_pred'])
        
        # 创建表格
        table_data = [[k, f"{v:.4f}"] for k, v in metrics.items()]
        table = ax6.table(cellText=table_data,
                         colLabels=['指标', '数值'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax6.set_title('性能指标摘要', pad=20)
    
    plt.suptitle('模型训练与预测仪表板', fontsize=20, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"仪表板已保存到: {save_path}")
    
    plt.show()
