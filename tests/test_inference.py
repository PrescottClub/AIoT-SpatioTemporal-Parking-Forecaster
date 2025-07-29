"""
推理系统测试模块

测试推理器、评估器和后处理器的功能。

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

from src.inference.predictor import InferencePredictor, BatchPredictor
from src.inference.evaluator import ModelEvaluator, PerformanceAnalyzer
from src.inference.postprocessor import PredictionPostProcessor
from src.models.spatiotemporal_model import SpatioTemporalPredictor as TrainingPredictor
from src.data.preprocessor import DataPreprocessor
from src.config import Config


class TestInferencePredictor:
    """推理预测器测试类"""
    
    @pytest.fixture
    def inference_setup(self, test_config, device):
        """设置推理测试环境"""
        # 调整配置
        test_config.model.node_feature_dim = 8
        test_config.model.hidden_dim = 16
        test_config.data.sequence_length = 12
        test_config.data.prediction_length = 6
        test_config.system.device = 'cpu'  # 强制使用CPU
        
        # 创建训练模型
        training_model = TrainingPredictor(test_config)
        
        # 创建推理预测器
        predictor = InferencePredictor(model=training_model, config=test_config)
        
        # 创建测试数据
        node_features = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        time_series = torch.randn(4, 12, 8)
        
        return predictor, node_features, edge_index, time_series
    
    def test_predictor_initialization(self, inference_setup):
        """测试预测器初始化"""
        predictor, _, _, _ = inference_setup
        
        assert predictor.model is not None
        assert predictor.device is not None
        assert predictor.config is not None
    
    def test_single_prediction(self, inference_setup):
        """测试单次预测"""
        predictor, node_features, edge_index, time_series = inference_setup
        
        predictions = predictor.predict(node_features, edge_index, time_series)
        
        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape == (4, 6, 3, 1)  # [batch, pred_len, nodes, output_dim]
    
    def test_prediction_with_confidence(self, inference_setup):
        """测试带置信度的预测"""
        predictor, node_features, edge_index, time_series = inference_setup
        
        predictions, confidence = predictor.predict(
            node_features, edge_index, time_series, return_confidence=True
        )
        
        assert isinstance(predictions, torch.Tensor)
        assert isinstance(confidence, torch.Tensor)
        assert predictions.shape == confidence.shape
    
    def test_batch_prediction(self, inference_setup):
        """测试批量预测"""
        predictor, node_features, edge_index, time_series = inference_setup
        
        # 创建批量数据
        batch_data = []
        for i in range(8):
            batch_data.append({
                'node_features': node_features,
                'edge_index': edge_index,
                'time_series': time_series[i % 4]
            })
        
        predictions = predictor.predict_batch(batch_data, batch_size=4)
        
        assert len(predictions) == 8
        assert all(isinstance(pred, torch.Tensor) for pred in predictions)
        assert all(pred.shape == (6, 3, 1) for pred in predictions)
    
    def test_realtime_prediction(self, inference_setup):
        """测试实时预测"""
        predictor, node_features, edge_index, time_series = inference_setup
        
        result = predictor.predict_realtime(
            node_features, edge_index, time_series[:1]
        )
        
        assert 'predictions' in result
        assert 'inference_time' in result
        assert 'timestamp' in result
        assert 'input_shape' in result
        assert 'output_shape' in result
        
        assert isinstance(result['inference_time'], float)
        assert result['inference_time'] > 0


class TestBatchPredictor:
    """批量预测器测试类"""
    
    @pytest.fixture
    def batch_predictor_setup(self, test_config):
        """设置批量预测器测试环境"""
        test_config.system.device = 'cpu'
        training_model = TrainingPredictor(test_config)
        inference_predictor = InferencePredictor(model=training_model, config=test_config)
        
        batch_predictor = BatchPredictor(inference_predictor, batch_size=4)
        
        # 创建测试数据集
        dataset = []
        for i in range(10):
            dataset.append({
                'node_features': torch.randn(3, 8),
                'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]]),
                'time_series': torch.randn(12, 8)
            })
        
        return batch_predictor, dataset
    
    def test_batch_predictor_initialization(self, batch_predictor_setup):
        """测试批量预测器初始化"""
        batch_predictor, _ = batch_predictor_setup
        
        assert batch_predictor.predictor is not None
        assert batch_predictor.batch_size == 4
        assert batch_predictor.num_workers == 1
    
    def test_predict_dataset(self, batch_predictor_setup):
        """测试数据集预测"""
        batch_predictor, dataset = batch_predictor_setup
        
        predictions = batch_predictor.predict_dataset(dataset, show_progress=False)
        
        assert len(predictions) == len(dataset)
        assert all(isinstance(pred, torch.Tensor) for pred in predictions)


class TestModelEvaluator:
    """模型评估器测试类"""
    
    @pytest.fixture
    def evaluator_setup(self, test_config):
        """设置评估器测试环境"""
        test_config.system.device = 'cpu'
        training_model = TrainingPredictor(test_config)
        predictor = InferencePredictor(model=training_model, config=test_config)
        
        evaluator = ModelEvaluator(predictor)
        
        # 创建测试数据
        test_data = []
        targets = []
        for i in range(5):
            test_data.append({
                'node_features': torch.randn(3, 8),
                'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]]),
                'time_series': torch.randn(12, 8)
            })
            targets.append(torch.randn(6, 3, 1))
        
        return evaluator, test_data, targets
    
    def test_evaluator_initialization(self, evaluator_setup):
        """测试评估器初始化"""
        evaluator, _, _ = evaluator_setup
        
        assert evaluator.predictor is not None
        assert evaluator.metrics == ['mae', 'rmse', 'mape', 'r2']
    
    def test_model_evaluation(self, evaluator_setup):
        """测试模型评估"""
        evaluator, test_data, targets = evaluator_setup
        
        results = evaluator.evaluate(test_data, targets, detailed=False)
        
        assert 'metrics' in results
        assert 'inference_time' in results
        assert 'samples_per_second' in results
        assert 'num_samples' in results
        
        assert isinstance(results['metrics'], dict)
        # 检查指标是否存在（键名是大写的）
        expected_metrics = [metric.upper() for metric in evaluator.metrics]
        assert all(metric in results['metrics'] for metric in expected_metrics)
    
    def test_detailed_evaluation(self, evaluator_setup):
        """测试详细评估"""
        evaluator, test_data, targets = evaluator_setup
        
        results = evaluator.evaluate(test_data, targets, detailed=True)
        
        assert 'timestep_metrics' in results
        assert 'node_metrics' in results
        assert 'error_statistics' in results
        assert 'prediction_shape' in results
        assert 'target_shape' in results


class TestPerformanceAnalyzer:
    """性能分析器测试类"""
    
    @pytest.fixture
    def analyzer_setup(self, test_config):
        """设置性能分析器测试环境"""
        test_config.system.device = 'cpu'
        training_model = TrainingPredictor(test_config)
        predictor = InferencePredictor(model=training_model, config=test_config)
        
        analyzer = PerformanceAnalyzer(predictor)
        
        # 创建测试数据
        test_data = []
        for i in range(8):
            test_data.append({
                'node_features': torch.randn(3, 8),
                'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]]),
                'time_series': torch.randn(12, 8)
            })
        
        return analyzer, test_data
    
    def test_analyzer_initialization(self, analyzer_setup):
        """测试性能分析器初始化"""
        analyzer, _ = analyzer_setup
        
        assert analyzer.predictor is not None
    
    def test_benchmark_inference(self, analyzer_setup):
        """测试推理性能基准测试"""
        analyzer, test_data = analyzer_setup
        
        results = analyzer.benchmark_inference(
            test_data, batch_sizes=[1, 2, 4], num_runs=2
        )
        
        assert 'batch_size_1' in results
        assert 'batch_size_2' in results
        assert 'batch_size_4' in results
        
        for batch_result in results.values():
            assert 'avg_time' in batch_result
            assert 'std_time' in batch_result
            assert 'throughput' in batch_result
            assert 'times' in batch_result
    
    def test_profile_model_components(self, analyzer_setup):
        """测试模型组件性能分析"""
        analyzer, test_data = analyzer_setup
        
        results = analyzer.profile_model_components(test_data[0], num_runs=3)
        
        assert 'total_time_avg' in results
        assert 'total_time_std' in results
        assert 'total_time_min' in results
        assert 'total_time_max' in results


class TestPredictionPostProcessor:
    """预测结果后处理器测试类"""
    
    @pytest.fixture
    def postprocessor_setup(self):
        """设置后处理器测试环境"""
        processor = PredictionPostProcessor(output_format='dataframe')
        
        # 创建测试预测结果
        predictions = torch.randn(2, 6, 3, 1)
        confidence = torch.randn(2, 6, 3, 1) * 0.1 + 0.8
        
        metadata = {
            'node_ids': ['node_0', 'node_1', 'node_2'],
            'start_time': pd.Timestamp('2025-01-01 00:00:00'),
            'time_interval': pd.Timedelta(minutes=10)
        }
        
        return processor, predictions, confidence, metadata
    
    def test_postprocessor_initialization(self, postprocessor_setup):
        """测试后处理器初始化"""
        processor, _, _, _ = postprocessor_setup
        
        assert processor.output_format == 'dataframe'
        assert processor.preprocessor is None
    
    def test_process_to_dataframe(self, postprocessor_setup):
        """测试转换为DataFrame"""
        processor, predictions, confidence, metadata = postprocessor_setup
        
        result = processor.process(predictions, metadata, confidence)
        
        assert isinstance(result, pd.DataFrame)
        assert 'batch_id' in result.columns
        assert 'timestamp' in result.columns
        assert 'node_id' in result.columns
        assert 'prediction' in result.columns
        assert 'confidence' in result.columns
        
        assert len(result) == 2 * 6 * 3 * 1  # batch * time * nodes * dims
    
    def test_process_to_json(self, postprocessor_setup):
        """测试转换为JSON"""
        processor, predictions, confidence, metadata = postprocessor_setup
        processor.output_format = 'json'
        
        result = processor.process(predictions, metadata, confidence)
        
        assert isinstance(result, dict)
        assert 'metadata' in result
        assert 'predictions' in result
        
        assert result['metadata']['batch_size'] == 2
        assert result['metadata']['prediction_length'] == 6
        assert result['metadata']['num_nodes'] == 3
        assert result['metadata']['has_confidence'] == True
    
    def test_confidence_intervals(self, postprocessor_setup):
        """测试置信区间计算"""
        processor, predictions, confidence, _ = postprocessor_setup
        
        lower, upper = processor.calculate_confidence_intervals(
            predictions, confidence, confidence_level=0.95
        )
        
        assert isinstance(lower, torch.Tensor)
        assert isinstance(upper, torch.Tensor)
        assert lower.shape == predictions.shape
        assert upper.shape == predictions.shape
        assert torch.all(lower <= predictions)
        assert torch.all(upper >= predictions)
    
    def test_aggregate_predictions(self, postprocessor_setup):
        """测试预测结果聚合"""
        processor, predictions, _, _ = postprocessor_setup
        
        # 按时间维度聚合
        aggregated = processor.aggregate_predictions(
            predictions, aggregation_method='mean', axis=1
        )
        
        assert aggregated.shape == (2, 1, 3, 1)
    
    def test_create_summary_report(self, postprocessor_setup):
        """测试创建摘要报告"""
        processor, predictions, _, metadata = postprocessor_setup
        
        summary = processor.create_summary_report(predictions, metadata)
        
        assert 'shape' in summary
        assert 'statistics' in summary
        assert 'percentiles' in summary
        assert 'timestamp' in summary
        assert 'metadata' in summary
        
        assert summary['statistics']['mean'] is not None
        assert summary['statistics']['std'] is not None
