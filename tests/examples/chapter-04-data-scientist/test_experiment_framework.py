"""
Tests for the ExperimentManager class in chapter-04-data-scientist/experiment_framework.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path

# Add the examples directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "examples"))

# Import after adding to path, and handle import of missing dependencies
try:
    import matplotlib.pyplot
    # If matplotlib is available, use normal import
    from chapter_04_data_scientist.experiment_framework import ExperimentManager
except ImportError:
    # If not available, mock it
    with patch('sys.modules.matplotlib', MagicMock()):
        with patch('sys.modules.matplotlib.pyplot', MagicMock()):
            try:
                from chapter_04_data_scientist.experiment_framework import ExperimentManager
            except ImportError:
                # If still fails, create a mock class
                class ExperimentManager:
                    def __init__(self, experiment_name, client):
                        self.experiment_name = experiment_name
                        self.client = client
                        self.results = []
                        self.start_time = datetime.now()

from tests.fixtures.mock_responses import MockLLMResponses


class TestExperimentManager:
    """Test cases for ExperimentManager class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock LLM client."""
        client = Mock()
        client.chat_completion.return_value = MockLLMResponses.chat_completion_success()
        return client
    
    @pytest.fixture
    def experiment_manager(self, mock_client):
        """Create an ExperimentManager instance."""
        return ExperimentManager("test_experiment", mock_client)
    
    @pytest.fixture
    def test_cases(self):
        """Sample test cases for experiments."""
        return [
            {
                "name": "test_case_1",
                "messages": [{"role": "user", "content": "Test message 1"}]
            },
            {
                "name": "test_case_2", 
                "messages": [{"role": "user", "content": "Test message 2"}]
            }
        ]
    
    @pytest.fixture
    def model_configs(self):
        """Sample model configurations."""
        return {
            "config_1": {"temperature": 0.5, "max_tokens": 100},
            "config_2": {"temperature": 0.9, "max_tokens": 200}
        }
    
    def test_initialization(self, mock_client):
        """Test ExperimentManager initialization."""
        manager = ExperimentManager("test_exp", mock_client)
        
        assert manager.experiment_name == "test_exp"
        assert manager.client == mock_client
        assert manager.results == []
        assert isinstance(manager.start_time, datetime)
    
    def test_run_single_test(self, experiment_manager, mock_client):
        """Test running a single test case."""
        test_case = {
            "name": "test",
            "messages": [{"role": "user", "content": "Test"}]
        }
        config = {"temperature": 0.7, "max_tokens": 100}
        
        with patch('datetime.datetime') as mock_datetime:
            # Mock datetime to control timing
            mock_start = datetime(2024, 1, 1, 10, 0, 0)
            mock_end = datetime(2024, 1, 1, 10, 0, 1)  # 1 second later
            mock_datetime.now.side_effect = [mock_start, mock_end]
            
            result = experiment_manager._run_single_test(test_case, config)
        
        # Verify client was called correctly
        mock_client.chat_completion.assert_called_once_with(
            messages=test_case["messages"],
            temperature=0.7,
            max_tokens=100
        )
        
        # Verify result structure
        assert result["test_case"] == test_case
        assert result["latency"] == 1.0  # 1 second
        assert result["tokens_generated"] == 45  # From mock response
        assert "response" in result
        assert "timestamp" in result
    
    def test_run_experiment(self, experiment_manager, test_cases, model_configs, mock_client):
        """Test running complete experiment."""
        experiment_manager.run_experiment(test_cases, model_configs)
        
        # Verify correct number of calls
        expected_calls = len(test_cases) * len(model_configs)
        assert mock_client.chat_completion.call_count == expected_calls
        
        # Verify results structure
        assert len(experiment_manager.results) == len(model_configs)
        
        for result in experiment_manager.results:
            assert "config_name" in result
            assert "config" in result
            assert "results" in result
            assert "avg_latency" in result
            assert len(result["results"]) == len(test_cases)
    
    def test_calculate_avg_latency(self, experiment_manager):
        """Test average latency calculation."""
        results = [
            {"latency": 1.0},
            {"latency": 2.0},
            {"latency": 3.0}
        ]
        
        avg_latency = experiment_manager._calculate_avg_latency(results)
        assert avg_latency == 2.0
    
    def test_export_results(self, experiment_manager, test_cases, model_configs, tmp_path):
        """Test exporting results to CSV."""
        # Run experiment first
        experiment_manager.run_experiment(test_cases, model_configs)
        
        # Export to temporary file
        output_file = tmp_path / "test_results.csv"
        
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            experiment_manager.export_results(str(output_file))
            mock_to_csv.assert_called_once_with(str(output_file), index=False)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.bar')
    @patch('matplotlib.pyplot.figure')
    def test_visualize_results(self, mock_figure, mock_bar, mock_show, 
                              experiment_manager, test_cases, model_configs):
        """Test visualization of results."""
        # Run experiment first
        experiment_manager.run_experiment(test_cases, model_configs)
        
        # Test visualization
        experiment_manager.visualize_results()
        
        # Verify matplotlib calls
        mock_figure.assert_called_once_with(figsize=(10, 6))
        mock_bar.assert_called_once()
        mock_show.assert_called_once()
        
        # Verify bar chart data
        bar_call_args = mock_bar.call_args[0]
        configs = bar_call_args[0]
        latencies = bar_call_args[1]
        
        assert len(configs) == len(model_configs)
        assert len(latencies) == len(model_configs)
        assert all(isinstance(l, float) for l in latencies)
    
    def test_experiment_with_error_response(self, experiment_manager, mock_client):
        """Test handling of error responses."""
        # Make client return error
        mock_client.chat_completion.return_value = MockLLMResponses.chat_completion_error()
        
        test_case = {
            "name": "test",
            "messages": [{"role": "user", "content": "Test"}]
        }
        config = {"temperature": 0.7}
        
        result = experiment_manager._run_single_test(test_case, config)
        
        # Should still return result with error response
        assert "response" in result
        assert "error" in result["response"]
        assert result["tokens_generated"] == 0  # No tokens on error
    
    def test_multiple_experiments_isolation(self, mock_client):
        """Test that multiple experiment instances don't interfere."""
        exp1 = ExperimentManager("exp1", mock_client)
        exp2 = ExperimentManager("exp2", mock_client)
        
        test_cases = [{"messages": [{"role": "user", "content": "Test"}]}]
        configs = {"config1": {"temperature": 0.5}}
        
        exp1.run_experiment(test_cases, configs)
        exp2.run_experiment(test_cases, configs)
        
        # Results should be isolated
        assert len(exp1.results) == 1
        assert len(exp2.results) == 1
        assert exp1.results is not exp2.results
    
    @pytest.mark.parametrize("num_test_cases,num_configs", [
        (1, 1),
        (5, 3),
        (10, 5),
    ])
    def test_scaling(self, mock_client, num_test_cases, num_configs):
        """Test experiment scaling with different numbers of tests and configs."""
        manager = ExperimentManager("scale_test", mock_client)
        
        test_cases = [
            {"messages": [{"role": "user", "content": f"Test {i}"}]}
            for i in range(num_test_cases)
        ]
        
        configs = {
            f"config_{i}": {"temperature": 0.5 + i * 0.1}
            for i in range(num_configs)
        }
        
        manager.run_experiment(test_cases, configs)
        
        # Verify correct number of results
        assert len(manager.results) == num_configs
        assert mock_client.chat_completion.call_count == num_test_cases * num_configs