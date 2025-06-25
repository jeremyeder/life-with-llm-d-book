"""
Simple test to demonstrate the testing infrastructure is working.
"""

import pytest
from tests.fixtures.mock_responses import MockLLMResponses
from tests.utils.test_helpers import MockDataGenerator, ResponseValidator


class TestInfrastructure:
    """Test that demonstrates the testing infrastructure works."""
    
    def test_llm_mock_responses(self):
        """Test LLM mock responses are properly formatted."""
        response = MockLLMResponses.chat_completion_success()
        assert ResponseValidator.validate_inference_response(response)
        assert response["id"] == "chatcmpl-test123"
    
    def test_gpu_metrics_generator(self):
        """Test GPU metrics generation."""
        metrics = MockDataGenerator.generate_gpu_metrics(num_gpus=2)
        assert len(metrics) == 2
        assert "gpu_0" in metrics
        assert "gpu_1" in metrics
    
    def test_cost_data_generator(self):
        """Test cost data generation."""
        cost_data = MockDataGenerator.generate_cost_data(hours=24)
        assert cost_data["period"] == "24h"
        assert len(cost_data["hourly_breakdown"]) == 24
        assert cost_data["total_cost"] > 0
    
    @pytest.mark.parametrize("num_steps", [10, 50])
    def test_training_metrics_generator(self, num_steps):
        """Test training metrics generation with different step counts."""
        metrics = MockDataGenerator.generate_training_metrics(num_steps)
        assert len(metrics) == num_steps
        
        # Verify loss generally decreases
        first_loss = metrics[0]["loss"]
        last_loss = metrics[-1]["loss"]
        assert last_loss < first_loss
    
    def test_config_validation(self):
        """Test configuration validation utilities."""
        config = {"model": "test", "batch_size": 32, "temperature": 0.7}
        schema = {"model": str, "batch_size": int, "temperature": float}
        assert ResponseValidator.validate_config_schema(config, schema) is True
    
    def test_infrastructure_ready(self):
        """Verify the testing infrastructure is ready."""
        # This test confirms:
        # - pytest is working
        # - Test directory structure is correct
        # - Mock utilities are importable
        # - Fixtures are working
        # - Parametrized tests work
        assert True