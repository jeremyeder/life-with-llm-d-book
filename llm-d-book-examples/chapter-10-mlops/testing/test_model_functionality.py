# Model functionality test suite
# Comprehensive testing framework for LLM model functionality
# Tests model loading, inference, batch processing, and memory constraints

import pytest
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any

class ModelFunctionalityTester:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        
    def setup_model(self):
        """Load model and tokenizer for testing"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            return True
        except Exception as e:
            pytest.fail(f"Failed to load model: {e}")
            return False
    
    def test_model_loading(self):
        """Test that model loads successfully"""
        assert self.setup_model(), "Model should load without errors"
        assert self.model is not None, "Model should be initialized"
        assert self.tokenizer is not None, "Tokenizer should be initialized"
    
    def test_basic_inference(self):
        """Test basic inference functionality"""
        if not self.model:
            self.setup_model()
            
        test_prompts = [
            "The quick brown fox",
            "Once upon a time",
            "In the field of artificial intelligence",
            "def hello_world():"
        ]
        
        for prompt in test_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Validate output
            assert len(generated_text) > len(prompt), f"Should generate new tokens for prompt: {prompt}"
            assert generated_text.startswith(prompt), f"Output should start with input prompt: {prompt}"
    
    def test_batch_inference(self):
        """Test batch inference capabilities"""
        if not self.model:
            self.setup_model()
            
        batch_prompts = [
            "Hello world",
            "Machine learning is",
            "The future of AI"
        ]
        
        # Tokenize batch
        inputs = self.tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Validate batch output
        assert outputs.shape[0] == len(batch_prompts), "Should generate output for each input"
        
        # Decode and validate each output
        for i, generated_ids in enumerate(outputs):
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            assert batch_prompts[i] in generated_text, f"Output should contain input prompt: {batch_prompts[i]}"
    
    def test_memory_constraints(self):
        """Test model respects memory constraints"""
        if not self.model:
            self.setup_model()
            
        # Test with very long sequence
        long_prompt = "This is a test. " * 100  # ~400 tokens
        inputs = self.tokenizer(long_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Monitor GPU memory before and after
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    do_sample=False
                )
            
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used_gb = (peak_memory - initial_memory) / (1024**3)
            
            # Should not use excessive memory (model-dependent threshold)
            max_memory_gb = 20.0  # Adjust based on model size
            assert memory_used_gb < max_memory_gb, f"Memory usage {memory_used_gb:.2f}GB exceeds limit {max_memory_gb}GB"

# Pytest fixtures and test runner
@pytest.fixture(scope="module", params=["llama-3.1-7b", "llama-3.1-13b"])
def model_tester(request):
    """Fixture to test multiple model variants"""
    model_path = f"s3://model-registry/{request.param}/latest"
    return ModelFunctionalityTester(model_path)

def test_model_functionality_suite(model_tester):
    """Run complete model functionality test suite"""
    model_tester.test_model_loading()
    model_tester.test_basic_inference()
    model_tester.test_batch_inference()
    model_tester.test_memory_constraints()