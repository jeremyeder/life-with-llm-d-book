# Model benchmarking framework
# Comprehensive performance benchmarking for different model sizes
# Measures inference speed, memory usage, and system resource utilization

import time
import torch
import psutil
import json
from typing import Dict, List
import GPUtil

class ModelBenchmarker:
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.results = {}
        
    def benchmark_inference_speed(self, batch_sizes: List[int] = [1, 2, 4, 8]) -> Dict:
        """Benchmark inference speed across different batch sizes"""
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"📊 Benchmarking batch size {batch_size}...")
            
            # Prepare batch
            prompts = [f"Test prompt {i} for benchmarking" for i in range(batch_size)]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            
            # Warm up
            for _ in range(3):
                with torch.no_grad():
                    _ = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=20,
                        do_sample=False
                    )
            
            # Benchmark
            latencies = []
            tokens_per_second = []
            
            for run in range(10):  # 10 benchmark runs
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=50,
                        do_sample=False
                    )
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                # Calculate metrics
                run_latency = end_time - start_time
                tokens_generated = (outputs.shape[1] - inputs.input_ids.shape[1]) * batch_size
                tokens_per_sec = tokens_generated / run_latency
                
                latencies.append(run_latency)
                tokens_per_second.append(tokens_per_sec)
            
            # Calculate statistics
            results[f"batch_{batch_size}"] = {
                "avg_latency_ms": (sum(latencies) / len(latencies)) * 1000,
                "min_latency_ms": min(latencies) * 1000,
                "max_latency_ms": max(latencies) * 1000,
                "avg_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
                "throughput_requests_per_second": batch_size / (sum(latencies) / len(latencies))
            }
        
        return results
    
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage patterns"""
        
        # System memory before model loading
        initial_ram = psutil.virtual_memory().used / (1024**3)
        initial_gpu_memory = 0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        
        # Load model and measure memory
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Memory after loading
        loaded_ram = psutil.virtual_memory().used / (1024**3)
        loaded_gpu_memory = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        # Run inference to measure peak memory
        test_input = tokenizer("Test memory usage with a longer prompt to see peak allocation", return_tensors="pt")
        
        with torch.no_grad():
            _ = model.generate(
                test_input.input_ids,
                max_new_tokens=100,
                do_sample=False
            )
        
        # Peak memory usage
        peak_ram = psutil.virtual_memory().used / (1024**3)
        peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        return {
            "ram_usage_gb": {
                "initial": initial_ram,
                "after_loading": loaded_ram,
                "peak": peak_ram,
                "model_overhead": loaded_ram - initial_ram
            },
            "gpu_memory_gb": {
                "initial": initial_gpu_memory,
                "after_loading": loaded_gpu_memory,
                "peak": peak_gpu_memory,
                "model_size": loaded_gpu_memory - initial_gpu_memory
            }
        }
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        
        print(f"🔬 Running full benchmark for {self.model_name}")
        
        benchmark_results = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "timestamp": time.time(),
            "system_info": {
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
                "cpu_count": psutil.cpu_count(),
                "total_ram_gb": psutil.virtual_memory().total / (1024**3)
            }
        }
        
        # Run benchmarks
        try:
            benchmark_results["inference_speed"] = self.benchmark_inference_speed()
            benchmark_results["memory_usage"] = self.benchmark_memory_usage()
            benchmark_results["status"] = "completed"
            
        except Exception as e:
            benchmark_results["status"] = "failed"
            benchmark_results["error"] = str(e)
        
        return benchmark_results

def run_model_benchmarks():
    """Run benchmarks for all model variants"""
    
    models_to_benchmark = [
        {"name": "llama-3.1-7b", "path": "s3://model-registry/llama-3.1-7b/latest"},
        {"name": "llama-3.1-13b", "path": "s3://model-registry/llama-3.1-13b/latest"}
    ]
    
    all_results = []
    
    for model_config in models_to_benchmark:
        benchmarker = ModelBenchmarker(model_config["name"], model_config["path"])
        results = benchmarker.run_full_benchmark()
        all_results.append(results)
        
        # Save individual results
        with open(f"benchmark_{model_config['name']}.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Save combined results
    with open("benchmark_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n📈 Benchmark Summary:")
    for result in all_results:
        if result["status"] == "completed":
            print(f"  {result['model_name']}:")
            print(f"    Memory: {result['memory_usage']['gpu_memory_gb']['model_size']:.1f}GB GPU")
            print(f"    Speed: {result['inference_speed']['batch_1']['avg_tokens_per_second']:.1f} tokens/sec")
        else:
            print(f"  {result['model_name']}: FAILED - {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    run_model_benchmarks()