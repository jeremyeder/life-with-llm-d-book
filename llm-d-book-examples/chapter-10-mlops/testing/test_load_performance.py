# Load testing framework for LLM models
# Comprehensive async load testing with performance metrics
# Tests concurrent requests, sustained load, and latency requirements

import asyncio
import aiohttp
import time
import statistics
import json
from typing import List, Dict
import pytest

class LoadTester:
    def __init__(self, endpoint: str, model_name: str):
        self.endpoint = endpoint
        self.model_name = model_name
        
    async def single_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict:
        """Send a single inference request"""
        payload = {
            "prompt": f"Test request {request_id}: Generate a response about artificial intelligence",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.endpoint}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                
                result = await response.json()
                
                return {
                    "request_id": request_id,
                    "status_code": response.status,
                    "latency_ms": (end_time - start_time) * 1000,
                    "success": response.status == 200,
                    "tokens_generated": len(result.get("choices", [{}])[0].get("text", "").split()) if response.status == 200 else 0,
                    "error": None
                }
                
        except Exception as e:
            end_time = time.time()
            return {
                "request_id": request_id,
                "status_code": 0,
                "latency_ms": (end_time - start_time) * 1000,
                "success": False,
                "tokens_generated": 0,
                "error": str(e)
            }
    
    async def load_test(self, total_requests: int, concurrent_requests: int) -> Dict:
        """Run load test with specified parameters"""
        
        print(f"🚀 Starting load test: {total_requests} requests, {concurrent_requests} concurrent")
        
        results = []
        
        # Configure session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=concurrent_requests * 2,
            limit_per_host=concurrent_requests
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # Send requests in batches to maintain concurrency level
            for batch_start in range(0, total_requests, concurrent_requests):
                batch_end = min(batch_start + concurrent_requests, total_requests)
                batch_size = batch_end - batch_start
                
                # Create tasks for this batch
                tasks = [
                    self.single_request(session, request_id)
                    for request_id in range(batch_start, batch_end)
                ]
                
                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, Exception):
                        results.append({
                            "request_id": len(results),
                            "status_code": 0,
                            "latency_ms": 0,
                            "success": False,
                            "tokens_generated": 0,
                            "error": str(result)
                        })
                    else:
                        results.append(result)
        
        # Calculate metrics
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        if successful_requests:
            latencies = [r["latency_ms"] for r in successful_requests]
            tokens_per_request = [r["tokens_generated"] for r in successful_requests]
            
            metrics = {
                "total_requests": total_requests,
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(successful_requests) / total_requests,
                "latency_stats": {
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "mean_ms": statistics.mean(latencies),
                    "median_ms": statistics.median(latencies),
                    "p95_ms": self._percentile(latencies, 95),
                    "p99_ms": self._percentile(latencies, 99)
                },
                "throughput": {
                    "avg_tokens_per_request": statistics.mean(tokens_per_request) if tokens_per_request else 0,
                    "total_tokens_generated": sum(tokens_per_request)
                }
            }
        else:
            metrics = {
                "total_requests": total_requests,
                "successful_requests": 0,
                "failed_requests": len(failed_requests),
                "success_rate": 0.0,
                "latency_stats": {},
                "throughput": {"avg_tokens_per_request": 0, "total_tokens_generated": 0}
            }
        
        return metrics
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a dataset"""
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_data) - 1)
        
        if lower_index == upper_index:
            return sorted_data[lower_index]
        
        weight = index - lower_index
        return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight

@pytest.mark.parametrize("concurrent_requests", [1, 5, 10, 20])
async def test_concurrent_load(concurrent_requests):
    """Test model performance under different concurrency levels"""
    
    endpoint = "http://llm-gateway.staging.svc.cluster.local"
    model_name = "llama-3.1-7b"
    
    tester = LoadTester(endpoint, model_name)
    
    # Run load test
    results = await tester.load_test(
        total_requests=50,
        concurrent_requests=concurrent_requests
    )
    
    # Performance assertions
    assert results["success_rate"] >= 0.95, f"Success rate {results['success_rate']:.2f} below threshold"
    
    if results["latency_stats"]:
        assert results["latency_stats"]["p95_ms"] <= 5000, f"P95 latency {results['latency_stats']['p95_ms']:.0f}ms exceeds 5s threshold"
        assert results["latency_stats"]["mean_ms"] <= 2000, f"Mean latency {results['latency_stats']['mean_ms']:.0f}ms exceeds 2s threshold"
    
    print(f"✅ Concurrency {concurrent_requests}: Success rate {results['success_rate']:.2%}, P95 latency {results['latency_stats'].get('p95_ms', 0):.0f}ms")

async def test_sustained_load():
    """Test sustained load over extended period"""
    
    endpoint = "http://llm-gateway.staging.svc.cluster.local"
    model_name = "llama-3.1-7b"
    
    tester = LoadTester(endpoint, model_name)
    
    # Run sustained load test
    results = await tester.load_test(
        total_requests=200,
        concurrent_requests=10
    )
    
    # Sustained load assertions
    assert results["success_rate"] >= 0.99, f"Sustained load success rate {results['success_rate']:.2f} below threshold"
    assert results["latency_stats"]["p99_ms"] <= 10000, f"P99 latency {results['latency_stats']['p99_ms']:.0f}ms exceeds 10s threshold"
    
    print(f"✅ Sustained load: {results['total_requests']} requests, {results['success_rate']:.2%} success rate")