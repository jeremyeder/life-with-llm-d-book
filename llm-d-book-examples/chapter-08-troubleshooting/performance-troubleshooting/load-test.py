#!/usr/bin/env python3
"""
Load Testing Framework for LLM Services

This script provides a comprehensive load testing framework for LLM inference services,
measuring latency, throughput, and error rates under various load conditions.

Usage:
    python load-test.py

Run this script to perform load testing on your LLM service endpoints.
"""

import asyncio
import aiohttp
import time
import numpy as np
from typing import List, Dict

class LoadTester:
    def __init__(self, endpoint: str, num_requests: int, concurrency: int):
        self.endpoint = endpoint
        self.num_requests = num_requests
        self.concurrency = concurrency
        self.latencies = []
        self.errors = 0
    
    async def make_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict:
        """Make a single request and measure latency"""
        payload = {
            "prompt": "Once upon a time",
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        start_time = time.time()
        try:
            async with session.post(self.endpoint, json=payload) as response:
                result = await response.json()
                latency = time.time() - start_time
                self.latencies.append(latency)
                
                return {
                    "request_id": request_id,
                    "latency": latency,
                    "status": response.status,
                    "tokens": len(result.get("text", "").split())
                }
        except Exception as e:
            self.errors += 1
            return {
                "request_id": request_id,
                "error": str(e)
            }
    
    async def run_test(self):
        """Run the load test"""
        print(f"Starting load test: {self.num_requests} requests, {self.concurrency} concurrent")
        
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            start_time = time.time()
            
            for i in range(self.num_requests):
                task = self.make_request(session, i)
                tasks.append(task)
                
                # Control concurrency
                if len(tasks) >= self.concurrency:
                    await asyncio.gather(*tasks)
                    tasks = []
            
            # Wait for remaining tasks
            if tasks:
                await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
        
        # Calculate statistics
        self.print_results(total_time)
    
    def print_results(self, total_time: float):
        """Print test results"""
        if not self.latencies:
            print("No successful requests")
            return
        
        latencies_array = np.array(self.latencies)
        
        print(f"\n=== Load Test Results ===")
        print(f"Total requests: {self.num_requests}")
        print(f"Successful: {len(self.latencies)}")
        print(f"Failed: {self.errors}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {len(self.latencies)/total_time:.2f} req/s")
        print(f"\nLatency Statistics:")
        print(f"  Min: {np.min(latencies_array):.3f}s")
        print(f"  Max: {np.max(latencies_array):.3f}s")
        print(f"  Mean: {np.mean(latencies_array):.3f}s")
        print(f"  P50: {np.percentile(latencies_array, 50):.3f}s")
        print(f"  P95: {np.percentile(latencies_array, 95):.3f}s")
        print(f"  P99: {np.percentile(latencies_array, 99):.3f}s")

# Run load test
if __name__ == "__main__":
    tester = LoadTester(
        endpoint="http://llm-model-service:8080/v1/completions",
        num_requests=1000,
        concurrency=50
    )
    asyncio.run(tester.run_test())