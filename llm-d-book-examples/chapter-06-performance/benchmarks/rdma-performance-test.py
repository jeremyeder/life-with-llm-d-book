#!/usr/bin/env python3
"""
RDMA Performance Validation for llm-d Deployments
Tests RDMA bandwidth, latency, and integration with inference workloads

This comprehensive test suite validates RDMA networking performance for
high-throughput LLM inference deployments, including:
- Raw RDMA bandwidth testing using ib_send_bw
- RDMA latency measurement using ib_send_lat  
- End-to-end inference performance validation
- Comprehensive reporting and analysis

Prerequisites:
- RDMA drivers installed (use setup-rdma-nodes.sh)
- Perftest tools (ib_send_bw, ib_send_lat)
- Remote RDMA host with server processes running
- llm-d inference service with RDMA networking

Usage:
    python rdma-performance-test.py --remote-host 192.168.100.2 --service-url http://llm-d-service:8000

Expected Results:
- Bandwidth: >50 Gbps for production workloads
- Latency: <5μs for real-time applications  
- Inference throughput: 40-60% improvement vs TCP

Source: Chapter 6 - Performance Optimization
"""

import subprocess
import time
import json
from typing import Dict, List
import requests

class RDMAPerformanceTester:
    def __init__(self, rdma_interface: str = "mlx5_0"):
        self.rdma_interface = rdma_interface
        self.test_results = {}
    
    def test_rdma_bandwidth(self, remote_host: str) -> Dict:
        """Test RDMA bandwidth using ib_send_bw"""
        print(f"Testing RDMA bandwidth to {remote_host}...")
        
        try:
            # Run bandwidth test (server should be running on remote_host)
            cmd = [
                "ib_send_bw", 
                "-d", self.rdma_interface,
                "-s", "1048576",  # 1MB message size
                "-n", "10000",    # Number of iterations
                remote_host
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Parse results
            lines = result.stdout.split('\n')
            for line in lines:
                if "bytes" in line and "BW average" in line:
                    parts = line.split()
                    bandwidth_gbps = float(parts[-2])
                    
                    return {
                        "bandwidth_gbps": bandwidth_gbps,
                        "bandwidth_bytes_per_sec": bandwidth_gbps * 1e9 / 8,
                        "status": "success" if bandwidth_gbps > 50 else "warning",
                        "raw_output": result.stdout
                    }
            
            return {"status": "error", "message": "Could not parse bandwidth results"}
            
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Bandwidth test timed out"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def test_rdma_latency(self, remote_host: str) -> Dict:
        """Test RDMA latency using ib_send_lat"""
        print(f"Testing RDMA latency to {remote_host}...")
        
        try:
            cmd = [
                "ib_send_lat",
                "-d", self.rdma_interface,
                "-s", "64",       # Small message size for latency
                "-n", "10000",
                remote_host
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Parse latency results
            lines = result.stdout.split('\n')
            for line in lines:
                if "usec" in line and "typical" in line:
                    parts = line.split()
                    latency_us = float(parts[0])
                    
                    return {
                        "latency_microseconds": latency_us,
                        "status": "success" if latency_us < 5 else "warning",
                        "raw_output": result.stdout
                    }
            
            return {"status": "error", "message": "Could not parse latency results"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def test_inference_with_rdma(self, service_url: str) -> Dict:
        """Test inference performance with RDMA networking"""
        print(f"Testing inference performance with RDMA...")
        
        test_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "Describe the key differences between RDMA and TCP networking."
        ]
        
        results = []
        for prompt in test_prompts:
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{service_url}/v1/completions",
                    json={
                        "model": "llama-3.1-70b-rdma",
                        "prompt": prompt,
                        "max_tokens": 150,
                        "temperature": 0.7
                    },
                    timeout=30
                )
                
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        "prompt_length": len(prompt),
                        "response_length": len(data["choices"][0]["text"]),
                        "latency_seconds": end_time - start_time,
                        "tokens_per_second": len(data["choices"][0]["text"]) / (end_time - start_time),
                        "status": "success"
                    })
                else:
                    results.append({
                        "status": "error",
                        "error_code": response.status_code,
                        "error_message": response.text
                    })
                    
            except Exception as e:
                results.append({
                    "status": "error",
                    "error_message": str(e)
                })
        
        # Calculate summary statistics
        successful_tests = [r for r in results if r.get("status") == "success"]
        if successful_tests:
            avg_latency = sum(r["latency_seconds"] for r in successful_tests) / len(successful_tests)
            avg_throughput = sum(r["tokens_per_second"] for r in successful_tests) / len(successful_tests)
            
            return {
                "average_latency_seconds": avg_latency,
                "average_tokens_per_second": avg_throughput,
                "success_rate": len(successful_tests) / len(results),
                "test_results": results,
                "status": "success" if len(successful_tests) == len(results) else "partial"
            }
        else:
            return {
                "status": "error",
                "message": "All inference tests failed",
                "test_results": results
            }
    
    def comprehensive_test(self, remote_host: str, service_url: str) -> Dict:
        """Run comprehensive RDMA performance validation"""
        print("=== Starting Comprehensive RDMA Performance Test ===")
        
        results = {
            "timestamp": time.time(),
            "rdma_interface": self.rdma_interface,
            "remote_host": remote_host,
            "service_url": service_url
        }
        
        # 1. RDMA bandwidth test
        print("\n1. Testing RDMA bandwidth...")
        results["bandwidth_test"] = self.test_rdma_bandwidth(remote_host)
        
        # 2. RDMA latency test
        print("\n2. Testing RDMA latency...")
        results["latency_test"] = self.test_rdma_latency(remote_host)
        
        # 3. Inference performance test
        print("\n3. Testing inference with RDMA...")
        results["inference_test"] = self.test_inference_with_rdma(service_url)
        
        # 4. Generate performance report
        self._generate_report(results)
        
        return results
    
    def _generate_report(self, results: Dict):
        """Generate human-readable performance report"""
        print("\n" + "="*60)
        print("RDMA PERFORMANCE REPORT")
        print("="*60)
        
        # Bandwidth results
        bw_test = results.get("bandwidth_test", {})
        if bw_test.get("status") == "success":
            print(f"✓ RDMA Bandwidth: {bw_test['bandwidth_gbps']:.2f} Gbps")
        else:
            print(f"✗ RDMA Bandwidth: {bw_test.get('message', 'Failed')}")
        
        # Latency results
        lat_test = results.get("latency_test", {})
        if lat_test.get("status") == "success":
            print(f"✓ RDMA Latency: {lat_test['latency_microseconds']:.2f} μs")
        else:
            print(f"✗ RDMA Latency: {lat_test.get('message', 'Failed')}")
        
        # Inference results
        inf_test = results.get("inference_test", {})
        if inf_test.get("status") in ["success", "partial"]:
            print(f"✓ Inference Latency: {inf_test['average_latency_seconds']:.2f} seconds")
            print(f"✓ Inference Throughput: {inf_test['average_tokens_per_second']:.2f} tokens/sec")
            print(f"✓ Success Rate: {inf_test['success_rate']*100:.1f}%")
        else:
            print(f"✗ Inference Test: {inf_test.get('message', 'Failed')}")
        
        print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RDMA Performance Tester for llm-d")
    parser.add_argument("--remote-host", required=True, help="Remote host for RDMA testing")
    parser.add_argument("--service-url", required=True, help="llm-d inference service URL")
    parser.add_argument("--interface", default="mlx5_0", help="RDMA interface name")
    
    args = parser.parse_args()
    
    tester = RDMAPerformanceTester(args.interface)
    results = tester.comprehensive_test(args.remote_host, args.service_url)
    
    # Save results to file
    with open("rdma-performance-results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to rdma-performance-results.json")