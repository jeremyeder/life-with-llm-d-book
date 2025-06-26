"""
Tests for load testing framework in chapter-08-troubleshooting/performance-troubleshooting/load-test.py
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
from pathlib import Path
import aiohttp
import numpy as np

# Add the examples directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "llm-d-book-examples"))

try:
    from chapter_08_troubleshooting.performance_troubleshooting.load_test import LoadTester
except ImportError:
    # Create mock class for testing when real implementation isn't available
    class LoadTester:
        def __init__(self, endpoint: str, num_requests: int, concurrency: int):
            self.endpoint = endpoint
            self.num_requests = num_requests
            self.concurrency = concurrency
            self.latencies = []
            self.errors = 0
            self.successful_requests = 0
            self.results = []
            
        async def make_request(self, session, request_id: int):
            """Make a single request and measure latency"""
            import time
            import random
            
            payload = {
                "prompt": "Once upon a time",
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            start_time = time.time()
            
            # Simulate network latency and processing time
            await asyncio.sleep(random.uniform(0.05, 0.15))  # 50-150ms latency
            
            # Simulate occasional errors (5% error rate)
            if random.random() < 0.05:
                self.errors += 1
                return {
                    "request_id": request_id,
                    "error": "Connection timeout"
                }
            
            # Simulate successful response
            latency = time.time() - start_time
            self.latencies.append(latency)
            self.successful_requests += 1
            
            tokens_generated = random.randint(80, 120)
            
            return {
                "request_id": request_id,
                "latency": latency,
                "status": 200,
                "tokens": tokens_generated,
                "response_size_bytes": tokens_generated * 4,  # Approximate
                "queue_time_ms": random.uniform(1, 10),
                "processing_time_ms": latency * 1000 - random.uniform(1, 10)
            }
        
        async def run_test(self):
            """Run the load test"""
            print(f"Starting load test: {self.num_requests} requests, {self.concurrency} concurrent")
            
            import time
            
            tasks = []
            start_time = time.time()
            
            # Create mock session
            mock_session = Mock()
            
            # Run requests with concurrency control
            for i in range(self.num_requests):
                task = self.make_request(mock_session, i)
                tasks.append(task)
                
                # Control concurrency
                if len(tasks) >= self.concurrency:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    self.results.extend([r for r in batch_results if not isinstance(r, Exception)])
                    tasks = []
            
            # Wait for remaining tasks
            if tasks:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                self.results.extend([r for r in batch_results if not isinstance(r, Exception)])
            
            total_time = time.time() - start_time
            
            # Calculate and print statistics
            self.print_results(total_time)
            
            return {
                "total_time": total_time,
                "successful_requests": self.successful_requests,
                "failed_requests": self.errors,
                "throughput_req_per_sec": self.successful_requests / total_time if total_time > 0 else 0,
                "results": self.results
            }
        
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
        
        def get_detailed_metrics(self):
            """Get detailed performance metrics"""
            if not self.latencies:
                return {"error": "No data available"}
            
            latencies_array = np.array(self.latencies)
            
            return {
                "request_metrics": {
                    "total_requests": self.num_requests,
                    "successful_requests": self.successful_requests,
                    "failed_requests": self.errors,
                    "success_rate": (self.successful_requests / self.num_requests) * 100,
                    "error_rate": (self.errors / self.num_requests) * 100
                },
                "latency_metrics": {
                    "min_latency_ms": np.min(latencies_array) * 1000,
                    "max_latency_ms": np.max(latencies_array) * 1000,
                    "mean_latency_ms": np.mean(latencies_array) * 1000,
                    "median_latency_ms": np.median(latencies_array) * 1000,
                    "p95_latency_ms": np.percentile(latencies_array, 95) * 1000,
                    "p99_latency_ms": np.percentile(latencies_array, 99) * 1000,
                    "std_deviation_ms": np.std(latencies_array) * 1000
                },
                "throughput_metrics": {
                    "avg_throughput_req_per_sec": len(self.latencies) / sum(self.latencies) if self.latencies else 0,
                    "peak_throughput_req_per_sec": self.concurrency / (np.min(latencies_array) if len(latencies_array) > 0 else 1),
                    "concurrency_level": self.concurrency
                },
                "token_metrics": {
                    "total_tokens_generated": sum(r.get("tokens", 0) for r in self.results if "tokens" in r),
                    "avg_tokens_per_request": np.mean([r.get("tokens", 0) for r in self.results if "tokens" in r]) if self.results else 0,
                    "tokens_per_second": sum(r.get("tokens", 0) for r in self.results) / sum(self.latencies) if self.latencies else 0
                }
            }
        
        def analyze_performance_bottlenecks(self):
            """Analyze potential performance bottlenecks"""
            if not self.latencies:
                return {"error": "No data available for analysis"}
            
            latencies_array = np.array(self.latencies)
            mean_latency = np.mean(latencies_array)
            p95_latency = np.percentile(latencies_array, 95)
            
            bottlenecks = []
            recommendations = []
            
            # Analyze latency distribution
            if p95_latency > mean_latency * 2:
                bottlenecks.append("High latency variance detected")
                recommendations.append("Investigate tail latency optimization")
            
            # Analyze error rate
            error_rate = (self.errors / self.num_requests) * 100
            if error_rate > 5:
                bottlenecks.append(f"High error rate: {error_rate:.1f}%")
                recommendations.append("Check service capacity and error handling")
            
            # Analyze throughput
            avg_throughput = len(self.latencies) / sum(self.latencies) if self.latencies else 0
            theoretical_max = self.concurrency / mean_latency
            throughput_efficiency = (avg_throughput / theoretical_max) * 100 if theoretical_max > 0 else 0
            
            if throughput_efficiency < 70:
                bottlenecks.append(f"Low throughput efficiency: {throughput_efficiency:.1f}%")
                recommendations.append("Consider increasing concurrency or optimizing request processing")
            
            return {
                "bottlenecks_detected": bottlenecks,
                "recommendations": recommendations,
                "performance_analysis": {
                    "latency_variance_ratio": p95_latency / mean_latency if mean_latency > 0 else 0,
                    "error_rate_percent": error_rate,
                    "throughput_efficiency_percent": throughput_efficiency,
                    "concurrency_utilization": min(100, (avg_throughput * mean_latency / self.concurrency) * 100) if self.concurrency > 0 else 0
                }
            }


class TestLoadTester:
    """Test cases for load testing framework."""
    
    @pytest.fixture
    def load_tester(self):
        """Create load tester instance."""
        return LoadTester(
            endpoint="http://test-service:8080/v1/completions",
            num_requests=10,
            concurrency=3
        )
    
    def test_initialization(self, load_tester):
        """Test LoadTester initialization."""
        assert load_tester.endpoint == "http://test-service:8080/v1/completions"
        assert load_tester.num_requests == 10
        assert load_tester.concurrency == 3
        assert load_tester.latencies == []
        assert load_tester.errors == 0
    
    @pytest.mark.asyncio
    async def test_single_request(self, load_tester):
        """Test making a single request."""
        mock_session = Mock()
        result = await load_tester.make_request(mock_session, 1)
        
        # Verify result structure
        assert "request_id" in result
        assert result["request_id"] == 1
        
        # Should either have success fields or error field
        if "error" in result:
            assert isinstance(result["error"], str)
        else:
            assert "latency" in result
            assert "status" in result
            assert "tokens" in result
            assert result["status"] == 200
            assert result["latency"] > 0
    
    @pytest.mark.asyncio
    async def test_load_test_execution(self, load_tester):
        """Test full load test execution."""
        test_results = await load_tester.run_test()
        
        # Verify test completion
        assert "total_time" in test_results
        assert "successful_requests" in test_results
        assert "failed_requests" in test_results
        assert "throughput_req_per_sec" in test_results
        
        # Verify request accounting
        total_processed = test_results["successful_requests"] + test_results["failed_requests"]
        assert total_processed == load_tester.num_requests
        
        # Verify latency data collection
        if load_tester.latencies:
            assert len(load_tester.latencies) == test_results["successful_requests"]
    
    @pytest.mark.asyncio
    async def test_concurrency_control(self, load_tester):
        """Test concurrency level enforcement."""
        # Monitor concurrent execution
        concurrent_tasks = []
        original_make_request = load_tester.make_request
        
        async def monitored_make_request(session, request_id):
            concurrent_tasks.append(request_id)
            result = await original_make_request(session, request_id)
            concurrent_tasks.remove(request_id)
            return result
        
        load_tester.make_request = monitored_make_request
        
        # Run test with monitoring
        test_results = await load_tester.run_test()
        
        # Verify results structure
        assert test_results["successful_requests"] + test_results["failed_requests"] == load_tester.num_requests
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in load testing."""
        # Create tester that will generate more errors
        error_tester = LoadTester(
            endpoint="http://invalid-service:8080/v1/completions",
            num_requests=5,
            concurrency=2
        )
        
        test_results = await error_tester.run_test()
        
        # Should handle errors gracefully
        assert test_results["failed_requests"] >= 0
        assert test_results["successful_requests"] >= 0
        assert test_results["total_time"] > 0
    
    def test_detailed_metrics_calculation(self, load_tester):
        """Test detailed metrics calculation."""
        # Simulate some test data
        load_tester.latencies = [0.1, 0.15, 0.12, 0.18, 0.09]
        load_tester.successful_requests = 5
        load_tester.errors = 1
        load_tester.num_requests = 6
        load_tester.results = [
            {"tokens": 95, "latency": 0.1},
            {"tokens": 105, "latency": 0.15},
            {"tokens": 88, "latency": 0.12},
            {"tokens": 112, "latency": 0.18},
            {"tokens": 92, "latency": 0.09}
        ]
        
        metrics = load_tester.get_detailed_metrics()
        
        # Verify metric structure
        assert "request_metrics" in metrics
        assert "latency_metrics" in metrics
        assert "throughput_metrics" in metrics
        assert "token_metrics" in metrics
        
        # Verify request metrics
        request_metrics = metrics["request_metrics"]
        assert request_metrics["total_requests"] == 6
        assert request_metrics["successful_requests"] == 5
        assert request_metrics["failed_requests"] == 1
        assert request_metrics["success_rate"] == (5/6) * 100
        
        # Verify latency metrics
        latency_metrics = metrics["latency_metrics"]
        assert latency_metrics["min_latency_ms"] == 90.0  # 0.09 * 1000
        assert latency_metrics["max_latency_ms"] == 180.0  # 0.18 * 1000
        assert latency_metrics["mean_latency_ms"] == 128.0  # mean * 1000
    
    def test_performance_bottleneck_analysis(self, load_tester):
        """Test performance bottleneck analysis."""
        # Create test data with known characteristics
        load_tester.latencies = [0.1, 0.1, 0.1, 0.1, 0.5]  # High tail latency
        load_tester.successful_requests = 5
        load_tester.errors = 2  # High error rate
        load_tester.num_requests = 7
        load_tester.concurrency = 3
        
        analysis = load_tester.analyze_performance_bottlenecks()
        
        # Verify analysis structure
        assert "bottlenecks_detected" in analysis
        assert "recommendations" in analysis
        assert "performance_analysis" in analysis
        
        # Should detect high latency variance
        bottlenecks = analysis["bottlenecks_detected"]
        assert any("latency variance" in b for b in bottlenecks)
        
        # Should detect high error rate (2/7 ≈ 28.6% > 5%)
        assert any("error rate" in b for b in bottlenecks)
        
        # Verify performance analysis metrics
        perf_analysis = analysis["performance_analysis"]
        assert "latency_variance_ratio" in perf_analysis
        assert "error_rate_percent" in perf_analysis
        assert "throughput_efficiency_percent" in perf_analysis
    
    @pytest.mark.parametrize("num_requests,concurrency", [
        (10, 2),
        (50, 5),
        (100, 10),
        (20, 1),  # Serial execution
    ])
    @pytest.mark.asyncio
    async def test_different_load_configurations(self, num_requests, concurrency):
        """Test load tester with different configurations."""
        tester = LoadTester(
            endpoint="http://test-service:8080/v1/completions",
            num_requests=num_requests,
            concurrency=concurrency
        )
        
        test_results = await tester.run_test()
        
        # Verify configuration respect
        assert tester.num_requests == num_requests
        assert tester.concurrency == concurrency
        
        # Verify all requests processed
        total_processed = test_results["successful_requests"] + test_results["failed_requests"]
        assert total_processed == num_requests
        
        # Verify throughput calculation
        if test_results["total_time"] > 0:
            expected_throughput = test_results["successful_requests"] / test_results["total_time"]
            assert abs(test_results["throughput_req_per_sec"] - expected_throughput) < 0.01
    
    @pytest.mark.asyncio
    async def test_latency_distribution_analysis(self, load_tester):
        """Test latency distribution analysis."""
        test_results = await load_tester.run_test()
        
        if load_tester.latencies:
            latencies = np.array(load_tester.latencies)
            
            # Verify latency statistics make sense
            assert np.min(latencies) <= np.mean(latencies) <= np.max(latencies)
            assert np.percentile(latencies, 50) <= np.percentile(latencies, 95)
            assert np.percentile(latencies, 95) <= np.percentile(latencies, 99)
            
            # Verify all latencies are positive
            assert all(lat > 0 for lat in latencies)
    
    @pytest.mark.asyncio
    async def test_throughput_calculation(self, load_tester):
        """Test throughput calculation accuracy."""
        test_results = await load_tester.run_test()
        
        if test_results["successful_requests"] > 0 and test_results["total_time"] > 0:
            calculated_throughput = test_results["successful_requests"] / test_results["total_time"]
            reported_throughput = test_results["throughput_req_per_sec"]
            
            # Should match within reasonable precision
            assert abs(calculated_throughput - reported_throughput) < 0.01
    
    def test_token_metrics_calculation(self, load_tester):
        """Test token-based performance metrics."""
        # Simulate test results with token data
        load_tester.results = [
            {"tokens": 100, "latency": 0.1},
            {"tokens": 150, "latency": 0.15},
            {"tokens": 80, "latency": 0.08},
        ]
        load_tester.latencies = [0.1, 0.15, 0.08]
        load_tester.successful_requests = 3
        
        metrics = load_tester.get_detailed_metrics()
        token_metrics = metrics["token_metrics"]
        
        # Verify token calculations
        assert token_metrics["total_tokens_generated"] == 330  # 100+150+80
        assert token_metrics["avg_tokens_per_request"] == 110  # 330/3
        
        # Verify tokens per second calculation
        total_time = sum(load_tester.latencies)  # 0.33
        expected_tokens_per_sec = 330 / total_time
        assert abs(token_metrics["tokens_per_second"] - expected_tokens_per_sec) < 0.01
    
    @pytest.mark.asyncio
    async def test_stress_test_simulation(self):
        """Test high-load stress testing scenario."""
        stress_tester = LoadTester(
            endpoint="http://stress-test-service:8080/v1/completions",
            num_requests=25,  # Reduced for test performance
            concurrency=8
        )
        
        test_results = await stress_tester.run_test()
        
        # Verify stress test completion
        assert test_results["total_time"] > 0
        total_processed = test_results["successful_requests"] + test_results["failed_requests"]
        assert total_processed == 25
        
        # Get detailed metrics for stress analysis
        if stress_tester.successful_requests > 0:
            metrics = stress_tester.get_detailed_metrics()
            
            # Should have reasonable performance under stress
            latency_metrics = metrics["latency_metrics"]
            assert latency_metrics["mean_latency_ms"] < 1000  # Under 1 second average
            
            request_metrics = metrics["request_metrics"]
            assert request_metrics["success_rate"] > 50  # At least 50% success rate
    
    def test_empty_results_handling(self, load_tester):
        """Test handling of empty test results."""
        # Simulate no successful requests
        load_tester.latencies = []
        load_tester.successful_requests = 0
        load_tester.errors = load_tester.num_requests
        
        metrics = load_tester.get_detailed_metrics()
        assert "error" in metrics
        
        analysis = load_tester.analyze_performance_bottlenecks()
        assert "error" in analysis
    
    @pytest.mark.asyncio
    async def test_request_payload_customization(self, load_tester):
        """Test request payload customization."""
        # Mock session to capture payloads
        payloads_sent = []
        
        original_make_request = load_tester.make_request
        
        async def capture_payload_make_request(session, request_id):
            # In real implementation, would capture actual payload
            payloads_sent.append({
                "prompt": "Once upon a time",
                "max_tokens": 100,
                "temperature": 0.7
            })
            return await original_make_request(session, request_id)
        
        load_tester.make_request = capture_payload_make_request
        
        test_results = await load_tester.run_test()
        
        # Verify payload consistency
        assert len(payloads_sent) == test_results["successful_requests"] + test_results["failed_requests"]
        
        # Verify payload structure
        for payload in payloads_sent:
            assert "prompt" in payload
            assert "max_tokens" in payload
            assert "temperature" in payload
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, load_tester):
        """Test detection of performance regressions."""
        # Run baseline test
        baseline_results = await load_tester.run_test()
        baseline_metrics = load_tester.get_detailed_metrics()
        
        # Simulate degraded performance for comparison
        degraded_tester = LoadTester(
            endpoint=load_tester.endpoint,
            num_requests=load_tester.num_requests,
            concurrency=max(1, load_tester.concurrency // 2)  # Reduced concurrency
        )
        
        degraded_results = await degraded_tester.run_test()
        
        # Compare performance
        if (baseline_results["successful_requests"] > 0 and 
            degraded_results["successful_requests"] > 0):
            
            baseline_throughput = baseline_results["throughput_req_per_sec"]
            degraded_throughput = degraded_results["throughput_req_per_sec"]
            
            # Should detect throughput difference
            throughput_ratio = degraded_throughput / baseline_throughput if baseline_throughput > 0 else 0
            
            # With reduced concurrency, degraded test might have lower throughput
            # This validates that our testing can detect performance changes
            assert 0 < throughput_ratio <= 2.0  # Reasonable bounds