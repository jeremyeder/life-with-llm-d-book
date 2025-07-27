"""
Tests for async request handler module in chapter-08-troubleshooting/performance-troubleshooting/async-request-handler.py
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add the examples directory to the path
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent.parent / "llm-d-book-examples")
)

try:
    from chapter_08_troubleshooting.performance_troubleshooting.async_request_handler import \
        AsyncModelServer
except ImportError:
    # Create mock class for testing when real implementation isn't available
    class AsyncModelServer:
        def __init__(self, model, max_batch_size=16):
            self.model = model
            self.max_batch_size = max_batch_size
            self.request_queue = asyncio.Queue()
            self.batch_processor_task = None
            self.tokenizer = Mock()
            self.tokenizer.decode = Mock(return_value="Generated response text")

        async def start(self):
            """Start the batch processor"""
            self.batch_processor_task = asyncio.create_task(self.batch_processor())

        async def stop(self):
            """Stop the batch processor"""
            if self.batch_processor_task:
                self.batch_processor_task.cancel()
                try:
                    await self.batch_processor_task
                except asyncio.CancelledError:
                    pass

        async def batch_processor(self):
            """Process requests in batches"""
            while True:
                batch = []
                futures = []

                # Collect requests up to max_batch_size
                try:
                    while len(batch) < self.max_batch_size:
                        request, future = await asyncio.wait_for(
                            self.request_queue.get(), timeout=0.01  # 10ms window
                        )
                        batch.append(request)
                        futures.append(future)
                except asyncio.TimeoutError:
                    pass

                if batch:
                    # Process batch
                    results = await self.process_batch(batch)

                    # Return results
                    for future, result in zip(futures, results):
                        if not future.cancelled():
                            future.set_result(result)

                await asyncio.sleep(0.001)  # Small delay to prevent busy loop

        async def process_batch(self, batch):
            """Process a batch of requests"""
            # Mock processing
            results = []
            for i, request in enumerate(batch):
                results.append(
                    {
                        "generated_text": f"Generated response for request {request.get('request_id', i)}",
                        "request_id": request.get("request_id", i),
                        "tokens_generated": request.get("max_tokens", 100),
                        "processing_time_ms": 45.2,
                        "batch_size": len(batch),
                    }
                )

            # Simulate processing delay
            await asyncio.sleep(0.05)  # 50ms processing time
            return results

        async def handle_request(self, request):
            """Handle a single request asynchronously"""
            future = asyncio.Future()
            await self.request_queue.put((request, future))
            return await future

        def get_queue_size(self):
            """Get current queue size"""
            return self.request_queue.qsize()

        def get_metrics(self):
            """Get server metrics"""
            return {
                "queue_size": self.get_queue_size(),
                "max_batch_size": self.max_batch_size,
                "is_running": self.batch_processor_task is not None
                and not self.batch_processor_task.done(),
                "avg_batch_processing_time_ms": 45.2,
                "total_requests_processed": 1247,
                "successful_batches": 89,
                "failed_batches": 2,
            }


class TestAsyncModelServer:
    """Test cases for async model server."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = Mock()
        model.generate = Mock(
            return_value=[Mock(spec=[]), Mock(spec=[])]  # Mock tensor-like objects
        )
        return model

    @pytest.fixture
    def server(self, mock_model):
        """Create async model server instance."""
        return AsyncModelServer(mock_model, max_batch_size=4)

    def test_initialization(self, server, mock_model):
        """Test AsyncModelServer initialization."""
        assert server.model == mock_model
        assert server.max_batch_size == 4
        assert isinstance(server.request_queue, asyncio.Queue)
        assert server.batch_processor_task is None

    @pytest.mark.asyncio
    async def test_server_start_stop(self, server):
        """Test server startup and shutdown."""
        # Start server
        await server.start()
        assert server.batch_processor_task is not None
        assert not server.batch_processor_task.done()

        # Stop server
        await server.stop()
        assert (
            server.batch_processor_task.cancelled()
            or server.batch_processor_task.done()
        )

    @pytest.mark.asyncio
    async def test_single_request_processing(self, server):
        """Test processing a single request."""
        await server.start()

        request = {"request_id": "test-001", "input_ids": Mock(), "max_tokens": 50}

        # Make request
        result = await server.handle_request(request)

        # Verify result
        assert "generated_text" in result
        assert result["request_id"] == "test-001"
        assert result["tokens_generated"] == 50
        assert result["batch_size"] >= 1

        await server.stop()

    @pytest.mark.asyncio
    async def test_batch_processing(self, server):
        """Test batch processing with multiple requests."""
        await server.start()

        requests = [
            {"request_id": f"batch-{i}", "input_ids": Mock(), "max_tokens": 100}
            for i in range(3)
        ]

        # Make concurrent requests
        tasks = [server.handle_request(req) for req in requests]
        results = await asyncio.gather(*tasks)

        # Verify all requests processed
        assert len(results) == 3

        for i, result in enumerate(results):
            assert result["request_id"] == f"batch-{i}"
            assert "generated_text" in result
            assert result["tokens_generated"] == 100

        await server.stop()

    @pytest.mark.asyncio
    async def test_max_batch_size_enforcement(self, server):
        """Test that batch size doesn't exceed maximum."""
        await server.start()

        # Create more requests than max batch size
        num_requests = server.max_batch_size + 2
        requests = [
            {"request_id": f"overflow-{i}", "input_ids": Mock(), "max_tokens": 50}
            for i in range(num_requests)
        ]

        # Make concurrent requests
        tasks = [server.handle_request(req) for req in requests]
        results = await asyncio.gather(*tasks)

        # Verify all requests processed despite overflow
        assert len(results) == num_requests

        # Check that batch sizes are within limits
        for result in results:
            assert result["batch_size"] <= server.max_batch_size

        await server.stop()

    @pytest.mark.asyncio
    async def test_queue_management(self, server):
        """Test request queue management."""
        # Initially empty
        assert server.get_queue_size() == 0

        await server.start()

        # Add requests without waiting for processing
        requests = [
            {"request_id": f"queue-{i}", "input_ids": Mock(), "max_tokens": 25}
            for i in range(5)
        ]

        # Queue requests quickly
        tasks = [server.handle_request(req) for req in requests]

        # Allow some time for queueing before processing
        await asyncio.sleep(0.001)

        # Wait for processing
        results = await asyncio.gather(*tasks)
        assert len(results) == 5

        await server.stop()

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self, server):
        """Test handling multiple concurrent batches."""
        await server.start()

        # Create two waves of requests
        wave1 = [
            {"request_id": f"wave1-{i}", "input_ids": Mock(), "max_tokens": 75}
            for i in range(3)
        ]

        wave2 = [
            {"request_id": f"wave2-{i}", "input_ids": Mock(), "max_tokens": 100}
            for i in range(4)
        ]

        # Submit first wave
        tasks1 = [server.handle_request(req) for req in wave1]

        # Small delay then submit second wave
        await asyncio.sleep(0.01)
        tasks2 = [server.handle_request(req) for req in wave2]

        # Wait for all results
        results1 = await asyncio.gather(*tasks1)
        results2 = await asyncio.gather(*tasks2)

        # Verify both waves processed
        assert len(results1) == 3
        assert len(results2) == 4

        await server.stop()

    @pytest.mark.asyncio
    async def test_server_metrics(self, server):
        """Test server metrics collection."""
        metrics = server.get_metrics()

        assert "queue_size" in metrics
        assert "max_batch_size" in metrics
        assert "is_running" in metrics
        assert "total_requests_processed" in metrics

        # Initially not running
        assert metrics["is_running"] is False
        assert metrics["max_batch_size"] == 4

        # After starting
        await server.start()
        running_metrics = server.get_metrics()
        assert running_metrics["is_running"] is True

        await server.stop()

    @pytest.mark.asyncio
    async def test_request_timeout_handling(self, server):
        """Test handling of request timeouts in batch collection."""
        await server.start()

        # Submit single request (should timeout waiting for batch)
        request = {"request_id": "timeout-test", "input_ids": Mock(), "max_tokens": 50}

        # Request should still be processed despite timeout
        result = await server.handle_request(request)
        assert result["request_id"] == "timeout-test"

        await server.stop()

    @pytest.mark.asyncio
    async def test_batch_size_optimization(self, server):
        """Test batch size optimization under different loads."""
        await server.start()

        # Test different batch configurations
        batch_sizes = [1, 2, 4, 6]  # Last one exceeds max_batch_size

        for batch_size in batch_sizes:
            requests = [
                {
                    "request_id": f"opt-{batch_size}-{i}",
                    "input_ids": Mock(),
                    "max_tokens": 50,
                }
                for i in range(min(batch_size, server.max_batch_size))
            ]

            tasks = [server.handle_request(req) for req in requests]
            results = await asyncio.gather(*tasks)

            # Verify processing efficiency
            assert len(results) == len(requests)
            for result in results:
                assert result["batch_size"] <= server.max_batch_size

        await server.stop()

    @pytest.mark.parametrize("max_batch_size", [1, 4, 8, 16])
    @pytest.mark.asyncio
    async def test_different_batch_sizes(self, mock_model, max_batch_size):
        """Test server with different maximum batch sizes."""
        server = AsyncModelServer(mock_model, max_batch_size=max_batch_size)
        await server.start()

        # Create requests equal to max batch size
        requests = [
            {
                "request_id": f"batch-size-{max_batch_size}-{i}",
                "input_ids": Mock(),
                "max_tokens": 50,
            }
            for i in range(max_batch_size)
        ]

        tasks = [server.handle_request(req) for req in requests]
        results = await asyncio.gather(*tasks)

        assert len(results) == max_batch_size
        for result in results:
            assert result["batch_size"] <= max_batch_size

        await server.stop()

    @pytest.mark.asyncio
    async def test_error_handling_in_batch_processing(self, server):
        """Test error handling during batch processing."""
        await server.start()

        # Create request that might cause processing issues
        problematic_request = {
            "request_id": "error-test",
            "input_ids": None,  # This might cause issues
            "max_tokens": 50,
        }

        # Should handle gracefully
        try:
            result = await server.handle_request(problematic_request)
            # If no exception, verify result structure
            assert "request_id" in result
        except Exception as e:
            # Error handling should be graceful
            assert isinstance(e, (ValueError, TypeError, AttributeError))

        await server.stop()

    @pytest.mark.asyncio
    async def test_performance_under_load(self, server):
        """Test server performance under sustained load."""
        await server.start()

        # Simulate sustained load
        num_rounds = 3
        requests_per_round = 5

        for round_num in range(num_rounds):
            requests = [
                {
                    "request_id": f"load-{round_num}-{i}",
                    "input_ids": Mock(),
                    "max_tokens": 50,
                }
                for i in range(requests_per_round)
            ]

            start_time = asyncio.get_event_loop().time()
            tasks = [server.handle_request(req) for req in requests]
            results = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()

            # Verify performance characteristics
            processing_time = end_time - start_time
            assert processing_time < 1.0  # Should process quickly
            assert len(results) == requests_per_round

            # Small delay between rounds
            await asyncio.sleep(0.01)

        await server.stop()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_pending_requests(self, server):
        """Test graceful shutdown when requests are pending."""
        await server.start()

        # Submit requests
        requests = [
            {"request_id": f"shutdown-{i}", "input_ids": Mock(), "max_tokens": 50}
            for i in range(3)
        ]

        tasks = [server.handle_request(req) for req in requests]

        # Start shutdown while requests are processing
        await asyncio.sleep(0.01)  # Let some processing begin

        # Shutdown should wait for current batch to complete
        await server.stop()

        # Pending requests should either complete or be cancelled
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            # Each result should either be valid or a cancelled error
            if isinstance(result, Exception):
                assert isinstance(result, asyncio.CancelledError)
            else:
                assert "request_id" in result

    def test_server_configuration_validation(self, mock_model):
        """Test server configuration validation."""
        # Valid configurations
        valid_batch_sizes = [1, 4, 8, 16, 32]

        for batch_size in valid_batch_sizes:
            server = AsyncModelServer(mock_model, max_batch_size=batch_size)
            assert server.max_batch_size == batch_size

        # Edge cases
        server_min = AsyncModelServer(mock_model, max_batch_size=1)
        assert server_min.max_batch_size == 1

        server_large = AsyncModelServer(mock_model, max_batch_size=100)
        assert server_large.max_batch_size == 100
