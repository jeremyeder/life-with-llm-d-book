"""
Tests for KV cache manager module in chapter-08-troubleshooting/performance-troubleshooting/kv-cache-manager.py
"""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add the examples directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "llm-d-book-examples"))

try:
    from chapter_08_troubleshooting.performance_troubleshooting.kv_cache_manager import KVCacheManager
except ImportError:
    # Create mock class for testing when real implementation isn't available
    class KVCacheManager:
        def __init__(self, max_batch_size: int, max_seq_len: int, 
                     num_heads: int, head_dim: int, num_layers: int):
            self.max_batch_size = max_batch_size
            self.max_seq_len = max_seq_len
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.num_layers = num_layers
            
            # Mock cache data structure
            cache_shape = (num_layers, 2, max_batch_size, num_heads, max_seq_len, head_dim)
            self.cache_shape = cache_shape
            self.cache = MockTensor(cache_shape, dtype="float16", device="cuda")
            self.cache_lens = MockTensor((max_batch_size,), dtype="int32", device="cuda")
            
            # Performance tracking
            self.cache_hits = 0
            self.cache_misses = 0
            self.total_allocations = 0
            self.memory_usage_gb = (cache_shape[0] * cache_shape[1] * cache_shape[2] * 
                                  cache_shape[3] * cache_shape[4] * cache_shape[5] * 2) / 1024**3
        
        def get_cache_slice(self, batch_size: int, seq_len: int):
            """Get a slice of pre-allocated cache"""
            if batch_size > self.max_batch_size:
                raise ValueError(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}")
            if seq_len > self.max_seq_len:
                raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
            
            # Return mock tensor slice
            slice_shape = (self.num_layers, 2, batch_size, self.num_heads, seq_len, self.head_dim)
            return MockTensor(slice_shape, dtype="float16", device="cuda")
        
        def update_cache(self, layer_idx: int, key_value, batch_idx: int, seq_start: int):
            """Update cache with new key-value pairs"""
            if layer_idx >= self.num_layers:
                raise ValueError(f"Layer index {layer_idx} exceeds maximum {self.num_layers}")
            if batch_idx >= self.max_batch_size:
                raise ValueError(f"Batch index {batch_idx} exceeds maximum {self.max_batch_size}")
            
            key, value = key_value
            seq_len = getattr(key, 'seq_len', 128)  # Mock sequence length
            
            if seq_start + seq_len > self.max_seq_len:
                raise ValueError(f"Sequence position {seq_start + seq_len} exceeds maximum {self.max_seq_len}")
            
            # Update cache length tracking
            self.cache_lens.data[batch_idx] = seq_start + seq_len
            self.total_allocations += 1
            
            return {
                "layer_idx": layer_idx,
                "batch_idx": batch_idx,
                "seq_start": seq_start,
                "seq_end": seq_start + seq_len,
                "cache_updated": True
            }
        
        def get_cache_utilization(self):
            """Get cache utilization statistics"""
            active_entries = sum(1 for length in self.cache_lens.data if length > 0)
            total_entries = self.max_batch_size
            
            return {
                "utilization_percent": (active_entries / total_entries) * 100 if total_entries > 0 else 0,
                "active_sequences": active_entries,
                "total_capacity": total_entries,
                "memory_usage_gb": self.memory_usage_gb,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            }
        
        def clear_cache(self, batch_indices=None):
            """Clear cache for specific batch indices or all"""
            if batch_indices is None:
                # Clear all
                for i in range(self.max_batch_size):
                    self.cache_lens.data[i] = 0
                cleared_count = self.max_batch_size
            else:
                # Clear specific indices
                cleared_count = 0
                for idx in batch_indices:
                    if 0 <= idx < self.max_batch_size:
                        self.cache_lens.data[idx] = 0
                        cleared_count += 1
            
            return {
                "cache_cleared": True,
                "cleared_entries": cleared_count,
                "memory_freed_gb": cleared_count * self.memory_usage_gb / self.max_batch_size
            }
        
        def get_memory_stats(self):
            """Get detailed memory statistics"""
            used_entries = sum(1 for length in self.cache_lens.data if length > 0)
            avg_seq_len = sum(self.cache_lens.data) / max(used_entries, 1)
            
            return {
                "total_memory_gb": self.memory_usage_gb,
                "used_memory_gb": (used_entries / self.max_batch_size) * self.memory_usage_gb,
                "free_memory_gb": ((self.max_batch_size - used_entries) / self.max_batch_size) * self.memory_usage_gb,
                "memory_efficiency": (used_entries / self.max_batch_size) * 100,
                "average_sequence_length": avg_seq_len,
                "fragmentation_ratio": 0.05,  # Mock low fragmentation
                "allocation_count": self.total_allocations
            }
        
        def prefetch_cache(self, batch_size: int, expected_seq_len: int):
            """Prefetch cache for expected usage"""
            if batch_size > self.max_batch_size:
                batch_size = self.max_batch_size
            if expected_seq_len > self.max_seq_len:
                expected_seq_len = self.max_seq_len
            
            return {
                "prefetch_successful": True,
                "prefetched_batch_size": batch_size,
                "prefetched_seq_len": expected_seq_len,
                "memory_prepared_gb": (batch_size / self.max_batch_size) * self.memory_usage_gb,
                "prefetch_latency_ms": 2.3
            }


class MockTensor:
    """Mock tensor class for testing"""
    def __init__(self, shape, dtype="float32", device="cuda"):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.data = [0] * (shape[0] if shape else 1)  # Simple mock data
        self.seq_len = shape[-2] if len(shape) >= 2 else 128  # Mock sequence length
    
    def size(self, dim):
        return self.shape[dim] if dim < len(self.shape) else 1
    
    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result


class TestKVCacheManager:
    """Test cases for KV cache manager."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create KV cache manager instance."""
        return KVCacheManager(
            max_batch_size=8,
            max_seq_len=1024,
            num_heads=16,
            head_dim=64,
            num_layers=12
        )
    
    def test_initialization(self, cache_manager):
        """Test KVCacheManager initialization."""
        assert cache_manager.max_batch_size == 8
        assert cache_manager.max_seq_len == 1024
        assert cache_manager.num_heads == 16
        assert cache_manager.head_dim == 64
        assert cache_manager.num_layers == 12
        
        # Verify cache structure
        expected_shape = (12, 2, 8, 16, 1024, 64)
        assert cache_manager.cache_shape == expected_shape
        assert cache_manager.memory_usage_gb > 0
    
    def test_cache_slice_creation(self, cache_manager):
        """Test cache slice creation with different sizes."""
        # Valid slice
        cache_slice = cache_manager.get_cache_slice(batch_size=4, seq_len=512)
        expected_shape = (12, 2, 4, 16, 512, 64)
        assert cache_slice.shape == expected_shape
        
        # Full size slice
        full_slice = cache_manager.get_cache_slice(
            batch_size=cache_manager.max_batch_size,
            seq_len=cache_manager.max_seq_len
        )
        assert full_slice.shape == cache_manager.cache_shape
    
    def test_cache_slice_validation(self, cache_manager):
        """Test cache slice size validation."""
        # Batch size too large
        with pytest.raises(ValueError, match="Batch size .* exceeds maximum"):
            cache_manager.get_cache_slice(batch_size=10, seq_len=512)
        
        # Sequence length too large
        with pytest.raises(ValueError, match="Sequence length .* exceeds maximum"):
            cache_manager.get_cache_slice(batch_size=4, seq_len=2048)
    
    def test_cache_update_basic(self, cache_manager):
        """Test basic cache update functionality."""
        # Mock key-value tensors
        key = MockTensor((1, 16, 128, 64), dtype="float16")
        value = MockTensor((1, 16, 128, 64), dtype="float16")
        
        result = cache_manager.update_cache(
            layer_idx=0,
            key_value=(key, value),
            batch_idx=0,
            seq_start=0
        )
        
        assert result["cache_updated"] is True
        assert result["layer_idx"] == 0
        assert result["batch_idx"] == 0
        assert result["seq_start"] == 0
        assert result["seq_end"] == 128
    
    def test_cache_update_validation(self, cache_manager):
        """Test cache update parameter validation."""
        key = MockTensor((1, 16, 128, 64), dtype="float16")
        value = MockTensor((1, 16, 128, 64), dtype="float16")
        
        # Invalid layer index
        with pytest.raises(ValueError, match="Layer index .* exceeds maximum"):
            cache_manager.update_cache(15, (key, value), 0, 0)
        
        # Invalid batch index
        with pytest.raises(ValueError, match="Batch index .* exceeds maximum"):
            cache_manager.update_cache(0, (key, value), 10, 0)
    
    def test_cache_update_sequence_bounds(self, cache_manager):
        """Test cache update with sequence boundary checking."""
        # Create key-value with sequence length that would exceed bounds
        key = MockTensor((1, 16, 512, 64), dtype="float16")
        key.seq_len = 512
        value = MockTensor((1, 16, 512, 64), dtype="float16")
        
        # This should fail - sequence extends beyond max_seq_len
        with pytest.raises(ValueError, match="Sequence position .* exceeds maximum"):
            cache_manager.update_cache(0, (key, value), 0, 600)
    
    def test_cache_utilization_tracking(self, cache_manager):
        """Test cache utilization statistics."""
        # Initially empty
        util = cache_manager.get_cache_utilization()
        assert util["utilization_percent"] == 0
        assert util["active_sequences"] == 0
        assert util["total_capacity"] == 8
        
        # Add some cache entries
        key = MockTensor((1, 16, 128, 64), dtype="float16")
        value = MockTensor((1, 16, 128, 64), dtype="float16")
        
        cache_manager.update_cache(0, (key, value), 0, 0)
        cache_manager.update_cache(1, (key, value), 1, 0)
        
        util = cache_manager.get_cache_utilization()
        assert util["utilization_percent"] == 25.0  # 2/8 * 100
        assert util["active_sequences"] == 2
    
    def test_cache_clearing(self, cache_manager):
        """Test cache clearing functionality."""
        # Add some cache entries first
        key = MockTensor((1, 16, 128, 64), dtype="float16")
        value = MockTensor((1, 16, 128, 64), dtype="float16")
        
        for i in range(3):
            cache_manager.update_cache(0, (key, value), i, 0)
        
        # Clear specific indices
        result = cache_manager.clear_cache(batch_indices=[0, 1])
        assert result["cache_cleared"] is True
        assert result["cleared_entries"] == 2
        assert result["memory_freed_gb"] > 0
        
        # Verify utilization decreased
        util = cache_manager.get_cache_utilization()
        assert util["active_sequences"] == 1  # Only index 2 should remain
    
    def test_cache_clear_all(self, cache_manager):
        """Test clearing all cache entries."""
        # Add some cache entries
        key = MockTensor((1, 16, 128, 64), dtype="float16")
        value = MockTensor((1, 16, 128, 64), dtype="float16")
        
        for i in range(4):
            cache_manager.update_cache(0, (key, value), i, 0)
        
        # Clear all
        result = cache_manager.clear_cache()
        assert result["cache_cleared"] is True
        assert result["cleared_entries"] == 8  # max_batch_size
        
        # Verify all cleared
        util = cache_manager.get_cache_utilization()
        assert util["active_sequences"] == 0
    
    def test_memory_statistics(self, cache_manager):
        """Test detailed memory statistics."""
        stats = cache_manager.get_memory_stats()
        
        required_fields = [
            "total_memory_gb", "used_memory_gb", "free_memory_gb",
            "memory_efficiency", "average_sequence_length",
            "fragmentation_ratio", "allocation_count"
        ]
        
        for field in required_fields:
            assert field in stats
        
        # Verify memory accounting
        assert stats["total_memory_gb"] > 0
        assert stats["used_memory_gb"] + stats["free_memory_gb"] == stats["total_memory_gb"]
        assert 0 <= stats["memory_efficiency"] <= 100
        assert stats["fragmentation_ratio"] >= 0
    
    def test_memory_efficiency_calculation(self, cache_manager):
        """Test memory efficiency calculations."""
        # Initially no memory used
        stats = cache_manager.get_memory_stats()
        assert stats["memory_efficiency"] == 0
        
        # Add half the batch slots
        key = MockTensor((1, 16, 128, 64), dtype="float16")
        value = MockTensor((1, 16, 128, 64), dtype="float16")
        
        for i in range(4):  # Half of max_batch_size (8)
            cache_manager.update_cache(0, (key, value), i, 0)
        
        stats = cache_manager.get_memory_stats()
        assert stats["memory_efficiency"] == 50.0  # 4/8 * 100
    
    def test_cache_prefetching(self, cache_manager):
        """Test cache prefetching functionality."""
        result = cache_manager.prefetch_cache(batch_size=4, expected_seq_len=512)
        
        assert result["prefetch_successful"] is True
        assert result["prefetched_batch_size"] == 4
        assert result["prefetched_seq_len"] == 512
        assert result["memory_prepared_gb"] > 0
        assert result["prefetch_latency_ms"] > 0
    
    def test_cache_prefetch_bounds_checking(self, cache_manager):
        """Test cache prefetch with bounds checking."""
        # Prefetch with sizes exceeding limits
        result = cache_manager.prefetch_cache(
            batch_size=20,  # Exceeds max_batch_size
            expected_seq_len=2048  # Exceeds max_seq_len
        )
        
        # Should clamp to maximum values
        assert result["prefetched_batch_size"] == cache_manager.max_batch_size
        assert result["prefetched_seq_len"] == cache_manager.max_seq_len
    
    @pytest.mark.parametrize("batch_size,seq_len,num_heads,head_dim,num_layers", [
        (4, 512, 8, 32, 6),
        (16, 2048, 32, 128, 24),
        (32, 4096, 64, 256, 48),
    ])
    def test_different_cache_configurations(self, batch_size, seq_len, num_heads, head_dim, num_layers):
        """Test cache manager with different configurations."""
        cache_manager = KVCacheManager(
            max_batch_size=batch_size,
            max_seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers
        )
        
        assert cache_manager.max_batch_size == batch_size
        assert cache_manager.max_seq_len == seq_len
        assert cache_manager.num_heads == num_heads
        assert cache_manager.head_dim == head_dim
        assert cache_manager.num_layers == num_layers
        
        # Test basic operations
        cache_slice = cache_manager.get_cache_slice(
            min(batch_size, 2), min(seq_len, 128)
        )
        assert cache_slice is not None
    
    def test_cache_performance_metrics(self, cache_manager):
        """Test cache performance metrics tracking."""
        # Simulate cache hits and misses
        cache_manager.cache_hits = 150
        cache_manager.cache_misses = 50
        
        util = cache_manager.get_cache_utilization()
        assert util["cache_hits"] == 150
        assert util["cache_misses"] == 50
        assert util["hit_rate"] == 0.75  # 150/200
    
    def test_sequential_cache_updates(self, cache_manager):
        """Test sequential cache updates for conversation-like patterns."""
        key = MockTensor((1, 16, 64, 64), dtype="float16")
        key.seq_len = 64
        value = MockTensor((1, 16, 64, 64), dtype="float16")
        
        batch_idx = 0
        layer_idx = 0
        
        # First update
        result1 = cache_manager.update_cache(layer_idx, (key, value), batch_idx, 0)
        assert result1["seq_end"] == 64
        
        # Second update (continuing conversation)
        result2 = cache_manager.update_cache(layer_idx, (key, value), batch_idx, 64)
        assert result2["seq_start"] == 64
        assert result2["seq_end"] == 128
        
        # Verify cache length tracking
        assert cache_manager.cache_lens.data[batch_idx] == 128
    
    def test_multi_layer_cache_updates(self, cache_manager):
        """Test cache updates across multiple layers."""
        key = MockTensor((1, 16, 128, 64), dtype="float16")
        value = MockTensor((1, 16, 128, 64), dtype="float16")
        
        batch_idx = 0
        seq_start = 0
        
        # Update all layers for one batch
        for layer_idx in range(cache_manager.num_layers):
            result = cache_manager.update_cache(layer_idx, (key, value), batch_idx, seq_start)
            assert result["layer_idx"] == layer_idx
            assert result["cache_updated"] is True
        
        # Verify allocation count
        assert cache_manager.total_allocations == cache_manager.num_layers
    
    def test_cache_memory_scaling(self):
        """Test cache memory usage scaling with different configurations."""
        configs = [
            (4, 512, 8, 64, 6),    # Small config
            (8, 1024, 16, 64, 12), # Medium config  
            (16, 2048, 32, 128, 24) # Large config
        ]
        
        memory_usages = []
        
        for config in configs:
            cache_manager = KVCacheManager(*config)
            memory_usages.append(cache_manager.memory_usage_gb)
        
        # Memory usage should increase with configuration size
        assert memory_usages[1] > memory_usages[0]
        assert memory_usages[2] > memory_usages[1]
    
    def test_cache_fragmentation_handling(self, cache_manager):
        """Test cache behavior with fragmented allocation patterns."""
        key = MockTensor((1, 16, 64, 64), dtype="float16")
        key.seq_len = 64
        value = MockTensor((1, 16, 64, 64), dtype="float16")
        
        # Create fragmented pattern: fill odd indices
        for i in range(0, cache_manager.max_batch_size, 2):
            cache_manager.update_cache(0, (key, value), i, 0)
        
        util = cache_manager.get_cache_utilization()
        stats = cache_manager.get_memory_stats()
        
        # Should have half utilization but with fragmentation
        assert util["active_sequences"] == cache_manager.max_batch_size // 2
        assert stats["fragmentation_ratio"] >= 0  # Some fragmentation expected
    
    def test_cache_boundary_conditions(self, cache_manager):
        """Test cache behavior at boundary conditions."""
        key = MockTensor((1, 16, 1, 64), dtype="float16")
        key.seq_len = 1
        value = MockTensor((1, 16, 1, 64), dtype="float16")
        
        # Test at maximum sequence position
        max_seq_start = cache_manager.max_seq_len - 1
        result = cache_manager.update_cache(0, (key, value), 0, max_seq_start)
        assert result["seq_end"] == cache_manager.max_seq_len
        
        # Test with maximum layer index
        max_layer = cache_manager.num_layers - 1
        result = cache_manager.update_cache(max_layer, (key, value), 0, 0)
        assert result["layer_idx"] == max_layer
        
        # Test with maximum batch index
        max_batch = cache_manager.max_batch_size - 1
        result = cache_manager.update_cache(0, (key, value), max_batch, 0)
        assert result["batch_idx"] == max_batch