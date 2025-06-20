#!/usr/bin/env python3
"""
GPU Memory Optimizer for Large Language Models

This script provides utilities for optimizing GPU memory usage in LLM deployments,
including memory profiling, cleanup, and efficient allocation strategies.

Usage:
    python gpu-memory-optimizer.py

Run this script to optimize GPU memory usage and profile memory consumption.
"""

import torch
import gc

class GPUMemoryOptimizer:
    @staticmethod
    def optimize_memory():
        """Optimize GPU memory usage"""
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Enable memory efficient attention
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    @staticmethod
    def profile_memory_usage(model, input_batch):
        """Profile GPU memory usage"""
        torch.cuda.reset_peak_memory_stats()
        
        # Forward pass
        with torch.no_grad():
            output = model(input_batch)
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"GPU Memory Stats:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Peak: {peak:.2f} GB")
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'peak_gb': peak
        }

if __name__ == "__main__":
    # Example usage
    optimizer = GPUMemoryOptimizer()
    optimizer.optimize_memory()
    
    # Profile memory usage (replace with your model and input)
    # memory_stats = optimizer.profile_memory_usage(model, input_batch)
    # print(memory_stats)