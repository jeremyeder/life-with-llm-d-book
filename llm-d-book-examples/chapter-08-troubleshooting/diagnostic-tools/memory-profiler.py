#!/usr/bin/env python3
"""
Memory Usage Profiler for Model Inference

This script provides detailed memory usage analysis during model loading
and inference, helping identify memory bottlenecks and leaks.

Usage:
    python memory-profiler.py

Run this script to analyze memory usage patterns during model operations.
"""

import tracemalloc
import torch
import gc

def profile_model_memory():
    tracemalloc.start()
    
    # Your model loading code here
    model = load_model()
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
    # Detailed snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("\nTop 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)
    
    tracemalloc.stop()

# GPU memory tracking
def track_gpu_memory():
    torch.cuda.reset_peak_memory_stats()
    
    # Your inference code here
    output = model(input)
    
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    profile_model_memory()
    track_gpu_memory()