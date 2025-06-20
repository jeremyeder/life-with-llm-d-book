#!/usr/bin/env python3
"""
GPU Memory Profiler for LLM Deployments

This script analyzes GPU memory usage patterns in llm-d deployments,
providing detailed insights into memory allocation and utilization.

Usage:
    python gpu-memory-profile.py

Run this script inside a pod with GPU access to analyze memory usage.
"""

import torch
import psutil
import GPUtil

def analyze_gpu_memory():
    # System memory
    print(f"System RAM: {psutil.virtual_memory().percent}% used")
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            
            # Detailed GPU stats
            gpu = GPUtil.getGPUs()[i]
            print(f"GPU Load: {gpu.load * 100:.1f}%")
            print(f"GPU Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
            print(f"GPU Temp: {gpu.temperature}°C")

if __name__ == "__main__":
    analyze_gpu_memory()