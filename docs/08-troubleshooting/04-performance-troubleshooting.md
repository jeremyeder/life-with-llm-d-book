---
title: Performance Troubleshooting
description: Detailed guide for diagnosing and resolving performance issues
sidebar_position: 4
---

# Performance Troubleshooting

This section focuses on identifying and resolving performance bottlenecks in llm-d deployments, from inference latency to throughput optimization.

## Performance Metrics Overview

### Key Performance Indicators

Understanding which metrics to monitor:

```yaml
# Critical metrics for LLM performance
metrics:
  latency:
    - first_token_latency    # Time to first token (TTFT)
    - completion_latency      # Total request time
    - p50, p95, p99 latencies # Percentile measurements
  
  throughput:
    - requests_per_second     # Overall RPS
    - tokens_per_second       # Generation speed
    - batch_efficiency        # Batching effectiveness
  
  resource_utilization:
    - gpu_utilization         # GPU compute usage
    - gpu_memory_usage        # VRAM utilization
    - cpu_usage              # CPU percentage
    - memory_usage           # System RAM
  
  quality:
    - error_rate             # Failed requests
    - timeout_rate           # Timed out requests
    - queue_depth            # Pending requests
```

### Baseline Establishment

```bash
# Performance baseline script
#!/bin/bash

# Collect baseline metrics
echo "Collecting 5-minute baseline..."
kubectl exec -n <namespace> <pod> -- \
  curl -s localhost:9090/metrics > baseline_start.txt

sleep 300

kubectl exec -n <namespace> <pod> -- \
  curl -s localhost:9090/metrics > baseline_end.txt

# Analyze differences
python3 analyze_metrics.py baseline_start.txt baseline_end.txt
```

## Latency Optimization

### First Token Latency (TTFT)

Optimizing time to first token:

```yaml
# Model configuration for TTFT optimization
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
spec:
  model:
    # Enable model quantization
    quantization:
      enabled: true
      precision: "int8"
    
    # Optimize loading
    loading:
      preload: true
      pinMemory: true
      warmupSamples: 50
    
    # KV cache optimization
    cache:
      enabled: true
      maxSize: "8Gi"
      preallocation: true
```

### Inference Optimization

```python
# inference_optimizer.py
import torch
from torch.profiler import profile, ProfilerActivity

class InferenceOptimizer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # Enable optimizations
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
    
    def optimize_with_compile(self):
        """Use torch.compile for optimization"""
        self.model = torch.compile(
            self.model,
            mode="reduce-overhead",
            fullgraph=True
        )
    
    def profile_inference(self, input_batch):
        """Profile inference to find bottlenecks"""
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                output = self.model(input_batch)
        
        # Print profiling results
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # Export for visualization
        prof.export_chrome_trace("inference_trace.json")
        
        return output
```

### Batching Strategies

```yaml
# Dynamic batching configuration
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
spec:
  serving:
    batching:
      enabled: true
      maxBatchSize: 32
      maxLatencyMs: 50
      # Continuous batching for LLMs
      continuous:
        enabled: true
        windowSizeMs: 10
```

## Throughput Optimization

### Request Handling

```python
# async_request_handler.py
import asyncio
from aiohttp import web
import torch
from typing import List, Dict

class AsyncModelServer:
    def __init__(self, model, max_batch_size=16):
        self.model = model
        self.max_batch_size = max_batch_size
        self.request_queue = asyncio.Queue()
        self.batch_processor_task = None
    
    async def start(self):
        """Start the batch processor"""
        self.batch_processor_task = asyncio.create_task(self.batch_processor())
    
    async def batch_processor(self):
        """Process requests in batches"""
        while True:
            batch = []
            futures = []
            
            # Collect requests up to max_batch_size
            try:
                while len(batch) < self.max_batch_size:
                    request, future = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=0.01  # 10ms window
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
                    future.set_result(result)
    
    async def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of requests"""
        # Prepare inputs
        input_ids = torch.stack([req['input_ids'] for req in batch])
        
        # Run inference
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=batch[0]['max_tokens'],
                do_sample=True,
                temperature=0.7
            )
        
        # Format results
        results = []
        for i, output in enumerate(outputs):
            results.append({
                'generated_text': self.tokenizer.decode(output),
                'request_id': batch[i]['request_id']
            })
        
        return results
```

### Horizontal Scaling

```yaml
# HPA configuration for throughput
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-model-hpa
spec:
  scaleTargetRef:
    apiVersion: inference.llm-d.io/v1alpha1
    kind: LLMDeployment
    name: gpt-model
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

## GPU Performance

### GPU Memory Optimization

```python
# gpu_memory_optimizer.py
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
```

### Multi-GPU Optimization

```yaml
# Multi-GPU deployment configuration
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
spec:
  model:
    parallelism:
      tensor: 2      # Tensor parallelism
      pipeline: 1    # Pipeline parallelism
      data: 4        # Data parallelism
  
  resources:
    limits:
      nvidia.com/gpu: "8"
  
  # GPU placement strategy
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - llm-model
        topologyKey: kubernetes.io/hostname
```

## Network Performance

### Load Balancer Optimization

```yaml
# Service configuration for performance
apiVersion: v1
kind: Service
metadata:
  name: llm-model-service
  annotations:
    # AWS NLB optimizations
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
    # Connection pooling
    service.beta.kubernetes.io/aws-load-balancer-proxy-protocol: "*"
spec:
  type: LoadBalancer
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800
  ports:
  - port: 8080
    targetPort: 8080
    protocol: TCP
    nodePort: 30080
```

### gRPC Performance

```python
# grpc_server_optimized.py
import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc

class OptimizedInferenceService(inference_pb2_grpc.InferenceServicer):
    def __init__(self, model, max_workers=10):
        self.model = model
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
    
    def Predict(self, request, context):
        # Set compression
        context.set_compression(grpc.Compression.Gzip)
        
        # Process request
        result = self.model.predict(request.input)
        
        return inference_pb2.PredictResponse(output=result)

def serve():
    # Server options for performance
    options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ('grpc.keepalive_time_ms', 10000),
        ('grpc.keepalive_timeout_ms', 5000),
        ('grpc.keepalive_permit_without_calls', True),
        ('grpc.http2.max_pings_without_data', 0),
        ('grpc.http2.min_time_between_pings_ms', 10000),
    ]
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=100),
        options=options
    )
    
    inference_pb2_grpc.add_InferenceServicer_to_server(
        OptimizedInferenceService(model),
        server
    )
    
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
```

## Caching Strategies

### Model Cache

```yaml
# Redis-based model cache
apiVersion: v1
kind: ConfigMap
metadata:
  name: cache-config
data:
  cache.yaml: |
    redis:
      host: redis-master
      port: 6379
      db: 0
      ttl: 3600
    
    cache_keys:
      - model_outputs
      - embeddings
      - attention_states
---
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
spec:
  cache:
    enabled: true
    backend: redis
    configMap:
      name: cache-config
      key: cache.yaml
```

### KV Cache Optimization

```python
# kv_cache_manager.py
import torch
from typing import Optional, Tuple

class KVCacheManager:
    def __init__(self, max_batch_size: int, max_seq_len: int, 
                 num_heads: int, head_dim: int, num_layers: int):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        
        # Pre-allocate cache
        cache_shape = (num_layers, 2, max_batch_size, num_heads, max_seq_len, head_dim)
        self.cache = torch.zeros(cache_shape, dtype=torch.float16, device='cuda')
        self.cache_lens = torch.zeros(max_batch_size, dtype=torch.int32, device='cuda')
    
    def get_cache_slice(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Get a slice of pre-allocated cache"""
        return self.cache[:, :, :batch_size, :, :seq_len, :]
    
    def update_cache(self, layer_idx: int, key_value: Tuple[torch.Tensor, torch.Tensor],
                     batch_idx: int, seq_start: int):
        """Update cache with new key-value pairs"""
        key, value = key_value
        seq_len = key.size(2)
        
        self.cache[layer_idx, 0, batch_idx, :, seq_start:seq_start+seq_len, :] = key
        self.cache[layer_idx, 1, batch_idx, :, seq_start:seq_start+seq_len, :] = value
        self.cache_lens[batch_idx] = seq_start + seq_len
```

## Performance Testing

### Load Testing Framework

```python
# load_test.py
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
```

### Performance Monitoring Dashboard

```yaml
# Grafana dashboard for LLM performance
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-performance-dashboard
data:
  dashboard.json: |
    {
      "dashboard": {
        "title": "LLM Performance Monitoring",
        "panels": [
          {
            "title": "Request Latency",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))"
              }
            ]
          },
          {
            "title": "GPU Utilization",
            "targets": [
              {
                "expr": "avg(llm_gpu_utilization_percent) by (pod)"
              }
            ]
          },
          {
            "title": "Throughput",
            "targets": [
              {
                "expr": "sum(rate(llm_requests_total[1m]))"
              }
            ]
          },
          {
            "title": "Token Generation Rate",
            "targets": [
              {
                "expr": "sum(rate(llm_tokens_generated_total[1m]))"
              }
            ]
          }
        ]
      }
    }
```

## Optimization Checklist

### Quick Wins

- [ ] Enable model quantization (int8/int4)
- [ ] Implement request batching
- [ ] Configure KV cache appropriately
- [ ] Enable GPU memory pooling
- [ ] Use continuous batching for LLMs
- [ ] Optimize batch sizes
- [ ] Enable model compilation (torch.compile)
- [ ] Configure proper resource limits

### Advanced Optimizations

- [ ] Implement tensor parallelism
- [ ] Use FlashAttention/PagedAttention
- [ ] Deploy with vLLM or TGI
- [ ] Implement speculative decoding
- [ ] Use mixed precision (fp16/bf16)
- [ ] Optimize network stack
- [ ] Implement result caching
- [ ] Use dedicated inference GPUs

## Next Steps

- Review [Error Patterns](./05-error-patterns.md) for performance-related errors
- Check [Emergency Procedures](./06-emergency-procedures.md) for performance crisis handling
- See [Case Studies](./07-case-studies.md) for real-world optimization examples
