# Runbook: High Inference Latency

## Overview
This runbook covers troubleshooting steps for when LLM-D inference latency exceeds acceptable thresholds.

## Severity: P2 (Medium)
**SLO Impact:** Latency SLO breach (>2s p95)
**Response Time:** 30 minutes

## Investigation Steps

### 1. Verify the Alert
```bash
# Check current latency metrics
kubectl exec -n monitoring prometheus-0 -- \
  promtool query instant \
  'histogram_quantile(0.95, rate(llm_d_request_duration_seconds_bucket[5m]))'

# Check if latency is consistently high
kubectl exec -n monitoring prometheus-0 -- \
  promtool query range \
  'histogram_quantile(0.95, rate(llm_d_request_duration_seconds_bucket[5m]))' \
  --start $(date -d '1 hour ago' --iso-8601) \
  --end $(date --iso-8601) \
  --step 1m
```

### 2. Check System Resources

```bash
# Check GPU utilization
kubectl top pods -n production --sort-by='gpu'

# Check memory usage
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check queue depths
kubectl exec -n monitoring prometheus-0 -- \
  promtool query instant \
  'llm_d_queue_depth'
```

### 3. Analyze Request Patterns

```bash
# Check request rate
kubectl exec -n monitoring prometheus-0 -- \
  promtool query instant \
  'rate(llm_d_requests_total[5m])'

# Check request size distribution
kubectl logs -n production deployment/llm-d-prefill \
  | grep "request_tokens" \
  | tail -50
```

### 4. Check Cache Performance

```bash
# Check cache hit ratio
kubectl exec -n monitoring prometheus-0 -- \
  promtool query instant \
  'rate(llm_d_cache_hits_total[5m]) / rate(llm_d_cache_requests_total[5m])'

# Check cache memory usage
kubectl exec -n production deployment/redis-cluster -- \
  redis-cli info memory
```

## Common Root Causes and Solutions

### High Queue Depth

**Symptoms:** Queue depth > 10, increased latency
**Solution:** Scale up decode pods

```bash
kubectl scale deployment llm-d-decode \
  --replicas=16 \
  -n production
```

### GPU Resource Exhaustion

**Symptoms:** GPU utilization > 95%, OOM errors
**Solution:** Optimize batch size or scale prefill pods

```bash
# Check for OOM events
kubectl get events -n production \
  --field-selector reason=OOMKilled

# Scale prefill pods if needed
kubectl scale deployment llm-d-prefill \
  --replicas=6 \
  -n production
```

### Cache Misses

**Symptoms:** Low cache hit ratio < 60%
**Solution:** Investigate cache configuration

```bash
# Check cache eviction patterns
kubectl exec -n production deployment/redis-cluster -- \
  redis-cli info stats | grep evicted

# Review cache sizing
kubectl exec -n production deployment/redis-cluster -- \
  redis-cli config get maxmemory
```

### Model Loading Issues

**Symptoms:** Cold start latency spikes
**Solution:** Ensure model warmup

```bash
# Check model loading times
kubectl logs -n production deployment/llm-d-prefill \
  | grep "model_load_time"

# Trigger model warmup
curl -X POST http://llm-d-service:8080/v1/warmup \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3-8b"}'
```

## Escalation Criteria

- Latency remains >5s p95 after 30 minutes
- GPU utilization cannot be reduced below 90%
- Queue depth continues growing despite scaling
- Multiple service dependencies affected

## Post-Incident Actions

1. Update capacity planning based on findings
2. Review and update autoscaling policies
3. Consider infrastructure improvements
4. Update monitoring thresholds if needed