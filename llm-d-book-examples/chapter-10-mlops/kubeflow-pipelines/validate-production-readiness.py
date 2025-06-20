# Production readiness validation script
# Validates model performance in staging and checks resource capacity
# Used before production deployments to ensure SLO compliance

import argparse
import requests
import time
import json
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect

def validate_staging_performance(
    model_name: str,
    staging_namespace: str,
    prometheus_url: str = "http://prometheus.monitoring.svc.cluster.local:9090"
) -> dict:
    """Validate model performance in staging environment"""
    
    prom = PrometheusConnect(url=prometheus_url)
    
    # Query metrics for the last 24 hours
    metrics = {}
    
    # Availability SLO: 99.9% uptime
    uptime_query = f'avg_over_time(up{{job="llm-model",model="{model_name}",namespace="{staging_namespace}"}}[24h])'
    uptime_result = prom.custom_query(uptime_query)
    metrics['uptime'] = float(uptime_result[0]['value'][1]) if uptime_result else 0
    
    # Latency SLO: P95 < 2 seconds
    latency_query = f'histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket{{model="{model_name}",namespace="{staging_namespace}"}}[24h]))'
    latency_result = prom.custom_query(latency_query)
    metrics['p95_latency'] = float(latency_result[0]['value'][1]) if latency_result else 999
    
    # Error rate SLO: < 1% errors
    error_query = f'rate(llm_requests_failed_total{{model="{model_name}",namespace="{staging_namespace}"}}[24h]) / rate(llm_requests_total{{model="{model_name}",namespace="{staging_namespace}"}}[24h])'
    error_result = prom.custom_query(error_query)
    metrics['error_rate'] = float(error_result[0]['value'][1]) if error_result else 1
    
    # Validate SLOs
    slo_checks = {
        'uptime_slo': metrics['uptime'] >= 0.999,
        'latency_slo': metrics['p95_latency'] <= 2.0,
        'error_rate_slo': metrics['error_rate'] <= 0.01
    }
    
    return {
        'metrics': metrics,
        'slo_checks': slo_checks,
        'all_passed': all(slo_checks.values())
    }

def validate_resource_capacity(
    model_name: str,
    model_version: str,
    target_namespace: str = "production"
) -> dict:
    """Validate sufficient resources for production deployment"""
    
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    
    # Get model resource requirements
    with open(f"models/{model_name}/model.yaml", 'r') as f:
        model_config = yaml.safe_load(f)
    
    required_gpu = model_config['resources']['gpu_count']
    required_memory_gb = model_config['resources']['memory_gb']
    
    # Check available resources in production namespace
    nodes = v1.list_node()
    available_gpu = 0
    available_memory_gb = 0
    
    for node in nodes.items:
        if node.spec.taints:
            # Skip tainted nodes (likely GPU nodes reserved for specific workloads)
            continue
            
        gpu_capacity = node.status.allocatable.get('nvidia.com/gpu', '0')
        memory_capacity = node.status.allocatable.get('memory', '0Gi')
        
        # Simple capacity calculation (would be more sophisticated in practice)
        available_gpu += int(gpu_capacity)
        available_memory_gb += int(memory_capacity.replace('Gi', ''))
    
    capacity_checks = {
        'gpu_available': available_gpu >= required_gpu,
        'memory_available': available_memory_gb >= required_memory_gb
    }
    
    return {
        'required': {
            'gpu': required_gpu,
            'memory_gb': required_memory_gb
        },
        'available': {
            'gpu': available_gpu,
            'memory_gb': available_memory_gb
        },
        'capacity_checks': capacity_checks,
        'sufficient_capacity': all(capacity_checks.values())
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--model-version', required=True)
    parser.add_argument('--staging-namespace', default='staging')
    
    args = parser.parse_args()
    
    print(f"🔍 Validating production readiness for {args.model_name} v{args.model_version}")
    
    # Validate staging performance
    performance_result = validate_staging_performance(
        args.model_name,
        args.staging_namespace
    )
    
    # Validate resource capacity
    capacity_result = validate_resource_capacity(
        args.model_name,
        args.model_version
    )
    
    # Overall validation result
    validation_passed = (
        performance_result['all_passed'] and 
        capacity_result['sufficient_capacity']
    )
    
    print(f"📊 Performance Validation:")
    print(f"   Uptime: {performance_result['metrics']['uptime']:.3f} (>= 0.999: {performance_result['slo_checks']['uptime_slo']})")
    print(f"   P95 Latency: {performance_result['metrics']['p95_latency']:.3f}s (<= 2.0s: {performance_result['slo_checks']['latency_slo']})")
    print(f"   Error Rate: {performance_result['metrics']['error_rate']:.3f} (<= 0.01: {performance_result['slo_checks']['error_rate_slo']})")
    
    print(f"💾 Capacity Validation:")
    print(f"   GPU: {capacity_result['available']['gpu']} available, {capacity_result['required']['gpu']} required")
    print(f"   Memory: {capacity_result['available']['memory_gb']}GB available, {capacity_result['required']['memory_gb']}GB required")
    
    if validation_passed:
        print("✅ Production readiness validation PASSED")
        print("::set-output name=result::passed")
    else:
        print("❌ Production readiness validation FAILED")
        print("::set-output name=result::failed")
        exit(1)

if __name__ == "__main__":
    main()