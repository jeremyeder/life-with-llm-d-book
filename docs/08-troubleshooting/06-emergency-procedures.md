---
title: Emergency Procedures
description: Critical incident response procedures for llm-d deployments
sidebar_position: 6
---

# Emergency Procedures

This section provides step-by-step emergency response procedures for critical incidents in llm-d deployments.

## Incident Classification

### Severity Levels

#### P0 - Critical (Response: Immediate)

- Complete service outage
- Data loss or corruption
- Security breach
- GPU cluster failure

#### P1 - High (Response: 15 minutes)

- Significant performance degradation (>50% latency increase)
- Partial service outage (>50% requests failing)
- High error rates (>10%)
- Resource exhaustion

#### P2 - Medium (Response: 1 hour)

- Minor performance issues
- Single pod failures
- Non-critical feature unavailable
- Warning-level alerts

#### P3 - Low (Response: Next business day)

- Cosmetic issues
- Documentation problems
- Non-urgent optimizations

## Emergency Response Team

### Roles and Responsibilities

```yaml
# Emergency contacts
emergency_contacts:
  primary_oncall:
    role: "Site Reliability Engineer"
    contact: "+1-555-0123"
    escalation_time: "15 minutes"
  
  secondary_oncall:
    role: "Platform Engineer"
    contact: "+1-555-0124"
    escalation_time: "30 minutes"
  
  escalation_manager:
    role: "Engineering Manager"
    contact: "+1-555-0125"
    escalation_time: "60 minutes"
  
  incident_commander:
    role: "Senior SRE"
    contact: "+1-555-0126"
    escalation_time: "When P0 declared"

# Communication channels
channels:
  incident_room: "#incident-response"
  status_page: "https://status.company.com"
  war_room: "Zoom: https://company.zoom.us/j/emergency"
```

## P0 - Critical Incident Procedures

### Complete Service Outage

**Immediate Actions (0-5 minutes):**

```bash
#!/bin/bash
# P0_service_outage.sh - Execute immediately

echo "=== P0 INCIDENT: COMPLETE SERVICE OUTAGE ==="
echo "Timestamp: $(date)"
echo "Incident ID: INC-$(date +%Y%m%d-%H%M%S)"

# 1. Verify outage scope
echo "1. Checking service status..."
kubectl get llmdeployments -A
kubectl get pods -A -l app.kubernetes.io/name=llm-d

# 2. Check cluster health
echo "2. Checking cluster health..."
kubectl get nodes
kubectl cluster-info

# 3. Check operator status
echo "3. Checking operator status..."
kubectl get pods -n llm-d-system
kubectl logs -n llm-d-system deployment/llm-d-operator --tail=50

# 4. Enable maintenance mode
echo "4. Enabling maintenance mode..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: maintenance-mode
  namespace: default
spec:
  selector:
    app: maintenance-page
  ports:
  - port: 80
    targetPort: 8080
EOF

# 5. Notify stakeholders
echo "5. Sending notifications..."
curl -X POST "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK" \
  -H 'Content-type: application/json' \
  --data '{
    "text": "🚨 P0 INCIDENT: Complete LLM service outage detected",
    "channel": "#incident-response",
    "username": "llm-d-monitor"
  }'
```

**Recovery Actions (5-15 minutes):**

```bash
# Check if it's an operator issue
if kubectl get pods -n llm-d-system | grep -q "0/1.*Running"; then
  echo "Operator unhealthy - restarting..."
  kubectl rollout restart deployment/llm-d-operator -n llm-d-system
  kubectl rollout status deployment/llm-d-operator -n llm-d-system --timeout=300s
fi

# Check for node issues
unhealthy_nodes=$(kubectl get nodes | grep -v Ready | grep -v NAME | wc -l)
if [ $unhealthy_nodes -gt 0 ]; then
  echo "Node issues detected - checking GPU nodes..."
  kubectl get nodes -l nvidia.com/gpu=true
  # May need manual intervention
fi

# Restart all deployments as last resort
echo "Performing full restart of all LLM deployments..."
kubectl rollout restart deployment -l app.kubernetes.io/name=llm-d -A
```

### GPU Cluster Failure

**Immediate Response:**

```bash
#!/bin/bash
# gpu_cluster_failure.sh

echo "=== P0 INCIDENT: GPU CLUSTER FAILURE ==="

# 1. Assess GPU node status
echo "1. Checking GPU nodes..."
kubectl get nodes -l nvidia.com/gpu=true

# 2. Check device plugin
echo "2. Checking NVIDIA device plugin..."
kubectl get ds -n kube-system nvidia-device-plugin-daemonset
kubectl logs -n kube-system -l name=nvidia-device-plugin-ds --tail=50

# 3. Immediate mitigation - Scale to CPU-only
echo "3. Emergency CPU fallback..."
for deployment in $(kubectl get llmdeployments -A -o jsonpath='{.items[*].metadata.name}'); do
  namespace=$(kubectl get llmdeployment $deployment -A -o jsonpath='{.items[0].metadata.namespace}')
  kubectl patch llmdeployment $deployment -n $namespace --type='merge' -p='{
    "spec": {
      "resources": {
        "limits": {"nvidia.com/gpu": "0"},
        "requests": {"nvidia.com/gpu": "0"}
      },
      "model": {
        "quantization": {
          "enabled": true,
          "precision": "int8"
        }
      }
    }
  }'
done

# 4. Contact cloud provider if using managed service
echo "4. Contact cloud provider support..."
echo "AWS: aws support create-case --service-code 'amazon-ec2' --severity-code 'urgent'"
echo "GCP: gcloud support cases create --display-name='GPU cluster failure'"
echo "Azure: az support tickets create --ticket-name 'GPU cluster failure'"
```

## P1 - High Severity Procedures

### High Error Rate Response

```bash
#!/bin/bash
# high_error_rate.sh

echo "=== P1 INCIDENT: HIGH ERROR RATE ==="

# 1. Identify error patterns
echo "1. Analyzing error patterns..."
kubectl logs -A -l app.kubernetes.io/name=llm-d --tail=1000 | \
  grep -i error | \
  awk '{print $NF}' | \
  sort | \
  uniq -c | \
  sort -rn | \
  head -10

# 2. Check resource utilization
echo "2. Checking resource utilization..."
kubectl top nodes
kubectl top pods -A --containers | grep llm-d

# 3. Scale up if resource constrained
echo "3. Emergency scaling..."
for hpa in $(kubectl get hpa -A -o jsonpath='{.items[*].metadata.name}'); do
  namespace=$(kubectl get hpa $hpa -A -o jsonpath='{.items[0].metadata.namespace}')
  current_max=$(kubectl get hpa $hpa -n $namespace -o jsonpath='{.spec.maxReplicas}')
  new_max=$((current_max * 2))
  kubectl patch hpa $hpa -n $namespace --type='merge' -p="{\"spec\":{\"maxReplicas\":$new_max}}"
done

# 4. Implement circuit breaker
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: llm-circuit-breaker
spec:
  host: "*.local"
  trafficPolicy:
    outlierDetection:
      consecutiveGatewayErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
EOF
```

### Memory Pressure Emergency

```bash
#!/bin/bash
# memory_pressure.sh

echo "=== P1 INCIDENT: MEMORY PRESSURE ==="

# 1. Identify memory-hungry pods
echo "1. Finding high-memory pods..."
kubectl top pods -A --sort-by=memory | head -20

# 2. Check for memory leaks
echo "2. Checking for memory leaks..."
for pod in $(kubectl get pods -A -l app.kubernetes.io/name=llm-d -o jsonpath='{.items[*].metadata.name}'); do
  namespace=$(kubectl get pod $pod -A -o jsonpath='{.items[0].metadata.namespace}')
  echo "Pod: $pod"
  kubectl exec -n $namespace $pod -- cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable"
done

# 3. Emergency memory cleanup
echo "3. Emergency memory cleanup..."
# Clear caches
kubectl exec -A -l app.kubernetes.io/name=llm-d -- sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

# 4. Reduce batch sizes
echo "4. Reducing batch sizes..."
for deployment in $(kubectl get llmdeployments -A -o jsonpath='{.items[*].metadata.name}'); do
  namespace=$(kubectl get llmdeployment $deployment -A -o jsonpath='{.items[0].metadata.namespace}')
  kubectl patch llmdeployment $deployment -n $namespace --type='merge' -p='{
    "spec": {
      "serving": {
        "batchSize": 1,
        "maxConcurrency": 2
      }
    }
  }'
done

# 5. Restart highest memory consumers
echo "5. Restarting high memory pods..."
kubectl delete pods -A -l app.kubernetes.io/name=llm-d --field-selector 'status.phase=Running'
```

## Recovery Procedures

### Service Recovery Checklist

```bash
#!/bin/bash
# service_recovery_checklist.sh

echo "=== SERVICE RECOVERY CHECKLIST ==="

# Health checks
checks=(
  "kubectl get llmdeployments -A"
  "kubectl get pods -A -l app.kubernetes.io/name=llm-d"
  "kubectl get nodes -l nvidia.com/gpu=true"
  "kubectl get hpa -A"
)

for check in "${checks[@]}"; do
  echo "Running: $check"
  if $check; then
    echo "✅ PASS"
  else
    echo "❌ FAIL"
  fi
  echo ""
done

# Performance validation
echo "=== PERFORMANCE VALIDATION ==="
kubectl port-forward -n default svc/llm-model-service 8080:8080 &
PF_PID=$!
sleep 5

# Test latency
response_time=$(curl -w "%{time_total}" -o /dev/null -s http://localhost:8080/health)
echo "Health check response time: ${response_time}s"

if (( $(echo "$response_time < 1.0" | bc -l) )); then
  echo "✅ Response time acceptable"
else
  echo "❌ Response time too high"
fi

kill $PF_PID
```

### Data Integrity Verification

```python
#!/usr/bin/env python3
# data_integrity_check.py

import subprocess
import json
import sys

class DataIntegrityChecker:
    def __init__(self):
        self.issues = []
    
    def check_model_files(self):
        """Verify model files are intact"""
        try:
            # Get all model storage PVCs
            result = subprocess.run([
                'kubectl', 'get', 'pvc', '-A', 
                '-l', 'app.kubernetes.io/component=model-storage',
                '-o', 'json'
            ], capture_output=True, text=True)
            
            pvcs = json.loads(result.stdout)
            
            for pvc in pvcs['items']:
                namespace = pvc['metadata']['namespace']
                name = pvc['metadata']['name']
                
                # Mount PVC and check files
                check_cmd = f"""
                kubectl run model-check-{name} -n {namespace} \\
                  --image=busybox --rm -i --restart=Never \\
                  --overrides='{{
                    "spec": {{
                      "volumes": [{{
                        "name": "model-data",
                        "persistentVolumeClaim": {{"claimName": "{name}"}}
                      }}],
                      "containers": [{{
                        "name": "checker",
                        "image": "busybox",
                        "command": ["find", "/models", "-name", "*.bin", "-exec", "ls", "-la", "{{}}", ";"],
                        "volumeMounts": [{{
                          "name": "model-data",
                          "mountPath": "/models"
                        }}]
                      }}]
                    }}
                  }}' -- find /models -name "*.bin" -exec ls -la {{}} ;
                """
                
                result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    self.issues.append(f"Model file check failed for PVC {name} in {namespace}")
        
        except Exception as e:
            self.issues.append(f"Error checking model files: {e}")
    
    def check_configuration(self):
        """Verify configurations are valid"""
        try:
            result = subprocess.run([
                'kubectl', 'get', 'llmdeployments', '-A', '-o', 'json'
            ], capture_output=True, text=True)
            
            deployments = json.loads(result.stdout)
            
            for deployment in deployments['items']:
                name = deployment['metadata']['name']
                namespace = deployment['metadata']['namespace']
                
                # Check if deployment is healthy
                status = deployment.get('status', {})
                if status.get('phase') != 'Ready':
                    self.issues.append(f"Deployment {name} in {namespace} not ready")
        
        except Exception as e:
            self.issues.append(f"Error checking configurations: {e}")
    
    def run_checks(self):
        """Run all integrity checks"""
        print("Running data integrity checks...")
        
        self.check_model_files()
        self.check_configuration()
        
        if self.issues:
            print("❌ Issues found:")
            for issue in self.issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ All integrity checks passed")
            return True

if __name__ == "__main__":
    checker = DataIntegrityChecker()
    success = checker.run_checks()
    sys.exit(0 if success else 1)
```

## Post-Incident Procedures

### Incident Report Template

```markdown
# Incident Report: [INCIDENT_ID]

## Summary
- **Date**: [DATE]
- **Duration**: [START_TIME] - [END_TIME] ([DURATION])
- **Severity**: P[0-3]
- **Impact**: [DESCRIPTION]
- **Root Cause**: [ROOT_CAUSE]

## Timeline
| Time | Action | Owner |
|------|--------|-------|
| 00:00 | Incident detected | Monitoring |
| 00:05 | Initial response | OnCall SRE |
| 00:15 | Escalation | Incident Commander |
| ... | ... | ... |

## Impact Assessment
- **Users Affected**: [NUMBER/PERCENTAGE]
- **Revenue Impact**: [AMOUNT]
- **SLA Breach**: [YES/NO]
- **Data Loss**: [YES/NO]

## Root Cause Analysis
[Detailed analysis of what went wrong]

## Resolution
[Steps taken to resolve the incident]

## Action Items
- [ ] [ACTION_ITEM_1] - Owner: [NAME] - Due: [DATE]
- [ ] [ACTION_ITEM_2] - Owner: [NAME] - Due: [DATE]

## Lessons Learned
[What we learned and how we can improve]
```

### Automated Post-Incident Analysis

```python
#!/usr/bin/env python3
# post_incident_analysis.py

import subprocess
import json
from datetime import datetime, timedelta

class PostIncidentAnalyzer:
    def __init__(self, incident_start: str, incident_end: str):
        self.start_time = datetime.fromisoformat(incident_start)
        self.end_time = datetime.fromisoformat(incident_end)
    
    def analyze_logs(self):
        """Analyze logs during incident window"""
        print("Analyzing logs during incident...")
        
        # Get logs from incident window
        since = f"{int((datetime.now() - self.start_time).total_seconds())}s"
        
        result = subprocess.run([
            'kubectl', 'logs', '-A', '--since', since,
            '-l', 'app.kubernetes.io/name=llm-d'
        ], capture_output=True, text=True)
        
        logs = result.stdout.split('\n')
        
        # Analyze error patterns
        errors = [line for line in logs if 'error' in line.lower()]
        warnings = [line for line in logs if 'warning' in line.lower()]
        
        print(f"Found {len(errors)} errors and {len(warnings)} warnings")
        
        # Top error patterns
        error_patterns = {}
        for error in errors:
            # Simple pattern extraction
            if 'CUDA' in error:
                error_patterns['CUDA'] = error_patterns.get('CUDA', 0) + 1
            elif 'memory' in error.lower():
                error_patterns['Memory'] = error_patterns.get('Memory', 0) + 1
            elif 'network' in error.lower():
                error_patterns['Network'] = error_patterns.get('Network', 0) + 1
        
        print("Top error patterns:")
        for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {count}")
    
    def analyze_metrics(self):
        """Analyze metrics during incident"""
        print("Analyzing metrics during incident...")
        
        # This would integrate with your metrics system
        # Example for Prometheus
        metrics_queries = [
            'llm_request_duration_seconds',
            'llm_gpu_utilization_percent',
            'llm_memory_usage_bytes'
        ]
        
        # Would make actual Prometheus queries here
        print("Metrics analysis would go here...")
    
    def generate_recommendations(self):
        """Generate improvement recommendations"""
        recommendations = [
            "Implement better monitoring alerts",
            "Add circuit breakers",
            "Improve resource allocation",
            "Enhance error handling",
            "Create better runbooks"
        ]
        
        print("Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

# Usage
if __name__ == "__main__":
    analyzer = PostIncidentAnalyzer(
        "2024-01-15T10:00:00Z",
        "2024-01-15T11:30:00Z"
    )
    analyzer.analyze_logs()
    analyzer.analyze_metrics()
    analyzer.generate_recommendations()
```

## Emergency Contacts Quick Reference

```bash
# Save as emergency_contacts.sh
#!/bin/bash

echo "=== EMERGENCY CONTACTS ==="
echo "Primary OnCall: +1-555-0123"
echo "Secondary OnCall: +1-555-0124"
echo "Escalation Manager: +1-555-0125"
echo "Incident Commander: +1-555-0126"
echo ""
echo "=== SLACK CHANNELS ==="
echo "#incident-response"
echo "#llm-d-alerts"
echo "#sre-oncall"
echo ""
echo "=== IMPORTANT URLS ==="
echo "Status Page: https://status.company.com"
echo "Runbooks: https://wiki.company.com/runbooks"
echo "Monitoring: https://grafana.company.com"
echo "Logs: https://kibana.company.com"
```

## Next Steps

- Review [Case Studies](./07-case-studies.md) for real-world incident examples
- Update emergency procedures based on lessons learned
