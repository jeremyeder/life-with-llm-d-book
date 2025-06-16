---
title: End-to-End Case Study
description: Complete implementation walkthrough of MLOps workflow for a small team
sidebar_position: 8
---

# End-to-End Case Study: Building MLOps from Scratch

This case study walks through implementing the complete MLOps workflow covered in this chapter, following a small team as they build a production-ready LLM deployment system.

## Team and Context

### Meet the Team

**TechStart AI** - 3-person startup building AI-powered customer support

- **Sarah** (CTO/ML Engineer): Former FAANG ML engineer, responsible for model selection and optimization
- **Mike** (Platform Engineer/SRE): Infrastructure and operations background, handles deployment and monitoring  
- **Alex** (Product Lead): Business requirements and user experience, manages priorities

### Business Requirements

- **Primary Use Case**: Customer support chatbot for SaaS platform
- **Expected Load**: 1,000 concurrent users, 10,000 requests/day initially
- **Budget Constraints**: Limited cloud spend, need cost-effective scaling
- **Compliance**: SOC 2 Type II required for enterprise customers
- **Timeline**: 8 weeks to production deployment

### Technical Constraints

- **Infrastructure**: CoreWeave GPU cloud for cost-effectiveness
- **Team Capacity**: Part-time ML focus, need high automation
- **Expertise**: Strong on traditional software, learning LLM operations
- **Scaling**: Start small, plan for 10x growth

## Implementation Timeline

### Week 1-2: Foundation Setup

#### Day 1: Infrastructure Bootstrap

Mike sets up the foundational infrastructure:

```bash
# Initial cluster setup on CoreWeave
cat <<EOF > cluster-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-config
data:
  cluster.yaml: |
    cluster:
      name: techstart-ai-prod
      region: ord1
      gpu_nodes:
        node_pools:
        - name: inference-pool
          machine_type: gpu-a40-2
          min_nodes: 2
          max_nodes: 10
          gpu_type: "A40"
          gpu_count: 2
      
      storage:
        s3_compatible:
          endpoint: "object.ord1.coreweave.com"
          bucket: "techstart-models"
          region: "ord1"
EOF

# Deploy essential components
kubectl apply -f https://github.com/kubeflow/kubeflow/releases/download/v1.7.0/kubeflow-manifests.yaml
kubectl apply -f https://github.com/llm-d/llm-d/releases/latest/download/crds.yaml

# Install Istio for traffic management
curl -L https://istio.io/downloadIstio | sh -
istioctl install --set values.defaultRevision=default
```

#### Day 3: CI/CD Pipeline Setup

Sarah creates the GitHub Actions workflow:

```yaml
# .github/workflows/model-deployment.yml
name: Model Deployment Pipeline

on:
  push:
    branches: [main]
    paths: ['models/**']
  pull_request:
    paths: ['models/**']

env:
  KUBEFLOW_ENDPOINT: ${{ secrets.KUBEFLOW_ENDPOINT }}
  MODEL_REGISTRY_BUCKET: "techstart-models"

jobs:
  model-validation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install torch transformers pytest
        pip install -r requirements.txt
    
    - name: Run model tests
      run: |
        pytest tests/test_model_functionality.py -v
        python scripts/benchmark_model.py --model-path models/llama-3.1-8b
    
    - name: Security scan
      run: |
        pip install safety
        safety check
        
  deploy-staging:
    needs: model-validation
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to staging
      run: |
        python scripts/deploy_model.py \
          --model-name llama-3.1-8b \
          --environment staging \
          --replicas 1
```

#### Day 5: Model Registry Setup

The team configures Kubeflow model registry:

```python
# scripts/setup_model_registry.py
from kubeflow.model_registry import ModelRegistry
import boto3

def setup_model_registry():
    """Initialize model registry with S3 backend"""
    
    # Configure S3 client for CoreWeave
    s3_client = boto3.client(
        's3',
        endpoint_url='https://object.ord1.coreweave.com',
        aws_access_key_id=os.environ['COREWEAVE_ACCESS_KEY'],
        aws_secret_access_key=os.environ['COREWEAVE_SECRET_KEY'],
        region_name='ord1'
    )
    
    # Create bucket if not exists
    try:
        s3_client.create_bucket(Bucket='techstart-models')
        print("✅ Model storage bucket created")
    except:
        print("✅ Model storage bucket already exists")
    
    # Initialize registry
    registry = ModelRegistry(
        server_address="model-registry.kubeflow.svc.cluster.local:8080",
        s3_config={
            "endpoint": "https://object.ord1.coreweave.com",
            "bucket": "techstart-models",
            "region": "ord1"
        }
    )
    
    print("✅ Model registry configured")
    return registry

if __name__ == "__main__":
    setup_model_registry()
```

### Week 3-4: Model Development and Testing

#### Model Selection Process

Sarah evaluates different models:

```python
# experiments/model_evaluation.py
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelEvaluator:
    def __init__(self):
        self.test_prompts = [
            "Hello, I need help with my account login.",
            "How do I cancel my subscription?",
            "What are your pricing plans?",
            "I'm having trouble with the API integration."
        ]
    
    def evaluate_model(self, model_name: str, model_path: str):
        """Evaluate model for customer support use case"""
        
        print(f"🔍 Evaluating {model_name}")
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        results = {
            "model_name": model_name,
            "latency_ms": [],
            "memory_usage_gb": 0,
            "response_quality": []
        }
        
        # Test each prompt
        for prompt in self.test_prompts:
            start_time = time.time()
            
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )
            
            end_time = time.time()
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Record metrics
            latency = (end_time - start_time) * 1000
            results["latency_ms"].append(latency)
            
            # Simple quality assessment
            quality_score = self._assess_response_quality(prompt, response)
            results["response_quality"].append(quality_score)
        
        # Memory usage
        if torch.cuda.is_available():
            results["memory_usage_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        
        # Calculate averages
        results["avg_latency_ms"] = sum(results["latency_ms"]) / len(results["latency_ms"])
        results["avg_quality_score"] = sum(results["response_quality"]) / len(results["response_quality"])
        
        return results
    
    def _assess_response_quality(self, prompt: str, response: str) -> float:
        """Simple response quality assessment"""
        
        # Basic criteria
        score = 0.0
        
        # Response should be longer than prompt
        if len(response) > len(prompt):
            score += 0.3
        
        # Should be helpful (contains relevant keywords)
        helpful_keywords = ["help", "support", "assist", "resolve", "contact"]
        if any(keyword in response.lower() for keyword in helpful_keywords):
            score += 0.4
        
        # Should be professional (no offensive content)
        offensive_keywords = ["hate", "stupid", "idiot"]
        if not any(keyword in response.lower() for keyword in offensive_keywords):
            score += 0.3
        
        return min(score, 1.0)

# Evaluation results
def run_model_comparison():
    """Compare different model options"""
    
    evaluator = ModelEvaluator()
    
    models_to_test = [
        ("Llama4-7B", "meta-llama/Llama-2-7b-chat-hf"),
        ("Llama4-13B", "meta-llama/Llama-2-13b-chat-hf")
    ]
    
    results = []
    for model_name, model_path in models_to_test:
        result = evaluator.evaluate_model(model_name, model_path)
        results.append(result)
    
    # Decision matrix
    print("\n📊 Model Comparison Results:")
    print(f"{'Model':<15} {'Latency (ms)':<12} {'Memory (GB)':<12} {'Quality':<10} {'Cost Score':<10}")
    print("-" * 65)
    
    for result in results:
        cost_score = 10 - (result["memory_usage_gb"] / 2)  # Simple cost calculation
        print(f"{result['model_name']:<15} {result['avg_latency_ms']:<12.0f} {result['memory_usage_gb']:<12.1f} {result['avg_quality_score']:<10.2f} {cost_score:<10.1f}")
    
    # Team decision: Llama4-7B for cost-effectiveness
    print("\n🎯 Decision: Llama4-7B selected for production deployment")
    print("   Rationale: Best cost/performance ratio for startup budget")

if __name__ == "__main__":
    run_model_comparison()
```

### Week 5-6: Production Deployment

#### Progressive Deployment Implementation

Mike implements the progressive deployment:

```yaml
# deployments/llama-3.1-8b-production.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: llama-3.1-8b-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/techstart-ai/mlops-config
    targetRevision: main
    path: applications/llama-3.1-8b/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
---
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b
  namespace: production
spec:
  model:
    name: llama-3.1-8b
    source:
      modelUri: s3://techstart-models/llama-3.1-8b/v1.0.0
  
  replicas: 3
  
  resources:
    requests:
      nvidia.com/gpu: "1"
      memory: "16Gi"
    limits:
      nvidia.com/gpu: "1" 
      memory: "20Gi"
  
  serving:
    protocol: http
    port: 8080
    batchSize: 4
  
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetGPUUtilization: 70
  
  nodeSelector:
    node-pool: "inference-pool"
```

#### Monitoring Setup

Alex sets up business metrics tracking:

```python
# monitoring/business_metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import asyncio

class BusinessMetricsCollector:
    def __init__(self):
        # Customer satisfaction metrics
        self.customer_satisfaction = Gauge(
            'customer_support_satisfaction_score',
            'Customer satisfaction score (1-5)',
            ['conversation_id']
        )
        
        # Response quality metrics
        self.response_quality = Histogram(
            'support_response_quality_score',
            'Quality score of support responses',
            buckets=[0.2, 0.4, 0.6, 0.8, 1.0]
        )
        
        # Business impact metrics
        self.resolution_rate = Counter(
            'support_issues_resolved_total',
            'Total number of support issues resolved',
            ['category', 'resolution_type']
        )
        
        # Cost metrics
        self.cost_per_interaction = Gauge(
            'support_cost_per_interaction_dollars',
            'Cost per customer interaction in dollars'
        )
        
        # User engagement
        self.active_conversations = Gauge(
            'support_active_conversations',
            'Number of active support conversations'
        )
    
    def record_interaction(self, conversation_id: str, quality_score: float, 
                          category: str, resolved: bool, cost: float):
        """Record a customer support interaction"""
        
        # Update metrics
        self.response_quality.observe(quality_score)
        
        if resolved:
            self.resolution_rate.labels(
                category=category,
                resolution_type='automated'
            ).inc()
        
        self.cost_per_interaction.set(cost)
        
        print(f"📊 Recorded interaction: {conversation_id}, quality: {quality_score:.2f}")

# Integration with application
class CustomerSupportBot:
    def __init__(self):
        self.metrics = BusinessMetricsCollector()
        self.model_endpoint = "http://llama-3.1-8b-service.production.svc.cluster.local:8080"
    
    async def handle_customer_query(self, user_id: str, query: str) -> dict:
        """Handle customer support query"""
        
        conversation_id = f"conv_{user_id}_{int(time.time())}"
        
        # Generate response
        response = await self._generate_response(query)
        
        # Assess response quality
        quality_score = self._assess_response_quality(query, response)
        
        # Determine if issue was resolved
        resolved = self._is_issue_resolved(query, response)
        
        # Calculate cost (simplified)
        cost = 0.05  # $0.05 per interaction
        
        # Record business metrics
        self.metrics.record_interaction(
            conversation_id=conversation_id,
            quality_score=quality_score,
            category=self._categorize_query(query),
            resolved=resolved,
            cost=cost
        )
        
        return {
            "conversation_id": conversation_id,
            "response": response,
            "quality_score": quality_score,
            "resolved": resolved
        }

if __name__ == "__main__":
    # Start metrics server
    start_http_server(8000)
    
    bot = CustomerSupportBot()
    print("✅ Customer support bot with business metrics started")
```

### Week 7-8: Optimization and Monitoring

#### A/B Testing Implementation

Sarah implements A/B testing for model optimization:

```python
# ab_testing/customer_support_experiment.py
from experiments.experiment_manager import ExperimentManager
import asyncio

async def run_response_length_experiment():
    """A/B test different response lengths for customer satisfaction"""
    
    experiment = Experiment(
        id="response-length-optimization",
        name="Optimize Response Length for Customer Satisfaction", 
        description="Test short vs detailed responses",
        traffic_split={
            "short-responses": 50,  # max_tokens=50
            "detailed-responses": 50  # max_tokens=150
        },
        target_metrics={
            "customer_satisfaction": 4.0,  # Target 4.0/5.0
            "resolution_rate": 0.8,       # 80% resolution rate
            "avg_response_time": 1500      # 1.5s response time
        },
        duration_hours=72,
        success_criteria={
            "satisfaction_improvement": 0.2,  # 20% improvement
            "resolution_rate_maintained": 0.75
        },
        rollback_criteria={
            "satisfaction_drop": 0.1,
            "resolution_rate_drop": 0.7
        }
    )
    
    manager = ExperimentManager()
    
    # Create Istio configuration for experiment
    istio_config = {
        "apiVersion": "networking.istio.io/v1beta1",
        "kind": "VirtualService",
        "metadata": {
            "name": "support-bot-experiment",
            "namespace": "production"
        },
        "spec": {
            "hosts": ["support-api.techstart.ai"],
            "http": [{
                "match": [{
                    "headers": {
                        "x-experiment-id": {
                            "exact": "response-length-optimization"
                        }
                    }
                }],
                "route": [{
                    "destination": {
                        "host": "llama-3.1-8b-service",
                        "subset": "short-responses"
                    },
                    "weight": 50,
                    "headers": {
                        "request": {
                            "set": {
                                "x-max-tokens": "50"
                            }
                        }
                    }
                }, {
                    "destination": {
                        "host": "llama-3.1-8b-service", 
                        "subset": "detailed-responses"
                    },
                    "weight": 50,
                    "headers": {
                        "request": {
                            "set": {
                                "x-max-tokens": "150"
                            }
                        }
                    }
                }]
            }]
        }
    }
    
    # Start experiment
    success = manager.create_experiment(experiment)
    if success:
        print("✅ A/B test started successfully")
        
        # Monitor for 3 days
        results = await manager.monitor_experiment(experiment)
        print(f"📊 Experiment results: {results}")
    
    return results

# Run the experiment
if __name__ == "__main__":
    asyncio.run(run_response_length_experiment())
```

## Results and Lessons Learned

### Week 8: Production Results

After 8 weeks, TechStart AI achieved their goals:

#### Key Metrics Achieved

```python
# Week 8 production metrics summary
production_metrics = {
    "performance": {
        "avg_response_time_ms": 1200,  # Target: <2000ms
        "p95_response_time_ms": 1800,
        "availability": 0.997,         # Target: >99.5%
        "error_rate": 0.008           # Target: <1%
    },
    
    "business": {
        "customer_satisfaction": 4.1,  # Target: >4.0
        "issue_resolution_rate": 0.82, # Target: >80%
        "cost_per_interaction": 0.03,  # Target: <$0.05
        "daily_interactions": 8500     # Growing from 1000
    },
    
    "operational": {
        "deployment_success_rate": 0.95,    # Target: >95%
        "mean_time_to_recovery": 180,       # 3 minutes
        "change_failure_rate": 0.05,        # Target: <10%
        "deployment_frequency": "daily"      # Target: daily releases
    }
}

print("🎯 Production Metrics Summary:")
for category, metrics in production_metrics.items():
    print(f"\n{category.upper()}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
```

### Lessons Learned

#### What Worked Well

1. **Kubeflow + llm-d Integration**: Seamless model lifecycle management
2. **Progressive Deployment**: Caught issues early with 7B → 13B scaling
3. **CoreWeave Partnership**: Cost-effective GPU access for startup budget
4. **Automated Testing**: Prevented 3 potential production issues
5. **Business Metrics**: Clear ROI demonstration for stakeholders

#### Challenges Overcome

1. **GPU Resource Management**: Initial over-provisioning, solved with autoscaling
2. **Model Loading Times**: Implemented preloading and warm-up strategies  
3. **Cost Control**: Implemented scheduling to scale down during off-hours
4. **Monitoring Complexity**: Started simple, added sophistication gradually

#### Key Success Factors

```python
# Success factors for small team MLOps
success_factors = {
    "automation_first": {
        "description": "Automate everything possible from day 1",
        "impact": "Reduced manual work by 80%"
    },
    
    "start_simple": {
        "description": "Begin with single model, single environment",
        "impact": "Faster time to value, easier debugging"
    },
    
    "monitor_business_metrics": {
        "description": "Track customer satisfaction, not just technical metrics",
        "impact": "Clear business value demonstration"
    },
    
    "leverage_managed_services": {
        "description": "Use Kubeflow, CoreWeave instead of building from scratch",
        "impact": "3-month development time saved"
    },
    
    "iterative_improvement": {
        "description": "Weekly retrospectives and continuous optimization",
        "impact": "40% performance improvement over 8 weeks"
    }
}
```

### Scaling Plan

The team's plan for scaling to 10x load:

```yaml
# scaling_plan.yaml
scaling_phases:
  phase_1: # Current state
    capacity: "10K interactions/day"
    infrastructure: "3 A40 GPUs"
    cost: "$500/month"
    
  phase_2: # 3-month target
    capacity: "50K interactions/day"
    infrastructure: "Llama4-13B + auto-scaling"
    cost: "$2,000/month"
    improvements:
    - Model quantization for efficiency
    - Multi-region deployment
    - Advanced caching strategies
    
  phase_3: # 6-month target
    capacity: "100K interactions/day"
    infrastructure: "Multi-tier model serving"
    cost: "$4,000/month"
    improvements:
    - Llama4-70B for complex queries
    - Intelligent routing
    - Edge deployment for latency
```

## Conclusion

This case study demonstrates that a small team can successfully implement enterprise-grade MLOps workflows by:

1. **Leveraging the right tools**: Kubeflow, llm-d, and cloud partnerships
2. **Starting simple and scaling gradually**: Single model → multi-model progression
3. **Automating from day one**: CI/CD, testing, and deployment automation
4. **Monitoring what matters**: Business metrics alongside technical metrics
5. **Continuous improvement**: A/B testing and iterative optimization

The result: a production-ready LLM deployment system that scales with business growth while maintaining high reliability and cost-effectiveness.

## Next Steps

With Chapter 10 complete, teams can now implement comprehensive MLOps workflows for their LLM deployments. The next logical chapters would cover:

- [Chapter 11: Cost Optimization](../11-cost-optimization.md) - Advanced cost management strategies
- [Chapter 12: MLOps for SREs](../12-mlops-for-sres.md) - Day-to-day operational workflows
