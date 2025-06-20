# A/B Testing Experiment Manager
# Manages creation, monitoring, and cleanup of A/B testing experiments
# Integrates with Istio for traffic management and Kubernetes for deployment

from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
import kubernetes
from kubernetes import client, config

@dataclass
class Experiment:
    id: str
    name: str
    description: str
    traffic_split: Dict[str, int]  # variant -> percentage
    target_metrics: Dict[str, float]
    duration_hours: int
    success_criteria: Dict[str, float]
    rollback_criteria: Dict[str, float]

class ExperimentManager:
    def __init__(self, namespace: str = "production"):
        config.load_incluster_config()
        self.k8s_client = client.CustomObjectsApi()
        self.namespace = namespace
        
    def create_experiment(self, experiment: Experiment) -> bool:
        """Create A/B testing experiment with Istio configuration"""
        
        # Generate VirtualService for experiment
        virtual_service = self._generate_virtual_service(experiment)
        
        # Apply configuration
        try:
            self.k8s_client.create_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="virtualservices",
                body=virtual_service
            )
            
            print(f"✅ Experiment {experiment.id} created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create experiment: {e}")
            return False
    
    def _generate_virtual_service(self, experiment: Experiment) -> Dict:
        """Generate Istio VirtualService for A/B test"""
        
        # Build routing rules based on traffic split
        routes = []
        for variant, weight in experiment.traffic_split.items():
            route = {
                "destination": {
                    "host": f"llama-3.1-7b-service",  # Base service
                    "subset": variant
                },
                "weight": weight,
                "headers": {
                    "response": {
                        "set": {
                            "x-experiment-id": experiment.id,
                            "x-variant": variant
                        }
                    }
                }
            }
            routes.append(route)
        
        virtual_service = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": f"experiment-{experiment.id}",
                "namespace": self.namespace,
                "labels": {
                    "experiment-id": experiment.id,
                    "managed-by": "experiment-manager"
                }
            },
            "spec": {
                "hosts": ["api.llm-platform.com"],
                "gateways": ["llm-gateway"],
                "http": [{
                    "match": [{
                        "headers": {
                            "x-experiment-id": {
                                "exact": experiment.id
                            }
                        }
                    }],
                    "route": routes,
                    "timeout": "30s",
                    "retries": {
                        "attempts": 3,
                        "perTryTimeout": "10s"
                    }
                }]
            }
        }
        
        return virtual_service
    
    def update_traffic_split(self, experiment_id: str, new_split: Dict[str, int]) -> bool:
        """Update traffic split for running experiment"""
        
        try:
            # Get existing VirtualService
            vs = self.k8s_client.get_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="virtualservices",
                name=f"experiment-{experiment_id}"
            )
            
            # Update traffic split
            for i, route in enumerate(vs["spec"]["http"][0]["route"]):
                variant = route["headers"]["response"]["set"]["x-variant"]
                if variant in new_split:
                    vs["spec"]["http"][0]["route"][i]["weight"] = new_split[variant]
            
            # Apply update
            self.k8s_client.patch_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="virtualservices",
                name=f"experiment-{experiment_id}",
                body=vs
            )
            
            print(f"✅ Traffic split updated for experiment {experiment_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update traffic split: {e}")
            return False
    
    def rollback_experiment(self, experiment_id: str) -> bool:
        """Rollback experiment to baseline"""
        
        return self.update_traffic_split(experiment_id, {
            "baseline": 100,
            "candidate": 0
        })
    
    def cleanup_experiment(self, experiment_id: str) -> bool:
        """Clean up experiment resources"""
        
        try:
            self.k8s_client.delete_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="virtualservices",
                name=f"experiment-{experiment_id}"
            )
            
            print(f"✅ Experiment {experiment_id} cleaned up")
            return True
            
        except Exception as e:
            print(f"❌ Failed to cleanup experiment: {e}")
            return False

# Example usage
def run_model_comparison_experiment():
    """Example: Compare two model versions"""
    
    experiment = Experiment(
        id="llama-3.1-7b-v1-1-comparison",
        name="Llama4 7B v1.1 Performance Test",
        description="Compare v1.0 baseline against v1.1 candidate",
        traffic_split={
            "v1-0": 70,  # Baseline gets 70% traffic
            "v1-1": 30   # Candidate gets 30% traffic
        },
        target_metrics={
            "latency_p95_ms": 2000,
            "success_rate": 0.99,
            "tokens_per_second": 150
        },
        duration_hours=24,
        success_criteria={
            "latency_improvement": 0.1,  # 10% improvement
            "success_rate_maintained": 0.99
        },
        rollback_criteria={
            "latency_degradation": 0.2,  # 20% degradation triggers rollback
            "success_rate_drop": 0.95
        }
    )
    
    manager = ExperimentManager()
    manager.create_experiment(experiment)