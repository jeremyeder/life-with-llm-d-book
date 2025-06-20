# Automated Experiment Controller
# Monitors A/B tests and automatically adjusts traffic based on performance
# Integrates with Prometheus for metrics and provides automatic rollback

import time
import asyncio
from prometheus_api_client import PrometheusConnect
from typing import Dict, List
import logging

class ExperimentController:
    def __init__(self, prometheus_url: str = "http://prometheus.monitoring.svc.cluster.local:9090"):
        self.prometheus = PrometheusConnect(url=prometheus_url)
        self.experiment_manager = ExperimentManager()
        self.logger = logging.getLogger(__name__)
        
    async def monitor_experiment(self, experiment: Experiment):
        """Monitor experiment and automatically adjust traffic"""
        
        self.logger.info(f"Starting monitoring for experiment {experiment.id}")
        
        start_time = time.time()
        end_time = start_time + (experiment.duration_hours * 3600)
        
        while time.time() < end_time:
            # Collect metrics for all variants
            metrics = await self._collect_experiment_metrics(experiment)
            
            # Analyze performance
            analysis = self._analyze_performance(experiment, metrics)
            
            # Make traffic adjustment decisions
            if analysis["should_rollback"]:
                self.logger.warning(f"Rolling back experiment {experiment.id} due to poor performance")
                self.experiment_manager.rollback_experiment(experiment.id)
                break
                
            elif analysis["should_adjust_traffic"]:
                new_split = analysis["recommended_split"]
                self.logger.info(f"Adjusting traffic split for experiment {experiment.id}: {new_split}")
                self.experiment_manager.update_traffic_split(experiment.id, new_split)
            
            # Log current status
            self._log_experiment_status(experiment, metrics, analysis)
            
            # Wait before next check
            await asyncio.sleep(300)  # Check every 5 minutes
        
        # Experiment completed
        self.logger.info(f"Experiment {experiment.id} completed")
        final_analysis = self._generate_final_report(experiment, metrics)
        
        return final_analysis
    
    async def _collect_experiment_metrics(self, experiment: Experiment) -> Dict:
        """Collect Prometheus metrics for experiment variants"""
        
        metrics = {}
        
        for variant in experiment.traffic_split.keys():
            variant_metrics = {}
            
            # Latency metrics
            latency_query = f'''
            histogram_quantile(0.95, 
                rate(istio_request_duration_milliseconds_bucket{{
                    destination_service_name="llama-3.1-7b-service",
                    destination_version="{variant}"
                }}[5m])
            )
            '''
            latency_result = self.prometheus.custom_query(latency_query)
            variant_metrics["latency_p95"] = float(latency_result[0]["value"][1]) if latency_result else 0
            
            # Success rate
            success_query = f'''
            rate(istio_requests_total{{
                destination_service_name="llama-3.1-7b-service",
                destination_version="{variant}",
                response_code!~"5.*"
            }}[5m]) / 
            rate(istio_requests_total{{
                destination_service_name="llama-3.1-7b-service",
                destination_version="{variant}"
            }}[5m])
            '''
            success_result = self.prometheus.custom_query(success_query)
            variant_metrics["success_rate"] = float(success_result[0]["value"][1]) if success_result else 0
            
            # Request rate
            rate_query = f'''
            rate(istio_requests_total{{
                destination_service_name="llama-3.1-7b-service",
                destination_version="{variant}"
            }}[5m])
            '''
            rate_result = self.prometheus.custom_query(rate_query)
            variant_metrics["request_rate"] = float(rate_result[0]["value"][1]) if rate_result else 0
            
            # Token generation rate (custom metric)
            tokens_query = f'''
            rate(llm_tokens_generated_total{{
                model_name="llama-3.1-7b",
                version="{variant}"
            }}[5m])
            '''
            tokens_result = self.prometheus.custom_query(tokens_query)
            variant_metrics["tokens_per_second"] = float(tokens_result[0]["value"][1]) if tokens_result else 0
            
            metrics[variant] = variant_metrics
        
        return metrics
    
    def _analyze_performance(self, experiment: Experiment, metrics: Dict) -> Dict:
        """Analyze experiment performance and make recommendations"""
        
        analysis = {
            "should_rollback": False,
            "should_adjust_traffic": False,
            "recommended_split": experiment.traffic_split.copy(),
            "performance_summary": {}
        }
        
        baseline_variant = list(experiment.traffic_split.keys())[0]  # Assume first is baseline
        candidate_variants = list(experiment.traffic_split.keys())[1:]
        
        baseline_metrics = metrics.get(baseline_variant, {})
        
        for candidate in candidate_variants:
            candidate_metrics = metrics.get(candidate, {})
            
            if not candidate_metrics:
                continue
            
            # Check rollback criteria
            if candidate_metrics.get("success_rate", 0) < experiment.rollback_criteria.get("success_rate_drop", 0.95):
                analysis["should_rollback"] = True
                analysis["rollback_reason"] = f"Success rate too low: {candidate_metrics['success_rate']:.3f}"
                break
            
            if candidate_metrics.get("latency_p95", 999999) > baseline_metrics.get("latency_p95", 0) * (1 + experiment.rollback_criteria.get("latency_degradation", 0.2)):
                analysis["should_rollback"] = True
                analysis["rollback_reason"] = f"Latency degradation too high: {candidate_metrics['latency_p95']:.0f}ms"
                break
            
            # Check if candidate is performing better
            latency_improvement = (baseline_metrics.get("latency_p95", 999999) - candidate_metrics.get("latency_p95", 999999)) / baseline_metrics.get("latency_p95", 1)
            success_rate_improvement = candidate_metrics.get("success_rate", 0) - baseline_metrics.get("success_rate", 0)
            
            # Gradual traffic increase for good performers
            if (latency_improvement > 0.05 and  # 5% latency improvement
                success_rate_improvement >= 0 and  # No success rate degradation
                candidate_metrics.get("success_rate", 0) > 0.98):  # High success rate
                
                current_candidate_traffic = experiment.traffic_split[candidate]
                
                if current_candidate_traffic < 50:  # Gradually increase
                    new_candidate_traffic = min(current_candidate_traffic + 10, 50)
                    new_baseline_traffic = 100 - sum(experiment.traffic_split[v] for v in candidate_variants if v != candidate) - new_candidate_traffic
                    
                    analysis["should_adjust_traffic"] = True
                    analysis["recommended_split"][baseline_variant] = new_baseline_traffic
                    analysis["recommended_split"][candidate] = new_candidate_traffic
            
            # Store performance comparison
            analysis["performance_summary"][candidate] = {
                "latency_improvement": latency_improvement,
                "success_rate_improvement": success_rate_improvement,
                "tokens_per_second": candidate_metrics.get("tokens_per_second", 0)
            }
        
        return analysis
    
    def _log_experiment_status(self, experiment: Experiment, metrics: Dict, analysis: Dict):
        """Log current experiment status"""
        
        self.logger.info(f"Experiment {experiment.id} status:")
        
        for variant, variant_metrics in metrics.items():
            traffic_percentage = experiment.traffic_split.get(variant, 0)
            self.logger.info(f"  {variant} ({traffic_percentage}% traffic):")
            self.logger.info(f"    Latency P95: {variant_metrics.get('latency_p95', 0):.1f}ms")
            self.logger.info(f"    Success Rate: {variant_metrics.get('success_rate', 0):.3f}")
            self.logger.info(f"    Tokens/sec: {variant_metrics.get('tokens_per_second', 0):.1f}")
        
        if analysis.get("performance_summary"):
            self.logger.info("  Performance comparison:")
            for variant, perf in analysis["performance_summary"].items():
                self.logger.info(f"    {variant}: Latency {perf['latency_improvement']:+.1%}, Success Rate {perf['success_rate_improvement']:+.3f}")

    def _generate_final_report(self, experiment: Experiment, final_metrics: Dict) -> Dict:
        """Generate final experiment report"""
        
        report = {
            "experiment_id": experiment.id,
            "duration_hours": experiment.duration_hours,
            "final_metrics": final_metrics,
            "recommendations": [],
            "winner": None
        }
        
        # Determine winning variant
        best_variant = None
        best_score = -1
        
        for variant, metrics in final_metrics.items():
            # Simple scoring: weighted combination of latency and success rate
            score = (metrics.get("success_rate", 0) * 0.6) + \
                   (1 / max(metrics.get("latency_p95", 999999), 1) * 1000 * 0.4)
            
            if score > best_score:
                best_score = score
                best_variant = variant
        
        report["winner"] = best_variant
        
        # Generate recommendations
        if best_variant:
            report["recommendations"].append(f"Promote {best_variant} to production")
            report["recommendations"].append(f"Update baseline deployment to use {best_variant}")
        
        return report

# Example experiment execution
async def run_automated_experiment():
    """Run an automated A/B test"""
    
    experiment = Experiment(
        id="auto-llama-3.1-optimization",
        name="Automated Llama4 7B Optimization Test",
        description="Automatically optimize traffic split based on performance",
        traffic_split={"v1-0": 80, "v1-1": 20},
        target_metrics={"latency_p95_ms": 1500, "success_rate": 0.99},
        duration_hours=6,
        success_criteria={"latency_improvement": 0.1},
        rollback_criteria={"latency_degradation": 0.15, "success_rate_drop": 0.97}
    )
    
    controller = ExperimentController()
    
    # Start experiment
    controller.experiment_manager.create_experiment(experiment)
    
    # Monitor and automatically adjust
    final_report = await controller.monitor_experiment(experiment)
    
    print("📊 Final Experiment Report:")
    print(f"  Winner: {final_report['winner']}")
    print(f"  Recommendations: {final_report['recommendations']}")
    
    # Cleanup
    controller.experiment_manager.cleanup_experiment(experiment.id)

if __name__ == "__main__":
    asyncio.run(run_automated_experiment())