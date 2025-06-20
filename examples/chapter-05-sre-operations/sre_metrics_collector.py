class SREMetricsCollector:
    def __init__(self):
        self.metrics = {
            "reliability": self._collect_reliability_metrics,
            "performance": self._collect_performance_metrics,
            "operational": self._collect_operational_metrics,
            "cost": self._collect_cost_metrics
        }
    
    def generate_sre_report(self, period="monthly"):
        """Generate comprehensive SRE metrics report"""
        report = {}
        
        for category, collector in self.metrics.items():
            report[category] = collector(period)
        
        # Calculate overall SRE score
        report["overall_score"] = self._calculate_sre_score(report)
        
        # Generate insights and recommendations
        report["insights"] = self._generate_insights(report)
        
        return report
    
    def _collect_reliability_metrics(self, period):
        """Collect reliability-focused metrics"""
        return {
            "availability": {
                "slo_target": 99.9,
                "actual": 99.95,
                "error_budget_remaining": 80.0
            },
            "mttr": {
                "target_minutes": 30,
                "actual_minutes": 22,
                "trend": "improving"
            },
            "mtbf": {
                "target_hours": 720,  # 30 days
                "actual_hours": 896,
                "trend": "stable"
            },
            "incident_count": {
                "p0": 0,
                "p1": 2,
                "p2": 8,
                "p3": 15
            }
        }
    
    def _collect_performance_metrics(self, period):
        """Collect performance-focused metrics"""
        return {
            "latency": {
                "p50_ms": 850,
                "p95_ms": 1800,
                "p99_ms": 3200,
                "slo_target_ms": 2000
            },
            "throughput": {
                "rps_avg": 250,
                "rps_peak": 890,
                "capacity_utilization": 0.68
            },
            "resource_efficiency": {
                "cpu_utilization": 0.72,
                "gpu_utilization": 0.85,
                "memory_utilization": 0.78
            }
        }
    
    def _collect_operational_metrics(self, period):
        """Collect operational efficiency metrics"""
        return {
            "deployment_metrics": {
                "deployment_frequency": "2.3/week",
                "deployment_success_rate": 0.96,
                "rollback_rate": 0.04,
                "lead_time_hours": 4.2
            },
            "automation": {
                "automated_responses": 0.85,
                "manual_interventions": 12,
                "runbook_coverage": 0.92
            },
            "monitoring": {
                "alert_noise_ratio": 0.15,
                "false_positive_rate": 0.08,
                "monitoring_coverage": 0.94
            }
        }
    
    def _calculate_sre_score(self, report):
        """Calculate overall SRE effectiveness score"""
        weights = {
            "availability": 0.4,
            "performance": 0.3,
            "automation": 0.2,
            "cost_efficiency": 0.1
        }
        
        scores = {
            "availability": min(report["reliability"]["availability"]["actual"] / 99.9, 1.0),
            "performance": min(2000 / report["performance"]["latency"]["p95_ms"], 1.0),
            "automation": report["operational"]["automation"]["automated_responses"],
            "cost_efficiency": min(report["cost"]["efficiency_score"], 1.0)
        }
        
        overall_score = sum(score * weights[metric] for metric, score in scores.items())
        return round(overall_score * 100, 1)  # Convert to percentage

# Example usage
metrics_collector = SREMetricsCollector()
monthly_report = metrics_collector.generate_sre_report("monthly")

print(f"Overall SRE Score: {monthly_report['overall_score']}%")
print(f"Availability: {monthly_report['reliability']['availability']['actual']}%")
print(f"P95 Latency: {monthly_report['performance']['latency']['p95_ms']}ms")