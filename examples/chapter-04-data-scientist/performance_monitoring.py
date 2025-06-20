class ModelPerformanceMonitor:
    def __init__(self, model_name, prometheus_client):
        self.model_name = model_name
        self.prometheus = prometheus_client
        self.alerts = []
        
    def setup_monitoring_dashboard(self):
        """Set up Grafana dashboard for model monitoring"""
        dashboard_config = {
            "dashboard": {
                "title": f"Model Performance - {self.model_name}",
                "panels": [
                    {
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [{
                            "expr": f"rate(llm_d_requests_total{{model='{self.model_name}'}}[5m])",
                            "legendFormat": "Requests/sec"
                        }]
                    },
                    {
                        "title": "Latency Distribution",
                        "type": "heatmap",
                        "targets": [{
                            "expr": f"histogram_quantile(0.95, rate(llm_d_request_duration_seconds_bucket{{model='{self.model_name}'}}[5m]))",
                            "legendFormat": "P95 Latency"
                        }]
                    },
                    {
                        "title": "Quality Metrics",
                        "type": "stat",
                        "targets": [{
                            "expr": f"llm_d_model_quality_score{{model='{self.model_name}'}}",
                            "legendFormat": "Quality Score"
                        }]
                    },
                    {
                        "title": "Resource Utilization",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": f"llm_d_gpu_utilization{{model='{self.model_name}'}}",
                                "legendFormat": "GPU Utilization"
                            },
                            {
                                "expr": f"llm_d_memory_utilization{{model='{self.model_name}'}}",
                                "legendFormat": "Memory Utilization"
                            }
                        ]
                    }
                ]
            }
        }
        
        return dashboard_config
    
    def setup_alerting_rules(self):
        """Configure alerting rules for model performance"""
        alerting_rules = [
            {
                "alert": "HighModelLatency",
                "expr": f"histogram_quantile(0.95, rate(llm_d_request_duration_seconds_bucket{{model='{self.model_name}'}}[5m])) > 2",
                "for": "5m",
                "labels": {
                    "severity": "warning",
                    "model": self.model_name
                },
                "annotations": {
                    "summary": "High model latency detected",
                    "description": f"Model {self.model_name} P95 latency is above 2 seconds"
                }
            },
            {
                "alert": "ModelQualityDegradation",
                "expr": f"llm_d_model_quality_score{{model='{self.model_name}'}} < 0.8",
                "for": "10m",
                "labels": {
                    "severity": "critical",
                    "model": self.model_name
                },
                "annotations": {
                    "summary": "Model quality degradation detected",
                    "description": f"Model {self.model_name} quality score has dropped below 0.8"
                }
            },
            {
                "alert": "HighErrorRate",
                "expr": f"rate(llm_d_errors_total{{model='{self.model_name}'}}[5m]) / rate(llm_d_requests_total{{model='{self.model_name}'}}[5m]) > 0.05",
                "for": "5m",
                "labels": {
                    "severity": "critical",
                    "model": self.model_name
                },
                "annotations": {
                    "summary": "High error rate detected",
                    "description": f"Model {self.model_name} error rate is above 5%"
                }
            }
        ]
        
        return alerting_rules