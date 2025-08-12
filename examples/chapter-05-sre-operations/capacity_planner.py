class CapacityPlanner:
    def __init__(self, prometheus_client, metrics_retention="30d"):
        self.prometheus = prometheus_client
        self.retention = metrics_retention

    def analyze_historical_usage(self, service_name):
        """Analyze historical resource usage patterns"""
        queries = {
            "request_rate": f"rate(llm_d_requests_total{{service='{service_name}'}}[5m])",
            "gpu_utilization": f"avg(llm_d_gpu_utilization{{service='{service_name}'}}) by (instance)",
            "memory_utilization": f"avg(llm_d_memory_utilization{{service='{service_name}'}}) by (instance)",
            "latency_p95": f"histogram_quantile(0.95, rate(llm_d_request_duration_seconds_bucket{{service='{service_name}'}}[5m]))",
            "queue_depth": f"avg(llm_d_queue_depth{{service='{service_name}'}})",
        }

        historical_data = {}
        for metric, query in queries.items():
            data = self.prometheus.query_range(
                query=query, start_time=f"-{self.retention}", end_time="now", step="1h"
            )
            historical_data[metric] = self._process_time_series(data)

        return self._analyze_patterns(historical_data)

    def _analyze_patterns(self, data):
        """Analyze usage patterns and identify trends"""
        analysis = {
            "peak_hours": self._identify_peak_hours(data["request_rate"]),
            "growth_trend": self._calculate_growth_trend(data["request_rate"]),
            "resource_correlation": self._analyze_resource_correlation(data),
            "capacity_utilization": self._calculate_capacity_utilization(data),
            "forecasting": self._forecast_requirements(data),
        }

        return analysis

    def _forecast_requirements(self, data, forecast_period="90d"):
        """Forecast future resource requirements"""
        import numpy as np
        from sklearn.linear_model import LinearRegression

        # Prepare time series data
        request_rates = data["request_rate"]
        timestamps = [point["timestamp"] for point in request_rates]
        values = [point["value"] for point in request_rates]

        # Simple linear regression for trend
        X = np.array(range(len(values))).reshape(-1, 1)
        y = np.array(values)

        model = LinearRegression()
        model.fit(X, y)

        # Forecast future points
        future_points = 90 * 24  # 90 days of hourly data
        future_X = np.array(range(len(values), len(values) + future_points)).reshape(
            -1, 1
        )
        forecast = model.predict(future_X)

        # Calculate resource requirements based on forecast
        peak_forecast = np.max(forecast) * 1.2  # 20% safety margin

        return {
            "forecast_peak_rps": peak_forecast,
            "recommended_replicas": self._calculate_required_replicas(peak_forecast),
            "recommended_gpu_count": self._calculate_required_gpus(peak_forecast),
            "growth_rate_per_day": model.coef_[0] * 24,
        }

    def _calculate_required_replicas(self, peak_rps):
        """Calculate required replicas based on peak RPS"""
        # Assume each replica can handle 50 RPS at 80% utilization
        replicas_needed = peak_rps / (50 * 0.8)
        return max(int(np.ceil(replicas_needed)), 2)  # Minimum 2 replicas

    def _calculate_required_gpus(self, peak_rps):
        """Calculate required GPU count based on peak RPS"""
        # Assume each GPU can handle 25 RPS for LLM inference
        gpus_needed = peak_rps / 25
        return max(int(np.ceil(gpus_needed)), 1)

    def generate_capacity_plan(self, analysis_results):
        """Generate comprehensive capacity plan"""
        plan = {
            "current_state": {
                "avg_utilization": analysis_results["capacity_utilization"]["avg"],
                "peak_utilization": analysis_results["capacity_utilization"]["peak"],
                "current_replicas": self._get_current_replicas(),
                "current_gpu_count": self._get_current_gpu_count(),
            },
            "recommendations": {
                "immediate": self._immediate_recommendations(analysis_results),
                "short_term": self._short_term_recommendations(analysis_results),
                "long_term": self._long_term_recommendations(analysis_results),
            },
            "scaling_triggers": {
                "scale_up": {
                    "cpu_threshold": "80%",
                    "gpu_threshold": "85%",
                    "memory_threshold": "85%",
                    "latency_threshold": "2s",
                    "queue_depth_threshold": "10",
                },
                "scale_down": {
                    "cpu_threshold": "30%",
                    "gpu_threshold": "40%",
                    "memory_threshold": "40%",
                    "sustained_duration": "15m",
                },
            },
            "cost_analysis": self._analyze_costs(analysis_results),
        }

        return plan


# Example usage
planner = CapacityPlanner(prometheus_client)
usage_analysis = planner.analyze_historical_usage("llm-d-production")
capacity_plan = planner.generate_capacity_plan(usage_analysis)

print("Capacity Planning Results:")
print(f"Current utilization: {capacity_plan['current_state']['avg_utilization']:.1%}")
print(
    f"Recommended replicas: {capacity_plan['recommendations']['immediate']['replicas']}"
)
print(f"Forecasted peak RPS: {usage_analysis['forecasting']['forecast_peak_rps']:.1f}")
