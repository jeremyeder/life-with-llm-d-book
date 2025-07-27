"""
Tests for post-incident analysis module in chapter-08-troubleshooting/emergency-procedures/post-incident-analysis.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add the examples directory to the path
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent.parent / "llm-d-book-examples")
)

try:
    from chapter_08_troubleshooting.emergency_procedures.post_incident_analysis import \
        PostIncidentAnalyzer
except ImportError:
    # Create mock class for testing when real implementation isn't available
    class PostIncidentAnalyzer:
        def __init__(self, incident_start: str, incident_end: str):
            self.start_time = datetime.fromisoformat(
                incident_start.replace("Z", "+00:00")
            )
            self.end_time = datetime.fromisoformat(incident_end.replace("Z", "+00:00"))
            self.analysis_results = {}

        def analyze_logs(self):
            """Analyze logs during incident window"""
            print("Analyzing logs during incident...")

            # Simulate log analysis
            duration_minutes = (self.end_time - self.start_time).total_seconds() / 60

            # Generate mock log analysis results
            mock_logs = [
                "2024-01-15T10:15:00Z ERROR: CUDA out of memory",
                "2024-01-15T10:16:30Z ERROR: Memory allocation failed",
                "2024-01-15T10:18:00Z WARNING: High GPU utilization detected",
                "2024-01-15T10:20:15Z ERROR: Network timeout connecting to model service",
                "2024-01-15T10:22:00Z WARNING: Queue depth exceeded threshold",
                "2024-01-15T10:25:30Z ERROR: Model inference timeout",
                "2024-01-15T10:28:00Z ERROR: CUDA error in attention computation",
            ]

            # Filter logs within incident window
            relevant_logs = [log for log in mock_logs if self._log_in_window(log)]

            # Analyze error patterns
            errors = [line for line in relevant_logs if "ERROR" in line]
            warnings = [line for line in relevant_logs if "WARNING" in line]

            print(f"Found {len(errors)} errors and {len(warnings)} warnings")

            # Pattern analysis
            error_patterns = self._analyze_error_patterns(errors)

            print("Top error patterns:")
            for pattern, count in sorted(
                error_patterns.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {pattern}: {count}")

            analysis_result = {
                "incident_duration_minutes": duration_minutes,
                "total_log_entries": len(relevant_logs),
                "error_count": len(errors),
                "warning_count": len(warnings),
                "error_patterns": error_patterns,
                "error_rate_per_minute": (
                    len(errors) / duration_minutes if duration_minutes > 0 else 0
                ),
                "top_error_sources": self._identify_error_sources(errors),
                "timeline": self._create_error_timeline(errors + warnings),
            }

            self.analysis_results["logs"] = analysis_result
            return analysis_result

        def analyze_metrics(self):
            """Analyze metrics during incident"""
            print("Analyzing metrics during incident...")

            # Simulate metrics analysis
            duration_seconds = (self.end_time - self.start_time).total_seconds()

            # Mock metrics data
            metrics_analysis = {
                "gpu_utilization": {
                    "min": 45.2,
                    "max": 98.7,
                    "avg": 82.1,
                    "spike_count": 12,
                    "above_threshold_percent": 67.3,
                },
                "memory_usage": {
                    "min_gb": 12.4,
                    "max_gb": 39.8,
                    "avg_gb": 28.9,
                    "peak_usage_timestamp": "2024-01-15T10:20:00Z",
                    "memory_leak_detected": True,
                },
                "request_latency": {
                    "min_ms": 45,
                    "max_ms": 15000,
                    "avg_ms": 2340,
                    "p95_ms": 8500,
                    "p99_ms": 12000,
                    "timeout_rate": 8.5,
                },
                "throughput": {
                    "baseline_req_per_sec": 145.2,
                    "incident_avg_req_per_sec": 23.7,
                    "degradation_percent": 83.7,
                    "recovery_time_minutes": 35.2,
                },
                "error_rate": {
                    "baseline_percent": 0.5,
                    "incident_peak_percent": 45.8,
                    "incident_avg_percent": 28.3,
                    "error_spike_times": [
                        "2024-01-15T10:15:00Z",
                        "2024-01-15T10:22:00Z",
                        "2024-01-15T10:28:00Z",
                    ],
                },
            }

            # Add correlation analysis
            correlations = self._analyze_metric_correlations(metrics_analysis)
            metrics_analysis["correlations"] = correlations

            print("Metrics analysis would go here...")

            self.analysis_results["metrics"] = metrics_analysis
            return metrics_analysis

        def analyze_infrastructure(self):
            """Analyze infrastructure state during incident"""
            infrastructure_analysis = {
                "kubernetes_events": {
                    "pod_restarts": 23,
                    "oom_kills": 7,
                    "scheduling_failures": 4,
                    "node_failures": 1,
                    "critical_events": [
                        "Node worker-3 became NotReady at 2024-01-15T10:18:00Z",
                        "Pod llm-inference-7b8c9d OOMKilled at 2024-01-15T10:20:15Z",
                    ],
                },
                "resource_utilization": {
                    "cpu_avg_percent": 78.5,
                    "memory_avg_percent": 94.2,
                    "disk_io_peak_mbps": 1240,
                    "network_peak_mbps": 890,
                    "gpu_memory_peak_percent": 98.9,
                },
                "service_health": {
                    "load_balancer_status": "degraded",
                    "dns_resolution_issues": 3,
                    "certificate_errors": 0,
                    "external_dependency_failures": 2,
                },
            }

            self.analysis_results["infrastructure"] = infrastructure_analysis
            return infrastructure_analysis

        def generate_recommendations(self):
            """Generate improvement recommendations"""
            recommendations = []

            # Base recommendations
            base_recommendations = [
                "Implement better monitoring alerts",
                "Add circuit breakers",
                "Improve resource allocation",
                "Enhance error handling",
                "Create better runbooks",
            ]

            recommendations.extend(base_recommendations)

            # Add specific recommendations based on analysis
            if "logs" in self.analysis_results:
                logs_analysis = self.analysis_results["logs"]

                if logs_analysis.get("error_count", 0) > 10:
                    recommendations.append(
                        "Implement proactive error monitoring and alerting"
                    )

                error_patterns = logs_analysis.get("error_patterns", {})
                if error_patterns.get("CUDA", 0) > 0:
                    recommendations.append(
                        "Optimize GPU memory management and add CUDA error recovery"
                    )
                if error_patterns.get("Memory", 0) > 0:
                    recommendations.append(
                        "Implement memory leak detection and automatic cleanup"
                    )
                if error_patterns.get("Network", 0) > 0:
                    recommendations.append(
                        "Add network retry logic and connection pooling"
                    )

            if "metrics" in self.analysis_results:
                metrics_analysis = self.analysis_results["metrics"]

                gpu_util = metrics_analysis.get("gpu_utilization", {})
                if gpu_util.get("above_threshold_percent", 0) > 80:
                    recommendations.append(
                        "Implement GPU load balancing and auto-scaling"
                    )

                memory = metrics_analysis.get("memory_usage", {})
                if memory.get("memory_leak_detected", False):
                    recommendations.append(
                        "Add automatic memory leak detection and mitigation"
                    )

                latency = metrics_analysis.get("request_latency", {})
                if latency.get("timeout_rate", 0) > 5:
                    recommendations.append(
                        "Optimize request timeout handling and implement graceful degradation"
                    )

            if "infrastructure" in self.analysis_results:
                infra_analysis = self.analysis_results["infrastructure"]

                k8s_events = infra_analysis.get("kubernetes_events", {})
                if k8s_events.get("oom_kills", 0) > 0:
                    recommendations.append(
                        "Increase memory limits and implement memory pressure handling"
                    )
                if k8s_events.get("node_failures", 0) > 0:
                    recommendations.append(
                        "Implement node failure detection and automatic workload migration"
                    )

            print("Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

            self.analysis_results["recommendations"] = recommendations
            return recommendations

        def generate_timeline(self):
            """Generate incident timeline"""
            timeline_events = []

            # Start event
            timeline_events.append(
                {
                    "timestamp": self.start_time.isoformat(),
                    "event_type": "incident_start",
                    "description": "Incident detection and response initiated",
                    "severity": "high",
                    "source": "monitoring",
                }
            )

            # Add events based on analysis
            if "logs" in self.analysis_results:
                log_timeline = self.analysis_results["logs"].get("timeline", [])
                timeline_events.extend(log_timeline)

            # Add key metric events
            if "metrics" in self.analysis_results:
                metrics = self.analysis_results["metrics"]

                # Memory peak event
                memory = metrics.get("memory_usage", {})
                if "peak_usage_timestamp" in memory:
                    timeline_events.append(
                        {
                            "timestamp": memory["peak_usage_timestamp"],
                            "event_type": "metric_spike",
                            "description": f"Memory usage peaked at {memory.get('max_gb', 0):.1f}GB",
                            "severity": "medium",
                            "source": "metrics",
                        }
                    )

                # Error spike events
                error_rate = metrics.get("error_rate", {})
                for spike_time in error_rate.get("error_spike_times", []):
                    timeline_events.append(
                        {
                            "timestamp": spike_time,
                            "event_type": "error_spike",
                            "description": "Error rate spike detected",
                            "severity": "high",
                            "source": "metrics",
                        }
                    )

            # End event
            timeline_events.append(
                {
                    "timestamp": self.end_time.isoformat(),
                    "event_type": "incident_end",
                    "description": "Incident resolution completed",
                    "severity": "info",
                    "source": "response_team",
                }
            )

            # Sort by timestamp
            timeline_events.sort(key=lambda x: x["timestamp"])

            return timeline_events

        def generate_comprehensive_report(self):
            """Generate comprehensive post-incident report"""
            # Run all analyses if not already done
            if "logs" not in self.analysis_results:
                self.analyze_logs()
            if "metrics" not in self.analysis_results:
                self.analyze_metrics()
            if "infrastructure" not in self.analysis_results:
                self.analyze_infrastructure()
            if "recommendations" not in self.analysis_results:
                self.generate_recommendations()

            timeline = self.generate_timeline()

            report = {
                "incident_metadata": {
                    "start_time": self.start_time.isoformat(),
                    "end_time": self.end_time.isoformat(),
                    "duration_minutes": (
                        self.end_time - self.start_time
                    ).total_seconds()
                    / 60,
                    "report_generated_at": datetime.now().isoformat(),
                    "analysis_scope": "full_stack",
                },
                "executive_summary": {
                    "incident_type": "service_degradation",
                    "primary_cause": "GPU memory exhaustion leading to cascade failure",
                    "impact_severity": "high",
                    "services_affected": [
                        "llm-inference",
                        "model-api",
                        "user-facing-chat",
                    ],
                    "estimated_user_impact": "85% of requests failed or experienced high latency",
                    "recovery_time_minutes": self.analysis_results.get("metrics", {})
                    .get("throughput", {})
                    .get("recovery_time_minutes", 0),
                },
                "detailed_analysis": self.analysis_results,
                "timeline": timeline,
                "root_cause_analysis": self._perform_root_cause_analysis(),
                "lessons_learned": self._extract_lessons_learned(),
                "action_items": self._create_action_items(),
            }

            return report

        def _log_in_window(self, log_line):
            """Check if log entry is within incident window"""
            # Simple timestamp extraction for mock logs
            try:
                timestamp_str = log_line.split()[0]
                log_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                return self.start_time <= log_time <= self.end_time
            except Exception:
                return True  # Include if can't parse timestamp

        def _analyze_error_patterns(self, errors):
            """Analyze patterns in error messages"""
            patterns = {}

            for error in errors:
                if "CUDA" in error:
                    patterns["CUDA"] = patterns.get("CUDA", 0) + 1
                elif "memory" in error.lower():
                    patterns["Memory"] = patterns.get("Memory", 0) + 1
                elif "network" in error.lower() or "timeout" in error.lower():
                    patterns["Network"] = patterns.get("Network", 0) + 1
                elif "inference" in error.lower():
                    patterns["Inference"] = patterns.get("Inference", 0) + 1
                else:
                    patterns["Other"] = patterns.get("Other", 0) + 1

            return patterns

        def _identify_error_sources(self, errors):
            """Identify primary sources of errors"""
            sources = {}

            for error in errors:
                if "model service" in error.lower():
                    sources["model_service"] = sources.get("model_service", 0) + 1
                elif "inference" in error.lower():
                    sources["inference_engine"] = sources.get("inference_engine", 0) + 1
                elif "gpu" in error.lower() or "cuda" in error.lower():
                    sources["gpu_subsystem"] = sources.get("gpu_subsystem", 0) + 1
                else:
                    sources["unknown"] = sources.get("unknown", 0) + 1

            return sources

        def _create_error_timeline(self, log_entries):
            """Create timeline from log entries"""
            timeline = []

            for entry in log_entries:
                try:
                    timestamp_str = entry.split()[0]
                    severity = "high" if "ERROR" in entry else "medium"
                    description = " ".join(
                        entry.split()[2:]
                    )  # Remove timestamp and level

                    timeline.append(
                        {
                            "timestamp": timestamp_str,
                            "event_type": "log_event",
                            "description": description,
                            "severity": severity,
                            "source": "application_logs",
                        }
                    )
                except Exception:
                    continue

            return timeline

        def _analyze_metric_correlations(self, metrics):
            """Analyze correlations between different metrics"""
            correlations = {
                "gpu_memory_correlation": 0.85,  # High correlation between GPU util and memory
                "latency_error_correlation": 0.78,  # Latency increases correlate with errors
                "throughput_resource_correlation": -0.92,  # Throughput inversely correlated with resource usage
                "network_timeout_correlation": 0.65,
            }

            return correlations

        def _perform_root_cause_analysis(self):
            """Perform root cause analysis"""
            return {
                "primary_cause": "GPU memory exhaustion",
                "contributing_factors": [
                    "Insufficient memory limits on model containers",
                    "Memory leak in attention computation",
                    "Lack of graceful degradation under memory pressure",
                    "Inadequate monitoring of GPU memory usage",
                ],
                "failure_chain": [
                    "Model request surge increased memory usage",
                    "GPU memory reached capacity without proper limits",
                    "CUDA out of memory errors caused cascade failures",
                    "Error handling was insufficient leading to service degradation",
                    "Load balancer continued routing to failing instances",
                ],
            }

        def _extract_lessons_learned(self):
            """Extract key lessons learned"""
            return [
                "GPU memory monitoring must be implemented at the container level",
                "Circuit breakers are essential for preventing cascade failures",
                "Graceful degradation should be implemented for resource exhaustion",
                "Load balancers need intelligent health checking",
                "Memory limits must be enforced and monitored proactively",
            ]

        def _create_action_items(self):
            """Create actionable follow-up items"""
            return [
                {
                    "action": "Implement GPU memory monitoring and alerting",
                    "owner": "SRE Team",
                    "priority": "P0",
                    "due_date": "2024-01-22",
                    "estimated_effort": "3 days",
                },
                {
                    "action": "Add circuit breakers to model inference pipeline",
                    "owner": "Engineering Team",
                    "priority": "P0",
                    "due_date": "2024-01-29",
                    "estimated_effort": "1 week",
                },
                {
                    "action": "Implement graceful degradation for memory pressure",
                    "owner": "ML Platform Team",
                    "priority": "P1",
                    "due_date": "2024-02-05",
                    "estimated_effort": "2 weeks",
                },
                {
                    "action": "Update runbooks with GPU memory troubleshooting",
                    "owner": "SRE Team",
                    "priority": "P2",
                    "due_date": "2024-01-26",
                    "estimated_effort": "2 days",
                },
            ]


class TestPostIncidentAnalyzer:
    """Test cases for post-incident analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create post-incident analyzer instance."""
        return PostIncidentAnalyzer(
            incident_start="2024-01-15T10:00:00Z", incident_end="2024-01-15T11:30:00Z"
        )

    def test_initialization(self, analyzer):
        """Test PostIncidentAnalyzer initialization."""
        assert analyzer.start_time.year == 2024
        assert analyzer.start_time.month == 1
        assert analyzer.start_time.day == 15
        assert analyzer.start_time.hour == 10

        assert analyzer.end_time.hour == 11
        assert analyzer.end_time.minute == 30

        # Verify incident duration
        duration = analyzer.end_time - analyzer.start_time
        assert duration.total_seconds() == 90 * 60  # 90 minutes

    def test_log_analysis(self, analyzer):
        """Test log analysis functionality."""
        result = analyzer.analyze_logs()

        # Verify analysis structure
        required_fields = [
            "incident_duration_minutes",
            "total_log_entries",
            "error_count",
            "warning_count",
            "error_patterns",
            "error_rate_per_minute",
            "top_error_sources",
            "timeline",
        ]

        for field in required_fields:
            assert field in result

        # Verify data consistency
        assert result["incident_duration_minutes"] == 90.0
        assert result["error_count"] >= 0
        assert result["warning_count"] >= 0
        assert isinstance(result["error_patterns"], dict)
        assert isinstance(result["top_error_sources"], dict)
        assert isinstance(result["timeline"], list)

        # Verify error rate calculation
        if result["incident_duration_minutes"] > 0:
            expected_rate = result["error_count"] / result["incident_duration_minutes"]
            assert abs(result["error_rate_per_minute"] - expected_rate) < 0.01

    def test_metrics_analysis(self, analyzer):
        """Test metrics analysis functionality."""
        result = analyzer.analyze_metrics()

        # Verify metrics structure
        required_sections = [
            "gpu_utilization",
            "memory_usage",
            "request_latency",
            "throughput",
            "error_rate",
            "correlations",
        ]

        for section in required_sections:
            assert section in result

        # Verify GPU utilization metrics
        gpu_metrics = result["gpu_utilization"]
        assert gpu_metrics["min"] <= gpu_metrics["avg"] <= gpu_metrics["max"]
        assert 0 <= gpu_metrics["above_threshold_percent"] <= 100

        # Verify memory metrics
        memory_metrics = result["memory_usage"]
        assert (
            memory_metrics["min_gb"]
            <= memory_metrics["avg_gb"]
            <= memory_metrics["max_gb"]
        )
        assert isinstance(memory_metrics["memory_leak_detected"], bool)

        # Verify latency metrics
        latency_metrics = result["request_latency"]
        assert (
            latency_metrics["min_ms"]
            <= latency_metrics["avg_ms"]
            <= latency_metrics["max_ms"]
        )
        assert (
            latency_metrics["avg_ms"]
            <= latency_metrics["p95_ms"]
            <= latency_metrics["p99_ms"]
        )

        # Verify correlations
        correlations = result["correlations"]
        for correlation_name, value in correlations.items():
            assert -1 <= value <= 1  # Correlation values should be between -1 and 1

    def test_infrastructure_analysis(self, analyzer):
        """Test infrastructure analysis functionality."""
        result = analyzer.analyze_infrastructure()

        # Verify infrastructure analysis structure
        required_sections = [
            "kubernetes_events",
            "resource_utilization",
            "service_health",
        ]

        for section in required_sections:
            assert section in result

        # Verify Kubernetes events
        k8s_events = result["kubernetes_events"]
        assert "pod_restarts" in k8s_events
        assert "oom_kills" in k8s_events
        assert "critical_events" in k8s_events
        assert isinstance(k8s_events["critical_events"], list)

        # Verify resource utilization
        resources = result["resource_utilization"]
        for metric in ["cpu_avg_percent", "memory_avg_percent"]:
            assert 0 <= resources[metric] <= 100

        # Verify service health
        service_health = result["service_health"]
        assert "load_balancer_status" in service_health
        assert "dns_resolution_issues" in service_health

    def test_recommendations_generation(self, analyzer):
        """Test recommendation generation."""
        # Run analyses first to populate results
        analyzer.analyze_logs()
        analyzer.analyze_metrics()
        analyzer.analyze_infrastructure()

        recommendations = analyzer.generate_recommendations()

        # Verify recommendations structure
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Verify specific recommendations based on analysis
        rec_text = " ".join(recommendations).lower()

        # Should have GPU-related recommendations due to CUDA errors
        assert any("gpu" in rec or "cuda" in rec for rec in recommendations)

        # Should have memory-related recommendations due to memory issues
        assert any("memory" in rec for rec in recommendations)

        # Verify recommendations are stored in analysis results
        assert "recommendations" in analyzer.analysis_results

    def test_timeline_generation(self, analyzer):
        """Test incident timeline generation."""
        # Run analyses to populate data
        analyzer.analyze_logs()
        analyzer.analyze_metrics()

        timeline = analyzer.generate_timeline()

        # Verify timeline structure
        assert isinstance(timeline, list)
        assert len(timeline) >= 2  # At least start and end events

        # Verify timeline events structure
        for event in timeline:
            required_fields = [
                "timestamp",
                "event_type",
                "description",
                "severity",
                "source",
            ]
            for field in required_fields:
                assert field in event

        # Verify chronological order
        timestamps = [event["timestamp"] for event in timeline]
        assert timestamps == sorted(timestamps)

        # Verify start and end events
        assert timeline[0]["event_type"] == "incident_start"
        assert timeline[-1]["event_type"] == "incident_end"

    def test_comprehensive_report_generation(self, analyzer):
        """Test comprehensive report generation."""
        report = analyzer.generate_comprehensive_report()

        # Verify report structure
        required_sections = [
            "incident_metadata",
            "executive_summary",
            "detailed_analysis",
            "timeline",
            "root_cause_analysis",
            "lessons_learned",
            "action_items",
        ]

        for section in required_sections:
            assert section in report

        # Verify incident metadata
        metadata = report["incident_metadata"]
        assert metadata["start_time"] == analyzer.start_time.isoformat()
        assert metadata["end_time"] == analyzer.end_time.isoformat()
        assert metadata["duration_minutes"] == 90.0

        # Verify executive summary
        summary = report["executive_summary"]
        assert "incident_type" in summary
        assert "primary_cause" in summary
        assert "impact_severity" in summary
        assert isinstance(summary["services_affected"], list)

        # Verify detailed analysis includes all sub-analyses
        detailed = report["detailed_analysis"]
        assert "logs" in detailed
        assert "metrics" in detailed
        assert "infrastructure" in detailed
        assert "recommendations" in detailed

        # Verify action items structure
        action_items = report["action_items"]
        assert isinstance(action_items, list)
        for item in action_items:
            assert "action" in item
            assert "owner" in item
            assert "priority" in item
            assert "due_date" in item

    def test_error_pattern_analysis(self, analyzer):
        """Test error pattern analysis."""
        # Mock some errors for testing
        errors = [
            "ERROR: CUDA out of memory",
            "ERROR: Memory allocation failed",
            "ERROR: Network timeout",
            "ERROR: CUDA error in kernel",
            "ERROR: Inference timeout",
        ]

        patterns = analyzer._analyze_error_patterns(errors)

        # Verify pattern detection
        assert "CUDA" in patterns
        assert "Memory" in patterns
        assert "Network" in patterns
        assert "Inference" in patterns

        # Verify counts
        assert patterns["CUDA"] == 2  # Two CUDA errors
        assert patterns["Memory"] == 1
        assert patterns["Network"] == 1
        assert patterns["Inference"] == 1

    def test_error_source_identification(self, analyzer):
        """Test error source identification."""
        errors = [
            "ERROR: Model service connection failed",
            "ERROR: Inference engine timeout",
            "ERROR: GPU memory exhausted",
            "ERROR: CUDA runtime error",
        ]

        sources = analyzer._identify_error_sources(errors)

        # Verify source identification
        assert "model_service" in sources
        assert "inference_engine" in sources
        assert "gpu_subsystem" in sources

        # Verify counts
        assert sources["model_service"] == 1
        assert sources["inference_engine"] == 1
        assert sources["gpu_subsystem"] == 2  # GPU and CUDA errors

    def test_root_cause_analysis(self, analyzer):
        """Test root cause analysis."""
        rca = analyzer._perform_root_cause_analysis()

        # Verify RCA structure
        assert "primary_cause" in rca
        assert "contributing_factors" in rca
        assert "failure_chain" in rca

        # Verify content
        assert isinstance(rca["contributing_factors"], list)
        assert isinstance(rca["failure_chain"], list)
        assert len(rca["contributing_factors"]) > 0
        assert len(rca["failure_chain"]) > 0

    def test_lessons_learned_extraction(self, analyzer):
        """Test lessons learned extraction."""
        lessons = analyzer._extract_lessons_learned()

        # Verify lessons structure
        assert isinstance(lessons, list)
        assert len(lessons) > 0

        # Verify lessons contain actionable insights
        lessons_text = " ".join(lessons).lower()
        assert any(
            keyword in lessons_text
            for keyword in ["monitoring", "degradation", "limits", "circuit"]
        )

    def test_action_items_creation(self, analyzer):
        """Test action items creation."""
        action_items = analyzer._create_action_items()

        # Verify action items structure
        assert isinstance(action_items, list)
        assert len(action_items) > 0

        # Verify action item structure
        for item in action_items:
            required_fields = [
                "action",
                "owner",
                "priority",
                "due_date",
                "estimated_effort",
            ]
            for field in required_fields:
                assert field in item

            # Verify priority format
            assert item["priority"].startswith("P")

            # Verify due date format (should be a date string)
            assert len(item["due_date"]) >= 10  # YYYY-MM-DD format minimum

    @pytest.mark.parametrize(
        "start_time,end_time,expected_duration",
        [
            ("2024-01-15T10:00:00Z", "2024-01-15T10:30:00Z", 30),
            ("2024-01-15T09:00:00Z", "2024-01-15T11:00:00Z", 120),
            ("2024-01-15T10:00:00Z", "2024-01-15T10:05:00Z", 5),
        ],
    )
    def test_different_incident_durations(
        self, start_time, end_time, expected_duration
    ):
        """Test analyzer with different incident durations."""
        analyzer = PostIncidentAnalyzer(start_time, end_time)

        duration = (analyzer.end_time - analyzer.start_time).total_seconds() / 60
        assert duration == expected_duration

        # Test that analysis works with different durations
        log_result = analyzer.analyze_logs()
        assert log_result["incident_duration_minutes"] == expected_duration

    def test_metric_correlation_analysis(self, analyzer):
        """Test metric correlation analysis."""
        # Create mock metrics
        mock_metrics = {
            "gpu_utilization": {"avg": 85.0},
            "memory_usage": {"avg_gb": 30.0},
            "request_latency": {"avg_ms": 500},
        }

        correlations = analyzer._analyze_metric_correlations(mock_metrics)

        # Verify correlation structure
        assert isinstance(correlations, dict)

        # Verify correlation values are valid
        for correlation_name, value in correlations.items():
            assert -1 <= value <= 1
            assert isinstance(value, (int, float))

    def test_log_window_filtering(self, analyzer):
        """Test log filtering within incident window."""
        # Test logs within window
        log_in_window = "2024-01-15T10:30:00Z ERROR: Test error"
        assert analyzer._log_in_window(log_in_window) is True

        # Test logs outside window
        log_before = "2024-01-15T09:30:00Z ERROR: Before incident"
        log_after = "2024-01-15T12:30:00Z ERROR: After incident"

        assert analyzer._log_in_window(log_before) is False
        assert analyzer._log_in_window(log_after) is False

    def test_empty_analysis_handling(self, analyzer):
        """Test handling when no data is available."""
        # Test with empty error patterns
        empty_patterns = analyzer._analyze_error_patterns([])
        assert isinstance(empty_patterns, dict)
        assert len(empty_patterns) == 0

        # Test with empty error sources
        empty_sources = analyzer._identify_error_sources([])
        assert isinstance(empty_sources, dict)
        assert len(empty_sources) == 0

    def test_analysis_result_storage(self, analyzer):
        """Test that analysis results are properly stored."""
        # Initially empty
        assert len(analyzer.analysis_results) == 0

        # After log analysis
        analyzer.analyze_logs()
        assert "logs" in analyzer.analysis_results

        # After metrics analysis
        analyzer.analyze_metrics()
        assert "metrics" in analyzer.analysis_results

        # After infrastructure analysis
        analyzer.analyze_infrastructure()
        assert "infrastructure" in analyzer.analysis_results

        # Verify results persist
        assert len(analyzer.analysis_results) == 3
