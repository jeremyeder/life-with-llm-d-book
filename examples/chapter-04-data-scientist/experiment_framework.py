class ExperimentManager:
    def __init__(self, experiment_name, client):
        self.experiment_name = experiment_name
        self.client = client
        self.results = []
        self.start_time = datetime.now()

    def run_experiment(self, test_cases, model_configs):
        """Run experiments across multiple configurations"""
        for config_name, config in model_configs.items():
            print(f"Running experiment: {config_name}")

            config_results = []
            for test_case in test_cases:
                result = self._run_single_test(test_case, config)
                config_results.append(result)

            self.results.append(
                {
                    "config_name": config_name,
                    "config": config,
                    "results": config_results,
                    "avg_latency": self._calculate_avg_latency(config_results),
                    "avg_quality": self._calculate_avg_quality(config_results),
                }
            )

    def _run_single_test(self, test_case, config):
        """Execute individual test case"""
        start_time = datetime.now()

        response = self.client.chat_completion(messages=test_case["messages"], **config)

        end_time = datetime.now()
        latency = (end_time - start_time).total_seconds()

        return {
            "test_case": test_case,
            "response": response,
            "latency": latency,
            "tokens_generated": response.get("usage", {}).get("completion_tokens", 0),
            "timestamp": start_time.isoformat(),
        }

    def _calculate_avg_latency(self, results):
        return sum(r["latency"] for r in results) / len(results)

    def _calculate_avg_quality(self, results):
        # Implement quality scoring based on your metrics
        pass

    def export_results(self, filename):
        """Export experiment results to CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

    def visualize_results(self):
        """Create visualizations of experiment results"""
        configs = [r["config_name"] for r in self.results]
        latencies = [r["avg_latency"] for r in self.results]

        plt.figure(figsize=(10, 6))
        plt.bar(configs, latencies)
        plt.title(f"Experiment: {self.experiment_name} - Average Latency")
        plt.xlabel("Configuration")
        plt.ylabel("Latency (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# Example usage
experiment = ExperimentManager("temperature_comparison", client)

# Define test cases
test_cases = [
    {
        "name": "creative_writing",
        "messages": [
            {
                "role": "user",
                "content": "Write a creative short story about a robot learning to paint.",
            }
        ],
    },
    {
        "name": "technical_explanation",
        "messages": [
            {
                "role": "user",
                "content": "Explain how neural networks work in simple terms.",
            }
        ],
    },
    {
        "name": "code_generation",
        "messages": [
            {
                "role": "user",
                "content": "Write a Python function to calculate the Fibonacci sequence.",
            }
        ],
    },
]

# Define model configurations to test
model_configs = {
    "low_temp": {"temperature": 0.3, "max_tokens": 500},
    "medium_temp": {"temperature": 0.7, "max_tokens": 500},
    "high_temp": {"temperature": 1.0, "max_tokens": 500},
}

# Run experiments
experiment.run_experiment(test_cases, model_configs)
experiment.visualize_results()
experiment.export_results(
    f"experiment_{experiment.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
