# Integration test runner for model deployments
# Comprehensive testing of deployed models including health checks,
# inference validation, and load handling tests

import argparse
import requests
import time
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

class ModelIntegrationTester:
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url
        self.timeout = timeout
        
    def test_health_endpoint(self, model_name: str) -> dict:
        """Test model health endpoint"""
        url = f"{self.base_url}/{model_name}/health"
        
        try:
            response = requests.get(url, timeout=10)
            return {
                'test': 'health_check',
                'status': 'passed' if response.status_code == 200 else 'failed',
                'response_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000
            }
        except Exception as e:
            return {
                'test': 'health_check',
                'status': 'failed',
                'error': str(e)
            }
    
    def test_inference_endpoint(self, model_name: str) -> dict:
        """Test model inference endpoint"""
        url = f"{self.base_url}/{model_name}/v1/completions"
        
        payload = {
            "prompt": "The quick brown fox",
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'test': 'inference',
                    'status': 'passed',
                    'response_time_ms': (end_time - start_time) * 1000,
                    'output_length': len(result.get('choices', [{}])[0].get('text', '')),
                    'tokens_generated': result.get('usage', {}).get('completion_tokens', 0)
                }
            else:
                return {
                    'test': 'inference',
                    'status': 'failed',
                    'response_code': response.status_code,
                    'error': response.text
                }
                
        except Exception as e:
            return {
                'test': 'inference',
                'status': 'failed',
                'error': str(e)
            }
    
    def test_load_handling(self, model_name: str, concurrent_requests: int = 5) -> dict:
        """Test model load handling"""
        url = f"{self.base_url}/{model_name}/v1/completions"
        
        payload = {
            "prompt": "Generate a short response",
            "max_tokens": 5,
            "temperature": 0.1
        }
        
        def send_request():
            try:
                start_time = time.time()
                response = requests.post(url, json=payload, timeout=30)
                end_time = time.time()
                
                return {
                    'status_code': response.status_code,
                    'response_time_ms': (end_time - start_time) * 1000,
                    'success': response.status_code == 200
                }
            except Exception as e:
                return {
                    'status_code': 0,
                    'response_time_ms': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # Send concurrent requests
        results = []
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(send_request) for _ in range(concurrent_requests)]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        # Analyze results
        successful_requests = sum(1 for r in results if r['success'])
        avg_response_time = sum(r['response_time_ms'] for r in results if r['success']) / max(successful_requests, 1)
        
        return {
            'test': 'load_handling',
            'status': 'passed' if successful_requests >= concurrent_requests * 0.8 else 'failed',
            'total_requests': concurrent_requests,
            'successful_requests': successful_requests,
            'success_rate': successful_requests / concurrent_requests,
            'avg_response_time_ms': avg_response_time
        }
    
    def run_integration_tests(self, models: list) -> dict:
        """Run complete integration test suite"""
        all_results = {}
        
        for model_name in models:
            print(f"🧪 Testing {model_name}...")
            
            model_results = []
            
            # Health check
            model_results.append(self.test_health_endpoint(model_name))
            
            # Basic inference
            model_results.append(self.test_inference_endpoint(model_name))
            
            # Load handling
            model_results.append(self.test_load_handling(model_name))
            
            all_results[model_name] = model_results
            
            # Print results
            for result in model_results:
                status_emoji = "✅" if result['status'] == 'passed' else "❌"
                print(f"  {status_emoji} {result['test']}: {result['status']}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', required=True, choices=['staging', 'production'])
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--models', nargs='+', help='Specific models to test')
    
    args = parser.parse_args()
    
    # Load environment configuration
    with open(f"config/{args.environment}.yaml", 'r') as f:
        env_config = yaml.safe_load(f)
    
    # Determine base URL
    if args.environment == 'staging':
        base_url = "http://llm-gateway.staging.svc.cluster.local"
    else:
        base_url = "http://llm-gateway.production.svc.cluster.local"
    
    # Determine models to test
    if args.models:
        models_to_test = args.models
    else:
        # Auto-discover deployed models
        models_to_test = ["llama-3.1-7b", "llama-3.1-13b"]  # Default models
    
    # Run tests
    tester = ModelIntegrationTester(base_url, args.timeout)
    results = tester.run_integration_tests(models_to_test)
    
    # Summary
    total_tests = sum(len(model_results) for model_results in results.values())
    passed_tests = sum(
        1 for model_results in results.values() 
        for result in model_results 
        if result['status'] == 'passed'
    )
    
    print(f"\n📊 Integration Test Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {passed_tests/total_tests:.1%}")
    
    if passed_tests == total_tests:
        print("🎉 All integration tests passed!")
        exit(0)
    else:
        print("❌ Some integration tests failed!")
        exit(1)

if __name__ == "__main__":
    main()