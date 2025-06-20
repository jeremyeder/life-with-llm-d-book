# Kubeflow pipeline trigger script
# Triggers model registration pipelines for changed models
# Integrates with GitHub Actions to automate model deployment

import kfp
import argparse
import json
import os
import yaml
from typing import List

def trigger_model_registration(
    endpoint: str,
    token: str,
    models_changed: List[str]
) -> None:
    """Trigger Kubeflow pipeline for changed models"""
    
    # Initialize Kubeflow client
    client = kfp.Client(
        host=endpoint,
        existing_token=token
    )
    
    for model_path in models_changed:
        if not model_path.startswith('models/'):
            continue
            
        model_dir = os.path.dirname(model_path)
        config_path = os.path.join(model_dir, "model.yaml")
        
        if not os.path.exists(config_path):
            print(f"⚠️  Skipping {model_path} - no model.yaml found")
            continue
        
        # Load model configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Prepare pipeline arguments
        pipeline_args = {
            'model_name': config['name'],
            'model_version': config['version'],
            'model_uri': config['source']['uri'],
            'architecture': config['architecture'],
            'memory_gb': config['resources']['memory_gb'],
            'gpu_count': config['resources']['gpu_count']
        }
        
        # Submit pipeline run
        try:
            run = client.create_run_from_pipeline_func(
                pipeline_func=model_registration_pipeline,
                arguments=pipeline_args,
                run_name=f"register-{config['name']}-{config['version']}-{os.environ.get('GITHUB_RUN_ID', 'manual')}"
            )
            
            print(f"✅ Triggered registration pipeline for {config['name']} v{config['version']}")
            print(f"   Run ID: {run.run_id}")
            print(f"   Run URL: {endpoint}/#/runs/details/{run.run_id}")
            
        except Exception as e:
            print(f"❌ Failed to trigger pipeline for {config['name']}: {e}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', required=True, help='Kubeflow endpoint')
    parser.add_argument('--token', required=True, help='Kubeflow auth token')
    parser.add_argument('--models-changed', required=True, help='JSON list of changed model files')
    
    args = parser.parse_args()
    
    models_changed = json.loads(args.models_changed)
    trigger_model_registration(args.endpoint, args.token, models_changed)