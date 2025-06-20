# Model configuration validation script
# Validates model YAML configuration against schema and business rules
# Used in CI/CD pipelines to ensure model configs are correct before deployment

import yaml
import jsonschema
import sys
import os

# Model configuration schema
MODEL_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["name", "version", "architecture", "resources"],
    "properties": {
        "name": {"type": "string", "pattern": "^[a-z0-9-]+$"},
        "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
        "architecture": {"type": "string"},
        "description": {"type": "string"},
        "resources": {
            "type": "object",
            "required": ["memory_gb", "gpu_count"],
            "properties": {
                "memory_gb": {"type": "number", "minimum": 1},
                "gpu_count": {"type": "integer", "minimum": 1},
                "gpu_type": {"type": "string"}
            }
        },
        "serving": {
            "type": "object",
            "properties": {
                "max_batch_size": {"type": "integer", "minimum": 1},
                "max_sequence_length": {"type": "integer", "minimum": 1},
                "protocol": {"type": "string", "enum": ["http", "grpc"]}
            }
        },
        "source": {
            "type": "object",
            "required": ["type", "uri"],
            "properties": {
                "type": {"type": "string", "enum": ["s3", "huggingface", "local"]},
                "uri": {"type": "string"}
            }
        }
    }
}

def validate_model_config(config_path: str) -> bool:
    """Validate model configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate against schema
        jsonschema.validate(config, MODEL_CONFIG_SCHEMA)
        
        # Additional business logic validation
        if config['resources']['memory_gb'] < config['resources']['gpu_count'] * 8:
            raise ValueError(f"Insufficient memory for {config['resources']['gpu_count']} GPUs")
        
        # Validate model name matches directory structure
        expected_model_name = os.path.basename(os.path.dirname(config_path))
        if config['name'] != expected_model_name:
            raise ValueError(f"Model name {config['name']} doesn't match directory {expected_model_name}")
        
        print(f"✅ Configuration valid: {config['name']} v{config['version']}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate-model-config.py <path-to-model-dir>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    config_path = os.path.join(model_dir, "model.yaml")
    
    if not os.path.exists(config_path):
        print(f"❌ Model configuration not found: {config_path}")
        sys.exit(1)
    
    if validate_model_config(config_path):
        sys.exit(0)
    else:
        sys.exit(1)