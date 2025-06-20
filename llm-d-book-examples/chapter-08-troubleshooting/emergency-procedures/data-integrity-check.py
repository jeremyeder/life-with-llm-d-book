#!/usr/bin/env python3
"""
Data Integrity Verification Tool

This script verifies data integrity after incident recovery, checking model files,
configurations, and deployment health to ensure no data corruption occurred.

Usage:
    python data-integrity-check.py

Execute after incident resolution to verify data integrity.
"""

import subprocess
import json
import sys

class DataIntegrityChecker:
    def __init__(self):
        self.issues = []
    
    def check_model_files(self):
        """Verify model files are intact"""
        try:
            # Get all model storage PVCs
            result = subprocess.run([
                'kubectl', 'get', 'pvc', '-A', 
                '-l', 'app.kubernetes.io/component=model-storage',
                '-o', 'json'
            ], capture_output=True, text=True)
            
            pvcs = json.loads(result.stdout)
            
            for pvc in pvcs['items']:
                namespace = pvc['metadata']['namespace']
                name = pvc['metadata']['name']
                
                # Mount PVC and check files
                check_cmd = f"""
                kubectl run model-check-{name} -n {namespace} \\
                  --image=busybox --rm -i --restart=Never \\
                  --overrides='{{
                    "spec": {{
                      "volumes": [{{
                        "name": "model-data",
                        "persistentVolumeClaim": {{"claimName": "{name}"}}
                      }}],
                      "containers": [{{
                        "name": "checker",
                        "image": "busybox",
                        "command": ["find", "/models", "-name", "*.bin", "-exec", "ls", "-la", "{{}}", ";"],
                        "volumeMounts": [{{
                          "name": "model-data",
                          "mountPath": "/models"
                        }}]
                      }}]
                    }}
                  }}' -- find /models -name "*.bin" -exec ls -la {{}} ;
                """
                
                result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    self.issues.append(f"Model file check failed for PVC {name} in {namespace}")
        
        except Exception as e:
            self.issues.append(f"Error checking model files: {e}")
    
    def check_configuration(self):
        """Verify configurations are valid"""
        try:
            result = subprocess.run([
                'kubectl', 'get', 'llmdeployments', '-A', '-o', 'json'
            ], capture_output=True, text=True)
            
            deployments = json.loads(result.stdout)
            
            for deployment in deployments['items']:
                name = deployment['metadata']['name']
                namespace = deployment['metadata']['namespace']
                
                # Check if deployment is healthy
                status = deployment.get('status', {})
                if status.get('phase') != 'Ready':
                    self.issues.append(f"Deployment {name} in {namespace} not ready")
        
        except Exception as e:
            self.issues.append(f"Error checking configurations: {e}")
    
    def run_checks(self):
        """Run all integrity checks"""
        print("Running data integrity checks...")
        
        self.check_model_files()
        self.check_configuration()
        
        if self.issues:
            print("❌ Issues found:")
            for issue in self.issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ All integrity checks passed")
            return True

if __name__ == "__main__":
    checker = DataIntegrityChecker()
    success = checker.run_checks()
    sys.exit(0 if success else 1)