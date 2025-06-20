#!/usr/bin/env python3
"""
Automated Diagnostics Collection Tool

This script automatically collects comprehensive diagnostic information
from llm-d deployments including deployment info, pod status, logs, and metrics.

Usage:
    python collect-diagnostics.py <namespace> <deployment>

Example:
    python collect-diagnostics.py production gpt-model
"""

import subprocess
import json
import datetime
import os
import sys

class LLMDiagnostics:
    def __init__(self, namespace, deployment):
        self.namespace = namespace
        self.deployment = deployment
        self.timestamp = datetime.datetime.now().isoformat()
        self.report = {
            "timestamp": self.timestamp,
            "namespace": namespace,
            "deployment": deployment,
            "diagnostics": {}
        }
    
    def run_command(self, cmd):
        """Execute command and return output"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def collect_deployment_info(self):
        """Collect deployment information"""
        cmd = f"kubectl get llmdeployment {self.deployment} -n {self.namespace} -o json"
        result = self.run_command(cmd)
        if result["success"]:
            self.report["diagnostics"]["deployment"] = json.loads(result["stdout"])
    
    def collect_pod_info(self):
        """Collect pod information"""
        cmd = f"kubectl get pods -n {self.namespace} -l app={self.deployment} -o json"
        result = self.run_command(cmd)
        if result["success"]:
            self.report["diagnostics"]["pods"] = json.loads(result["stdout"])
    
    def collect_logs(self, lines=100):
        """Collect recent logs"""
        cmd = f"kubectl logs -n {self.namespace} -l app={self.deployment} --tail={lines}"
        result = self.run_command(cmd)
        if result["success"]:
            self.report["diagnostics"]["logs"] = result["stdout"].split('\n')
    
    def collect_metrics(self):
        """Collect current metrics"""
        cmd = f"kubectl exec -n {self.namespace} -l app={self.deployment} -- curl -s localhost:9090/metrics"
        result = self.run_command(cmd)
        if result["success"]:
            self.report["diagnostics"]["metrics"] = result["stdout"]
    
    def save_report(self):
        """Save diagnostic report"""
        filename = f"llm-d-diagnostics-{self.deployment}-{self.timestamp.replace(':', '-')}.json"
        with open(filename, 'w') as f:
            json.dump(self.report, f, indent=2)
        print(f"Diagnostic report saved to: {filename}")
        return filename

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python collect_diagnostics.py <namespace> <deployment>")
        sys.exit(1)
    
    diag = LLMDiagnostics(sys.argv[1], sys.argv[2])
    diag.collect_deployment_info()
    diag.collect_pod_info()
    diag.collect_logs()
    diag.collect_metrics()
    diag.save_report()