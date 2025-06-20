#!/bin/bash
"""
SOP-003: GPU Not Available Response

Standardized procedure for restoring GPU availability for model deployments.
This SOP addresses GPU resource detection and allocation issues.

Usage:
    ./SOP-003-gpu-not-available-response.sh

Execute when GPU resources are not available to deployments.
"""

# SOP-003: GPU Not Available Response
# Objective: Restore GPU availability for model deployments

echo "=== SOP-003: GPU NOT AVAILABLE RESPONSE ==="
echo "Operator: ${USER}"
echo "Start Time: $(date)"

# Step 1: Verify Problem Scope (0-3 minutes)
echo "Step 1: Verify Problem Scope"

# Check GPU resource availability
echo "Checking GPU resource availability:"
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity."nvidia\.com/gpu" | grep -v '<none>'

# Check device plugin status
echo "Checking device plugin status:"
kubectl get ds -n kube-system nvidia-device-plugin-daemonset
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds

# Step 2: Check Driver Status (3-6 minutes)
echo "Step 2: Check Driver Status"

# For each GPU node
GPU_NODES=$(kubectl get nodes -l nvidia.com/gpu=true -o jsonpath='{.items[*].metadata.name}')
for node in $GPU_NODES; do
  echo "=== Checking $node ==="
  kubectl debug node/$node -it --image=ubuntu -- chroot /host nvidia-smi
done

# Step 3: Restart Device Plugin (6-8 minutes)
echo "Step 3: Restart Device Plugin"

# Restart device plugin daemonset
echo "Restarting NVIDIA device plugin..."
kubectl rollout restart ds nvidia-device-plugin-daemonset -n kube-system
kubectl rollout status ds nvidia-device-plugin-daemonset -n kube-system --timeout=180s

# Step 4: Verify GPU Visibility (8-10 minutes)
echo "Step 4: Verify GPU Visibility"

# Check GPU resources are back
echo "Verifying GPU resources are available:"
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity."nvidia\.com/gpu"

# Test GPU allocation
echo "Testing GPU allocation:"
kubectl run gpu-test --image=nvidia/cuda:11.0-base --rm -it --restart=Never \
  --limits='nvidia.com/gpu=1' -- nvidia-smi

echo "=== SOP-003 COMPLETED ==="
echo "End Time: $(date)"

# Verification steps
echo "=== VERIFICATION ==="
echo "All GPU nodes showing capacity:"
kubectl get nodes -l nvidia.com/gpu=true -o custom-columns=NAME:.metadata.name,GPU:.status.capacity."nvidia\.com/gpu" | grep -v "0\|<none>"

echo "Device plugin pods running:"
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds | grep Running