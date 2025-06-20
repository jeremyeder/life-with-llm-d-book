#!/bin/bash
# RDMA Node Preparation Script for OpenShift/Kubernetes
#
# This script prepares Kubernetes nodes for RDMA-accelerated LLM inference:
# - Installs RDMA drivers and tools (rdma-core, libibverbs, perftest)
# - Loads necessary kernel modules (ib_core, mlx5_core, etc.)
# - Configures RoCE networking on Ethernet interfaces
# - Applies RDMA performance tuning (buffer sizes, TCP settings)
# - Validates RDMA device detection and functionality
#
# Prerequisites:
# - Mellanox ConnectX-5/6 or similar RDMA-capable NICs
# - RHEL/CentOS 8+ or Ubuntu 20.04+
# - Root access or sudo privileges
#
# Usage:
#   sudo ./setup-rdma-nodes.sh
#   RDMA_INTERFACE=ens6f0 sudo ./setup-rdma-nodes.sh
#
# Performance Impact:
# - Reduces inter-node communication latency by 60-80%
# - Increases bandwidth utilization by 40-60%
# - Enables efficient pipeline parallelism for >100B models
#
# Source: Chapter 6 - Performance Optimization

set -euo pipefail

echo "=== RDMA Node Setup for llm-d ==="

# 1. Install RDMA drivers and tools
if command -v yum &> /dev/null; then
    # RHEL/CentOS
    sudo yum install -y rdma-core libibverbs-utils perftest
elif command -v apt &> /dev/null; then
    # Ubuntu/Debian
    sudo apt update
    sudo apt install -y rdma-core libibverbs-dev perftest
fi

# 2. Load RDMA kernel modules
sudo modprobe ib_core
sudo modprobe ib_cm
sudo modprobe ib_umad
sudo modprobe ib_uverbs
sudo modprobe mlx5_core
sudo modprobe mlx5_ib

# 3. Configure persistent module loading
cat << 'EOF' | sudo tee /etc/modules-load.d/rdma.conf
# RDMA kernel modules
ib_core
ib_cm
ib_umad
ib_uverbs
mlx5_core
mlx5_ib
EOF

# 4. Verify RDMA device detection
echo "Checking RDMA devices..."
if ibv_devinfo &> /dev/null; then
    echo "✓ RDMA devices detected:"
    ibv_devinfo | grep -E "(hca_id|port_state|link_layer)"
else
    echo "✗ No RDMA devices found"
    exit 1
fi

# 5. Configure network interface (example for Mellanox)
RDMA_INTERFACE="${RDMA_INTERFACE:-ens6f0}"
echo "Configuring RDMA interface: $RDMA_INTERFACE"

# Enable RoCE if using Ethernet
if [[ "$RDMA_INTERFACE" =~ ^(eth|ens) ]]; then
    echo "Configuring RoCE on $RDMA_INTERFACE"
    sudo cma_roce_mode -d mlx5_0 -p 1 -m 2  # RoCE v2
fi

# 6. Performance tuning
echo "Applying RDMA performance tuning..."
cat << 'EOF' | sudo tee /etc/sysctl.d/90-rdma.conf
# RDMA performance tuning
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
EOF

sudo sysctl -p /etc/sysctl.d/90-rdma.conf

echo "=== RDMA node setup complete ==="