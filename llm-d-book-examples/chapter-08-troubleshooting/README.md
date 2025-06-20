# Chapter 8: Troubleshooting - Code Examples

This directory contains comprehensive troubleshooting tools, scripts, and procedures extracted from Chapter 8 of the "Life with llm-d" book. These examples provide a complete troubleshooting toolkit for llm-d deployments.

## Directory Structure

### 📊 Diagnostic Tools (`diagnostic-tools/`)

Essential tools for system analysis and problem identification:

- **`gpu-memory-profile.py`** - Analyzes GPU memory usage patterns and allocation
- **`memory-profiler.py`** - Profiles system and model memory consumption
- **`llm-d-health-check.sh`** - Comprehensive health check for deployments
- **`collect-diagnostics.py`** - Automated diagnostic data collection tool

### 🚀 Performance Troubleshooting (`performance-troubleshooting/`)

Scripts for optimizing and troubleshooting performance issues:

- **`inference-optimizer.py`** - Model inference optimization with profiling
- **`async-request-handler.py`** - High-throughput async request processing
- **`gpu-memory-optimizer.py`** - GPU memory optimization utilities
- **`kv-cache-manager.py`** - Efficient KV cache management for attention
- **`load-test.py`** - Comprehensive load testing framework

### 🆘 Emergency Procedures (`emergency-procedures/`)

Critical incident response scripts for immediate action:

- **`P0-service-outage.sh`** - Complete service outage response
- **`gpu-cluster-failure.sh`** - GPU cluster failure mitigation
- **`high-error-rate.sh`** - High error rate incident response
- **`memory-pressure.sh`** - Memory pressure emergency procedures
- **`service-recovery-checklist.sh`** - Post-incident recovery verification
- **`data-integrity-check.py`** - Data integrity verification tool
- **`post-incident-analysis.py`** - Automated post-incident analysis

### 📋 Standard Operating Procedures (`standard-operating-procedures/`)

Standardized SOPs for consistent operational response:

- **`SOP-001-service-outage-response.sh`** - Service outage SOP
- **`SOP-002-high-memory-usage-response.sh`** - Memory pressure SOP
- **`SOP-003-gpu-not-available-response.sh`** - GPU availability SOP
- **`SOP-004-model-deployment-failure-response.sh`** - Deployment failure SOP

### 🔧 Common Issues (`common-issues/`)

Fix procedures for frequently encountered problems:

- **`fix-image-pull-errors.sh`** - Resolve image pull authentication issues
- **`fix-cuda-out-of-memory.sh`** - Handle CUDA memory exhaustion

## Quick Start Guide

### 1. Diagnostic Assessment

Start with a comprehensive health check:

```bash
# Check overall system health
./diagnostic-tools/llm-d-health-check.sh <namespace> <deployment>

# Collect detailed diagnostics
python ./diagnostic-tools/collect-diagnostics.py <namespace> <deployment>

# Profile GPU memory usage (run inside pod)
python ./diagnostic-tools/gpu-memory-profile.py
```

### 2. Performance Analysis

For performance issues:

```bash
# Run load test
python ./performance-troubleshooting/load-test.py

# Optimize GPU memory
python ./performance-troubleshooting/gpu-memory-optimizer.py

# Profile inference performance
python ./performance-troubleshooting/inference-optimizer.py
```

### 3. Emergency Response

For critical incidents:

```bash
# P0 Service Outage
./emergency-procedures/P0-service-outage.sh

# GPU Cluster Failure
./emergency-procedures/gpu-cluster-failure.sh

# Memory Pressure
./emergency-procedures/memory-pressure.sh
```

### 4. Standard Procedures

Follow SOPs for consistent response:

```bash
# Service outage response
./standard-operating-procedures/SOP-001-service-outage-response.sh

# Memory issues
./standard-operating-procedures/SOP-002-high-memory-usage-response.sh

# GPU problems
./standard-operating-procedures/SOP-003-gpu-not-available-response.sh
```

## Tool Categories

### 🔍 Diagnostic Tools
- **Purpose**: Identify and analyze problems
- **When to use**: First step in troubleshooting
- **Output**: Diagnostic reports and metrics

### ⚡ Performance Tools
- **Purpose**: Optimize and benchmark performance
- **When to use**: Performance degradation or optimization
- **Output**: Performance metrics and optimized configurations

### 🚨 Emergency Tools
- **Purpose**: Immediate incident response
- **When to use**: Critical outages and emergencies
- **Output**: Service restoration and impact mitigation

### 📝 Standard Procedures
- **Purpose**: Consistent operational response
- **When to use**: Routine operational issues
- **Output**: Systematic problem resolution

## Prerequisites

### Required Tools
- `kubectl` with cluster access
- Python 3.8+ with required packages
- Docker (for some registry operations)
- Basic shell utilities (`jq`, `curl`, `grep`, etc.)

### Python Dependencies
```bash
pip install torch psutil GPUtil aiohttp numpy
```

### Kubernetes Permissions
Ensure your service account has appropriate RBAC permissions:
- Read/write access to llm-d resources
- Pod exec and log access
- Node describe access
- Event reading permissions

## Usage Patterns

### Incident Response Workflow

1. **Detection** → Run diagnostic tools
2. **Assessment** → Use performance analysis
3. **Response** → Execute emergency procedures
4. **Resolution** → Follow standard procedures
5. **Recovery** → Verify with diagnostic tools

### Performance Optimization Workflow

1. **Baseline** → Establish performance metrics
2. **Profile** → Identify bottlenecks
3. **Optimize** → Apply performance improvements
4. **Test** → Validate with load testing
5. **Monitor** → Continuous performance tracking

## Best Practices

### 🎯 Diagnostic Best Practices
- Always collect diagnostics before making changes
- Save diagnostic outputs for post-incident analysis
- Use structured logging for better analysis
- Monitor trends, not just point-in-time metrics

### ⚡ Performance Best Practices
- Start with quick wins (quantization, batching)
- Profile before optimizing
- Test changes in non-production first
- Monitor impact of optimizations

### 🚨 Emergency Best Practices
- Follow escalation procedures
- Document all actions taken
- Communicate status regularly
- Preserve evidence for post-incident review

### 📋 Operational Best Practices
- Regular SOP training and drills
- Keep procedures updated
- Use automation where possible
- Learn from each incident

## Customization

### Adapting Scripts
Most scripts can be customized by:
- Modifying environment variables
- Adjusting thresholds and limits
- Adding organization-specific logic
- Integrating with existing tools

### Adding New Tools
When adding new troubleshooting tools:
1. Follow the established directory structure
2. Include comprehensive header documentation
3. Add usage examples
4. Update this README
5. Test thoroughly in non-production

## Support and Maintenance

### Regular Updates
- Review and update scripts quarterly
- Validate against new llm-d versions
- Incorporate lessons learned from incidents
- Test all procedures regularly

### Documentation
- Keep inline documentation current
- Update examples with real scenarios
- Document any customizations
- Share knowledge across team

## Related Resources

- **Chapter 8**: Full troubleshooting documentation
- **Monitoring Setup**: Observability configuration
- **Performance Tuning**: Optimization guides
- **Incident Response**: Broader incident management

---

**Note**: These tools are provided as examples and should be adapted to your specific environment and requirements. Always test in non-production environments before using in production.