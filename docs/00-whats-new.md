---
title: What's New
description: Latest updates and features in llm-d releases
sidebar_position: 1
---

# What's New in llm-d

Stay current with the latest developments in the llm-d community. This section highlights new features, improvements, and important changes in recent releases.

:::info Current Release
**llm-d v1.0.22** - Latest stable release  
Released: July 8, 2025
:::

## Latest Release Highlights

### llm-d v1.0.x Series - Foundation Release

*Latest: v1.0.22 released July 8, 2025*

**🚀 Core Features Introduced:**

- Kubernetes-native distributed inference serving stack
- vLLM-optimized inference scheduler with intelligent routing
- Disaggregated serving architecture for improved efficiency
- Prefix caching system for performance optimization
- Comprehensive Helm-based deployment framework

**🔧 Key Components:**

- **llm-d-deployer**: Single Helm chart deployment solution
- **llm-d-inference-scheduler**: Request routing and load balancing
- **llm-d-kv-cache-manager**: KV cache optimization and coordination
- **llm-d-benchmark**: Performance testing and validation framework

**📊 Performance Achievements:**

- Multi-hardware accelerator support (NVIDIA H100, AMD MI300X)
- Intelligent cache-aware routing for improved hit rates
- Scalable architecture supporting production workloads
- Integration with existing Kubernetes infrastructure

**🤝 Community Partnerships:**

- Founded by CoreWeave, Google, IBM Research, NVIDIA, and Red Hat
- Support from AMD, Cisco, Hugging Face, Intel, Lambda, Mistral AI
- Academic collaboration with UC Berkeley and University of Chicago

## Breaking Changes

### v1.0.x → v2.0.0 (Upcoming)

*No breaking changes expected in v2.0.0*

## Migration Guides

### Installing llm-d v1.0.x

For new installations, follow the [Installation and Setup](./02-installation-setup.md) guide. The quickstart process includes:

```bash
# Clone the deployer
git clone https://github.com/llm-d/llm-d-deployer.git
cd llm-d-deployer/quickstart

# Set Hugging Face token
export HF_TOKEN="your-token"

# Run installer
./llmd-installer.sh
```

## Community Updates

### Recent Announcements

- **July 2025**: llm-d v1.0.22 released with stability improvements
- **May 2025**: Initial open source release with Apache 2.0 license
- **May 2025**: Community infrastructure established (Slack, GitHub, weekly standups)

### Upcoming Milestones

See [What's Next](./00-whats-next.md) for detailed roadmap information.

## Documentation Updates

### New Content

- Comprehensive troubleshooting guide with decision trees
- MLOps workflows and CI/CD integration examples
- Security and compliance framework documentation
- Performance optimization and cost management strategies

### Improved Content

- Enhanced installation procedures with multiple deployment options
- Expanded API reference with complete CRD documentation
- Updated configuration templates for common use cases

## Getting Help

### Community Resources

- **Slack**: Join the llm-d community for real-time discussions
- **GitHub**: Report issues and contribute to development
- **Weekly Standups**: Participate in contributor meetings
- **Documentation**: Comprehensive guides and reference materials

### Support Channels

- GitHub Issues for bug reports and feature requests
- Community Slack for questions and discussions
- Weekly contributor calls for development coordination

---

**Next Update**: This section will be refreshed monthly with each llm-d release.  
**Last Updated**: July 16, 2025  
**Next Release**: llm-d v2.0.0 expected July 17, 2025
