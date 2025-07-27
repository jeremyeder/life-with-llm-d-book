# Troubleshooting Section Template v1

Use this template for creating systematic troubleshooting sections.

## Prompt Template

```
Write a troubleshooting section for [issue/problem] following the systematic approach established in Chapter 8:

1. **Problem Identification**
   - Clear symptom description
   - Common manifestations
   - Related error messages or logs

2. **Diagnostic Steps**
   - Step-by-step investigation process
   - Specific commands to run
   - What to look for in outputs

3. **Common Causes**
   - Most frequent root causes
   - Configuration issues
   - Resource constraints
   - Network/connectivity problems

4. **Resolution Steps**
   - Ordered solutions from simple to complex
   - Specific commands with expected outputs
   - Configuration changes needed

5. **Prevention**
   - Monitoring recommendations
   - Best practices to avoid recurrence
   - Related validation checks

Format Requirements:
- Use consistent command formatting with code blocks
- Include expected outputs where helpful
- Reference standard namespaces: production, staging, development, llm-d-system
- Follow resource specifications from docs/appendix/shared-config.md
- Link to related troubleshooting sections in other chapters

Template Variables:
- [issue/problem] = Specific technical issue being addressed
- [related-chapters] = Cross-references to related content
```

## Example Structure

```markdown
### GPU Memory Errors

**Problem**: Model fails to load with CUDA out of memory errors

**Symptoms**:
- Pod status shows OOMKilled
- Logs contain "RuntimeError: CUDA out of memory"

**Diagnostic Steps**:
1. Check GPU memory usage: `kubectl exec -n production pod-name -- nvidia-smi`
2. Review resource requests: `kubectl describe pod pod-name -n production`

**Common Causes**:
- Model size exceeds GPU memory
- Memory fragmentation
- Multiple models loaded simultaneously

**Resolution**:
1. Adjust memory requests in deployment
2. Enable memory optimization flags
3. Consider model sharding for large models

**Prevention**:
- Monitor GPU memory usage dashboards
- Follow sizing guidelines for model deployments
```

## Usage Notes

- Reference Chapter 8's systematic approach for consistency
- Include real kubectl commands and outputs
- Use standard namespace conventions