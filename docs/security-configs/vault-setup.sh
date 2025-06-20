#!/bin/bash
# HashiCorp Vault Setup Script for llm-d Integration
#
# This script configures HashiCorp Vault for secure secrets management
# with llm-d deployments including authentication, policies, and secret storage.
#
# Prerequisites:
# - Vault deployed and accessible
# - kubectl configured for target cluster
# - Vault CLI installed and configured
#
# Usage:
#   chmod +x vault-setup.sh
#   ./vault-setup.sh
#
# See: docs/07-security-compliance.md#llm-d-integration-patterns

set -euo pipefail

VAULT_ADDR="http://vault.llm-d-system.svc.cluster.local:8200"
VAULT_TOKEN="dev-token-123"

echo "Configuring Vault for llm-d integration..."

# Enable Kubernetes auth
vault auth enable kubernetes

# Configure Kubernetes auth
vault write auth/kubernetes/config \
    token_reviewer_jwt="$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)" \
    kubernetes_host="https://$KUBERNETES_PORT_443_TCP_ADDR:443" \
    kubernetes_ca_cert=@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt

# Create policy for llm-d secrets
vault policy write llm-d-policy - <<EOF
path "secret/data/llm-d/*" {
  capabilities = ["read", "list"]
}

path "secret/data/models/*" {
  capabilities = ["read", "list"]
}

path "auth/token/lookup-self" {
  capabilities = ["read"]
}
EOF

# Create role for llm-d service accounts
vault write auth/kubernetes/role/llm-d-role \
    bound_service_account_names=external-secrets-sa,inference-sa \
    bound_service_account_namespaces=llm-d-production,llm-d-staging \
    policies=llm-d-policy \
    ttl=24h

# Store model repository credentials
vault kv put secret/llm-d/model-repo \
    access_key_id="AKIA..." \
    secret_access_key="secret..." \
    region="us-west-2"

# Store API keys
vault kv put secret/llm-d/api-keys \
    huggingface_token="hf_..." \
    openai_api_key="sk-..." \
    anthropic_api_key="..."

# Store TLS certificates
vault kv put secret/llm-d/tls \
    certificate=@/certs/tls.crt \
    private_key=@/certs/tls.key

echo "Vault configuration complete"