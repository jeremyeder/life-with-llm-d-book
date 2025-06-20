#!/bin/bash
"""
Fix Image Pull Errors

This script provides comprehensive solutions for image pull errors including
missing secrets, incorrect credentials, and registry connectivity issues.

Usage:
    ./fix-image-pull-errors.sh <namespace> [registry_url] [username] [password] [email]

Example:
    ./fix-image-pull-errors.sh production registry.example.com myuser mypass user@example.com
"""

NAMESPACE=$1
REGISTRY_URL=${2:-"registry.example.com"}
USERNAME=${3:-""}
PASSWORD=${4:-""}
EMAIL=${5:-""}

if [ -z "$NAMESPACE" ]; then
  echo "Usage: $0 <namespace> [registry_url] [username] [password] [email]"
  exit 1
fi

echo "=== FIXING IMAGE PULL ERRORS ==="
echo "Namespace: $NAMESPACE"
echo "Registry: $REGISTRY_URL"

# 1. Check current pull secrets
echo "1. Checking current pull secrets..."
kubectl get secrets -n $NAMESPACE | grep docker

# 2. Create pull secret if credentials provided
if [ -n "$USERNAME" ] && [ -n "$PASSWORD" ] && [ -n "$EMAIL" ]; then
  echo "2. Creating docker registry secret..."
  kubectl create secret docker-registry model-registry \
    --docker-server=$REGISTRY_URL \
    --docker-username=$USERNAME \
    --docker-password=$PASSWORD \
    --docker-email=$EMAIL \
    -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
fi

# 3. Test registry access (if docker is available)
if command -v docker &> /dev/null && [ -n "$USERNAME" ] && [ -n "$PASSWORD" ]; then
  echo "3. Testing registry access..."
  echo "$PASSWORD" | docker login $REGISTRY_URL -u $USERNAME --password-stdin
  if [ $? -eq 0 ]; then
    echo "✅ Registry login successful"
  else
    echo "❌ Registry login failed"
  fi
fi

# 4. Add secret to existing deployments
echo "4. Adding pull secret to deployments..."
kubectl get llmdeployments -n $NAMESPACE -o json | jq -r '.items[].metadata.name' | while read deployment; do
  echo "Adding secret to deployment: $deployment"
  kubectl patch llmdeployment $deployment -n $NAMESPACE --type='json' \
    -p='[{"op": "add", "path": "/spec/imagePullSecrets", "value": [{"name": "model-registry"}]}]'
done

echo "=== IMAGE PULL ERROR FIX COMPLETED ==="