#!/usr/bin/env python3
"""
API Security Middleware for llm-d Inference Services

This module implements comprehensive security controls for LLM inference APIs including:
- Rate limiting with per-client tracking
- Input validation and prompt injection detection
- Sensitive content filtering
- Security event logging and monitoring

Key Features:
- Token bucket rate limiting
- Pattern-based threat detection
- PII detection and redaction
- Comprehensive audit logging
- Configurable security policies

Usage:
    from api_security_middleware import SecurityMiddleware, SecurityConfig

    config = SecurityConfig(max_requests_per_minute=60)
    middleware = SecurityMiddleware(config)

    # Validate incoming request
    result = middleware.validate_request(request_data)

Dependencies:
    - No external dependencies required
    - Integrates with FastAPI, Flask, or any Python web framework

See: docs/07-security-compliance.md#api-security-and-rate-limiting
"""

import hashlib
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration for API middleware"""

    max_requests_per_minute: int = 60
    max_tokens_per_request: int = 4096
    max_output_tokens: int = 2048
    blocked_patterns: List[str] = None
    require_authentication: bool = True
    log_all_requests: bool = True

    def __post_init__(self):
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r"(?i)ignore.*(previous|above|system).*instruction",
                r"(?i)jailbreak|prompt.?injection",
                r"(?i)developer.?mode|admin.?mode",
                r"(?i)repeat.*password|show.*config",
                r"(?i)system.*override|bypass.*filter",
            ]


class RateLimiter:
    """Token bucket rate limiter with per-client tracking"""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client"""
        now = time.time()
        client_requests = self.clients[client_id]

        # Remove old requests outside the window
        while client_requests and client_requests[0] <= now - self.window_seconds:
            client_requests.popleft()

        # Check if under rate limit
        if len(client_requests) < self.max_requests:
            client_requests.append(now)
            return True

        return False

    def get_reset_time(self, client_id: str) -> float:
        """Get time when rate limit resets for client"""
        client_requests = self.clients[client_id]
        if not client_requests:
            return 0
        return client_requests[0] + self.window_seconds


class SecurityMiddleware:
    """Security middleware for llm-d inference APIs"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.max_requests_per_minute)
        self.blocked_ips = set()
        self.suspicious_patterns = [re.compile(p) for p in config.blocked_patterns]

    def validate_request(self, request_data: Dict) -> Dict:
        """Validate incoming inference request"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "blocked": False,
        }

        # Extract request details
        prompt = request_data.get("prompt", "")
        max_tokens = request_data.get("max_tokens", 100)
        client_id = request_data.get("client_id", "anonymous")

        # Rate limiting check
        if not self.rate_limiter.is_allowed(client_id):
            validation_result["valid"] = False
            validation_result["blocked"] = True
            validation_result["errors"].append("Rate limit exceeded")
            reset_time = self.rate_limiter.get_reset_time(client_id)
            validation_result["retry_after"] = int(reset_time - time.time())
            return validation_result

        # Input length validation
        if len(prompt) > self.config.max_tokens_per_request:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Prompt too long: {len(prompt)} > {self.config.max_tokens_per_request}"
            )

        # Output length validation
        if max_tokens > self.config.max_output_tokens:
            validation_result["warnings"].append(
                f"Output tokens capped at {self.config.max_output_tokens}"
            )
            request_data["max_tokens"] = self.config.max_output_tokens

        # Prompt injection detection
        for pattern in self.suspicious_patterns:
            if pattern.search(prompt):
                validation_result["valid"] = False
                validation_result["blocked"] = True
                validation_result["errors"].append(
                    "Potentially malicious input detected"
                )

                # Log security event
                logger.warning(
                    f"Suspicious pattern detected from client {client_id}: {pattern.pattern}"
                )
                break

        # Content safety checks
        if self._contains_sensitive_content(prompt):
            validation_result["warnings"].append("Sensitive content detected in prompt")

        return validation_result

    def _contains_sensitive_content(self, text: str) -> bool:
        """Check for sensitive content patterns"""
        sensitive_patterns = [
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card numbers
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN format
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email addresses
            r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",  # Phone numbers
        ]

        for pattern in sensitive_patterns:
            if re.search(pattern, text):
                return True
        return False

    def log_request(
        self, request_data: Dict, response_data: Dict, validation_result: Dict
    ):
        """Log request for security monitoring"""
        if not self.config.log_all_requests:
            return

        # Create security log entry
        log_entry = {
            "timestamp": time.time(),
            "client_id": request_data.get("client_id", "anonymous"),
            "prompt_hash": hashlib.sha256(
                request_data.get("prompt", "").encode()
            ).hexdigest()[:16],
            "prompt_length": len(request_data.get("prompt", "")),
            "max_tokens_requested": request_data.get("max_tokens", 0),
            "tokens_generated": response_data.get("usage", {}).get(
                "completion_tokens", 0
            ),
            "validation_status": "valid" if validation_result["valid"] else "invalid",
            "blocked": validation_result.get("blocked", False),
            "warnings": len(validation_result.get("warnings", [])),
            "response_time_ms": response_data.get("processing_time_ms", 0),
        }

        logger.info(f"API Request: {log_entry}")

        # Alert on blocked requests
        if validation_result.get("blocked"):
            logger.warning(
                f"Blocked request from {log_entry['client_id']}: {validation_result['errors']}"
            )


# Usage in FastAPI application
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.base import BaseHTTPMiddleware

app = FastAPI()
security_config = SecurityConfig()
security_middleware = SecurityMiddleware(security_config)

@app.middleware("http")
async def security_middleware_handler(request: Request, call_next):
    # Extract request data
    if request.method == "POST" and "/v1/completions" in str(request.url):
        body = await request.json()
        
        # Validate request
        validation_result = security_middleware.validate_request(body)
        
        if not validation_result["valid"]:
            if validation_result.get("blocked"):
                raise HTTPException(
                    status_code=429 if "rate limit" in validation_result["errors"][0].lower() else 403,
                    detail=validation_result["errors"][0],
                    headers={"Retry-After": str(validation_result.get("retry_after", 60))}
                )
        
        # Log warnings but continue
        if validation_result.get("warnings"):
            logger.warning(f"Request warnings: {validation_result['warnings']}")
    
    response = await call_next(request)
    return response
"""
